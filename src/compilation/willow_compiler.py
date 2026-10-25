"""Compiler that lowers abstract Dicke state prototypes onto Google Willow.

Handles:
1. Subgraph selection on Willow105 using the measured error-rate dictionaries.
2. Decomposition of multi-qubit (>=3Q) gates into 1Q + 2Q unitaries.
3. Shortest-path SWAP routing if any 2-qubit gate acts on non-adjacent qubits.
4. Optimal KAK decomposition of arbitrary 2-qubit unitaries into at most 2-3 CZ
   gates and PhasedXZGates (`two_qubit_matrix_to_cz_operations`).
5. Merging of consecutive single-qubit gates into single `cirq.PhasedXZGate` ops.
6. Optional dynamical decoupling (X-X spin echo) on idle qubit slots.
"""

from typing import Callable, Optional, Sequence
import networkx as nx
import numpy as np
import cirq

from src.willow_utils import (
    build_connectivity_graph,
    find_best_connected_subgraph,
    find_best_linear_chain,
    lookup_coupler_error,
)


def _to_phased_xz(op: cirq.Operation, atol: float = 1e-12) -> Optional[cirq.Operation]:
    """Convert any 1-qubit unitary operation into a single cirq.PhasedXZGate."""
    if isinstance(op.gate, cirq.ResetChannel):
        return op
    u = cirq.unitary(op, None)
    if u is None:
        raise ValueError(f"Unsupported non-unitary 1Q operation: {op}")
    gate = cirq.PhasedXZGate.from_matrix(u)
    if np.isclose(gate.x_exponent % 2.0, 0.0, atol=atol) and np.isclose(
        gate.z_exponent % 2.0, 0.0, atol=atol
    ):
        return None
    return gate.on(op.qubits[0])


def _decompose_2q_op_to_cz_and_phxz(
    op: cirq.Operation, atol: float = 1e-12
) -> list[cirq.Operation]:
    """Decompose an arbitrary adjacent 2-qubit unitary into CZ and PhasedXZGate ops."""
    q0, q1 = op.qubits
    if isinstance(op.gate, cirq.CZPowGate) and np.isclose(op.gate.exponent, 1.0, atol=atol) and np.isclose(
        op.gate.global_shift, 0.0, atol=atol
    ):
        return [cirq.CZ(q0, q1)]

    u = cirq.unitary(op, None)
    if u is None:
        raise ValueError(f"Unsupported non-unitary 2Q operation: {op}")

    raw_ops = cirq.two_qubit_matrix_to_cz_operations(
        q0, q1, u, allow_partial_czs=False, atol=atol, clean_operations=True
    )
    native_ops: list[cirq.Operation] = []
    for sub_op in raw_ops:
        if len(sub_op.qubits) == 2:
            native_ops.append(cirq.CZ(sub_op.qubits[0], sub_op.qubits[1]))
        elif len(sub_op.qubits) == 1:
            pxz = _to_phased_xz(sub_op, atol=atol)
            if pxz is not None:
                native_ops.append(pxz)
    return native_ops


def _route_non_adjacent_2q_op(
    op: cirq.Operation,
    graph: nx.Graph,
) -> list[cirq.Operation]:
    """Route a 2Q operation across a shortest path of working couplers via SWAPs."""
    q0, q1 = op.qubits
    if graph.has_edge(q0, q1):
        return [op]

    if not nx.has_path(graph, q0, q1):
        raise ValueError(f"No working coupler path between {q0} and {q1}")

    path = nx.shortest_path(graph, source=q0, target=q1, weight="weight")
    forward_swaps = [cirq.SWAP(path[i], path[i + 1]) for i in range(len(path) - 2)]
    routed_op = op.gate.on(path[-2], path[-1])
    backward_swaps = list(reversed(forward_swaps))
    return [*forward_swaps, routed_op, *backward_swaps]


def compile_circuit_to_willow_gateset(
    circuit: cirq.Circuit,
    working_graph: Optional[nx.Graph] = None,
    atol: float = 1e-12,
) -> cirq.Circuit:
    """Lower a circuit on GridQubits strictly to cirq.CZ and cirq.PhasedXZGate."""
    # Step 1: Decompose any 3+ qubit operations into 1Q and 2Q operations
    two_q_ops: list[cirq.Operation] = []
    for op in circuit.all_operations():
        if len(op.qubits) <= 2:
            two_q_ops.append(op)
        else:
            decomposed = cirq.decompose(
                op,
                keep=lambda o: len(o.qubits) <= 2 and cirq.has_unitary(o),
            )
            two_q_ops.extend(decomposed)

    # Merge consecutive 2-qubit unitaries on the same pair before routing
    pre_merged = cirq.merge_k_qubit_unitaries(
        cirq.Circuit(two_q_ops),
        k=2,
        rewriter=lambda op: cirq.MatrixGate(cirq.unitary(op)).on(*op.qubits),
    )

    # Step 2: Route non-adjacent 2Q operations if working_graph is provided
    routed_ops: list[cirq.Operation] = []
    for op in pre_merged.all_operations():
        if len(op.qubits) == 2 and working_graph is not None:
            routed_ops.extend(_route_non_adjacent_2q_op(op, working_graph))
        else:
            routed_ops.append(op)

    # Merge consecutive 2-qubit unitaries on adjacent pairs after SWAP routing
    # (automatically cancels back-to-back inverse SWAPs and fuses adjacent 2Q blocks)
    post_merged = cirq.merge_k_qubit_unitaries(
        cirq.Circuit(routed_ops),
        k=2,
        rewriter=lambda op: cirq.MatrixGate(cirq.unitary(op)).on(*op.qubits),
    )

    # Step 3: Decompose all 2Q unitaries via KAK into CZ + PhasedXZGate
    raw_native_ops: list[cirq.Operation] = []
    for op in post_merged.all_operations():
        if len(op.qubits) == 2:
            raw_native_ops.extend(_decompose_2q_op_to_cz_and_phxz(op, atol=atol))
        elif len(op.qubits) == 1:
            pxz = _to_phased_xz(op, atol=atol)
            if pxz is not None:
                raw_native_ops.append(pxz)

    intermediate = cirq.Circuit(raw_native_ops)

    # Step 4: Merge consecutive 1-qubit gates into single PhasedXZGate operations
    merged = cirq.merge_single_qubit_gates_to_phxz(intermediate, atol=atol)

    # Step 5: Sanitize all moments so only CZ, PhasedXZGate, and ResetChannel remain
    clean_moments: list[cirq.Moment] = []
    for moment in merged:
        m_ops: list[cirq.Operation] = []
        for op in moment:
            if len(op.qubits) == 2:
                m_ops.append(cirq.CZ(op.qubits[0], op.qubits[1]))
            elif len(op.qubits) == 1:
                pxz = _to_phased_xz(op, atol=atol)
                if pxz is not None:
                    m_ops.append(pxz)
        if m_ops:
            clean_moments.append(cirq.Moment(m_ops))

    # Re-pack moments as early as possible to minimize circuit depth
    return cirq.Circuit( (op for m in clean_moments for op in m), strategy=cirq.InsertStrategy.EARLIEST )


def insert_dynamical_decoupling(
    circuit: cirq.Circuit,
    active_qubits: Sequence[cirq.GridQubit],
    min_idle_run: int = 2,
) -> cirq.Circuit:
    """Insert balanced X - X spin-echo pairs on qubits idle across >= min_idle_run moments.

    Because X * X = I in noiseless simulation, statevector fidelity is preserved while
    low-frequency dephasing on idle windows is echoed out.
    """
    moments = list(circuit)
    if len(moments) < min_idle_run:
        return circuit

    x_gate = cirq.PhasedXZGate(x_exponent=1.0, z_exponent=0.0, axis_phase_exponent=0.0)
    grid = [dict((op.qubits[0], op) for op in m if len(op.qubits) == 1) for m in moments]
    busy = [set(m.qubits) for m in moments]

    for q in active_qubits:
        idx = 0
        n_m = len(moments)
        while idx < n_m:
            if q in busy[idx]:
                idx += 1
                continue
            start = idx
            while idx < n_m and q not in busy[idx]:
                idx += 1
            run_len = idx - start
            # Insert pairs of X gates in interior idle slots
            if run_len >= min_idle_run and start > 0 and idx < n_m:
                m1 = start + (run_len // 3)
                m2 = start + (2 * run_len // 3)
                if m1 == m2 and run_len >= 2:
                    m1 = start
                    m2 = start + 1
                if m1 != m2:
                    grid[m1][q] = x_gate.on(q)
                    grid[m2][q] = x_gate.on(q)
                    busy[m1].add(q)
                    busy[m2].add(q)

    new_moments: list[cirq.Moment] = []
    for i, m in enumerate(moments):
        two_q_ops = [op for op in m if len(op.qubits) > 1]
        one_q_ops = list(grid[i].values())
        new_moments.append(cirq.Moment([*two_q_ops, *one_q_ops]))
    return cirq.Circuit(new_moments)


def compile_prototype_to_willow(
    prototype_fn: Callable[[int, int], tuple[cirq.Circuit, Sequence[cirq.Qid]]],
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: Optional[dict[cirq.GridQubit, tuple[float, float]]] = None,
    one_errors: Optional[dict[cirq.GridQubit, tuple[float, float]]] = None,
    enable_dynamical_decoupling: bool = False,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Compile a prototype (LineQubit) Dicke state generator onto Willow105."""
    abstract_circuit, abstract_qubits = prototype_fn(num_qubits, num_excitations)
    ordered_abstract_target = list(abstract_qubits)
    all_abstract = sorted(abstract_circuit.all_qubits())
    ancilla_abstract = [q for q in all_abstract if q not in set(ordered_abstract_target)]
    full_abstract_order = ordered_abstract_target + ancilla_abstract
    total_needed = len(full_abstract_order)

    graph = build_connectivity_graph(
        cz_pauli_error_rate=cz_pauli_error_rate,
        sq_pauli_error_rate=sq_pauli_error_rate,
        zero_errors=zero_errors,
        one_errors=one_errors,
    )

    try:
        physical_chain = find_best_linear_chain(graph, total_needed)
    except ValueError:
        try:
            physical_chain = find_best_connected_subgraph(graph, total_needed)
        except ValueError as exc:
            raise NotImplementedError(
                f"Unable to embed {total_needed} qubits on working Willow subgraph: {exc}"
            ) from exc

    qubit_map = dict(zip(full_abstract_order, physical_chain))
    mapped_circuit = abstract_circuit.transform_qubits(lambda q: qubit_map[q])
    sub_graph = graph.subgraph(physical_chain).copy()

    compiled = compile_circuit_to_willow_gateset(mapped_circuit, working_graph=sub_graph)
    target_physical = set(physical_chain[: len(ordered_abstract_target)])

    if len(compiled) == 0 and len(target_physical) > 0:
        # Ensure all target qubits are registered in the circuit or returned cleanly
        pass

    if enable_dynamical_decoupling:
        compiled = insert_dynamical_decoupling(compiled, physical_chain)

    return compiled, target_physical
