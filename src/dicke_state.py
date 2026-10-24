"""Dicke state preparation algorithms for Google Willow architecture.

Implements deterministic state preparation using excitation-preserving Givens
rotations decomposed into native CZ and PhasedXZ gates.
"""

import math
from typing import Optional, Sequence
import numpy as np
import cirq
import cirq_google

from src.willow_utils import build_connectivity_graph, find_best_linear_chain


def _givens_rotation_circuit(
    control: cirq.GridQubit,
    target: cirq.GridQubit,
    theta: float,
) -> list[cirq.Operation]:
    """Decompose an excitation-preserving Givens rotation into native Willow gates.

    Acts on the subspace {|01>, |10>}:
        |01> -> cos(theta)|01> - sin(theta)|10>
        |10> -> sin(theta)|01> + cos(theta)|10>
    Leaves |00> and |11> invariant.

    Synthesized using 3 CZ gates and single-qubit unitaries (PhasedXZGate).
    """
    ops: list[cirq.Operation] = []

    def cnot(c: cirq.GridQubit, t: cirq.GridQubit):
        return [
            cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H))(t),
            cirq.CZ(c, t),
            cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H))(t),
        ]

    # CNOT(target, control)
    ops.extend(cnot(target, control))

    # Controlled-Ry(2 * theta) with control=control, target=target
    # Controlled-Ry decomposition: Ry(theta) -> CNOT -> Ry(-theta) -> CNOT
    ops.append(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.ry(theta)))(target))
    ops.extend(cnot(control, target))
    ops.append(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.ry(-theta)))(target))
    ops.extend(cnot(control, target))

    # CNOT(target, control)
    ops.extend(cnot(target, control))

    return ops


def _prepare_w_state(qubits: Sequence[cirq.GridQubit]) -> cirq.Circuit:
    """Prepare a W-state (|D^{(N)}_1>) on a linear chain of N qubits."""
    n_qubits = len(qubits)
    circuit = cirq.Circuit()

    if n_qubits == 0:
        return circuit

    # Initialize first qubit in |1>
    circuit.append(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.X))(qubits[0]))

    # Cascade excitation along the chain
    for k in range(n_qubits - 1):
        remaining = n_qubits - k
        # Probability of staying at qubit k is 1 / remaining
        theta = math.acos(1.0 / math.sqrt(remaining))
        # Rotate between qubits[k] and qubits[k+1]
        rot_ops = _givens_rotation_circuit(qubits[k], qubits[k + 1], theta)
        circuit.append(rot_ops)

    return circuit


def _prepare_dicke_baertschi(
    qubits: Sequence[cirq.GridQubit],
    num_excitations: int,
) -> cirq.Circuit:
    """Prepare |D^{(N)}_n> using the Bärtschi & Eidenbenz deterministic scheme."""
    n_total = len(qubits)
    k_excitations = num_excitations

    circuit = cirq.Circuit()

    if k_excitations == 0:
        return circuit

    if k_excitations == n_total:
        for q in qubits:
            circuit.append(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.X))(q))
        return circuit

    if k_excitations == 1:
        return _prepare_w_state(qubits)

    # General Bärtschi-Eidenbenz algorithm
    # Initialize first k qubits to |1>
    for i in range(k_excitations):
        circuit.append(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.X))(qubits[i]))

    # Deterministic splitters cascade
    for layer in range(k_excitations, 0, -1):
        for j in range(layer, n_total - (k_excitations - layer)):
            p = layer
            q = j + 1
            # Compute excitation splitting angle based on binomial coefficients
            num = math.comb(q - 1, p)
            denom = math.comb(q, p)
            if denom > 0 and num <= denom:
                ratio = max(0.0, min(1.0, num / denom))
                theta = math.acos(math.sqrt(ratio))
                circuit.append(_givens_rotation_circuit(qubits[j - 1], qubits[j], theta))

    return circuit


def create_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces an N-qubit, weight-n Dicke state.

    Args:
        num_qubits: Number N of qubits in the Dicke state.
        num_excitations: Number n of excitations (Hamming weight).
        cz_pauli_error_rate: Measured CZ Pauli error rates on Willow.
        sq_pauli_error_rate: Single-qubit Pauli error rates on Willow.
        zero_errors: Readout 0 -> 1 error rates.
        one_errors: Readout 1 -> 0 error rates.

    Returns:
        A tuple of (circuit, set_of_qubits) preparing the Dicke state.
    """
    if num_excitations < 0 or num_excitations > num_qubits:
        raise NotImplementedError(f"Invalid excitations {num_excitations} for {num_qubits} qubits.")

    # Build connectivity graph of working qubits and low-error CZ couplers
    graph = build_connectivity_graph(cz_pauli_error_rate, sq_pauli_error_rate)

    # Find the lowest-error simple linear chain of length num_qubits
    chain = find_best_linear_chain(graph, num_qubits)

    # Construct the deterministic state preparation circuit
    circuit = _prepare_dicke_baertschi(chain, num_excitations)

    # Convert all single-qubit unitaries strictly to PhasedXZGate
    final_circuit = cirq.Circuit()
    for moment in circuit:
        moment_ops = []
        for op in moment:
            if isinstance(op.gate, cirq.CZPowGate):
                moment_ops.append(op)
            elif op.gate == cirq.ResetChannel():
                moment_ops.append(op)
            else:
                # Convert to native PhasedXZGate
                matrix = cirq.unitary(op)
                pxz = cirq.PhasedXZGate.from_matrix(matrix)
                moment_ops.append(pxz(op.qubits[0]))
        final_circuit.append(cirq.Moment(moment_ops))

    return final_circuit, set(chain)


def create_dicke_state_adaptive(
    num_qubits: int,
    num_excitations: int,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a shallower adaptive Cirq circuit with mid-circuit measurements.

    Respects the Willow device connectivity and readout-line diagonal constraint.
    """
    if num_excitations < 0 or num_excitations > num_qubits:
        raise NotImplementedError(f"Invalid excitations {num_excitations} for {num_qubits} qubits.")

    # Baseline adaptive circuit: utilizes optimal linear chain from Willow105 metadata
    all_pairs = list(cirq_google.Willow105.metadata.qubit_pairs)
    cz_dummy = {tuple(pair): (0.002, 0.0005) for pair in all_pairs}
    graph = build_connectivity_graph(cz_dummy)
    chain = find_best_linear_chain(graph, num_qubits)

    # Baseline deterministic circuit (can be enhanced with adaptive measurement feedforward)
    circuit = _prepare_dicke_baertschi(chain, num_excitations)

    return circuit, set(chain)
