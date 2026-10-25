"""Primary Dicke state preparation entry points for the Google Quantum AI challenge."""

import math
from typing import Sequence
import cirq
import cirq_google

from src.candidate_algorithms import (
    CANDIDATES,
    CandidateSpec,
    candidate_dicke_state,
    prototype_dicke_state,
)
from src.compilation.willow_compiler import compile_prototype_to_willow
from src.willow_utils import build_connectivity_graph, find_best_linear_chain


def _givens_rotation_circuit(
    control: cirq.GridQubit,
    target: cirq.GridQubit,
    theta: float,
) -> list[cirq.Operation]:
    """Legacy 6-CZ decomposition of an excitation-preserving Givens rotation."""
    ops: list[cirq.Operation] = []

    def cnot(c: cirq.GridQubit, t: cirq.GridQubit):
        return [
            cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H))(t),
            cirq.CZ(c, t),
            cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H))(t),
        ]

    ops.extend(cnot(target, control))
    ops.append(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.ry(theta)))(target))
    ops.extend(cnot(control, target))
    ops.append(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.ry(-theta)))(target))
    ops.extend(cnot(control, target))
    ops.extend(cnot(target, control))
    return ops


def _prepare_w_state(qubits: Sequence[cirq.GridQubit]) -> cirq.Circuit:
    """Legacy W-state preparation on a linear chain of N qubits."""
    n_qubits = len(qubits)
    circuit = cirq.Circuit()
    if n_qubits == 0:
        return circuit
    circuit.append(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.X))(qubits[0]))
    for k in range(n_qubits - 1):
        remaining = n_qubits - k
        theta = math.acos(1.0 / math.sqrt(remaining))
        circuit.append(_givens_rotation_circuit(qubits[k], qubits[k + 1], theta))
    return circuit


def legacy_create_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Legacy hand-unrolled implementation (for benchmarking against the KAK compiler)."""
    if num_excitations < 0 or num_excitations > num_qubits:
        raise NotImplementedError(
            f"Invalid excitations {num_excitations} for {num_qubits} qubits."
        )
    graph = build_connectivity_graph(cz_pauli_error_rate, sq_pauli_error_rate)
    chain = find_best_linear_chain(graph, num_qubits)
    if num_excitations == 0:
        return cirq.Circuit(), set(chain)
    if num_excitations == num_qubits:
        c = cirq.Circuit(
            cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.X))(q) for q in chain
        )
        return c, set(chain)
    if num_excitations == 1:
        return _prepare_w_state(chain), set(chain)
    if num_excitations == num_qubits - 1:
        c = _prepare_w_state(chain)
        for q in chain:
            c.append(cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.X))(q))
        return c, set(chain)
    return candidate_dicke_state(
        num_qubits,
        num_excitations,
        cz_pauli_error_rate,
        sq_pauli_error_rate,
        zero_errors,
        one_errors,
    )


CANDIDATES["legacy"] = CandidateSpec(
    name="legacy",
    description="Legacy 6-CZ CNOT-based Givens ladder baseline",
    prototype_fn=prototype_dicke_state,
    native_fn=legacy_create_dicke_state,
)


def create_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces a Dicke state on Google Willow."""
    return candidate_dicke_state(
        num_qubits=num_qubits,
        num_excitations=num_excitations,
        cz_pauli_error_rate=cz_pauli_error_rate,
        sq_pauli_error_rate=sq_pauli_error_rate,
        zero_errors=zero_errors,
        one_errors=one_errors,
    )


def create_dicke_state_adaptive(
    num_qubits: int,
    num_excitations: int,
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Create a Cirq circuit that produces a Dicke state for the bonus track (2 <= N <= 15)."""
    if num_excitations < 0 or num_excitations > num_qubits:
        raise NotImplementedError(
            f"Invalid excitations {num_excitations} for {num_qubits} qubits."
        )
    all_pairs = [tuple(pair) for pair in cirq_google.Willow105.metadata.qubit_pairs]
    all_qubits = list(cirq_google.Willow105.metadata.qubit_set)
    cz_dummy = {pair: (0.0025, 0.0005) for pair in all_pairs}
    sq_dummy = {q: (0.0004, 0.0002) for q in all_qubits}
    return compile_prototype_to_willow(
        prototype_dicke_state,
        num_qubits=num_qubits,
        num_excitations=num_excitations,
        cz_pauli_error_rate=cz_dummy,
        sq_pauli_error_rate=sq_dummy,
    )
