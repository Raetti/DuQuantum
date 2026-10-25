"""Pluggable candidate algorithm interface for Dicke state preparation.

Supports two modes:
1. Abstract / Prototype Mode (`prototype_dicke_state`):
   Uses `cirq.LineQubit.range(num_qubits)` and unrestricted theoretical gates
   (`cirq.ry`, `cirq.CNOT`, `cirq.X`, Givens rotations, controlled unitaries).
   Evaluated directly in Stage 1 (Pure Mathematical Logic) and automatically
   lowered onto Willow105 for Stages 2-4 via `compile_prototype_to_willow`.

2. Native Willow Mode (`candidate_dicke_state`):
   Conforms to the official Google Quantum AI challenge signature and imports
   only `cirq`, `cirq_google`, `numpy`, and Python's standard library.
"""

from dataclasses import dataclass
import math
from typing import Callable, Optional, Sequence
import numpy as np
import cirq
import cirq_google

from src.compilation.willow_compiler import compile_prototype_to_willow


PrototypeFn = Callable[[int, int], tuple[cirq.Circuit, Sequence[cirq.Qid]]]
NativeFn = Callable[
    [
        int,
        int,
        dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
        dict[cirq.GridQubit, tuple[float, float]],
        dict[cirq.GridQubit, tuple[float, float]],
        dict[cirq.GridQubit, tuple[float, float]],
    ],
    tuple[cirq.Circuit, set[cirq.GridQubit]],
]


def givens_gate(theta: float) -> cirq.MatrixGate:
    """Two-qubit excitation-preserving Givens rotation in the {|01>, |10>} subspace.

    In the 2-qubit basis {|00>, |01>, |10>, |11>}, maps:
        |00> -> |00>
        |01> ->  cos(theta)|01> + sin(theta)|10>
        |10> -> -sin(theta)|01> + cos(theta)|10>
        |11> -> |11>
    Wait: when shifting an excitation from |10> to |01> (left to right) with positive
    amplitudes:
        |10> -> cos(theta)|10> + sin(theta)|01>
        |01> -> -sin(theta)|10> + cos(theta)|01>
    """
    c = math.cos(theta)
    s = math.sin(theta)
    mat = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, s, 0.0],
            [0.0, -s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    return cirq.MatrixGate(mat, name=f"Givens({theta:.3f})")


def scs_givens_right_to_left(theta: float) -> cirq.MatrixGate:
    """Givens rotation shifting excitation from |01> into cos(theta)|01> + sin(theta)|10>."""
    c = math.cos(theta)
    s = math.sin(theta)
    mat = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    return cirq.MatrixGate(mat, name=f"SCSGivens({theta:.3f})")


def _append_controlled_scs_givens(
    circuit: cirq.Circuit,
    q_a: cirq.Qid,
    q_ctrl: cirq.Qid,
    q_c: cirq.Qid,
    theta: float,
) -> None:
    """Append 3-qubit controlled SCS Givens rotation on (q_a, q_c) controlled by q_ctrl.

    Uses the Bärtschi-Eidenbenz (2019) Lemma 2 decomposition ordered so that
    the inner controlled-Ry rotations act on the always-adjacent pair (q_ctrl, q_a),
    and the trailing CRy(theta) + CNOT(q_a, q_c) fuse into a single 2Q unitary.
    """
    circuit.append(cirq.CNOT(q_a, q_c))
    circuit.append(cirq.ry(theta).on(q_a).controlled_by(q_ctrl))
    circuit.append(cirq.CNOT(q_c, q_ctrl))
    circuit.append(cirq.ry(-theta).on(q_a).controlled_by(q_ctrl))
    circuit.append(cirq.CNOT(q_c, q_ctrl))
    circuit.append(cirq.ry(theta).on(q_a).controlled_by(q_c))
    circuit.append(cirq.CNOT(q_a, q_c))


def reference_w_state_prototype(
    num_qubits: int, num_excitations: int
) -> tuple[cirq.Circuit, Sequence[cirq.Qid]]:
    """Reference sanity-check prototype using Givens rotations for W states.

    Supports n in {0, 1, N-1, N} for any N, plus (N=4, n=2).
    """
    if num_excitations < 0 or num_excitations > num_qubits:
        raise NotImplementedError(
            f"Invalid (N={num_qubits}, n={num_excitations}) combination."
        )

    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()

    if num_excitations == 0:
        return circuit, qubits

    if num_excitations == num_qubits:
        circuit.append(cirq.X.on_each(*qubits))
        return circuit, qubits

    if num_excitations == 1:
        circuit.append(cirq.X(qubits[0]))
        for k in range(num_qubits - 1):
            remaining = num_qubits - k
            theta = math.acos(1.0 / math.sqrt(remaining))
            circuit.append(givens_gate(theta).on(qubits[k], qubits[k + 1]))
        return circuit, qubits

    if num_excitations == num_qubits - 1:
        w_circ, _ = reference_w_state_prototype(num_qubits, 1)
        circuit.append(w_circ)
        circuit.append(cirq.X.on_each(*qubits))
        return circuit, qubits

    if num_qubits == 4 and num_excitations == 2:
        circuit.append(cirq.H(qubits[0]))
        w3, _ = reference_w_state_prototype(3, 1)
        circuit.append(w3.transform_qubits(lambda q: qubits[q.x + 1]))
        for q in qubits[1:]:
            circuit.append(cirq.X(q))
        for q in qubits[1:]:
            circuit.append(cirq.CNOT(qubits[0], q))
        return circuit, qubits

    raise NotImplementedError(
        f"reference_w only implements n in {{0, 1, N-1, N}} and (4, 2); got ({num_qubits}, {num_excitations})."
    )


def reference_scs_prototype(
    num_qubits: int, num_excitations: int
) -> tuple[cirq.Circuit, Sequence[cirq.Qid]]:
    """General (N, n) deterministic Dicke state preparation via Split & Cyclic Shift.

    Implements the Bärtschi & Eidenbenz (arXiv:1904.07358) inductive SCS_{m, k}
    unitaries on `cirq.LineQubit.range(num_qubits)`. Uses particle-hole symmetry
    (X^N) whenever n > N // 2 to minimize circuit depth.
    """
    if num_excitations < 0 or num_excitations > num_qubits:
        raise NotImplementedError(
            f"Invalid (N={num_qubits}, n={num_excitations}) combination."
        )

    qubits = cirq.LineQubit.range(num_qubits)
    if num_excitations in (0, 1, num_qubits - 1, num_qubits) or (
        num_qubits == 4 and num_excitations == 2
    ):
        return reference_w_state_prototype(num_qubits, num_excitations)

    invert_at_end = False
    k = num_excitations
    if k > num_qubits // 2:
        k = num_qubits - k
        invert_at_end = True

    circuit = cirq.Circuit()
    # Initialize unary input state |0>^{N-k} |1>^k
    for q in qubits[num_qubits - k :]:
        circuit.append(cirq.X(q))

    # Apply SCS_{m, min(m-1, k)} on prefix qubits[0 : m] for m = N down to 2
    for m in range(num_qubits, 1, -1):
        l_max = min(m - 1, k)
        # Gate (i) for l = 1: acts on (qubits[m - 2], qubits[m - 1])
        theta_1 = math.acos(math.sqrt(1.0 / float(m)))
        circuit.append(
            scs_givens_right_to_left(theta_1).on(qubits[m - 2], qubits[m - 1])
        )
        # Gates (ii_l) for l = 2 .. l_max: acts on (qubits[m - 1 - l], qubits[m - 1])
        # controlled on qubits[m - l] == |1>
        for l_idx in range(2, l_max + 1):
            theta_l = math.acos(math.sqrt(float(l_idx) / float(m)))
            _append_controlled_scs_givens(
                circuit,
                q_a=qubits[m - 1 - l_idx],
                q_ctrl=qubits[m - l_idx],
                q_c=qubits[m - 1],
                theta=theta_l,
            )

    if invert_at_end:
        circuit.append(cirq.X.on_each(*qubits))

    return circuit, qubits


# ==============================================================================
# TEAMMATES' HANDWRITTEN ALGORITHM PLUG-IN SLOT
# ==============================================================================
def team_prototype_dicke_state(
    num_qubits: int, num_excitations: int
) -> tuple[cirq.Circuit, Sequence[cirq.Qid]]:
    """Plug in the physics team's derivation here (Stage 1 abstract mode).

    You can use `cirq.LineQubit.range(num_qubits)` and any standard Cirq gates:
      - `cirq.X`, `cirq.H`, `cirq.ry(theta)`, `cirq.rz(phi)`
      - `cirq.CNOT(q_ctrl, q_target)`, `cirq.CZ(q0, q1)`, `cirq.SWAP(q0, q1)`
      - `givens_gate(theta).on(q0, q1)`
      - `cirq.ry(theta).on(q_target).controlled_by(q_ctrl)`

    Once implemented, test immediately with:
        python scripts/test_algorithm.py --N 4 --n 2 --candidate team --stage 1
        python scripts/test_algorithm.py --N 4 --n 2 --candidate team --stage all
    """
    raise NotImplementedError(
        "team_prototype_dicke_state is awaiting the physics team's derivation. "
        "Paste their gate sequence into src/candidate_algorithms.py:team_prototype_dicke_state."
    )


def team_native_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Native Willow wrapper for the physics team's algorithm."""
    return compile_prototype_to_willow(
        team_prototype_dicke_state,
        num_qubits=num_qubits,
        num_excitations=num_excitations,
        cz_pauli_error_rate=cz_pauli_error_rate,
        sq_pauli_error_rate=sq_pauli_error_rate,
        zero_errors=zero_errors,
        one_errors=one_errors,
    )


def prototype_dicke_state(
    num_qubits: int, num_excitations: int
) -> tuple[cirq.Circuit, Sequence[cirq.Qid]]:
    """Default abstract prototype entry point (uses reference_scs_prototype)."""
    return reference_scs_prototype(num_qubits, num_excitations)


def candidate_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    """Default native Willow entry point conforming to Google's submission signature."""
    return compile_prototype_to_willow(
        prototype_dicke_state,
        num_qubits=num_qubits,
        num_excitations=num_excitations,
        cz_pauli_error_rate=cz_pauli_error_rate,
        sq_pauli_error_rate=sq_pauli_error_rate,
        zero_errors=zero_errors,
        one_errors=one_errors,
    )


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    description: str
    prototype_fn: Optional[PrototypeFn]
    native_fn: NativeFn


def _make_compiled_native(proto_fn: PrototypeFn, dd: bool = False) -> NativeFn:
    def _native(
        num_qubits: int,
        num_excitations: int,
        cz_pauli_error_rate: dict,
        sq_pauli_error_rate: dict,
        zero_errors: dict,
        one_errors: dict,
    ) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
        return compile_prototype_to_willow(
            proto_fn,
            num_qubits=num_qubits,
            num_excitations=num_excitations,
            cz_pauli_error_rate=cz_pauli_error_rate,
            sq_pauli_error_rate=sq_pauli_error_rate,
            zero_errors=zero_errors,
            one_errors=one_errors,
            enable_dynamical_decoupling=dd,
        )

    return _native


CANDIDATES: dict[str, CandidateSpec] = {
    "default": CandidateSpec(
        name="default",
        description="General Split & Cyclic Shift (SCS) + KAK 2-CZ compiler for arbitrary (N, n)",
        prototype_fn=prototype_dicke_state,
        native_fn=candidate_dicke_state,
    ),
    "default_dd": CandidateSpec(
        name="default_dd",
        description="General SCS + KAK 2-CZ compiler with X-X dynamical decoupling spin echoes",
        prototype_fn=prototype_dicke_state,
        native_fn=_make_compiled_native(prototype_dicke_state, dd=True),
    ),
    "reference_w": CandidateSpec(
        name="reference_w",
        description="Sanity-check Givens W-state ladder for n in {0, 1, N-1, N} and (4, 2)",
        prototype_fn=reference_w_state_prototype,
        native_fn=_make_compiled_native(reference_w_state_prototype),
    ),
    "team": CandidateSpec(
        name="team",
        description="Plug-in slot for the physics teammates' custom Dicke state derivation",
        prototype_fn=team_prototype_dicke_state,
        native_fn=team_native_dicke_state,
    ),
}
