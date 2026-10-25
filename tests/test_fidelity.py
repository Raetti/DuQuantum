"""Unit tests for Stage 1 and Stage 3 exact mathematical verification & diagnostics."""

import numpy as np
import pytest
import cirq

from src.candidate_algorithms import prototype_dicke_state
from src.verification.fidelity import verify_dicke_state


@pytest.mark.parametrize("n_qubits, n_exc", [(2, 1), (3, 1), (4, 2), (5, 2), (6, 3)])
def test_exact_prototype_dicke_states_pass(n_qubits: int, n_exc: int):
    circ, qubits = prototype_dicke_state(n_qubits, n_exc)
    rep = verify_dicke_state(circ, qubits, n_qubits, n_exc, run_official_test=True)
    assert rep.passed
    assert np.isclose(rep.fidelity, 1.0, atol=1e-6)
    assert rep.subspace_leakage < 1e-7
    assert rep.amplitude_uniformity_max_err < 1e-5
    assert rep.phase_spread_rad < 1e-5


def test_diagnostics_detect_angle_and_phase_errors():
    qubits = cirq.LineQubit.range(3)
    # Wrong rotation angle on W3
    bad_angle_circ = cirq.Circuit(
        cirq.X(qubits[0]),
        cirq.ry(0.4).on(qubits[1]).controlled_by(qubits[0]),
        cirq.CNOT(qubits[1], qubits[0]),
    )
    rep_amp = verify_dicke_state(bad_angle_circ, qubits, 3, 1, run_official_test=False)
    assert not rep_amp.passed
    assert rep_amp.amplitude_uniformity_max_err > 0.1

    # Relative phase flip on W3: apply Z on q[0] so |100> gets a -1 phase
    w3_circ, _ = prototype_dicke_state(3, 1)
    bad_phase_circ = w3_circ + cirq.Circuit(cirq.Z(qubits[0]))
    rep_phase = verify_dicke_state(bad_phase_circ, qubits, 3, 1, run_official_test=False)
    assert not rep_phase.passed
    assert np.isclose(rep_phase.subspace_prob, 1.0, atol=1e-6)
    assert rep_phase.phase_spread_rad > 3.0


def test_ancilla_disentanglement_checks():
    w3_circ, qubits = prototype_dicke_state(3, 1)
    anc = cirq.LineQubit(99)

    # Disentangled ancilla in |0> -> PASS
    clean_anc = w3_circ + cirq.Circuit(cirq.X(anc), cirq.X(anc))
    rep_clean = verify_dicke_state(clean_anc, qubits, 3, 1, run_official_test=True)
    assert rep_clean.passed
    assert not rep_clean.ancillas_entangled
    assert np.isclose(rep_clean.ancilla_zero_prob, 1.0)

    # Disentangled ancilla left in |1> -> PASS with warning
    one_anc = w3_circ + cirq.Circuit(cirq.X(anc))
    rep_one = verify_dicke_state(one_anc, qubits, 3, 1, run_official_test=True)
    assert rep_one.passed
    assert not rep_one.ancillas_entangled
    assert len(rep_one.warnings) > 0

    # Entangled ancilla -> FAIL
    entangled_anc = w3_circ + cirq.Circuit(cirq.CNOT(qubits[0], anc))
    rep_ent = verify_dicke_state(entangled_anc, qubits, 3, 1, run_official_test=False)
    assert not rep_ent.passed
    assert rep_ent.ancillas_entangled
