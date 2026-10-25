"""Unit tests for Stage 4 noisy Willow QVM simulation and official harness scoring."""

import numpy as np

from src.candidate_algorithms import candidate_dicke_state
from src.noise_models import sample_benchmark_environment
from src.simulation.noisy_evaluator import evaluate_noisy_circuit


def test_noisy_evaluator_qvm_and_ablation():
    env = sample_benchmark_environment(seed=42)
    circuit, qubits = candidate_dicke_state(
        3,
        1,
        env["cz_pauli_error_rate"],
        env["sq_pauli_error_rate"],
        env["zero_errors"],
        env["one_errors"],
    )
    res = evaluate_noisy_circuit(
        circuit=circuit,
        qubits=qubits,
        num_qubits=3,
        num_excitations=1,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
        zero_errors=env["zero_errors"],
        one_errors=env["one_errors"],
        n_shots=2000,
        noise_backend="qvm",
        seed=42,
        run_ablation=True,
    )
    assert res.z_subspace_fraction > 0.85
    assert res.z_subspace_fraction < 1.0
    assert res.overall_score > 0.0
    assert set(res.ablation_scores.keys()) == {
        "gate_only",
        "readout_only",
        "idle_t1_t2_only",
        "full_qvm",
    }
    assert res.ablation_scores["gate_only"]["z_subspace_fraction"] > res.ablation_scores["full_qvm"]["z_subspace_fraction"]


def test_noiseless_shots_backend_matches_ceiling():
    env = sample_benchmark_environment(seed=7)
    circuit, qubits = candidate_dicke_state(
        3,
        1,
        env["cz_pauli_error_rate"],
        env["sq_pauli_error_rate"],
        env["zero_errors"],
        env["one_errors"],
    )
    res = evaluate_noisy_circuit(
        circuit=circuit,
        qubits=qubits,
        num_qubits=3,
        num_excitations=1,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
        zero_errors=env["zero_errors"],
        one_errors=env["one_errors"],
        n_shots=5000,
        noise_backend="noiseless-shots",
        seed=7,
    )
    assert np.isclose(res.z_subspace_fraction, 1.0)
    assert res.x_score < 0.01
    assert res.overall_score > 0.5
