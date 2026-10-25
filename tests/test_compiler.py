"""Unit tests for the Willow compiler, subgraph routing, and dynamical decoupling."""

import pytest

from src.candidate_algorithms import CANDIDATES, prototype_dicke_state
from src.compilation.willow_compiler import compile_prototype_to_willow
from src.noise_models import sample_benchmark_environment
from src.verification.fidelity import verify_dicke_state
from src.verification.linter import lint_willow_circuit


@pytest.mark.parametrize("n_qubits, n_exc", [(2, 1), (3, 1), (4, 2), (5, 2), (6, 2)])
def test_compiler_produces_lint_clean_exact_willow_circuits(n_qubits: int, n_exc: int):
    env = sample_benchmark_environment(seed=17)
    circuit, qubits = compile_prototype_to_willow(
        prototype_dicke_state,
        num_qubits=n_qubits,
        num_excitations=n_exc,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
        zero_errors=env["zero_errors"],
        one_errors=env["one_errors"],
    )
    lint_res = lint_willow_circuit(
        circuit,
        qubits,
        num_qubits=n_qubits,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
    )
    assert lint_res.passed, f"Lint errors: {lint_res.errors}"

    fid_res = verify_dicke_state(circuit, qubits, n_qubits, n_exc, run_official_test=True)
    assert fid_res.passed, f"Fidelity failed: {fid_res.messages}"


def test_dynamical_decoupling_preserves_exact_fidelity_and_lints_clean():
    env = sample_benchmark_environment(seed=42)
    spec = CANDIDATES["default_dd"]
    circuit, qubits = spec.native_fn(
        5,
        1,
        env["cz_pauli_error_rate"],
        env["sq_pauli_error_rate"],
        env["zero_errors"],
        env["one_errors"],
    )
    lint_res = lint_willow_circuit(
        circuit,
        qubits,
        num_qubits=5,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
    )
    assert lint_res.passed
    fid_res = verify_dicke_state(circuit, qubits, 5, 1, run_official_test=True)
    assert fid_res.passed
