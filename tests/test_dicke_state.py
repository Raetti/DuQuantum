"""Unit tests for Dicke state generation using the official notebook helpers."""

import numpy as np
import pytest

from src.dicke_state import create_dicke_state, create_dicke_state_adaptive
from src.noise_models import sample_benchmark_environment
from src.official.notebook_helpers import (
    create_dicke_statevector,
    test_circuit as verify_circuit,
)


def test_create_dicke_statevector():
    """Verify ideal statevector calculations."""
    vec = create_dicke_statevector(2, 1)
    expected = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0])
    np.testing.assert_allclose(vec, expected)


@pytest.mark.parametrize(
    "n_qubits, excitations",
    [
        (2, 1),
        (3, 1),
        (4, 1),
        (3, 2),
        (4, 2),
        (5, 2),
    ],
)
def test_create_dicke_state_fidelity(n_qubits: int, excitations: int):
    """Verify circuit fidelity against statevector on the Willow105 benchmark environment."""
    env = sample_benchmark_environment(seed=42)
    circuit, qubits = create_dicke_state(
        num_qubits=n_qubits,
        num_excitations=excitations,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
        zero_errors=env["zero_errors"],
        one_errors=env["one_errors"],
    )

    assert len(qubits) == n_qubits
    assert len(circuit) <= 1750
    verify_circuit(circuit, excitations, qubits)


def test_create_dicke_state_adaptive():
    """Verify adaptive circuit generation on baseline parameters."""
    circuit, qubits = create_dicke_state_adaptive(num_qubits=4, num_excitations=2)
    assert len(qubits) == 4
    verify_circuit(circuit, 2, qubits)
