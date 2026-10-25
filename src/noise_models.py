"""Seeded wrappers around the official Willow105 error-rate generators."""

from typing import Optional
import numpy as np
import cirq
import cirq_google

from src.official.notebook_helpers import (
    generate_qubit_error_rates,
    generate_two_qubit_pauli_error_rates,
    pick_single_qubit_dropouts,
)


def get_willow_device():
    """Return the official cirq_google.Willow105 device."""
    return cirq_google.Willow105


def sample_benchmark_environment(seed: Optional[int] = 42) -> dict:
    """Generate a reproducible error-rate profile using the notebook's parameters."""
    rng = np.random.default_rng(seed)
    single_qubit_dropouts = pick_single_qubit_dropouts(0.01, rng=rng)
    sq_pauli_error_rate = generate_qubit_error_rates(
        4e-4, 0.5, single_qubit_dropouts, 2e-4, rng=rng
    )
    cz_pauli_error_rate = generate_two_qubit_pauli_error_rates(
        rng=rng, single_qubit_dropouts=single_qubit_dropouts
    )
    zero_errors = generate_qubit_error_rates(
        5e-3, 0.5, single_qubit_dropouts, 2e-4, rng=rng
    )
    one_errors = generate_qubit_error_rates(
        1e-2, 0.5, single_qubit_dropouts, 3e-4, rng=rng
    )
    return {
        "dropouts": single_qubit_dropouts,
        "sq_pauli_error_rate": sq_pauli_error_rate,
        "cz_pauli_error_rate": cz_pauli_error_rate,
        "zero_errors": zero_errors,
        "one_errors": one_errors,
    }


__all__ = [
    "get_willow_device",
    "pick_single_qubit_dropouts",
    "generate_qubit_error_rates",
    "generate_two_qubit_pauli_error_rates",
    "sample_benchmark_environment",
]
