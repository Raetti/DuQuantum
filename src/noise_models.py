"""Synthetic error generation utilities for the Google Willow processor.

Mirrors the challenge evaluation harness.
"""

from typing import Optional
import numpy as np
import cirq
import cirq_google


def pick_single_qubit_dropouts(
    dropout_rate: float,
    rng: Optional[np.random.Generator] = None,
) -> set[cirq.GridQubit]:
    """Pick qubits on Willow105 to be defective."""
    if rng is None:
        rng = np.random.default_rng()
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_dropout = rng.random(len(all_qubits)) < dropout_rate
    return set(all_qubits[is_dropout])


def generate_qubit_error_rates(
    median: float,
    spread: float,
    dropouts: set[cirq.GridQubit],
    uncertainty: float,
    rng: Optional[np.random.Generator] = None,
) -> dict[cirq.GridQubit, tuple[float, float]]:
    """Generate lognormal single-qubit error rates."""
    if rng is None:
        rng = np.random.default_rng()
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_working = np.array([q not in dropouts for q in all_qubits])
    working_qubits = all_qubits[is_working]
    error_rate = rng.lognormal(np.log(median), spread, len(working_qubits))
    return dict(
        zip(working_qubits, zip(error_rate, uncertainty * np.ones(len(working_qubits))))
    )


def generate_two_qubit_pauli_error_rates(
    median: float = 2.5e-3,
    spread: float = 0.5,
    dropout_rate: float = 0.01,
    uncertainty: float = 5e-4,
    rng: Optional[np.random.Generator] = None,
    single_qubit_dropouts: Optional[set[cirq.GridQubit]] = None,
) -> dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]]:
    """Generate lognormal two-qubit CZ Pauli error rates."""
    if rng is None:
        rng = np.random.default_rng()
    if single_qubit_dropouts is None:
        single_qubit_dropouts = set()

    all_pairs = np.array(list(cirq_google.Willow105.metadata.qubit_pairs))
    no_qubit_dropouts = np.array(
        [np.all([q not in single_qubit_dropouts for q in pair]) for pair in all_pairs]
    )
    is_working = (rng.random(len(all_pairs)) > dropout_rate) * no_qubit_dropouts
    working_pairs = [tuple(pair) for pair in all_pairs[is_working]]
    error_rate = rng.lognormal(np.log(median), spread, len(working_pairs))
    return dict(
        zip(working_pairs, zip(error_rate, uncertainty * np.ones(len(working_pairs))))
    )


def sample_benchmark_environment():
    """Produce a standard set of error rates for local testing."""
    dropouts = pick_single_qubit_dropouts(0.01)
    sq_err = generate_qubit_error_rates(4e-4, 0.5, dropouts, 2e-4)
    cz_err = generate_two_qubit_pauli_error_rates(single_qubit_dropouts=dropouts)
    zero_err = generate_qubit_error_rates(5e-3, 0.5, dropouts, 2e-4)
    one_err = generate_qubit_error_rates(1e-2, 0.5, dropouts, 3e-4)
    return {
        "dropouts": dropouts,
        "sq_pauli_error_rate": sq_err,
        "cz_pauli_error_rate": cz_err,
        "zero_errors": zero_err,
        "one_errors": one_err,
    }
