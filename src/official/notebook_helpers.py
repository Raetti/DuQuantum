"""Verbatim helper functions from DuQuantum_Google_Standard.py.

Only the official DuQuantum organizer erratum (`keep_indices = range(len(qubits))`
in `test_circuit`) is applied; all other helper ASTs are untouched and guarded by
`tests/test_official_helpers_sanctity.py`.
"""

import itertools
import math

import cirq
import cirq_google
import numpy as np


def create_dicke_statevector(num_qubits: int, num_excitations: int) -> np.ndarray:
    psi = np.zeros(2**num_qubits)
    normalization = 1 / np.sqrt(math.comb(num_qubits, num_excitations))
    qubits = range(num_qubits)
    for excited_qubits in itertools.combinations(qubits, num_excitations):
        idx = sum(2**qubit_number for qubit_number in excited_qubits)
        psi[idx] = normalization
    return psi


def test_circuit(circuit: cirq.Circuit, num_excitations: int, qubits: set[cirq.GridQubit], sampler = cirq.Simulator()):
    all_qubits = circuit.all_qubits()
    ancilla_qubits = all_qubits.difference(qubits)
    ordered_qubits = sorted(qubits) + sorted(ancilla_qubits)
    keep_indices = range(len(qubits))
    num_qubits = len(qubits)
    psi_circuit_full = sampler.simulate(circuit, qubit_order=ordered_qubits).final_state_vector
    psi_circuit = cirq.sub_state_vector(
        state_vector=psi_circuit_full,
        keep_indices=keep_indices
    ) if len(all_qubits) > len(qubits) else psi_circuit_full
    assert np.isclose(np.abs(psi_circuit @ create_dicke_statevector(num_qubits, num_excitations)) ** 2, 1.0)


test_circuit.__test__ = False


def check_z_bitstring_distribution(
    bitstrings: np.ndarray, num_excitations: int
) -> float:
    """Measure how close the observed Z-basis distribution of bitstrings is to
    the ideal one. The lower the score, the better.

    Args:
        bitstrings: An array of shape (n_shots, n_qubits) containing the n_shots Z-basis bitstrings.
        num_excitations: The weight, n, of the Dicke state.

    Returns:
        A measure of how close the distribution is to the ideal one.
    """
    # first, let's see how many bitstrings have one excitation
    n_shots, n_qubits = bitstrings.shape
    measured_excitations = np.sum(bitstrings, axis=1)
    fraction_in_subspace = np.mean(measured_excitations == num_excitations)

    # now, let's check how uniform the distribution is over states with n excitations
    chi2 = 0
    ideal_prob = 1 / math.comb(n_qubits, num_excitations)
    for indices in itertools.combinations(range(n_qubits), num_excitations):
        ideal_state = np.zeros(n_qubits)
        ideal_state[np.array(indices)] = 1
        match_count = np.sum(np.all(bitstrings == ideal_state, axis=1))
        expected_count = n_shots * fraction_in_subspace * ideal_prob
        chi2 += (match_count - expected_count) ** 2 / expected_count

    return ideal_prob * chi2 / fraction_in_subspace


def get_correlator_average(bitstrings: np.ndarray, correlator_size: int) -> float:
    """Get the average of all weight-correlator_size correlators in `bitstrings`.

    Args:
        bitstrings: The input bitstrings, shape (n_shots, n_qubits).
        correlator_size: Compute all correlators of this weight.

    Returns:
        The average of all weight-k correlators.
    """
    correlator_sum = 0.0
    n_shots, n_qubits = bitstrings.shape
    for indices in itertools.combinations(range(n_qubits), correlator_size):
        correlator_sum += 1 - 2 * np.mean(np.sum(bitstrings[:, indices], axis=1) % 2)
    return correlator_sum / math.comb(n_qubits, correlator_size)


def check_x_bitstring_distribution(
    bitstrings: np.ndarray, num_excitations: int, max_correlator_size: int
) -> float:
    """Check expected correlations in the X basis. The lower the score, the better.

    Args:
        bitstrings: An array of shape (n_shots, n_qubits) containing the n_shots X-basis bitstrings.
        max_correlator_size: The maximum weight of correlators to consider.
        num_excitations: The weight, n, of the Dicke state.

    Returns:
        A measure of how the correlations in the X-basis measurements compare to expectation.
    """
    comb = lambda n, k: math.comb(n, k) if (k >= 0 and n >= k) else 0
    n_shots, n_qubits = bitstrings.shape
    score = 0.0
    for correlator_size in range(1, max_correlator_size + 1):
        expected_correlator = (
            comb(correlator_size, correlator_size // 2)
            * comb(n_qubits - correlator_size, num_excitations - correlator_size // 2)
            / comb(n_qubits, num_excitations)
            if correlator_size % 2 == 0
            else 0.0
        )
        measured_correlator = get_correlator_average(bitstrings, correlator_size)
        score += (
            (expected_correlator - measured_correlator) ** 2 / expected_correlator**2
            if expected_correlator != 0
            else (expected_correlator - measured_correlator) ** 2
        )

    return score


def overall_score(
    z_bitstrings: np.ndarray,
    x_bitstrings: np.ndarray,
    num_excitations: int,
    max_correlator_size: int,
) -> float:
    """An overall score that checks how the Z and X basis bitstrings compare to their ideal
    distributions. The higher the score, the better.

    Args:
        z_bitstrings: The bitstrings measured in the Z basis
        x_bitstrings: The bitstrings measured in the X basis
        num_excitations: The weight, n, of the Dicke state.
        max_correlator_size: The maximum weight of correlators to consider

    Returns:
        A measure of how similar the bitstrings are to their ideal distributions.
    """
    return 1 / (
        check_z_bitstring_distribution(z_bitstrings, num_excitations)
        + check_x_bitstring_distribution(
            x_bitstrings, num_excitations, max_correlator_size
        )
    )


def pick_single_qubit_dropouts(
    dropout_rate: float,
    rng: np.random.Generator = np.random.default_rng()
) -> set[cirq.GridQubit]:
    """Pick some qubits to be broken.

    Args:
        dropout_rate: The probability that a qubit is broken.
        rng: The pseudo-random number generator

    Returns:
        The broken qubits.
    """
    all_qubits = np.array(list(cirq_google.Willow105.metadata.qubit_set))
    is_dropout = rng.random(len(all_qubits)) < dropout_rate
    return set(all_qubits[is_dropout])


def generate_qubit_error_rates(
    median: float,
    spread: float,
    dropouts: set[cirq.GridQubit],
    uncertainty: float,
    rng: np.random.Generator = np.random.default_rng(),
) -> dict[cirq.GridQubit, tuple[float, float]]:
    """Generate fake single-qubit error rates.

    Args:
        median: The median error rate.
        spread: The spread of the lognormal distribution
        dropouts: Qubits to exclude
        uncertainty: The uncertainty of the measured error rates.
        rng: The pseudo-random number generator

    Returns:
        A dictionary from qubits to the measured error rate and statistical uncertainty.
    """

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
    rng: np.random.Generator = np.random.default_rng(),
    single_qubit_dropouts: set[cirq.GridQubit] = None,
) -> dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]]:
    """Generate fake two-qubit error rates.

    Args:
        median: The median error rate.
        spread: The spread of the lognormal distribution
        dropout_rate: The probability of a broken coupler.
        uncertainty: The uncertainty of the measured error rates.
        rng: The pseudo-random number generator
        single_qubit_dropouts: The single-qubit dropouts to skip.

    Returns:
        A dictionary from qubits to the measured error rate and statistical uncertainty.
    """
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
