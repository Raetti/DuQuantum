"""Unit tests for the Stage 2 Willow hardware rule linter."""

import cirq
import pytest

from src.candidate_algorithms import candidate_dicke_state
from src.noise_models import sample_benchmark_environment
from src.verification.linter import MAX_MOMENTS, lint_willow_circuit


def test_clean_compiled_circuit_passes_linter():
    env = sample_benchmark_environment(seed=42)
    circuit, qubits = candidate_dicke_state(
        4,
        2,
        env["cz_pauli_error_rate"],
        env["sq_pauli_error_rate"],
        env["zero_errors"],
        env["one_errors"],
    )
    res = lint_willow_circuit(
        circuit,
        qubits,
        num_qubits=4,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
    )
    assert res.passed
    assert len(res.errors) == 0


def test_linter_catches_invalid_1q_and_2q_gates():
    env = sample_benchmark_environment(seed=42)
    pair = next(iter(env["cz_pauli_error_rate"].keys()))
    q0, q1 = pair
    bad_circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.CNOT(q0, q1),
        (cirq.CZ**0.5)(q0, q1),
    )
    res = lint_willow_circuit(
        bad_circuit,
        {q0, q1},
        num_qubits=2,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
    )
    assert not res.passed
    rules = {e.rule for e in res.errors}
    assert "INVALID_1Q_GATE" in rules
    assert "INVALID_2Q_GATE" in rules


def test_linter_catches_non_adjacent_or_broken_coupler():
    env = sample_benchmark_environment(seed=42)
    working_qubits = sorted(env["sq_pauli_error_rate"].keys())
    q_first, q_last = working_qubits[0], working_qubits[-1]
    bad_circuit = cirq.Circuit(cirq.CZ(q_first, q_last))
    res = lint_willow_circuit(
        bad_circuit,
        {q_first, q_last},
        num_qubits=2,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
    )
    assert not res.passed
    assert any(e.rule == "CZ_COUPLER_NOT_IN_ERROR_RATES" for e in res.errors)


def test_linter_accepts_reversed_coupler_tuple_order():
    env = sample_benchmark_environment(seed=42)
    q0, q1 = next(iter(env["cz_pauli_error_rate"].keys()))
    pxz = cirq.PhasedXZGate(x_exponent=0.5, z_exponent=0.0, axis_phase_exponent=0.0)
    circuit = cirq.Circuit(pxz.on(q0), cirq.CZ(q1, q0), pxz.on(q1))
    res = lint_willow_circuit(
        circuit,
        {q0, q1},
        num_qubits=2,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
    )
    assert res.passed


def test_linter_catches_measurement_and_moment_limit():
    env = sample_benchmark_environment(seed=42)
    q0 = next(iter(env["sq_pauli_error_rate"].keys()))
    pxz = cirq.PhasedXZGate(x_exponent=0.25, z_exponent=0.0, axis_phase_exponent=0.0)
    deep_circuit = cirq.Circuit([cirq.Moment(pxz.on(q0)) for _ in range(MAX_MOMENTS + 2)])
    deep_circuit.append(cirq.measure(q0, key="m"))
    res = lint_willow_circuit(
        deep_circuit,
        {q0},
        num_qubits=1,
        cz_pauli_error_rate=env["cz_pauli_error_rate"],
        sq_pauli_error_rate=env["sq_pauli_error_rate"],
    )
    assert not res.passed
    rules = {e.rule for e in res.errors}
    assert "MOMENT_LIMIT_EXCEEDED" in rules
    assert "MID_CIRCUIT_MEASUREMENT_FORBIDDEN" in rules
