"""Static hardware rule linter for Google Willow Dicke state circuits."""

from dataclasses import dataclass, field
from typing import Any, Iterable
import numpy as np
import cirq
import cirq_google

from src.willow_utils import lookup_coupler_error

MAX_MOMENTS = 1750


@dataclass(frozen=True)
class LintViolation:
    """Single hardware rule violation or warning with exact moment/qubit location."""

    rule: str
    severity: str
    moment: int | None
    qubits: tuple[cirq.Qid, ...]
    operation: str
    message: str


@dataclass
class LintResult:
    """Structured result of running the Willow hardware linter."""

    passed: bool
    errors: list[LintViolation] = field(default_factory=list)
    warnings: list[LintViolation] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _is_exact_cz(gate: cirq.Gate | None) -> bool:
    return (
        isinstance(gate, cirq.CZPowGate)
        and np.isclose(float(gate.exponent), 1.0, atol=1e-9)
        and np.isclose(float(gate.global_shift), 0.0, atol=1e-9)
    )


def lint_willow_circuit(
    circuit: cirq.Circuit,
    qubits: Iterable[cirq.Qid],
    num_qubits: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    max_moments: int = MAX_MOMENTS,
    validate_willow_device: bool = True,
) -> LintResult:
    """Statically validate that a circuit obeys every Google Willow hardware rule."""
    errors: list[LintViolation] = []
    warnings: list[LintViolation] = []

    if not isinstance(qubits, (set, frozenset)):
        warnings.append(
            LintViolation(
                rule="RETURN_TYPE_NOT_SET",
                severity="warning",
                moment=None,
                qubits=tuple(qubits),
                operation="return value",
                message=f"Returned qubits is {type(qubits).__name__}, expected set[cirq.GridQubit].",
            )
        )

    qubit_set = set(qubits)
    if len(qubit_set) != num_qubits:
        errors.append(
            LintViolation(
                rule="QUBIT_COUNT_MISMATCH",
                severity="error",
                moment=None,
                qubits=tuple(sorted(qubit_set, key=str)),
                operation="return value",
                message=f"Returned {len(qubit_set)} target qubits, expected num_qubits={num_qubits}.",
            )
        )

    non_grid = [q for q in qubit_set if not isinstance(q, cirq.GridQubit)]
    if non_grid:
        errors.append(
            LintViolation(
                rule="NON_GRID_QUBIT",
                severity="error",
                moment=None,
                qubits=tuple(non_grid),
                operation="return value",
                message=f"Target qubits must be cirq.GridQubit instances; found {[type(q).__name__ for q in non_grid]}.",
            )
        )

    dropped_targets = [
        q for q in qubit_set if isinstance(q, cirq.GridQubit) and q not in sq_pauli_error_rate
    ]
    if dropped_targets:
        errors.append(
            LintViolation(
                rule="TARGET_QUBIT_NOT_IN_SQ_RATES",
                severity="error",
                moment=None,
                qubits=tuple(dropped_targets),
                operation="return value",
                message=f"Target qubits {dropped_targets} are missing from sq_pauli_error_rate (broken/dropout qubits).",
            )
        )

    num_moments = len(circuit)
    if num_moments > max_moments:
        errors.append(
            LintViolation(
                rule="MOMENT_LIMIT_EXCEEDED",
                severity="error",
                moment=num_moments - 1,
                qubits=(),
                operation=f"Circuit({num_moments} moments)",
                message=f"Circuit has {num_moments} moments, exceeding the maximum allowed {max_moments}.",
            )
        )

    cz_count = 0
    cz_moments = 0
    sq_count = 0
    reset_count = 0

    for m_idx, moment in enumerate(circuit):
        moment_has_cz = False
        for raw_op in moment:
            op = raw_op.untagged if isinstance(raw_op, cirq.TaggedOperation) else raw_op
            if isinstance(raw_op, cirq.TaggedOperation):
                warnings.append(
                    LintViolation(
                        rule="TAGGED_OPERATION",
                        severity="warning",
                        moment=m_idx,
                        qubits=raw_op.qubits,
                        operation=str(raw_op),
                        message=f"Operation at moment {m_idx} carries tags {raw_op.tags}.",
                    )
                )

            if isinstance(op, cirq.ClassicallyControlledOperation) or getattr(
                op, "classical_controls", ()
            ):
                errors.append(
                    LintViolation(
                        rule="CLASSICALLY_CONDITIONED_OP",
                        severity="error",
                        moment=m_idx,
                        qubits=op.qubits,
                        operation=str(op),
                        message=f"Classically conditioned operation '{op}' at moment {m_idx} is forbidden in the standard track.",
                    )
                )
                continue

            if cirq.is_measurement(op) or isinstance(op.gate, cirq.MeasurementGate):
                errors.append(
                    LintViolation(
                        rule="MID_CIRCUIT_MEASUREMENT_FORBIDDEN",
                        severity="error",
                        moment=m_idx,
                        qubits=op.qubits,
                        operation=str(op),
                        message=f"Measurement operation '{op}' at moment {m_idx} is not allowed in the standard track.",
                    )
                )
                continue

            if cirq.is_parameterized(op):
                errors.append(
                    LintViolation(
                        rule="PARAMETERIZED_OPERATION",
                        severity="error",
                        moment=m_idx,
                        qubits=op.qubits,
                        operation=str(op),
                        message=f"Parameterized operation '{op}' at moment {m_idx} is not allowed.",
                    )
                )
                continue

            op_non_grid = [q for q in op.qubits if not isinstance(q, cirq.GridQubit)]
            if op_non_grid:
                errors.append(
                    LintViolation(
                        rule="NON_GRID_QUBIT",
                        severity="error",
                        moment=m_idx,
                        qubits=op.qubits,
                        operation=str(op),
                        message=f"Operation '{op}' at moment {m_idx} acts on non-GridQubit {op_non_grid}.",
                    )
                )
                continue

            n_q = len(op.qubits)
            if n_q == 1:
                q0 = op.qubits[0]
                if q0 not in sq_pauli_error_rate:
                    errors.append(
                        LintViolation(
                            rule="SQ_QUBIT_NOT_IN_ERROR_RATES",
                            severity="error",
                            moment=m_idx,
                            qubits=(q0,),
                            operation=str(op),
                            message=f"1Q operation '{op}' at moment {m_idx} acts on qubit {q0} missing from sq_pauli_error_rate.",
                        )
                    )
                if isinstance(op.gate, cirq.PhasedXZGate):
                    sq_count += 1
                elif isinstance(op.gate, cirq.ResetChannel):
                    reset_count += 1
                else:
                    errors.append(
                        LintViolation(
                            rule="INVALID_1Q_GATE",
                            severity="error",
                            moment=m_idx,
                            qubits=(q0,),
                            operation=str(op),
                            message=(
                                f"1Q gate '{type(op.gate).__name__}' on {q0} at moment {m_idx} is not allowed; "
                                "must be cirq.PhasedXZGate or cirq.ResetChannel."
                            ),
                        )
                    )
            elif n_q == 2:
                q0, q1 = op.qubits
                if not _is_exact_cz(op.gate):
                    errors.append(
                        LintViolation(
                            rule="INVALID_2Q_GATE",
                            severity="error",
                            moment=m_idx,
                            qubits=(q0, q1),
                            operation=str(op),
                            message=(
                                f"2Q gate '{op.gate}' on ({q0}, {q1}) at moment {m_idx} is not allowed; "
                                "must be cirq.CZ (CZPowGate with exponent=1.0)."
                            ),
                        )
                    )
                else:
                    cz_count += 1
                    moment_has_cz = True

                coupler_err = lookup_coupler_error(q0, q1, cz_pauli_error_rate)
                if coupler_err is None:
                    errors.append(
                        LintViolation(
                            rule="CZ_COUPLER_NOT_IN_ERROR_RATES",
                            severity="error",
                            moment=m_idx,
                            qubits=(q0, q1),
                            operation=str(op),
                            message=(
                                f"2Q operation '{op}' at moment {m_idx} uses coupler ({q0}, {q1}) "
                                "not present in cz_pauli_error_rate (non-adjacent or broken coupler)."
                            ),
                        )
                    )
            else:
                errors.append(
                    LintViolation(
                        rule="MULTI_QUBIT_GATE_FORBIDDEN",
                        severity="error",
                        moment=m_idx,
                        qubits=op.qubits,
                        operation=str(op),
                        message=f"{n_q}-qubit operation '{op}' at moment {m_idx} is not native to Willow.",
                    )
                )

        if moment_has_cz:
            cz_moments += 1

    all_circuit_qubits = circuit.all_qubits()
    ancilla_qubits = all_circuit_qubits.difference(qubit_set)
    untouched_targets = qubit_set.difference(all_circuit_qubits)
    if untouched_targets and num_qubits > 1:
        warnings.append(
            LintViolation(
                rule="UNTOUCHED_TARGET_QUBITS",
                severity="warning",
                moment=None,
                qubits=tuple(sorted(untouched_targets, key=str)),
                operation="Circuit",
                message=f"Target qubits {sorted(untouched_targets, key=str)} have no operations in the circuit.",
            )
        )

    if validate_willow_device and not errors:
        try:
            cirq_google.Willow105.validate_circuit(circuit)
        except Exception as exc:
            errors.append(
                LintViolation(
                    rule="WILLOW105_DEVICE_VALIDATION_FAILED",
                    severity="error",
                    moment=None,
                    qubits=(),
                    operation="Willow105.validate_circuit",
                    message=f"cirq_google.Willow105.validate_circuit raised: {exc}",
                )
            )

    active_total = len(all_circuit_qubits | qubit_set)
    busy_slots = sum(len(m.qubits) for m in circuit)
    idle_slots = max(0, num_moments * active_total - busy_slots)

    stats = {
        "num_moments": num_moments,
        "cz_count": cz_count,
        "cz_depth": cz_moments,
        "sq_count": sq_count,
        "reset_count": reset_count,
        "target_qubits": len(qubit_set),
        "ancilla_qubits": len(ancilla_qubits),
        "active_qubits": active_total,
        "idle_qubit_moments": idle_slots,
    }

    return LintResult(
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        stats=stats,
    )
