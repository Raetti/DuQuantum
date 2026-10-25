"""Static hardware linting and exact mathematical fidelity verification."""

from src.verification.fidelity import (
    FidelityReport,
    QubitSet,
    create_dicke_statevector,
    verify_dicke_state,
)
from src.verification.linter import (
    MAX_MOMENTS,
    LintResult,
    LintViolation,
    lint_willow_circuit,
)

__all__ = [
    "MAX_MOMENTS",
    "LintViolation",
    "LintResult",
    "lint_willow_circuit",
    "QubitSet",
    "FidelityReport",
    "create_dicke_statevector",
    "verify_dicke_state",
]
