"""AST sanctity guard verifying that official notebook helpers are untouched."""

import ast
from pathlib import Path
import warnings
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = REPO_ROOT / "reference" / "DuQuantum_Google_Standard.py"
HELPERS_PATH = REPO_ROOT / "src" / "official" / "notebook_helpers.py"

VERBATIM_HELPER_NAMES = [
    "create_dicke_statevector",
    "check_z_bitstring_distribution",
    "get_correlator_average",
    "check_x_bitstring_distribution",
    "overall_score",
    "pick_single_qubit_dropouts",
    "generate_qubit_error_rates",
    "generate_two_qubit_pauli_error_rates",
]


def _extract_functions(path: Path) -> dict[str, ast.FunctionDef]:
    raw = path.read_text(encoding="utf-8")
    clean_lines = [
        line for line in raw.splitlines() if not line.lstrip().startswith("!pip")
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse("\n".join(clean_lines))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


@pytest.mark.parametrize("func_name", VERBATIM_HELPER_NAMES)
def test_official_helper_ast_is_verbatim(func_name: str):
    """Ensure helper function AST is 100% identical to the original Colab file."""
    ref_funcs = _extract_functions(REFERENCE_PATH)
    mod_funcs = _extract_functions(HELPERS_PATH)
    assert func_name in ref_funcs, f"{func_name} missing in reference"
    assert func_name in mod_funcs, f"{func_name} missing in notebook_helpers.py"
    assert ast.dump(ref_funcs[func_name]) == ast.dump(mod_funcs[func_name])


def test_official_test_circuit_has_only_organizer_erratum():
    """Verify test_circuit differs from reference only by `range(qubits)` -> `range(len(qubits))`."""
    ref_raw = REFERENCE_PATH.read_text(encoding="utf-8")
    patched_ref = ref_raw.replace(
        "keep_indices = range(qubits)", "keep_indices = range(len(qubits))"
    )
    clean_lines = [
        line for line in patched_ref.splitlines() if not line.lstrip().startswith("!pip")
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        ref_tree = ast.parse("\n".join(clean_lines))
    ref_fn = next(
        n for n in ref_tree.body if isinstance(n, ast.FunctionDef) and n.name == "test_circuit"
    )
    mod_funcs = _extract_functions(HELPERS_PATH)
    assert ast.dump(ref_fn) == ast.dump(mod_funcs["test_circuit"])
