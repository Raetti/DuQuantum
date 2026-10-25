"""End-to-end tests for the unified CLI runner `scripts/test_algorithm.py`."""

from pathlib import Path
from scripts.test_algorithm import main


def test_cli_all_stages_passes(tmp_path: Path):
    json_out = tmp_path / "report.json"
    code = main(
        [
            "--N",
            "4",
            "--n",
            "2",
            "--stage",
            "all",
            "--shots",
            "2000",
            "--json",
            str(json_out),
        ]
    )
    assert code == 0
    assert json_out.exists()


def test_cli_team_stub_reports_not_implemented():
    code = main(["--N", "4", "--n", "2", "--stage", "1", "--candidate", "team"])
    assert code == 0


def test_cli_sweep_stages_1_to_3():
    code = main(["--sweep", "2-4", "--stage", "1-3"])
    assert code == 0
