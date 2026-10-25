#!/usr/bin/env python3
"""Unified 4-stage diagnostic CLI runner for DuQuantum Dicke state algorithms.

Stages:
  1. Pure Mathematical Logic (Abstract Noiseless on LineQubits)
  2. Willow Hardware Rule Linting (Gateset, Couplers, Dropouts, <=1750 Moments)
  3. Compiled Noiseless Fidelity & Official `test_circuit` Check
  4. Noisy Willow QVM Simulation & Official 10,000-Shot Scoring

Examples:
  python scripts/test_algorithm.py --N 4 --n 2 --stage all
  python scripts/test_algorithm.py --N 5 --n 2 --stage 1 --candidate team
  python scripts/test_algorithm.py --N 4 --n 2 --stage all --ablate
  python scripts/test_algorithm.py --sweep 2-6 --stage 1-3
"""

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Optional

# Ensure repository root is on sys.path when invoked as `python scripts/test_algorithm.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.candidate_algorithms import CANDIDATES, CandidateSpec
from src.noise_models import sample_benchmark_environment
from src.simulation.noisy_evaluator import (
    DEFAULT_T1_NS,
    DEFAULT_T2_NS,
    evaluate_noisy_circuit,
)
from src.verification.fidelity import verify_dicke_state
from src.verification.linter import MAX_MOMENTS, lint_willow_circuit

HARD_TIMEOUT_SECONDS = 60.0
console = Console()


def parse_stages(stage_arg: str) -> list[int]:
    """Parse --stage argument into a sorted list of stage numbers in {1, 2, 3, 4}."""
    val = stage_arg.strip().lower()
    if val == "all":
        return [1, 2, 3, 4]
    if val in ("1-3", "noiseless"):
        return [1, 2, 3]
    if val in ("2-3", "compiled"):
        return [2, 3]
    if val in ("2-4", "native"):
        return [2, 3, 4]
    stages: set[int] = set()
    for part in val.split(","):
        part = part.strip()
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            for s in range(int(lo_s), int(hi_s) + 1):
                if s not in (1, 2, 3, 4):
                    raise ValueError(f"Invalid stage number: {s}")
                stages.add(s)
        else:
            s = int(part)
            if s not in (1, 2, 3, 4):
                raise ValueError(f"Invalid stage number: {s}")
            stages.add(s)
    return sorted(stages)


def resolve_candidate(spec_name: str) -> CandidateSpec:
    """Resolve a candidate by registry name or `module:function` import path."""
    if spec_name in CANDIDATES:
        return CANDIDATES[spec_name]
    if ":" in spec_name:
        mod_name, fn_name = spec_name.split(":", 1)
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, fn_name)
        return CandidateSpec(
            name=spec_name,
            description=f"Custom imported function {spec_name}",
            prototype_fn=None,
            native_fn=fn,
        )
    raise ValueError(
        f"Unknown candidate '{spec_name}'. Available: {list(CANDIDATES.keys())} or module:function."
    )


def run_single_pair(
    candidate: CandidateSpec,
    num_qubits: int,
    num_excitations: int,
    stages: list[int],
    env: dict[str, Any],
    n_shots: int = 10_000,
    max_corr: Optional[int] = None,
    noise_backend: str = "qvm",
    t1_ns: float = DEFAULT_T1_NS,
    t2_ns: float = DEFAULT_T2_NS,
    seed: int = 42,
    ablate: bool = False,
    keep_going: bool = False,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the selected evaluation stages for a single (N, n) pair."""
    report: dict[str, Any] = {
        "candidate": candidate.name,
        "N": num_qubits,
        "n": num_excitations,
        "stages_requested": stages,
        "verdict": "PASS",
        "failed_stage": None,
        "stage1": None,
        "stage2": None,
        "stage3": None,
        "stage4": None,
    }

    if not quiet:
        console.rule(
            f"[bold cyan]Evaluating |D^({num_qubits})_{num_excitations}> "
            f"[white](candidate: [bold yellow]{candidate.name}[/bold yellow], stages: {stages})"
        )

    # ------------------------------------------------------------------
    # STAGE 1: Pure Mathematical Logic (Abstract Prototype)
    # ------------------------------------------------------------------
    if 1 in stages:
        if candidate.prototype_fn is None:
            if not quiet:
                console.print(
                    "[yellow]Stage 1 skipped:[/yellow] Candidate provides only a native Willow function."
                )
        else:
            t0 = time.perf_counter()
            try:
                proto_circ, proto_qubits = candidate.prototype_fn(
                    num_qubits, num_excitations
                )
                elapsed = time.perf_counter() - t0
            except NotImplementedError as exc:
                report["verdict"] = "NOT_IMPLEMENTED"
                report["stage1"] = {"status": "NOT_IMPLEMENTED", "message": str(exc)}
                if not quiet:
                    console.print(
                        Panel(
                            f"[yellow]NotImplementedError:[/yellow] {exc}",
                            title="Stage 1: Abstract Math (Not Implemented)",
                            border_style="yellow",
                        )
                    )
                return report

            s1_fid = verify_dicke_state(
                proto_circ,
                proto_qubits,
                num_qubits=num_qubits,
                num_excitations=num_excitations,
                run_official_test=False,
            )
            report["stage1"] = {
                "passed": s1_fid.passed,
                "compile_time_s": elapsed,
                "num_moments": len(proto_circ),
                "num_operations": sum(len(m) for m in proto_circ),
                **asdict(s1_fid),
            }

            if not quiet:
                status_str = "[bold green]PASS[/bold green]" if s1_fid.passed else "[bold red]FAIL[/bold red]"
                t1 = Table(title=f"Stage 1: Pure Mathematical Logic ({status_str})", show_header=True)
                t1.add_column("Metric", style="cyan")
                t1.add_column("Value", style="white")
                t1.add_row("Statevector Fidelity |<D|psi>|^2", f"{s1_fid.fidelity:.8f}")
                t1.add_row("Weight-n Subspace Probability", f"{s1_fid.subspace_prob:.8f}")
                t1.add_row("Max Amplitude Deviation", f"{s1_fid.amplitude_uniformity_max_err:.6e}")
                t1.add_row("Max Relative Phase Spread (rad)", f"{s1_fid.phase_spread_rad:.6e}")
                t1.add_row("Abstract Moments / Ops", f"{len(proto_circ)} / {sum(len(m) for m in proto_circ)}")
                t1.add_row("Generation Time", f"{elapsed * 1000:.2f} ms")
                console.print(t1)
                for msg in s1_fid.messages:
                    console.print(f"  - {msg}")

            if not s1_fid.passed:
                report["verdict"] = "STAGE_1_MATH_ERROR"
                report["failed_stage"] = 1
                if not keep_going:
                    return report

    # ------------------------------------------------------------------
    # Native Circuit Generation (Required for Stages 2, 3, 4)
    # ------------------------------------------------------------------
    native_stages = [s for s in stages if s in (2, 3, 4)]
    if not native_stages:
        return report

    t0 = time.perf_counter()
    try:
        native_circuit, native_qubits = candidate.native_fn(
            num_qubits,
            num_excitations,
            env["cz_pauli_error_rate"],
            env["sq_pauli_error_rate"],
            env["zero_errors"],
            env["one_errors"],
        )
        compile_elapsed = time.perf_counter() - t0
    except NotImplementedError as exc:
        report["verdict"] = "NOT_IMPLEMENTED"
        if not quiet:
            console.print(
                Panel(
                    f"[yellow]NotImplementedError:[/yellow] {exc}",
                    title="Native Willow Circuit (Not Implemented)",
                    border_style="yellow",
                )
            )
        return report

    timed_out = compile_elapsed > HARD_TIMEOUT_SECONDS

    # ------------------------------------------------------------------
    # STAGE 2: Willow Hardware Rule Linter
    # ------------------------------------------------------------------
    if 2 in stages:
        lint_res = lint_willow_circuit(
            circuit=native_circuit,
            qubits=native_qubits,
            num_qubits=num_qubits,
            cz_pauli_error_rate=env["cz_pauli_error_rate"],
            sq_pauli_error_rate=env["sq_pauli_error_rate"],
            max_moments=MAX_MOMENTS,
        )
        s2_passed = lint_res.passed and (not timed_out)
        report["stage2"] = {
            "passed": s2_passed,
            "compile_time_s": compile_elapsed,
            "timed_out": timed_out,
            "stats": lint_res.stats,
            "errors": [
                {
                    "rule": v.rule,
                    "moment": v.moment,
                    "qubits": [str(q) for q in v.qubits],
                    "operation": v.operation,
                    "message": v.message,
                }
                for v in lint_res.errors
            ],
            "warnings": [
                {
                    "rule": v.rule,
                    "moment": v.moment,
                    "qubits": [str(q) for q in v.qubits],
                    "operation": v.operation,
                    "message": v.message,
                }
                for v in lint_res.warnings
            ],
        }

        if not quiet:
            status_str = "[bold green]PASS[/bold green]" if s2_passed else "[bold red]FAIL[/bold red]"
            t2 = Table(title=f"Stage 2: Willow Hardware Rule Linter ({status_str})", show_header=True)
            t2.add_column("Check / Resource", style="cyan")
            t2.add_column("Value", style="white")
            st = lint_res.stats
            t2.add_row("Total Moments (limit <= 1750)", f"{st['num_moments']} / {MAX_MOMENTS}")
            t2.add_row("Native 2Q CZ Count / CZ Depth", f"{st['cz_count']} / {st['cz_depth']}")
            t2.add_row("Native 1Q PhasedXZ Count", f"{st['sq_count']}")
            t2.add_row("Target Qubits / Ancillas", f"{st['target_qubits']} / {st['ancilla_qubits']}")
            t2.add_row("Selected Willow Qubits", ", ".join(str(q) for q in sorted(native_qubits)))
            t2.add_row("Idle Qubit-Moments", f"{st['idle_qubit_moments']}")
            t2.add_row("Compilation Time (limit < 60s)", f"{compile_elapsed:.3f} s")
            console.print(t2)

            if timed_out:
                console.print(
                    f"  [bold red]- TIMEOUT:[/bold red] Compilation took {compile_elapsed:.2f}s (> 60s limit)."
                )
            for err in lint_res.errors:
                loc = f"moment {err.moment}" if err.moment is not None else "circuit"
                console.print(f"  [bold red]- [{err.rule}] ({loc}):[/bold red] {err.message}")
            for warn in lint_res.warnings:
                loc = f"moment {warn.moment}" if warn.moment is not None else "circuit"
                console.print(f"  [yellow]- [WARN:{warn.rule}] ({loc}):[/yellow] {warn.message}")

        if not s2_passed and report["verdict"] == "PASS":
            report["verdict"] = "STAGE_2_HARDWARE_LINT_ERROR"
            report["failed_stage"] = 2
            if not keep_going:
                return report

    # ------------------------------------------------------------------
    # STAGE 3: Compiled Noiseless Fidelity & Official `test_circuit`
    # ------------------------------------------------------------------
    if 3 in stages:
        s3_fid = verify_dicke_state(
            native_circuit,
            native_qubits,
            num_qubits=num_qubits,
            num_excitations=num_excitations,
            run_official_test=True,
        )
        report["stage3"] = asdict(s3_fid)

        if not quiet:
            status_str = "[bold green]PASS[/bold green]" if s3_fid.passed else "[bold red]FAIL[/bold red]"
            t3 = Table(title=f"Stage 3: Compiled Noiseless Fidelity ({status_str})", show_header=True)
            t3.add_column("Metric", style="cyan")
            t3.add_column("Value", style="white")
            t3.add_row("Compiled Fidelity |<D|psi>|^2", f"{s3_fid.fidelity:.8f}")
            t3.add_row(
                "Official notebook test_circuit()",
                "[green]PASS[/green]" if s3_fid.official_test_passed else "[red]FAIL[/red]",
            )
            t3.add_row("Ancilla Count / Disentangled", f"{s3_fid.num_ancillas} / {not s3_fid.ancillas_entangled}")
            t3.add_row("P(ancillas = |0...0>)", f"{s3_fid.ancilla_zero_prob:.8f}")
            t3.add_row("Weight-n Subspace Probability", f"{s3_fid.subspace_prob:.8f}")
            t3.add_row("Max Relative Phase Spread (rad)", f"{s3_fid.phase_spread_rad:.6e}")
            console.print(t3)
            for msg in s3_fid.messages:
                console.print(f"  - {msg}")
            for warn in s3_fid.warnings:
                console.print(f"  [yellow]- {warn}[/yellow]")

        if not s3_fid.passed and report["verdict"] == "PASS":
            report["verdict"] = "STAGE_3_COMPILATION_ERROR"
            report["failed_stage"] = 3
            if not keep_going:
                return report

    # ------------------------------------------------------------------
    # STAGE 4: Noisy Willow QVM Simulation & Official Harness Scoring
    # ------------------------------------------------------------------
    if 4 in stages:
        noisy_res = evaluate_noisy_circuit(
            circuit=native_circuit,
            qubits=native_qubits,
            num_qubits=num_qubits,
            num_excitations=num_excitations,
            cz_pauli_error_rate=env["cz_pauli_error_rate"],
            sq_pauli_error_rate=env["sq_pauli_error_rate"],
            zero_errors=env["zero_errors"],
            one_errors=env["one_errors"],
            n_shots=n_shots,
            max_correlator_size=max_corr,
            noise_backend=noise_backend,
            t1_ns=t1_ns,
            t2_ns=t2_ns,
            seed=seed,
            run_ablation=ablate,
        )
        report["stage4"] = asdict(noisy_res)

        if not quiet:
            t4 = Table(
                title=f"Stage 4: Noisy Willow QVM & Harness Score (backend={noise_backend}, shots={n_shots})",
                show_header=True,
            )
            t4.add_column("Metric", style="cyan")
            t4.add_column("Value", style="white")
            t4.add_row("Simulation Engine", noisy_res.simulation_method)
            t4.add_row("Z-Basis Subspace Fraction (weight == n)", f"{noisy_res.z_subspace_fraction:.4%}")
            t4.add_row("Official Z-Basis Score (lower is better)", f"{noisy_res.z_score:.6f}")
            t4.add_row(
                f"Official X-Basis Score (k_max={noisy_res.max_correlator_size}, lower is better)",
                f"{noisy_res.x_score:.6f}",
            )
            t4.add_row(
                "[bold yellow]Official Overall Score (higher is better)[/bold yellow]",
                f"[bold yellow]{noisy_res.overall_score:.4f}[/bold yellow]",
            )
            t4.add_row(
                "10k-Shot Noiseless Statistical Ceiling",
                f"{noisy_res.noiseless_ceiling_score:.4f}",
            )
            console.print(t4)

            if len(noisy_res.correlator_sensitivity) > 1:
                tk = Table(title="Undisclosed max_correlator_size Sensitivity", show_header=True)
                tk.add_column("max_correlator_size", style="cyan")
                tk.add_column("X-Basis Score", style="white")
                tk.add_column("Overall Score", style="yellow")
                for k_val, (xk, ovk) in sorted(noisy_res.correlator_sensitivity.items()):
                    tk.add_row(str(k_val), f"{xk:.6f}", f"{ovk:.4f}")
                console.print(tk)

            if noisy_res.ablation_scores:
                ta = Table(title="Noise Source Ablation Breakdown", show_header=True)
                ta.add_column("Noise Configuration", style="cyan")
                ta.add_column("Z Subspace %", style="white")
                ta.add_column("Z Score", style="white")
                ta.add_column("X Score", style="white")
                ta.add_column("Overall Score", style="yellow")
                for mode_name, vals in noisy_res.ablation_scores.items():
                    ta.add_row(
                        mode_name,
                        f"{vals['z_subspace_fraction']:.2%}",
                        f"{vals['z_score']:.6f}",
                        f"{vals['x_score']:.6f}",
                        f"{vals['overall_score']:.4f}",
                    )
                console.print(ta)

    return report


def parse_sweep_range(sweep_str: str) -> list[tuple[int, int]]:
    """Parse --sweep 'lo-hi' into all (N, n) pairs with lo <= N <= hi and 1 <= n <= N // 2."""
    if "-" in sweep_str:
        lo_s, hi_s = sweep_str.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
    else:
        lo = hi = int(sweep_str)
    pairs: list[tuple[int, int]] = []
    for n_q in range(max(2, lo), hi + 1):
        for n_ex in range(1, (n_q // 2) + 1):
            pairs.append((n_q, n_ex))
    return pairs


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-stage diagnostic test suite and Willow QVM runner for Dicke state algorithms."
    )
    parser.add_argument("--N", type=int, default=4, help="Number of qubits N (default: 4)")
    parser.add_argument("--n", type=int, default=2, help="Number of excitations n (default: 2)")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Stages to run: 1, 2, 3, 4, 1-3, 2-4, or all (default: all)",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="default",
        help=f"Candidate algorithm name ({', '.join(CANDIDATES.keys())}) or module:func",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="Sweep all (N, n<=N/2) in range e.g. '2-6' instead of a single (--N, --n)",
    )
    parser.add_argument(
        "--noise-backend",
        choices=["qvm", "gate-readout", "noiseless-shots"],
        default="qvm",
        help="Stage 4 noise model mode (default: qvm with Willow spec-sheet T1/T2)",
    )
    parser.add_argument("--shots", type=int, default=10_000, help="Shots per basis in Stage 4 (default: 10000)")
    parser.add_argument("--max-corr", type=int, default=None, help="max_correlator_size for X basis (default: N)")
    parser.add_argument("--t1", type=float, default=DEFAULT_T1_NS, help="Willow T1 in ns (default: 68000)")
    parser.add_argument("--t2", type=float, default=DEFAULT_T2_NS, help="Willow T2_CPMG in ns (default: 89000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for Willow error rates and sampling")
    parser.add_argument("--ablate", action="store_true", help="Run Stage 4 noise source ablation")
    parser.add_argument("--keep-going", action="store_true", help="Continue to later stages even if an earlier stage fails")
    parser.add_argument("--json", type=str, default=None, help="Write structured JSON report to path")

    args = parser.parse_args(argv)
    stages = parse_stages(args.stage)
    candidate = resolve_candidate(args.candidate)
    env = sample_benchmark_environment(seed=args.seed)

    pairs = parse_sweep_range(args.sweep) if args.sweep else [(args.N, args.n)]
    all_reports: list[dict[str, Any]] = []

    for n_q, n_ex in pairs:
        rep = run_single_pair(
            candidate=candidate,
            num_qubits=n_q,
            num_excitations=n_ex,
            stages=stages,
            env=env,
            n_shots=args.shots,
            max_corr=args.max_corr,
            noise_backend=args.noise_backend,
            t1_ns=args.t1,
            t2_ns=args.t2,
            seed=args.seed,
            ablate=args.ablate,
            keep_going=args.keep_going,
            quiet=False,
        )
        all_reports.append(rep)

    if len(all_reports) > 1:
        summary = Table(title=f"Sweep Summary ({candidate.name})", show_header=True)
        summary.add_column("(N, n)", style="cyan")
        summary.add_column("Verdict", style="white")
        summary.add_column("Moments", style="white")
        summary.add_column("CZ Count", style="white")
        summary.add_column("Noiseless Fidelity", style="white")
        summary.add_column("Overall Score", style="yellow")
        for r in all_reports:
            v = r["verdict"]
            v_str = f"[green]{v}[/green]" if v == "PASS" else (f"[yellow]{v}[/yellow]" if v == "NOT_IMPLEMENTED" else f"[red]{v}[/red]")
            moments_s = str(r["stage2"]["stats"]["num_moments"]) if r.get("stage2") else "-"
            cz_s = str(r["stage2"]["stats"]["cz_count"]) if r.get("stage2") else "-"
            fid_s = f"{r['stage3']['fidelity']:.6f}" if r.get("stage3") else (f"{r['stage1']['fidelity']:.6f}" if r.get("stage1") and "fidelity" in r["stage1"] else "-")
            score_s = f"{r['stage4']['overall_score']:.4f}" if r.get("stage4") else "-"
            summary.add_row(f"({r['N']}, {r['n']})", v_str, moments_s, cz_s, fid_s, score_s)
        console.print(summary)

    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_reports, indent=2, default=str), encoding="utf-8")
        console.print(f"[dim]Saved JSON diagnostic report to {out_path}[/dim]")

    any_failed = any(r["verdict"] not in ("PASS", "NOT_IMPLEMENTED") for r in all_reports)
    final_verdicts = [r["verdict"] for r in all_reports]
    if any_failed:
        console.print(
            Panel(
                f"[bold red]Diagnostic Failure Detected:[/bold red] {final_verdicts}",
                border_style="red",
            )
        )
        return 1

    console.print(
        Panel(
            f"[bold green]Completed Stages {stages}:[/bold green] {final_verdicts}",
            border_style="green",
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
