"""Exact statevector fidelity and physics-derivation diagnostic verifier."""

from dataclasses import dataclass, field
import itertools
import math
from typing import Iterable, Optional
import numpy as np
import cirq

from src.official.notebook_helpers import create_dicke_statevector, test_circuit


class QubitSet(frozenset):
    """Frozenset of qubits that also implements __index__ -> len(self).

    Allows even the unpatched notebook `keep_indices = range(qubits)` line to
    succeed without modifying the helper's source code.
    """

    def __index__(self) -> int:
        return len(self)


@dataclass
class FidelityReport:
    """Detailed noiseless mathematical verification and diagnostic report."""

    passed: bool
    fidelity: float
    num_qubits: int
    num_excitations: int
    num_ancillas: int
    ancillas_entangled: bool
    ancilla_purity: float
    ancilla_zero_prob: float
    subspace_prob: float
    subspace_leakage: float
    amplitude_uniformity_max_err: float
    phase_spread_rad: float
    official_test_passed: Optional[bool]
    leaked_states: list[tuple[str, float]] = field(default_factory=list)
    amplitude_outliers: list[tuple[str, float, float]] = field(default_factory=list)
    phase_outliers: list[tuple[str, float]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _bitstring_label(index: int, num_qubits: int) -> str:
    return format(index, f"0{num_qubits}b")


def verify_dicke_state(
    circuit: cirq.Circuit,
    qubits: Iterable[cirq.Qid],
    num_qubits: int,
    num_excitations: int,
    atol: float = 1e-5,
    run_official_test: bool = True,
    max_statevector_qubits: int = 22,
) -> FidelityReport:
    """Verify exact Dicke state preparation and diagnose mathematical errors."""
    target_set = set(qubits)
    if len(target_set) != num_qubits:
        return FidelityReport(
            passed=False,
            fidelity=0.0,
            num_qubits=num_qubits,
            num_excitations=num_excitations,
            num_ancillas=0,
            ancillas_entangled=False,
            ancilla_purity=0.0,
            ancilla_zero_prob=0.0,
            subspace_prob=0.0,
            subspace_leakage=1.0,
            amplitude_uniformity_max_err=1.0,
            phase_spread_rad=0.0,
            official_test_passed=False,
            messages=[
                f"Target qubit set has size {len(target_set)}, expected {num_qubits}."
            ],
        )

    all_qubits = circuit.all_qubits() | target_set
    ancilla_qubits = all_qubits.difference(target_set)
    ordered_targets = sorted(target_set)
    ordered_ancillas = sorted(ancilla_qubits)
    ordered_qubits = ordered_targets + ordered_ancillas
    total_qubits = len(ordered_qubits)

    if total_qubits > max_statevector_qubits:
        return FidelityReport(
            passed=False,
            fidelity=float("nan"),
            num_qubits=num_qubits,
            num_excitations=num_excitations,
            num_ancillas=len(ordered_ancillas),
            ancillas_entangled=False,
            ancilla_purity=float("nan"),
            ancilla_zero_prob=float("nan"),
            subspace_prob=float("nan"),
            subspace_leakage=float("nan"),
            amplitude_uniformity_max_err=float("nan"),
            phase_spread_rad=float("nan"),
            official_test_passed=None,
            messages=[
                f"Total qubits ({total_qubits}) exceeds statevector limit ({max_statevector_qubits})."
            ],
        )

    sampler = cirq.Simulator(dtype=np.complex128)
    sim_res = sampler.simulate(circuit, qubit_order=ordered_qubits)
    psi_full = np.asarray(sim_res.final_state_vector, dtype=np.complex128)

    messages: list[str] = []
    warnings: list[str] = []
    n_anc = len(ordered_ancillas)
    ancillas_entangled = False
    ancilla_purity = 1.0
    ancilla_zero_prob = 1.0

    if n_anc > 0:
        psi_mat = psi_full.reshape((2**num_qubits, 2**n_anc))
        ancilla_zero_prob = float(np.sum(np.abs(psi_mat[:, 0]) ** 2))
        singular_vals = np.linalg.svd(psi_mat, compute_uv=False)
        ancilla_purity = float(np.sum(singular_vals**4))
        dominant_weight = float(singular_vals[0] ** 2) if len(singular_vals) else 0.0

        if dominant_weight < 1.0 - atol:
            ancillas_entangled = True
            messages.append(
                f"Ancilla qubits failed to disentangle: target purity={ancilla_purity:.6f} "
                f"(dominant Schmidt weight={dominant_weight:.6f} < 1.0)."
            )
            # Project onto |0...0> ancilla subspace to inspect residual target state
            norm0 = np.linalg.norm(psi_mat[:, 0])
            psi_target = psi_mat[:, 0] / norm0 if norm0 > 1e-12 else psi_mat[:, 0]
        else:
            if ancilla_zero_prob < 1.0 - atol:
                warnings.append(
                    f"Ancillas disentangled cleanly into a product state, but P(ancilla=|0...0>)="
                    f"{ancilla_zero_prob:.6f} < 1.0."
                )
            psi_target = cirq.sub_state_vector(
                state_vector=psi_full,
                keep_indices=list(range(num_qubits)),
            )
    else:
        psi_target = psi_full

    ideal_psi = create_dicke_statevector(num_qubits, num_excitations)
    overlap = np.vdot(ideal_psi, psi_target)
    fidelity = float(np.abs(overlap) ** 2)

    # Diagnostic 1: Hamming-weight subspace purity & leaked basis states
    probs = np.abs(psi_target) ** 2
    dicke_indices = set()
    for comb_idx in itertools.combinations(range(num_qubits), num_excitations):
        # Cirq qubit_order Is big-endian:ordered_targets[0] is MSB (bit num_qubits-1),
        # while create_dicke_statevector is symmetric over all permutations.
        idx = sum(2**q for q in comb_idx)
        dicke_indices.add(idx)

    subspace_prob = float(sum(probs[i] for i in dicke_indices))
    subspace_leakage = max(0.0, 1.0 - subspace_prob)

    leaked_states: list[tuple[str, float]] = []
    if subspace_leakage > atol:
        for idx, p in enumerate(probs):
            if idx not in dicke_indices and p > 1e-6:
                leaked_states.append((_bitstring_label(idx, num_qubits), float(p)))
        leaked_states.sort(key=lambda item: item[1], reverse=True)
        leaked_states = leaked_states[:5]
        messages.append(
            f"Excitation leakage outside Hamming weight n={num_excitations}: "
            f"leakage={subspace_leakage:.6f} (top leaked: {leaked_states[:3]})."
        )

    # Diagnostic 2: Amplitude uniformity across the (N choose n) Dicke basis states
    ideal_amp = 1.0 / math.sqrt(math.comb(num_qubits, num_excitations))
    dicke_Sorted = sorted(dicke_indices)
    amps = np.abs(psi_target[dicke_Sorted])
    amp_diffs = np.abs(amps - ideal_amp)
    amplitude_uniformity_max_err = float(np.max(amp_diffs)) if len(amp_diffs) else 0.0

    amplitude_outliers: list[tuple[str, float, float]] = []
    if amplitude_uniformity_max_err > 1e-3:
        for idx in dicke_Sorted:
            actual_a = float(np.abs(psi_target[idx]))
            if abs(actual_a - ideal_amp) > 1e-3:
                amplitude_outliers.append(
                    (_bitstring_label(idx, num_qubits), actual_a, ideal_amp)
                )
        amplitude_outliers.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)
        amplitude_outliers = amplitude_outliers[:5]
        messages.append(
            f"Non-uniform amplitudes in weight-{num_excitations} subspace: "
            f"max |amp - ideal| = {amplitude_uniformity_max_err:.6f} (check rotation angles theta)."
        )

    # Diagnostic 3: Relative phase coherence across weight-n basis states
    nonzero_mask = amps > 1e-5
    phase_spread_rad = 0.0
    phase_outliers: list[tuple[str, float]] = []
    if np.any(nonzero_mask):
        active_vals = psi_target[np.array(dicke_Sorted)[nonzero_mask]]
        ref_phase = np.angle(active_vals[0])
        rel_phases = np.angle(active_vals * np.exp(-1j * ref_phase))
        phase_spread_rad = float(np.max(np.abs(rel_phases)))
        if phase_spread_rad > 1e-3:
            for idx in dicke_Sorted:
                val = psi_target[idx]
                if abs(val) > 1e-5:
                    rel_p = float(np.angle(val * np.exp(-1j * ref_phase)))
                    if abs(rel_p) > 1e-3:
                        phase_outliers.append((_bitstring_label(idx, num_qubits), rel_p))
            phase_outliers.sort(key=lambda x: abs(x[1]), reverse=True)
            phase_outliers = phase_outliers[:5]
            messages.append(
                f"Relative phase mismatch across Dicke basis states: "
                f"max phase deviation = {phase_spread_rad:.4f} rad (check sign/i factors; will degrade X-basis score)."
            )

    official_passed: Optional[bool] = None
    if run_official_test:
        try:
            test_circuit(circuit, num_excitations, QubitSet(target_set), sampler=sampler)
            official_passed = True
        except Exception as exc:
            official_passed = False
            messages.append(f"Official test_circuit failed: {type(exc).__name__}: {exc}")

    passed = (
        (not ancillas_entangled)
        and np.isclose(fidelity, 1.0, atol=atol)
        and (official_passed is not False)
    )
    if passed and not messages:
        messages.append(f"Exact Dicke state |D^({num_qubits})_{num_excitations}> verified (Fidelity = {fidelity:.8f}).")

    return FidelityReport(
        passed=bool(passed),
        fidelity=fidelity,
        num_qubits=num_qubits,
        num_excitations=num_excitations,
        num_ancillas=n_anc,
        ancillas_entangled=ancillas_entangled,
        ancilla_purity=ancilla_purity,
        ancilla_zero_prob=ancilla_zero_prob,
        subspace_prob=subspace_prob,
        subspace_leakage=subspace_leakage,
        amplitude_uniformity_max_err=amplitude_uniformity_max_err,
        phase_spread_rad=phase_spread_rad,
        official_test_passed=official_passed,
        leaked_states=leaked_states,
        amplitude_outliers=amplitude_outliers,
        phase_outliers=phase_outliers,
        messages=messages,
        warnings=warnings,
    )
