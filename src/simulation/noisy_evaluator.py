"""Noisy Willow QVM simulator and official challenge scoring harness.

Combines:
- Measured Pauli error dictionaries (`cz_pauli_error_rate`, `sq_pauli_error_rate`,
  `zero_errors`, `one_errors`) generated from `cirq_google.Willow105.metadata`.
- Willow spec-sheet coherence times (`T1 = 68 us`, `T2_CPMG = 89 us`) and
  `Willow105.metadata.gate_durations` (`25 ns` 1Q, `42 ns` CZ, `160 ns` reset).
- Exact 10,000-shot Z-basis and X-basis evaluation via the verbatim official
  scoring functions (`check_z_bitstring_distribution`, `check_x_bitstring_distribution`,
  `overall_score`).
"""

from dataclasses import dataclass, field
import math
from typing import Iterable, Optional, Sequence
import numpy as np
import cirq

from src.official.notebook_helpers import (
    check_x_bitstring_distribution,
    check_z_bitstring_distribution,
    create_dicke_statevector,
    overall_score,
)
from src.willow_utils import lookup_coupler_error

# Official Willow Spec Sheet (Dec 2024 / Acharya et al.) & Willow105.metadata.gate_durations
DEFAULT_T1_NS = 68_000.0
DEFAULT_T2_NS = 89_000.0
DEFAULT_SQ_NS = 25.0
DEFAULT_CZ_NS = 42.0
DEFAULT_RESET_NS = 160.0
DEFAULT_MEAS_NS = 600.0


class WillowNoiseModel(cirq.NoiseModel):
    """Per-moment Google Willow noise model combining gate Pauli errors and T1/T2 relaxation."""

    def __init__(
        self,
        cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
        sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
        enable_gate_noise: bool = True,
        enable_idle_noise: bool = True,
        t1_ns: float = DEFAULT_T1_NS,
        t2_ns: float = DEFAULT_T2_NS,
        sq_duration_ns: float = DEFAULT_SQ_NS,
        cz_duration_ns: float = DEFAULT_CZ_NS,
        reset_duration_ns: float = DEFAULT_RESET_NS,
    ):
        self.cz_pauli_error_rate = cz_pauli_error_rate
        self.sq_pauli_error_rate = sq_pauli_error_rate
        self.enable_gate_noise = enable_gate_noise
        self.enable_idle_noise = enable_idle_noise
        self.t1_ns = t1_ns
        self.t2_ns = t2_ns
        self.sq_duration_ns = sq_duration_ns
        self.cz_duration_ns = cz_duration_ns
        self.reset_duration_ns = reset_duration_ns

    def _moment_duration_ns(self, moment: cirq.Moment) -> float:
        dur = 0.0
        for op in moment:
            if isinstance(op.gate, cirq.ResetChannel):
                dur = max(dur, self.reset_duration_ns)
            elif len(op.qubits) == 2:
                dur = max(dur, self.cz_duration_ns)
            elif len(op.qubits) == 1:
                dur = max(dur, self.sq_duration_ns)
        return dur

    def _thermal_channels_for_duration(
        self, qubit: cirq.Qid, dt_ns: float
    ) -> list[cirq.Operation]:
        if dt_ns <= 0.0:
            return []
        ops: list[cirq.Operation] = []
        if self.t1_ns > 0.0:
            gamma_1 = float(np.clip(1.0 - math.exp(-dt_ns / self.t1_ns), 0.0, 1.0))
            if gamma_1 > 1e-12:
                ops.append(cirq.amplitude_damp(gamma_1).on(qubit))
        if self.t2_ns > 0.0:
            inv_tphi = (1.0 / self.t2_ns) - (
                (1.0 / (2.0 * self.t1_ns)) if self.t1_ns > 0.0 else 0.0
            )
            if inv_tphi > 1e-12:
                gamma_phi = float(np.clip(1.0 - math.exp(-2.0 * dt_ns * inv_tphi), 0.0, 1.0))
                if gamma_phi > 1e-12:
                    ops.append(cirq.phase_damp(gamma_phi).on(qubit))
        return ops

    def noisy_moment(
        self, moment: cirq.Moment, system_qubits: Sequence[cirq.Qid]
    ) -> cirq.OP_TREE:
        if len(moment) == 0:
            return [moment]

        gate_noise_ops: list[cirq.Operation] = []
        active_in_moment = set(moment.qubits)
        dt_ns = self._moment_duration_ns(moment)

        if self.enable_gate_noise:
            for op in moment:
                if len(op.qubits) == 1 and isinstance(op.gate, cirq.PhasedXZGate):
                    q = op.qubits[0]
                    if q in self.sq_pauli_error_rate:
                        p_sq = float(np.clip(self.sq_pauli_error_rate[q][0], 0.0, 0.999))
                        if p_sq > 0.0:
                            gate_noise_ops.append(cirq.depolarize(p=p_sq, n_qubits=1).on(q))
                elif len(op.qubits) == 2 and isinstance(op.gate, cirq.CZPowGate):
                    q0, q1 = op.qubits
                    pair_err = lookup_coupler_error(q0, q1, self.cz_pauli_error_rate)
                    if pair_err is not None:
                        p_cz = float(np.clip(pair_err[0], 0.0, 0.999))
                        if p_cz > 0.0:
                            gate_noise_ops.append(
                                cirq.depolarize(p=p_cz, n_qubits=2).on(q0, q1)
                            )

        idle_noise_ops: list[cirq.Operation] = []
        if self.enable_idle_noise and dt_ns > 0.0:
            for q in system_qubits:
                if q not in active_in_moment:
                    idle_noise_ops.extend(self._thermal_channels_for_duration(q, dt_ns))
                else:
                    # Wait time within a longer moment (e.g. 1Q 25ns gate inside a 42ns CZ moment)
                    op_on_q = moment.operation_at(q)
                    op_dt = (
                        self.cz_duration_ns
                        if (op_on_q and len(op_on_q.qubits) == 2)
                        else self.sq_duration_ns
                    )
                    slack_ns = max(0.0, dt_ns - op_dt)
                    if slack_ns > 0.0:
                        idle_noise_ops.extend(self._thermal_channels_for_duration(q, slack_ns))

        result: list[cirq.OP_TREE] = [moment]
        if gate_noise_ops:
            result.append(gate_noise_ops)
        if idle_noise_ops:
            result.append(idle_noise_ops)
        return result


@dataclass
class NoisyEvaluationResult:
    """Comprehensive Stage 4 noisy Willow benchmark result."""

    num_qubits: int
    num_excitations: int
    n_shots: int
    max_correlator_size: int
    noise_backend: str
    simulation_method: str
    z_subspace_fraction: float
    z_score: float
    x_score: float
    overall_score: float
    noiseless_ceiling_score: float
    correlator_sensitivity: dict[int, tuple[float, float]] = field(default_factory=dict)
    ablation_scores: dict[str, dict[str, float]] = field(default_factory=dict)


def _ints_to_bitstrings(indices: np.ndarray, num_qubits: int) -> np.ndarray:
    """Convert integer state indices in [0, 2^N) to big-endian (n_shots, num_qubits) bit array."""
    shifts = np.arange(num_qubits - 1, -1, -1, dtype=np.int64)
    return ((indices[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def _apply_readout_errors(
    bitstrings: np.ndarray,
    ordered_targets: Sequence[cirq.GridQubit],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply asymmetric 0->1 and 1->0 readout errors to sampled bitstrings."""
    noisy_bits = bitstrings.copy()
    n_shots, n_qubits = noisy_bits.shape
    u = rng.random((n_shots, n_qubits))

    for col_idx, q in enumerate(ordered_targets):
        p01 = float(np.clip(zero_errors.get(q, (0.0, 0.0))[0], 0.0, 1.0))
        p10 = float(np.clip(one_errors.get(q, (0.0, 0.0))[0], 0.0, 1.0))
        col = noisy_bits[:, col_idx]
        flip_mask = np.where(col == 0, u[:, col_idx] < p01, u[:, col_idx] < p10)
        noisy_bits[:, col_idx] = np.where(flip_mask, 1 - col, col)

    return noisy_bits


def _marginal_target_probabilities_from_dm(
    rho: np.ndarray, num_targets: int, num_ancillas: int
) -> np.ndarray:
    """Extract marginal computational-basis probabilities on the first num_targets qubits."""
    diag = np.real(np.diag(rho))
    diag = np.clip(diag, 0.0, None)
    if num_ancillas > 0:
        diag = diag.reshape((2**num_targets, 2**num_ancillas)).sum(axis=1)
    total = float(np.sum(diag))
    if total > 0.0:
        diag = diag / total
    else:
        diag = np.full(2**num_targets, 1.0 / (2**num_targets))
    return diag


def _sample_bitstrings_noisy(
    circuit: cirq.Circuit,
    ordered_targets: list[cirq.GridQubit],
    ordered_ancillas: list[cirq.GridQubit],
    noise_model: WillowNoiseModel,
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
    enable_readout_noise: bool,
    n_shots: int,
    rng: np.random.Generator,
    max_dm_qubits: int = 11,
    n_trajectories: int = 200,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Sample 10,000 Z-basis and 10,000 X-basis bitstrings under the Willow noise model."""
    ordered_qubits = ordered_targets + ordered_ancillas
    num_targets = len(ordered_targets)
    num_ancillas = len(ordered_ancillas)
    total_qubits = len(ordered_qubits)

    h_pxz = cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.H))
    x_basis_circuit = circuit + cirq.Circuit(
        cirq.Moment(h_pxz.on(q) for q in ordered_targets)
    )

    noisy_z_circuit = circuit.with_noise(noise_model)
    noisy_x_circuit = x_basis_circuit.with_noise(noise_model)

    if total_qubits <= max_dm_qubits:
        dm_sim = cirq.DensityMatrixSimulator()
        rho_z = dm_sim.simulate(noisy_z_circuit, qubit_order=ordered_qubits).final_density_matrix
        rho_x = dm_sim.simulate(noisy_x_circuit, qubit_order=ordered_qubits).final_density_matrix

        probs_z = _marginal_target_probabilities_from_dm(rho_z, num_targets, num_ancillas)
        probs_x = _marginal_target_probabilities_from_dm(rho_x, num_targets, num_ancillas)

        idx_z = rng.choice(2**num_targets, size=n_shots, p=probs_z)
        idx_x = rng.choice(2**num_targets, size=n_shots, p=probs_x)
        z_bits = _ints_to_bitstrings(idx_z, num_targets)
        x_bits = _ints_to_bitstrings(idx_x, num_targets)
        sim_method = "density_matrix"
    else:
        sv_sim = cirq.Simulator()
        shots_per_traj = max(1, n_shots // n_trajectories)
        z_chunks: list[np.ndarray] = []
        x_chunks: list[np.ndarray] = []
        collected = 0
        while collected < n_shots:
            batch = min(shots_per_traj, n_shots - collected)
            psi_z = sv_sim.simulate(noisy_z_circuit, qubit_order=ordered_qubits).final_state_vector
            probs_z = np.abs(psi_z) ** 2
            if num_ancillas > 0:
                probs_z = probs_z.reshape((2**num_targets, 2**num_ancillas)).sum(axis=1)
            probs_z = probs_z / np.sum(probs_z)
            z_chunks.append(_ints_to_bitstrings(rng.choice(2**num_targets, size=batch, p=probs_z), num_targets))

            psi_x = sv_sim.simulate(noisy_x_circuit, qubit_order=ordered_qubits).final_state_vector
            probs_x = np.abs(psi_x) ** 2
            if num_ancillas > 0:
                probs_x = probs_x.reshape((2**num_targets, 2**num_ancillas)).sum(axis=1)
            probs_x = probs_x / np.sum(probs_x)
            x_chunks.append(_ints_to_bitstrings(rng.choice(2**num_targets, size=batch, p=probs_x), num_targets))
            collected += batch

        z_bits = np.vstack(z_chunks)[:n_shots]
        x_bits = np.vstack(x_chunks)[:n_shots]
        sim_method = f"monte_carlo_trajectories({n_trajectories})"

    if enable_readout_noise:
        z_bits = _apply_readout_errors(z_bits, ordered_targets, zero_errors, one_errors, rng)
        x_bits = _apply_readout_errors(x_bits, ordered_targets, zero_errors, one_errors, rng)

    return z_bits, x_bits, sim_method


def _safe_score(
    z_bits: np.ndarray,
    x_bits: np.ndarray,
    num_excitations: int,
    max_correlator_size: int,
) -> tuple[float, float, float, float]:
    """Call the verbatim official scoring functions with protection against 0 subspace hits."""
    measured_excitations = np.sum(z_bits, axis=1)
    z_subspace_fraction = float(np.mean(measured_excitations == num_excitations))
    if z_subspace_fraction == 0.0:
        z_score = float("inf")
    else:
        z_score = float(check_z_bitstring_distribution(z_bits, num_excitations))
    x_score = float(
        check_x_bitstring_distribution(x_bits, num_excitations, max_correlator_size)
    )
    if math.isinf(z_score) or (z_score + x_score) == 0.0:
        total = 0.0 if math.isinf(z_score) else float("inf")
    else:
        total = float(
            overall_score(z_bits, x_bits, num_excitations, max_correlator_size)
        )
    return z_subspace_fraction, z_score, x_score, total


def _compute_noiseless_shot_ceiling(
    num_qubits: int,
    num_excitations: int,
    max_correlator_size: int,
    n_shots: int,
    rng: np.random.Generator,
) -> float:
    """Compute the expected 10,000-shot score ceiling on an ideal noiseless Dicke state."""
    ideal_z = create_dicke_statevector(num_qubits, num_excitations)
    probs_z = np.abs(ideal_z) ** 2
    probs_z = probs_z / np.sum(probs_z)

    # Apply H^{\otimes N} to ideal Dicke statevector for exact X-basis sampling
    psi_x = ideal_z.reshape((2,) * num_qubits)
    h_mat = np.array([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
    for axis in range(num_qubits):
        psi_x = np.tensordot(h_mat, psi_x, axes=([1], [axis]))
        psi_x = np.moveaxis(psi_x, 0, axis)
    probs_x = np.abs(psi_x.reshape(-1)) ** 2
    probs_x = probs_x / np.sum(probs_x)

    z_bits = _ints_to_bitstrings(rng.choice(2**num_qubits, size=n_shots, p=probs_z), num_qubits)
    x_bits = _ints_to_bitstrings(rng.choice(2**num_qubits, size=n_shots, p=probs_x), num_qubits)
    _, _, _, ceiling = _safe_score(z_bits, x_bits, num_excitations, max_correlator_size)
    return ceiling


def evaluate_noisy_circuit(
    circuit: cirq.Circuit,
    qubits: Iterable[cirq.GridQubit],
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
    n_shots: int = 10_000,
    max_correlator_size: Optional[int] = None,
    noise_backend: str = "qvm",
    t1_ns: float = DEFAULT_T1_NS,
    t2_ns: float = DEFAULT_T2_NS,
    seed: int = 42,
    run_ablation: bool = False,
    max_dm_qubits: int = 11,
    n_trajectories: int = 200,
) -> NoisyEvaluationResult:
    """Simulate a Dicke state circuit on the noisy Willow model and compute official scores.

    Args:
        noise_backend:
            - "qvm": Full Willow QVM (gate Pauli errors + readout errors + spec-sheet T1/T2 idle relaxation).
            - "gate-readout": Gate Pauli errors + readout errors only (no idle T1/T2).
            - "noiseless-shots": Ideal state evolution with 10,000 finite shots.
    """
    target_set = set(qubits)
    ordered_targets = sorted(target_set)
    all_qubits = circuit.all_qubits() | target_set
    ordered_ancillas = sorted(all_qubits.difference(target_set))

    if max_correlator_size is None:
        max_correlator_size = num_qubits

    rng = np.random.default_rng(seed)

    enable_gate = noise_backend in ("qvm", "gate-readout")
    enable_idle = noise_backend == "qvm"
    enable_readout = noise_backend in ("qvm", "gate-readout")

    noise_model = WillowNoiseModel(
        cz_pauli_error_rate=cz_pauli_error_rate,
        sq_pauli_error_rate=sq_pauli_error_rate,
        enable_gate_noise=enable_gate,
        enable_idle_noise=enable_idle,
        t1_ns=t1_ns,
        t2_ns=t2_ns,
    )

    z_bits, x_bits, sim_method = _sample_bitstrings_noisy(
        circuit=circuit,
        ordered_targets=ordered_targets,
        ordered_ancillas=ordered_ancillas,
        noise_model=noise_model,
        zero_errors=zero_errors,
        one_errors=one_errors,
        enable_readout_noise=enable_readout,
        n_shots=n_shots,
        rng=rng,
        max_dm_qubits=max_dm_qubits,
        n_trajectories=n_trajectories,
    )

    z_subspace, z_score, x_score, total_score = _safe_score(
        z_bits, x_bits, num_excitations, max_correlator_size
    )

    # Sensitivity across candidate undisclosed max_correlator_size values
    candidate_k_values = sorted({k for k in (2, 4, 6, num_qubits, max_correlator_size) if 1 <= k <= num_qubits})
    correlator_sensitivity: dict[int, tuple[float, float]] = {}
    for k_val in candidate_k_values:
        _, _, x_k, tot_k = _safe_score(z_bits, x_bits, num_excitations, k_val)
        correlator_sensitivity[k_val] = (x_k, tot_k)

    ceiling_score = _compute_noiseless_shot_ceiling(
        num_qubits=num_qubits,
        num_excitations=num_excitations,
        max_correlator_size=max_correlator_size,
        n_shots=n_shots,
        rng=np.random.default_rng(seed + 999),
    )

    ablation_scores: dict[str, dict[str, float]] = {}
    if run_ablation:
        for mode_name, (g_on, r_on, i_on) in [
            ("gate_only", (True, False, False)),
            ("readout_only", (False, True, False)),
            ("idle_t1_t2_only", (False, False, True)),
            ("full_qvm", (True, True, True)),
        ]:
            ab_model = WillowNoiseModel(
                cz_pauli_error_rate=cz_pauli_error_rate,
                sq_pauli_error_rate=sq_pauli_error_rate,
                enable_gate_noise=g_on,
                enable_idle_noise=i_on,
                t1_ns=t1_ns,
                t2_ns=t2_ns,
            )
            ab_z, ab_x, _ = _sample_bitstrings_noisy(
                circuit=circuit,
                ordered_targets=ordered_targets,
                ordered_ancillas=ordered_ancillas,
                noise_model=ab_model,
                zero_errors=zero_errors,
                one_errors=one_errors,
                enable_readout_noise=r_on,
                n_shots=n_shots,
                rng=np.random.default_rng(seed + 100),
                max_dm_qubits=max_dm_qubits,
                n_trajectories=n_trajectories,
            )
            sub_f, zs, xs, ov = _safe_score(
                ab_z, ab_x, num_excitations, max_correlator_size
            )
            ablation_scores[mode_name] = {
                "z_subspace_fraction": sub_f,
                "z_score": zs,
                "x_score": xs,
                "overall_score": ov,
            }

    return NoisyEvaluationResult(
        num_qubits=num_qubits,
        num_excitations=num_excitations,
        n_shots=n_shots,
        max_correlator_size=max_correlator_size,
        noise_backend=noise_backend,
        simulation_method=sim_method,
        z_subspace_fraction=z_subspace,
        z_score=z_score,
        x_score=x_score,
        overall_score=total_score,
        noiseless_ceiling_score=ceiling_score,
        correlator_sensitivity=correlator_sensitivity,
        ablation_scores=ablation_scores,
    )
