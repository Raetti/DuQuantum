# Google Quantum AI Challenge: Dicke State Preparation

## 1. Problem Statement

Given a target number of qubits $N$ and excitation weight $n$ ($1 \leq n \leq N$), prepare an $N$-qubit Dicke state:

$$\left|D^{(N)}_n\right\rangle = \binom{N}{n}^{-1/2} \sum_k \mathcal{P}_k \left(|0\rangle^{\otimes(N-n)} \otimes |1\rangle^{\otimes n}\right)$$

where the sum runs over all $\binom{N}{n}$ distinct computational basis states with Hamming weight $n$.

---

## 2. Official Organizer Updates & Errata

Per the DuQuantum organizer update:
1. **Bonus Problem Range**: Grading for the bonus adaptive track is over $2 \le N \le 15$, so `score_solution` iterates over `for num_qubits in range(2, 16)`.
2. **`test_circuit` Fix**: Line 77 in the starter notebook (`keep_indices = range(qubits)`) is corrected to `keep_indices = range(len(qubits))`.
3. **Undisclosed Grading Parameters**: The specific `(N, n)` pairs and `max_correlator_size` used during hardware grading are not disclosed ahead of the grading session. Our Stage 4 evaluator therefore reports a sensitivity table across `max_correlator_size` values ($k_{\max} \in \{2, 4, 6, N\}$).

---

## 3. Hardware Architecture & Constraints

- **Processor**: Google Willow (`cirq_google.Willow105`, 105 qubits, 182 couplers on a 13×15 grid).
  - Reference: [Willow Spec Sheet](https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf).
  - Coherence times (Chip 1 QEC spec / Acharya et al.): Mean $T_1 = 68\,\mu\text{s}$, $T_{2,\text{CPMG}} = 89\,\mu\text{s}$.
  - Native gate durations (`Willow105.metadata.gate_durations`): `PhasedXZGate` = **25 ns**, `CZ` = **42 ns**, `ResetChannel` = **160 ns**, `MeasurementGate` = **600 ns**.
- **Allowed Gates**:
  1. Two-qubit entangling gates: `cirq.CZ` exclusively between adjacent pairs present in `cz_pauli_error_rate`.
  2. Single-qubit unitaries: `cirq.PhasedXZGate` on qubits present in `sq_pauli_error_rate`.
  3. Mid-circuit reset gates: optionally permitted (`cirq.ResetChannel`).
  4. Classically-conditioned gates: **strictly forbidden** in the standard challenge.
- **Hard Limits**:
  - Maximum **1,750 moments**.
  - Compilation time under **60 seconds** per $(N, n)$ instance.
  - Only imports from `cirq`, `cirq_google`, their dependencies, and Python's standard library are permitted in submission code.
