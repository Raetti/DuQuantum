# Google Quantum AI Challenge: Dicke State Preparation

## 1. Problem Statement

Given a target number of qubits $N$ and excitation weight $n$ (where $1 \leq n \leq N$), prepare an $N$-qubit Dicke state:

$$\left|D^{(N)}_n\right\rangle = \binom{N}{n}^{-1/2} \sum_k \mathcal{P}_k \left(|0\rangle^{\otimes(N-n)} \otimes |1\rangle^{\otimes n}\right)$$

where the sum runs over all $\binom{N}{n}$ distinct computational basis states with Hamming weight $n$.
For $n = 1$, this represents the $W$ state:

$$|W\rangle = \frac{1}{\sqrt{N}} \sum_{i=1}^N |0\ldots 0 \underbrace{1}_{i\text{-th}} 0 \ldots 0\rangle$$

---

## 2. Hardware Architecture & Constraints

- **Processor**: Google Willow architecture (105 qubits, grid layout).
  - Reference: [Willow Spec Sheet](https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf).
- **Allowed Gates**:
  1. Two-qubit entangling gates: `cirq.CZ` exclusively between adjacent pairs present in `cz_pauli_error_rate`.
  2. Single-qubit unitaries: arbitrary single-qubit gates (`cirq.PhasedXZGate`) on qubits present in `sq_pauli_error_rate`.
  3. Mid-circuit reset gates: optionally permitted (`cirq.ResetChannel`).
  4. Classically-conditioned gates: **strictly forbidden** in the standard challenge.
- **Circuit Depth / Moments**:
  - Maximum 1750 moments. Any circuit exceeding 1750 moments will not be evaluated on hardware.
- **Execution Time**:
  - `create_dicke_state` must return within 60 seconds per $(N, n)$ instance.
- **State Fidelity**:
  - In ideal (noiseless) simulation, the state prepared on the specified subset of qubits must achieve fidelity 1.0 with the theoretical state vector, or raise `NotImplementedError`.

---

## 3. Input Arguments for `create_dicke_state`

```python
def create_dicke_state(
    num_qubits: int,
    num_excitations: int,
    cz_pauli_error_rate: dict[tuple[cirq.GridQubit, cirq.GridQubit], tuple[float, float]],
    sq_pauli_error_rate: dict[cirq.GridQubit, tuple[float, float]],
    zero_errors: dict[cirq.GridQubit, tuple[float, float]],
    one_errors: dict[cirq.GridQubit, tuple[float, float]],
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    ...
```

- `num_qubits`: Integer $N$.
- `num_excitations`: Integer $n$.
- `cz_pauli_error_rate`: Dictionary mapping qubit pairs to `(error_rate, uncertainty)`.
- `sq_pauli_error_rate`: Dictionary mapping qubits to `(error_rate, uncertainty)`.
- `zero_errors`: Dictionary mapping qubits to parallel $0 \to 1$ readout error rates.
- `one_errors`: Dictionary mapping qubits to parallel $1 \to 0$ readout error rates.

---

## 4. Hardware Scoring Metric

The circuit is executed on Google Willow hardware with 10,000 shots in the Z basis and 10,000 shots in the X basis.

$$\text{overall\_score} = \frac{1}{\text{score}_Z + \text{score}_X}$$

### Z-Basis Score ($\text{score}_Z$)
Measures subspace concentration (fraction of shots with exactly $n$ ones) and uniformity within that subspace:

$$\text{score}_Z = \frac{1}{\binom{N}{n}} \cdot \frac{\chi^2}{f_{\text{subspace}}}$$

where:
- $f_{\text{subspace}} = \text{mean}(\sum b_i == n)$
- $\chi^2 = \sum_{\text{perm}} \frac{(C_{\text{obs}} - C_{\text{exp}})^2}{C_{\text{exp}}}$
- $C_{\text{exp}} = n_{\text{shots}} \cdot f_{\text{subspace}} / \binom{N}{n}$

### X-Basis Score ($\text{score}_X$)
Measures parity correlators in the X basis up to `max_correlator_size`:

$$\langle C_k \rangle = \frac{1}{\binom{N}{k}} \sum_{|S|=k} \left(1 - 2 \cdot \text{mean}\left[\sum_{i \in S} b_i \pmod 2\right]\right)$$

The expected correlator for even $k$ is:
$$\mathbb{E}[C_k] = \frac{\binom{k}{k/2} \binom{N-k}{n - k/2}}{\binom{N}{n}}$$
and 0 for odd $k$.

$$\text{score}_X = \sum_{k=1}^{k_{\max}} \frac{(\mathbb{E}[C_k] - \langle C_k \rangle_{\text{obs}})^2}{\mathbb{E}[C_k]^2}$$

---

## 5. Bonus Challenge: Adaptive Circuits

Construct shallower circuits utilizing mid-circuit measurements and conditional feedforward gates:

```python
def create_dicke_state_adaptive(
    num_qubits: int, 
    num_excitations: int
) -> tuple[cirq.Circuit, set[cirq.GridQubit]]:
    ...
```

- Evaluated across $2 \leq N \leq 20$, $1 \leq n \leq \lfloor N/2 \rfloor$.
- Scored by total efficiency:
  $$\text{Score} = \sum_{N, n} \frac{N \cdot n}{\text{len(circuit)}}$$
- **Hardware Readout Line Constraint**:
  On Willow hardware, qubits on the diagonal ($r + c = \text{const}$) share a readout line. Measuring a single qubit on a line dephases the remaining qubits on that line. Therefore, mid-circuit measurements must measure either all or none of the qubits sharing that diagonal readout line.
