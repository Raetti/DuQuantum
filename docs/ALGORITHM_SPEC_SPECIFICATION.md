# Quantum Algorithm Specification Protocol (Physics-to-Engineering Handoff)

> **Context**: DuQuantum 2026 / Google Quantum AI Challenge — Dicke State Preparation $|D^{(N)}_n\rangle$ on Google Willow.  
> **Target Audience**: Theoretical Physicists & Algorithm Designers on the team.  
> **Purpose**: Eliminates translation bugs, phase errors, qubit indexing ambiguities, and hardware-infeasible assumptions before code implementation.

---

## 1. Non-Negotiable Requirements (The "Rules of Engagement")

Before writing the derivation, the physicists must adhere to these five operational constraints:

1. **Gate Matrix & Angle Convention (The Factor-of-2 Rule)**:
   - Cirq standardizes single-qubit rotations as $R_y(\theta) = \exp(-i \frac{\theta}{2} Y) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$.
   - Many theoretical papers (including Bärtschi & Eidenbenz 2019) define rotations without the $1/2$ factor: $R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$.
   - **Requirement**: **Every angle formula must explicitly specify whether it is the argument to Cirq's `cirq.ry(theta)` or the abstract rotation angle $\theta$.**
2. **Deterministic Phase & Sign Invariant**:
   - The target Dicke state has **purely real, strictly positive amplitudes**:
     $$\left|D^{(N)}_n\right\rangle = \frac{1}{\sqrt{\binom{N}{n}}} \sum_{|x|=n} |x\rangle$$
   - Any gate sequence that introduces relative signs (e.g., $-|01\rangle + |10\rangle$) or imaginary phases ($+i|01\rangle$) will fail Stage 1 fidelity verification ($\mathcal{F} = |\langle D^{(N)}_n | \psi \rangle|^2 < 1.0$).
3. **Big-Endian Qubit Indexing**:
   - Qubits are 0-indexed: $q_0, q_1, \dots, q_{N-1}$.
   - Cirq convention is **Big-Endian**: $|q_0 q_1 \dots q_{N-1}\rangle$, where $q_0$ corresponds to the most significant bit (MSB).
4. **Hardware Topology & Gate Set (Willow105)**:
   - Native gates: `cirq.CZ` (two-qubit) and `cirq.PhasedXZGate` (single-qubit).
   - Willow is a 2D planar grid (13×15); all 2Q gates must act on nearest-neighbors (or assume a 1D linear chain $q_0 - q_1 - \dots - q_{N-1}$ that will be routed).
   - **No mid-circuit classical feedforward / measurement-conditioned gates**. Mid-circuit `ResetChannel` is allowed only if ancillas are used.
5. **Ancilla Cleanliness**:
   - If ancilla qubits are introduced, they must start in $|0\rangle$ and be **deterministically returned to $|0\rangle$ (disentangled)** at circuit termination. Residual entanglement reduces state purity.

---

## 2. Fill-in-the-Blank Specification Template

*Copy the template below and have the physics sub-team fill in each numbered section.*

```markdown
# Algorithm Specification: [Name of Strategy, e.g., Bärtschi-Eidenbenz SCS / Tree / Recursive]

## 1. High-Level Concept & Subspace Strategy
- **Core Mechanism**: (e.g., Excitation propagation via Givens rotations, inductive splitting, Dicke tree)
- **Target Subspace**: Hamming weight subspace k = n
- **Ancilla Requirements**: Number of ancilla qubits required = [0 / fixed / O(N)]
- **Theoretical Scaling**:
  - 2-Qubit Gate Count: [e.g., n(N-n)]
  - Circuit Depth: [e.g., O(N) or O(n(N-n))]

---

## 2. Mathematical Definitions & Gate Matrices
Specify the exact matrix representation of all non-standard elementary gates used.

### 2.1 Two-Qubit Unitary / Givens Block: U(θ)
- **Matrix representation in computational basis {|00>, |01>, |10>, |11>}**:
  ```
  U(θ) = [
    [ 1,      0,       0,      0 ],
    [ 0,  cos(θ),  sin(θ),     0 ],
    [ 0, -sin(θ),  cos(θ),     0 ],
    [ 0,      0,       0,      1 ]
  ]
  ```
- **Action on state**: |01> -> cos(θ)|01> - sin(θ)|10>, |10> -> sin(θ)|10> + cos(θ)|01>
- **Decomposition into Cirq primitives (if known)**:
  - Number of CZs: [e.g., 2 CZs]
  - Explicit gate sequence: [e.g., Rz, Ry, CZ, Ry, CZ, ...]

### 2.2 Angle Calculation Formula
- For step (k, l) or level j:
  $$\theta(k, l) = 2 \cdot \arccos\left(\sqrt{\frac{k}{l}}\right) \quad \text{[or explicit Cirq angle]}$$
- Special boundary conditions (k = 0, k = l, l = 0):

---

## 3. General (N, n) Algorithmic Procedure (Pseudocode)

### Step 1: Initial State Preparation
- Starting state: |0...0> on qubits [q_0, ..., q_{N-1}]
- Single-qubit excitations: Which qubits get X gates applied?
  - E.g., Apply X to [q_{N-n}, ..., q_{N-1}] or [q_0, ..., q_{n-1}].

### Step 2: Gate Sequence / Main Loops
Provide exact loop ranges (inclusive vs exclusive) and target qubit indices:
```python
# Pseudocode representation
for l in range(start, end):
    for k in range(start, end):
        theta = calculate_angle(k, l)
        apply_gate(control=q[i], target=q[j], angle=theta)
```

### Step 3: Ancilla Cleanup / Disentanglement (if applicable)
- Steps taken to reset/uncompute ancillas to |0>:

---

## 4. Canonical Test Vectors (Ground Truth Verification)
To verify the implementation in Stage 1 without guessing, provide the exact intermediate statevectors for these two minimal cases:

### Test Case A: N = 3, n = 1 (W-State |W_3>)
- **Expected Final State**:
  $$\frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$$
- **Step-by-step gate execution**:
  1. Initial excitation: `X(q_2)` -> state = `|001>`
  2. Gate 1: `Givens(theta_1)` on `(q_1, q_2)` with `theta_1 = ...` -> state = `...`
  3. Gate 2: `Givens(theta_2)` on `(q_0, q_1)` with `theta_2 = ...` -> state = `...`

### Test Case B: N = 4, n = 2 (|D^(4)_2>)
- **Expected Final State**:
  $$\frac{1}{\sqrt{6}}(|0011\rangle + |0101\rangle + |0110\rangle + |1001\rangle + |1010\rangle + |1100\rangle)$$
- **Step-by-step gate execution table**:
  | Step | Gate Applied | Active Qubits | Angle Value (rad or deg) | Intermediate Statevector |
  |---|---|---|---|---|
  | 0 | `X(q_2), X(q_3)` | `q_2, q_3` | - | `|0011>` |
  | 1 | ... | ... | ... | ... |

---

## 5. Noise & Hardware Optimization Notes
- **Critical Path / Idle Qubits**: Which qubits remain idle for prolonged moments (candidates for Dynamical Decoupling $XY4$)?
- **Preferred Coupling Layout**: Does this algorithm prefer a linear chain ($q_0-q_1-q_2$), a star graph, or a 2D tree?
- **Commutation Opportunities**: Can any adjacent 2Q gates commute to allow moment parallelization?
```

---

## 3. Engineering Intake Checklist

When the physicists hand over the document, software engineers check off:

- [ ] **No Free-Parameter Ambiguity**: Every angle $\theta$ is an exact closed-form function of $(N, n, k, l)$ with no empirical tuning required.
- [ ] **Cirq Angle Factor Validated**: Confirmed whether $\theta$ uses Cirq convention ($2 \arccos$) or standard Hamiltonian evolution convention ($\arccos$).
- [ ] **Exact Statevector Check**: The Test Case A and B statevectors match the output of `src/verification/fidelity.py` with fidelity $> 0.999999$.
- [ ] **Linear Nearest-Neighbor Compatibility**: All two-qubit operations can be mapped to adjacent qubits on a path or tree without requiring all-to-all cross-chip connections.
- [ ] **Zero Classical Feedforward**: No conditional operations dependent on mid-circuit measurement results.
