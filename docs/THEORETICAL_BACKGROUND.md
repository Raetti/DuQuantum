# Theoretical Background & Algorithmic Strategies

## 1. What are Dicke States?

An $N$-qubit Dicke state with $n$ excitations, denoted $|D^{(N)}_n\rangle$, is the uniform, symmetric superposition of all $N$-qubit computational basis states having Hamming weight $n$:

$$\left|D^{(N)}_n\right\rangle = \frac{1}{\sqrt{\binom{N}{n}}} \sum_{x \in \{0, 1\}^N, \, |x|=n} |x\rangle$$

These states play a central role in quantum information:
- **Entanglement witness**: They exhibit multipartite entanglement that is exceptionally robust against qubit loss.
- **Quantum metrology**: Dicke states near $n \approx N/2$ achieve Heisenberg-limited phase sensitivity.
- **Quantum networking**: The $n = 1$ state (W state) enables genuine multipartite entanglement distribution across quantum nodes.

---

## 2. Deterministic Preparation Algorithms

### The Bärtschi and Eidenbenz Algorithm (arXiv:1904.07358)
Andreas Bärtschi and Stephan Eidenbenz introduced a deterministic scheme requiring only nearest-neighbor couplings on a linear chain:
- **Depth**: $\mathcal{O}(N)$ depth for fixed $n$, or $\mathcal{O}(n(N-n))$ in general.
- **Gate Count**: Exactly $n(N-n) + n$ two-qubit blocks.
- **Coupling Structure**: Uses controlled Givens rotations (or fermionic excitation-preserving beam splitters) that act non-trivially only on the subspace spanned by $\{|01\rangle, |10\rangle\}$:

$$U(\theta) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\theta & \sin\theta & 0 \\ 0 & -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

By choosing the rotation angles $\theta_{k, l} = \arccos\sqrt{\frac{k}{l}}$, excitations are partitioned deterministically into superpositions across neighboring qubits without exiting the excitation-preserving subspace.

---

## 3. Native Gate Synthesis on Google Willow

Google Willow processors natively support:
- Single-qubit unitaries: `cirq.PhasedXZGate(x_exponent, z_exponent, axis_phase_exponent)`
- Two-qubit entangling gates: `cirq.CZ`

To realize an arbitrary controlled two-qubit rotation or Givens rotation $U(\theta)$, we can decompose it into at most 2 to 3 CZ gates interspersed with single-qubit rotations:

### Standard Givens Rotation Decomposition
A Givens rotation $e^{i \theta (X \otimes Y - Y \otimes X)/2}$ can be synthesized using 2 CZ gates:
1. Local single-qubit rotations to map $\{|01\rangle, |10\rangle\}$ into a basis diagonalized by CZ.
2. First CZ gate.
3. Intermediate single-qubit rotations by angle $\pm \theta / 2$.
4. Second CZ gate.
5. Inverting the initial basis transformation.

This keeps two-qubit gate overhead strictly minimal, directly benefiting the hardware execution fidelity.

---

## 4. Hardware-Aware Qubit Selection on Willow

The Willow processor features a 105-qubit planar grid. Given:
- Single-qubit Pauli error rates: $\epsilon_i$
- CZ Pauli error rates: $\epsilon_{ij}$
- Readout error rates: $\epsilon_{0 \to 1}, \epsilon_{1 \to 0}$

We formulate the mapping of an $N$-qubit linear chain or tree onto Willow as an optimal subgraph problem:
1. Construct the device coupling graph $G = (V, E)$ filtering out broken couplers and dropped qubits.
2. Assign edge weights $w(e) = -\log(1 - \epsilon_{ij})$ representing infidelity costs.
3. Find an induced connected subgraph (path or minimum spanning tree) of size $N$ minimizing total expected error.
4. Qubits with the lowest readout errors are prioritized for critical measurement nodes.

---

## 5. Noise Mitigation: Dynamical Decoupling

Idle qubits during two-qubit gate operations suffer from environmental dephasing ($T_2^*$) and low-frequency $1/f$ flux noise.
Applying dynamical decoupling sequences (such as Hahn echo $X$ pulses or $XY4$ sequences $[X, Y, X, Y]$) during idle moments cancels dephasing noise and significantly elevates the hardware score.
