# DuQuantum 2026 - Google Quantum AI Challenge

Repository for the **DuQuantum 2026** Hackathon at Duke University (October 24 to 25, 2026).  
Challenge track: **Google Quantum AI - Dicke State Preparation on Willow Architecture**.

---

## Multi-Stage Diagnostic Test Suite & Willow QVM Runner

To isolate theoretical math bugs, compilation errors, hardware rule violations, and physical decoherence, this repository provides a 4-stage verification pipeline and CLI runner (`scripts/test_algorithm.py`):

| Stage | Module | What It Verifies |
|---|---|---|
| **Stage 1: Abstract Math** | `src/verification/fidelity.py` | Simulates the raw theoretical circuit on `cirq.LineQubit`s; checks exact statevector fidelity $\lvert\langle D^{(N)}_n\mid\psi\rangle\rvert^2$, Hamming-weight subspace leakage, amplitude uniformity, and relative phase coherence. |
| **Stage 2: Willow Hardware Linter** | `src/verification/linter.py` | Statically verifies that all qubits are valid `cirq.GridQubit`s in `sq_pauli_error_rate`, all 2Q gates are `cirq.CZ` on working couplers in `cz_pauli_error_rate`, all 1Q gates are `cirq.PhasedXZGate` or `cirq.ResetChannel`, and moments $\le 1{,}750$. |
| **Stage 3: Compiled Noiseless** | `src/verification/fidelity.py` | Verifies that the Willow-compiled circuit preserves exact fidelity $\mathcal{F} = 1.0$, disentangles any ancillas to $\lvert 0\rangle$, and passes the official notebook `test_circuit` helper. |
| **Stage 4: Noisy Willow QVM** | `src/simulation/noisy_evaluator.py` | Simulates 10,000 Z-basis and 10,000 X-basis shots under `cirq_google.Willow105` Pauli gate errors, asymmetric $0\to 1$ / $1\to 0$ readout errors, and Willow spec-sheet $T_1/T_2$ relaxation ($T_1 = 68\,\mu\text{s}, T_{2,\text{CPMG}} = 89\,\mu\text{s}$, 25 ns 1Q, 42 ns CZ), scoring via the verbatim official `overall_score`. |

### Quick CLI Usage

```bash
# Run all 4 stages on |D^(4)_2>
python scripts/test_algorithm.py --N 4 --n 2 --stage all

# Test only Stage 1 (pure paper math on LineQubits) for the team's derivation
python scripts/test_algorithm.py --N 5 --n 2 --stage 1 --candidate team

# Run noiseless verification (Stages 1, 2, and 3) across a sweep of (N, n) pairs
python scripts/test_algorithm.py --sweep 2-6 --stage 1-3

# Run noisy Willow QVM with noise-source ablation (gate vs. readout vs. T1/T2 idle)
python scripts/test_algorithm.py --N 4 --n 2 --stage all --noise-backend qvm --ablate
```

---

## Plugging In the Physics Team's Derivation

Open [`src/candidate_algorithms.py`](src/candidate_algorithms.py) and paste the gate sequence, rotation angles, and initial excitation logic into `team_prototype_dicke_state(num_qubits, num_excitations)`:
- **Abstract Mode**: Write directly on `cirq.LineQubit.range(num_qubits)` using `cirq.X`, `cirq.ry(theta)`, `cirq.CNOT`, `givens_gate(theta)`, or controlled rotations.
- **Automatic Willow Compilation**: `team_native_dicke_state` automatically selects the lowest-error connected subgraph on `cirq_google.Willow105`, routes non-adjacent gates if needed, decomposes 2Q unitaries via KAK into at most 2–3 `CZ` gates + `PhasedXZGate`, and merges single-qubit gates.

---

## Project Structure

```
DuQuantum/
├── README.md                           # Project overview and CLI guide
├── requirements.txt                    # Pinned cirq / cirq-google 1.8.0.dev (with Willow105)
├── reference/
│   └── DuQuantum_Google_Standard.py    # Original Colab download (byte-for-byte reference)
├── scripts/
│   └── test_algorithm.py               # Unified 4-stage diagnostic CLI runner
├── src/
│   ├── candidate_algorithms.py         # Pluggable prototype & native Willow candidate interface
│   ├── dicke_state.py                  # Primary challenge entry points
│   ├── noise_models.py                 # Seeded Willow105 error profile generator
│   ├── willow_utils.py                 # Willow connectivity graph, path search & readout check
│   ├── official/
│   │   └── notebook_helpers.py         # Verbatim helper functions from the official notebook
│   ├── compilation/
│   │   └── willow_compiler.py          # Subgraph embedding, KAK 2-CZ synthesis & DD echoes
│   ├── verification/
│   │   ├── linter.py                   # Stage 2 static hardware rule linter
│   │   └── fidelity.py                 # Stage 1 & 3 statevector verifier + math diagnostics
│   └── simulation/
│       └── noisy_evaluator.py          # Stage 4 Willow QVM simulator & official scorer
└── tests/
    ├── test_official_helpers_sanctity.py # AST test guaranteeing official helpers are untouched
    ├── test_linter.py                  # Hardware linter unit tests
    ├── test_fidelity.py                # Statevector & diagnostic unit tests
    ├── test_compiler.py                # Willow compiler & DD unit tests
    ├── test_noisy_evaluator.py         # Noisy QVM & ablation unit tests
    ├── test_cli.py                     # CLI runner integration tests
    └── test_dicke_state.py             # Submission signature unit tests
```
