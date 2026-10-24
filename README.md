# DuQuantum 2026 - Google Quantum AI Challenge

Repository for the **DuQuantum 2026** Hackathon at Duke University (October 24 to 25, 2026).
Challenge track: **Google Quantum AI - Dicke State Preparation on Willow Architecture**.

---

## Challenge Summary

The objective is to implement a quantum circuit in [Cirq](https://quantumai.google/cirq) that prepares an $N$-qubit, weight-$n$ Dicke state $|D^{(N)}_n\rangle$ on Google's superconducting architecture (Willow).

$$\left|D^{(N)}_n\right\rangle = \binom{N}{n}^{-1/2} \sum_k \mathcal{P}_k \left(|0\rangle^{\otimes(N-n)} \otimes |1\rangle^{\otimes n}\right)$$

For $n = 1$, this corresponds to the $W$ state:
$$|W\rangle = \frac{1}{\sqrt{N}} \left(|100\ldots 0\rangle + |010\ldots 0\rangle + \ldots + |00\ldots 01\rangle\right)$$

### Key Requirements
- **Gate Set**: Only CZ gates between adjacent qubit pairs in `cz_pauli_error_rate`, single-qubit unitaries (`cirq.PhasedXZGate`) on qubits in `sq_pauli_error_rate`, and optional mid-circuit resets.
- **No Classical Conditioning**: Classically-conditioned gates are prohibited in the primary challenge (allowed in the bonus challenge).
- **Noiseless Correctness**: In ideal simulation, the circuit must produce the exact Dicke state or raise `NotImplementedError`.
- **Runtime Limit**: Circuit generation function must return within 60 seconds per $(N, n)$ pair.
- **Moment Limit**: Maximum 1750 moments per circuit.
- **Evaluation**: Circuits run on physical Google Willow hardware with 10,000 shots in both Z and X bases, scored via `overall_score`.

---

## Project Structure

```
DuQuantum/
├── README.md                           # Project overview and instructions
├── requirements.txt                    # Python dependencies (cirq, cirq-google, numpy, pytest)
├── DuQuantum_Google_Standard.ipynb     # Starter Colab notebook from Google Quantum AI
├── docs/
│   ├── HACKER_GUIDE.md                 # Logistics, schedule, judging criteria, contacts
│   ├── GOOGLE_CHALLENGE.md             # Detailed technical rules and scoring breakdown
│   └── THEORETICAL_BACKGROUND.md       # Deterministic Dicke state preparation theory & Willow mapping
├── src/
│   ├── __init__.py
│   ├── dicke_state.py                  # Primary solutions: create_dicke_state & create_dicke_state_adaptive
│   ├── scoring.py                      # Verification, bitstring statistics, and overall_score metrics
│   ├── noise_models.py                 # Synthetic error generation simulating Willow105 device
│   └── willow_utils.py                 # Willow grid graph extraction and error-weighted qubit selection
└── tests/
    ├── __init__.py
    └── test_dicke_state.py             # Unit tests checking state vector fidelity
```

---

## Deliverables & Deadlines

- **Hacking End & Submission Deadline**: Sunday, October 25, 2026 at 11:30 AM EDT.
- **Required Deliverables**:
  1. **Live Presentation / Demo**: At most 8 minutes in designated judging room (Wilkinson Building).
  2. **Project Repository**: Submitted to Devpost.
  3. **Video Demo**: At most 4 minutes submitted to Devpost.
  4. **Google Challenge Submission**: Submit solution code to [Google Quantum AI Form](https://docs.google.com/forms/d/e/1FAIpQLScw93lwRb3OihfaHgpobs8TO6nXijwvkRVYjWPpNg7n92FjIA/viewform?usp=publish-editor).

---

## Setup & Getting Started

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # On Windows
   pip install -r requirements.txt
   ```

2. Run test verification:
   ```bash
   pytest tests/
   ```

3. Launch Jupyter or the notebook:
   ```bash
   jupyter lab DuQuantum_Google_Standard.ipynb
   ```
