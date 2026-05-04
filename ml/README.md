# Nautilus — ML Module

Quantum Kernel One-Class SVM for anomaly/outlier detection.
Dataset: Tox21 SR-MMP assay (toxic compound detection as proof-of-concept).

## Structure

```
ml/
├── README.md          ← you are here
├── requirements.txt   ← pip dependencies
├── run.py             ← entry point (orchestration only)
├── data_loader.py     ← Tox21 loading + ECFP extraction
├── preprocessing.py   ← PCA, normalization, one-class split
├── quantum_kernel.py  ← ZZFeatureMap + FidelityQuantumKernel
├── model.py           ← QuantumOCSVM + ClassicalOCSVM
└── evaluation.py      ← metrics + 6-panel results plot
```

## Setup

```bash
cd ml
pip install -r requirements.txt
python run.py
```

## Key design decisions

- `n_components = 6` (qubit budget) — keep low for fast statevector simulation
- `reps = 2` in ZZFeatureMap — minimum for cross-feature entanglement (ZZ gates)
- Training set: non-toxic compounds only (one-class SVM paradigm)
- Classical baseline uses RBF kernel on same PCA features — fair comparison
- Kernel computation uses `StatevectorSampler` (exact, fast) not Aer noisy sim

## Why quantum kernel?

RBF kernel computes `exp(-γ||x-y||²)` — no cross-feature interactions.
ZZFeatureMap encodes `x_i × x_j` interaction terms via CNOT entanglement.
Quantum kernel measures state overlap in a 2^n-dimensional Hilbert space.
Citation: Havlíček et al., *Nature* 567, 209–212 (2019).
