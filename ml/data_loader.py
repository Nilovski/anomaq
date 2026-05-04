"""
ml/data_loader.py
-----------------
Loads and preprocesses the Tox21 SR-MMP assay from DeepChem.

Responsibilities (single):
  - Pull dataset
  - Extract ECFP fingerprints
  - Filter missing labels
  - Return (X, y) numpy arrays

Nothing else. No PCA, no train/test logic, no model code here.
"""

from __future__ import annotations
import numpy as np
import deepchem as dc


ASSAY = "SR-MMP"


def load_tox21(
    assay: str = ASSAY,
    splitter: str = "random",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_train, y_train, X_test, y_test) for the given Tox21 assay.
    Rows with missing labels are dropped.
    X shape: (n, 1024)  — ECFP Morgan fingerprints
    y shape: (n,)       — 0=non-toxic, 1=toxic
    """
    tasks, datasets, _ = dc.molnet.load_tox21(
        featurizer="ECFP", splitter=splitter
    )
    assay_idx = list(tasks).index(assay)
    train_ds, _, test_ds = datasets

    def _extract(ds: dc.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
        X = ds.X
        y = ds.y[:, assay_idx]
        w = ds.w[:, assay_idx]
        mask = w != 0
        return X[mask].astype(np.float32), y[mask].astype(int)

    X_tr, y_tr = _extract(train_ds)
    X_te, y_te = _extract(test_ds)
    return X_tr, y_tr, X_te, y_te
