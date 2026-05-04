"""
ml/preprocessing.py
-------------------
PCA dimensionality reduction + normalization for quantum circuits.

Responsibilities (single):
  - Fit PCA on training data
  - Transform train/test to n_components dimensions
  - Normalize to [-1, 1] (required: ZZFeatureMap encodes values as rotation angles)
  - Subsample for tractable quantum kernel computation

Nothing else. No model code, no data loading, no plots.
"""

from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA


class QuantumPreprocessor:
    """
    Fits PCA on training data; normalizes to [-1, 1].
    Use fit_transform() on train, then transform() on test.
    n_components = qubit budget (keep <= 8 for reasonable circuit depth).
    """

    def __init__(self, n_components: int = 6):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        reduced = self.pca.fit_transform(X)
        self._min = reduced.min(axis=0)
        self._max = reduced.max(axis=0)
        return self._normalize(reduced)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._min is None:
            raise RuntimeError("Call fit_transform first.")
        reduced = self.pca.transform(X)
        return self._normalize(reduced)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        rng = self._max - self._min + 1e-8
        return 2.0 * (X - self._min) / rng - 1.0


def one_class_split(
    X: np.ndarray,
    y: np.ndarray,
    n_train: int = 200,
    n_test: int = 100,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X_train, X_test, y_test).
    X_train: non-toxic only (normal class for one-class SVM).
    X_test / y_test: balanced subsample for evaluation.
    """
    rng = np.random.default_rng(seed)

    non_toxic = np.where(y == 0)[0]
    train_idx = rng.choice(non_toxic, size=min(n_train, len(non_toxic)), replace=False)
    X_train = X[train_idx]

    test_idx = rng.choice(len(y), size=min(n_test, len(y)), replace=False)
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_test
