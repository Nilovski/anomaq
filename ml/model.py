"""
ml/model.py
-----------
One-Class SVM wrappers: quantum (precomputed kernel) and classical (RBF).

Responsibilities (single):
  - Fit and predict with OneClassSVM
  - Return binary predictions and raw decision scores

Outlier convention: toxic compound = outlier = label -1 from sklearn → mapped to 1.
Normal compound  = non-toxic = label +1 from sklearn → mapped to 0.
"""

from __future__ import annotations
import numpy as np
from sklearn.svm import OneClassSVM


class QuantumOCSVM:
    """
    One-Class SVM with precomputed quantum kernel matrix.
    Pass kernel matrices from QuantumKernelBuilder.evaluate().
    """

    def __init__(self, nu: float = 0.1):
        self._svm = OneClassSVM(kernel="precomputed", nu=nu)

    def fit(self, K_train: np.ndarray) -> "QuantumOCSVM":
        self._svm.fit(K_train)
        return self

    def predict(self, K_test: np.ndarray) -> np.ndarray:
        """Returns binary array: 1=toxic (outlier), 0=non-toxic (normal)."""
        raw = self._svm.predict(K_test)        # sklearn: +1 normal, -1 outlier
        return (raw == -1).astype(int)

    def decision_scores(self, K_test: np.ndarray) -> np.ndarray:
        """Higher score = more likely outlier (toxic). For ROC-AUC."""
        return -self._svm.decision_function(K_test)


class ClassicalOCSVM:
    """
    One-Class SVM with RBF kernel. Baseline comparison.
    Pass raw feature arrays (not kernel matrices).
    """

    def __init__(self, nu: float = 0.1):
        self._svm = OneClassSVM(kernel="rbf", nu=nu)

    def fit(self, X_train: np.ndarray) -> "ClassicalOCSVM":
        self._svm.fit(X_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        raw = self._svm.predict(X_test)
        return (raw == -1).astype(int)

    def decision_scores(self, X_test: np.ndarray) -> np.ndarray:
        return -self._svm.decision_function(X_test)
