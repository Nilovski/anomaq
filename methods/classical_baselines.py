"""Classical outlier-detection baselines wrapped in the OutlierDetector interface."""

from __future__ import annotations
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from .base import OutlierDetector


class IsolationForestDetector(OutlierDetector):
    name = "IsolationForest"

    def __init__(self, contamination: float | str = "auto", random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model = IsolationForest(contamination=self.contamination,
                                     random_state=self.random_state)
        self.model.fit(Xs)
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        return -self.model.score_samples(Xs)


class LOFDetector(OutlierDetector):
    """LOF in transductive (fit_predict on full dataset) mode."""

    name = "LOF"

    def __init__(self, n_neighbors: int = 20, contamination: float | str = "auto"):
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        self._X = self.scaler.transform(X)
        n_neighbors = min(self.n_neighbors, max(2, len(X) - 1))
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors,
                                        contamination=self.contamination)
        self.model.fit_predict(self._X)
        self._scores = -self.model.negative_outlier_factor_
        return self

    def score(self, X):
        # Transductive — scores are stored from fit on the same data.
        # If called with a different X, fall back to refitting.
        Xs = self.scaler.transform(X)
        if Xs.shape == self._X.shape and np.allclose(Xs, self._X):
            return self._scores
        # Refit (OK for benchmarks where train==test)
        n_neighbors = min(self.n_neighbors, max(2, len(X) - 1))
        m = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=self.contamination)
        m.fit_predict(Xs)
        return -m.negative_outlier_factor_


class OCSVMDetector(OutlierDetector):
    name = "OneClassSVM-RBF"

    def __init__(self, nu: float = 0.05, gamma: str | float = "scale"):
        self.nu = nu
        self.gamma = gamma

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model = OneClassSVM(kernel="rbf", nu=self.nu, gamma=self.gamma)
        self.model.fit(Xs)
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        return -self.model.decision_function(Xs)
