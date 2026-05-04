"""
Amplitude-encoding helpers shared by quantum methods that consume statevectors.

Naive normalisation `x / ||x||` is destructive for outlier detection: a sample
with magnitude 5 in every feature ends up with the *same* unit vector as a
sample with magnitude 1. We therefore use a *magnitude-preserving* encoding:

    1. Stack X with a final padding column.
    2. Globally rescale so the largest row norm equals R-eps (R<1).
    3. Set the padding column to sqrt(1 - ||rescaled_row||^2).

The result: every row has unit norm (required for a statevector), and the
amount of amplitude on the padding slot encodes how *small* the original
row was relative to the largest. Outliers in scale or in direction are both
distinguishable in the resulting density matrix.

The encoder also pads (or truncates via a fitted PCA) the feature dim to fit
exactly 2**n_qubits - 1 useful slots (last slot reserved for the magnitude
complement). For high-dim data we PCA-reduce *before* encoding.
"""

from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MagnitudePreservingEncoder:
    """Standardize -> (optional PCA) -> magnitude-preserving amplitude encoding."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        # last slot reserved for sqrt(1 - ||x||^2)
        self.usable_features = self.dim - 1

    def fit(self, X: np.ndarray) -> "MagnitudePreservingEncoder":
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)

        if X.shape[1] > self.usable_features:
            self.pca_ = PCA(n_components=self.usable_features).fit(Xs)
            Xr = self.pca_.transform(Xs)
        else:
            self.pca_ = None
            Xr = Xs

        # Rescale so all rows fit in a ball of radius (1 - epsilon).
        norms = np.linalg.norm(Xr, axis=1)
        self.scale_radius_ = float(norms.max()) + 1e-9
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        Xr = self.pca_.transform(Xs) if self.pca_ is not None else Xs

        Xp = np.zeros((Xr.shape[0], self.dim), dtype=float)
        n_feat = Xr.shape[1]
        Xp[:, :n_feat] = Xr / self.scale_radius_
        rescaled = np.linalg.norm(Xp[:, :n_feat], axis=1)
        rescaled = np.clip(rescaled, 0.0, 1.0 - 1e-12)
        Xp[:, -1] = np.sqrt(1.0 - rescaled ** 2)

        # Re-normalize defensively against floating-point drift.
        norms = np.linalg.norm(Xp, axis=1, keepdims=True)
        return Xp / norms

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
