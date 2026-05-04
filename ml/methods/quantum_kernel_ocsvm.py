"""
Method 1 — Quantum Kernel + One-Class SVM
==========================================

Encodes features through a ZZFeatureMap, computes the fidelity Gram matrix
K_ij = |<phi(x_i)|phi(x_j)>|^2 on a statevector simulator, then fits a
precomputed-kernel One-Class SVM. Anomaly score = -decision_function (higher
= more anomalous).

Paper section: 8.5 / 9.2.3 (natively quantum kernels).
Reference: Havlicek et al., Nature 567, 209-212 (2019).
"""

from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM

from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

from .base import OutlierDetector


class QuantumKernelOCSVM(OutlierDetector):
    name = "QuantumKernel-OCSVM"

    def __init__(self, n_qubits: int = 4, reps: int = 2, nu: float = 0.05):
        self.n_qubits = n_qubits
        self.reps = reps
        self.nu = nu

    def _preprocess_fit(self, X):
        self.std_scaler = StandardScaler().fit(X)
        Xs = self.std_scaler.transform(X)

        n_components = min(self.n_qubits, X.shape[1])
        self.n_components = n_components
        if X.shape[1] > n_components:
            self.pca = PCA(n_components=n_components).fit(Xs)
            Xr = self.pca.transform(Xs)
        else:
            self.pca = None
            Xr = Xs

        self.minmax = MinMaxScaler(feature_range=(0.0, np.pi)).fit(Xr)
        return self.minmax.transform(Xr)

    def _preprocess(self, X):
        Xs = self.std_scaler.transform(X)
        Xr = self.pca.transform(Xs) if self.pca is not None else Xs
        return self.minmax.transform(Xr)

    def fit(self, X):
        Xq = self._preprocess_fit(X)

        feature_map = zz_feature_map(feature_dimension=self.n_components,
                                     reps=self.reps, entanglement="linear")
        sampler = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=sampler)
        self.qkernel = FidelityQuantumKernel(feature_map=feature_map,
                                             fidelity=fidelity)

        K = self.qkernel.evaluate(x_vec=Xq)
        self.svm = OneClassSVM(kernel="precomputed", nu=self.nu)
        self.svm.fit(K)

        self._X_train = Xq
        return self

    def score(self, X):
        Xq = self._preprocess(X)
        K = self.qkernel.evaluate(x_vec=Xq, y_vec=self._X_train)
        return -self.svm.decision_function(K)
