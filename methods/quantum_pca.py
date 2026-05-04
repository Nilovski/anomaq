"""
Method 2 — Quantum PCA (variational)
=====================================

Variational adaptation of Lloyd-Mohseni-Rebentrost qPCA (2014). Each sample is
amplitude-encoded as a unit state vector |x_i>, then the data density matrix
rho = (1/N) sum_i |x_i><x_i| is built. We extract the top-k eigenvectors of
rho via a parameterized circuit U(theta): the optimum theta_k maximizes
<0|U(theta_k)^dagger rho U(theta_k)|0>, i.e. U(theta_k)|0> = principal
eigenvector. Subsequent components found by deflation.

Anomaly score = reconstruction residual ||x - V V^dagger x||^2 where V is
the matrix of recovered eigenvectors. This mirrors classical PCA-residual
outlier detection but uses a quantum circuit to obtain the principal axes.

Reference: Lloyd, Mohseni, Rebentrost (2014); variational qPCA / VQSVD
(Wang et al. 2021).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector

from .base import OutlierDetector
from ._encoding import MagnitudePreservingEncoder


class QuantumPCAResidual(OutlierDetector):
    name = "QuantumPCA"

    def __init__(self, n_qubits: int = 3, n_components: int = 2,
                 reps: int = 2, max_iter: int = 80, n_restarts: int = 2,
                 seed: int = 42):
        self.n_qubits = n_qubits
        self.n_components = n_components
        self.reps = reps
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.seed = seed

    def fit(self, X):
        self.encoder = MagnitudePreservingEncoder(n_qubits=self.n_qubits).fit(X)
        Xa = self.encoder.transform(X)

        d = 2 ** self.n_qubits
        rho = (Xa.T @ Xa) / Xa.shape[0]

        ansatz = efficient_su2(num_qubits=self.n_qubits, reps=self.reps,
                               entanglement="linear")
        n_params = ansatz.num_parameters
        rng = np.random.default_rng(self.seed)

        eigenvecs = []
        eigenvals = []
        rho_def = rho.copy()
        n_components = min(self.n_components, d)

        for k in range(n_components):
            def neg_rayleigh(theta):
                bound = ansatz.assign_parameters(theta)
                psi = Statevector(bound).data
                return -float(np.real(psi.conj() @ rho_def @ psi))

            best = None
            for _ in range(self.n_restarts):
                x0 = rng.uniform(0, 2 * np.pi, n_params)
                res = minimize(neg_rayleigh, x0, method="COBYLA",
                               options={"maxiter": self.max_iter, "rhobeg": 0.5})
                if best is None or res.fun < best.fun:
                    best = res

            bound = ansatz.assign_parameters(best.x)
            v = Statevector(bound).data.astype(complex)
            lam = float(np.real(v.conj() @ rho_def @ v))
            eigenvecs.append(v)
            eigenvals.append(lam)
            rho_def = rho_def - lam * np.outer(v, v.conj())

        self.V = np.column_stack(eigenvecs)            # (d, k) complex
        self.eigenvalues_ = np.array(eigenvals)
        return self

    def score(self, X):
        Xa = self.encoder.transform(X).astype(complex)
        proj = Xa @ self.V @ self.V.conj().T
        residual = np.linalg.norm(Xa - proj, axis=1) ** 2
        return np.real(residual)
