"""
Method 3 — Quantum Autoencoder (QAE)
=====================================

Romero-Olson-Aspuru-Guzik (2017) style autoencoder for unsupervised anomaly
detection. Each sample is amplitude-encoded as |psi(x)>; a parameterized
circuit U(theta) is trained so that the trash subsystem's reduced state
collapses to |0>^m for the *training distribution*. At inference, samples
that fail to compress (low <0|rho_trash|0>) are flagged as anomalies.

Loss (training):    L(theta) = 1 - mean_x <0|^m rho_trash(theta, x) |0>^m
Anomaly score:      s(x)     = 1 - <0|^m rho_trash(theta*, x) |0>^m

Paper section: 9.2.1.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import real_amplitudes
from qiskit.quantum_info import Statevector, partial_trace

from .base import OutlierDetector
from ._encoding import MagnitudePreservingEncoder


class QuantumAutoencoder(OutlierDetector):
    name = "QuantumAutoencoder"

    def __init__(self, n_qubits: int = 3, n_trash: int = 1, reps: int = 2,
                 max_iter: int = 80, n_restarts: int = 2,
                 max_train_samples: int = 60, seed: int = 42):
        if n_trash >= n_qubits:
            raise ValueError("n_trash must be < n_qubits")
        self.n_qubits = n_qubits
        self.n_trash = n_trash
        self.reps = reps
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.max_train_samples = max_train_samples
        self.seed = seed

    def _trash_fidelities(self, theta, Xa):
        bound = self.ansatz.assign_parameters(theta)
        # keep_qubits = first (n_qubits - n_trash); trash = last n_trash.
        # partial_trace(state, qargs) traces out qargs; pass keep to leave trash.
        keep_qubits = list(range(self.n_qubits - self.n_trash))
        fids = np.empty(len(Xa))
        for i, x in enumerate(Xa):
            psi = Statevector(x).evolve(bound)
            rho_trash = partial_trace(psi, keep_qubits)
            fids[i] = float(np.real(rho_trash.data[0, 0]))
        return fids

    def fit(self, X):
        self.encoder = MagnitudePreservingEncoder(n_qubits=self.n_qubits).fit(X)
        Xa = self.encoder.transform(X)

        # Subsample for training speed (autoencoder objective is averaged).
        rng = np.random.default_rng(self.seed)
        if len(Xa) > self.max_train_samples:
            idx = rng.choice(len(Xa), size=self.max_train_samples, replace=False)
            Xa_train = Xa[idx]
        else:
            Xa_train = Xa

        self.ansatz = real_amplitudes(num_qubits=self.n_qubits, reps=self.reps)
        n_params = self.ansatz.num_parameters

        def loss(theta):
            return 1.0 - float(np.mean(self._trash_fidelities(theta, Xa_train)))

        best = None
        for _ in range(self.n_restarts):
            x0 = rng.uniform(0, 2 * np.pi, n_params)
            res = minimize(loss, x0, method="COBYLA",
                           options={"maxiter": self.max_iter, "rhobeg": 0.5})
            if best is None or res.fun < best.fun:
                best = res

        self.theta_ = best.x
        self.train_loss_ = float(best.fun)
        return self

    def score(self, X):
        Xa = self.encoder.transform(X)
        fids = self._trash_fidelities(self.theta_, Xa)
        return 1.0 - fids
