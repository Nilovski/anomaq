"""
Method 4 — QAOA MaxCut Clustering
==================================

Each sample becomes a vertex in a complete graph with edge weights w_ij equal
to the pairwise dissimilarity (Euclidean distance after standardisation,
normalised to [0, 1]). MaxCut maximises sum_{cut edges} w_ij — so highly
dissimilar pairs prefer to land on opposite sides of the partition. An
isolated outlier (dissimilar to *all* others) tends to end up in the minority
partition.

Per-sample anomaly score = sum over high-probability bitstrings, weighted by
amplitude probability, of an indicator that this qubit lies in the minority
partition.

Hard limit: one qubit per sample. We cap at `max_samples` qubits and (for
larger benchmarks) score only that subsample; remaining samples receive 0.

Paper section: 10.
"""

from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.optimize import minimize

from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import Statevector, SparsePauliOp

from .base import OutlierDetector


class QAOAMaxCut(OutlierDetector):
    name = "QAOA-MaxCut"

    def __init__(self, max_samples: int = 12, p_layers: int = 2,
                 max_iter: int = 100, n_restarts: int = 3, seed: int = 42):
        self.max_samples = max_samples
        self.p_layers = p_layers
        self.max_iter = max_iter
        self.n_restarts = n_restarts
        self.seed = seed

    def _build_cost_op(self, W):
        """H_C = sum_{i<j} (w_ij/2) (Z_i Z_j - I).  Min(H_C) <=> Max(cut).

        We *minimise* expectation of H_C; ground state = max-cut bitstring.
        Constant -sum w_ij/2 dropped (does not affect optimisation)."""
        N = W.shape[0]
        terms = []
        for i in range(N):
            for j in range(i + 1, N):
                w = float(W[i, j])
                if w < 1e-9:
                    continue
                label = ["I"] * N
                label[i] = "Z"
                label[j] = "Z"
                # qiskit Pauli string is little-endian (qubit 0 is rightmost)
                terms.append(("".join(reversed(label)), 0.5 * w))
        return SparsePauliOp.from_list(terms)

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        n_full = len(Xs)

        rng = np.random.default_rng(self.seed)
        if n_full > self.max_samples:
            self.idx_ = np.sort(rng.choice(n_full, size=self.max_samples,
                                           replace=False))
        else:
            self.idx_ = np.arange(n_full)

        Xq = Xs[self.idx_]
        N = len(Xq)

        D = pairwise_distances(Xq)
        W = D / (D.max() + 1e-12)            # normalize to [0, 1]

        self.cost_op_ = self._build_cost_op(W)
        self.ansatz_ = qaoa_ansatz(cost_operator=self.cost_op_,
                                   reps=self.p_layers)
        n_params = self.ansatz_.num_parameters

        def expval(theta):
            bound = self.ansatz_.assign_parameters(theta)
            psi = Statevector(bound)
            return float(np.real(psi.expectation_value(self.cost_op_)))

        best = None
        for _ in range(self.n_restarts):
            x0 = rng.uniform(0, 2 * np.pi, n_params)
            res = minimize(expval, x0, method="COBYLA",
                           options={"maxiter": self.max_iter, "rhobeg": 0.5})
            if best is None or res.fun < best.fun:
                best = res

        self.theta_ = best.x
        bound = self.ansatz_.assign_parameters(self.theta_)
        psi = Statevector(bound)
        probs = psi.probabilities_dict()

        # Identify the (most-likely) bitstring's minority partition
        top = max(probs.items(), key=lambda kv: kv[1])[0]
        # qiskit bitstring is little-endian (rightmost = qubit 0); reverse.
        top_bits = np.array([int(b) for b in top[::-1]])
        n_ones = int(top_bits.sum())
        outlier_label = 1 if n_ones < N - n_ones else 0

        # Per-qubit minority-membership probability across all bitstrings
        scores_sub = np.zeros(N, dtype=float)
        for bs, p in probs.items():
            bv = np.array([int(b) for b in bs[::-1]])
            n_ones_bs = int(bv.sum())
            label = 1 if n_ones_bs < N - n_ones_bs else 0
            in_minority = (bv == label).astype(float)
            scores_sub += p * in_minority

        # If we ended up identifying outlier_label per-bitstring, that's already
        # consistent. Save full-length scores aligned with original X order.
        full_scores = np.zeros(n_full, dtype=float)
        full_scores[self.idx_] = scores_sub
        self._scores = full_scores
        self._outlier_partition = outlier_label
        return self

    def score(self, X):
        # Transductive: stored from fit. If called with different X, refit.
        if len(X) == len(self._scores):
            return self._scores
        return self.fit(X).score(X)
