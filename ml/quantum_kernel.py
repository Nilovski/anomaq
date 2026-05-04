"""
ml/quantum_kernel.py
--------------------
Builds the quantum kernel using Qiskit's ZZFeatureMap.

Responsibilities (single):
  - Construct ZZFeatureMap + FidelityQuantumKernel
  - Expose evaluate() wrapper for train/test kernel matrices

Fixes from v1:
  - StatevectorSampler/AerSimulator imports removed (not needed here)
  - FidelityQuantumKernel now uses ComputeUncompute fidelity (required >=0.7)
  - Kernel object is stateless — create once, call evaluate() as needed

Why ZZFeatureMap?
  Each feature maps to one qubit. reps=2 adds a second layer of ZZ entanglement
  gates, encoding pairwise cross-feature interactions (x_i × x_j terms) that an
  RBF kernel cannot represent explicitly. This is the core quantum advantage claim
  (Havlíček et al., Nature 2019).
"""

from __future__ import annotations
import numpy as np
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.kernels.algorithms import ComputeUncompute
from qiskit.primitives import StatevectorSampler


class QuantumKernelBuilder:
    """
    Wraps FidelityQuantumKernel for easy reuse.

    Parameters
    ----------
    n_features : int
        Number of qubits = number of PCA components.
    reps : int
        ZZFeatureMap repetitions. 2 is the minimum for cross-feature entanglement.
    """

    def __init__(self, n_features: int = 6, reps: int = 2):
        self.feature_map = ZZFeatureMap(
            feature_dimension=n_features, reps=reps
        )
        sampler = StatevectorSampler()
        fidelity = ComputeUncompute(sampler=sampler)
        self.kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            fidelity=fidelity,
        )

    def evaluate(
        self,
        X1: np.ndarray,
        X2: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute kernel matrix K where K[i,j] = |<phi(X1_i)|phi(X2_j)>|^2.
        If X2 is None, computes the symmetric training kernel (X1 vs X1).
        """
        if X2 is None:
            X2 = X1
        return self.kernel.evaluate(x_vec=X1, y_vec=X2)
