"""Common interface for all outlier-detection methods."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class OutlierDetector(ABC):
    """All methods produce per-sample anomaly scores, higher = more anomalous."""

    name: str = "OutlierDetector"

    @abstractmethod
    def fit(self, X: np.ndarray) -> "OutlierDetector":
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return shape-(n,) anomaly scores (higher = more anomalous)."""
        ...

    def fit_score(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).score(X)
