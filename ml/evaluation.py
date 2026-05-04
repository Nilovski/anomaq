"""
ml/evaluation.py
----------------
Metrics and plots for quantum vs classical OCSVM comparison.

Responsibilities (single):
  - Compute precision, recall, F1, ROC-AUC
  - Render ROC curves, confusion matrices, scatter plots, kernel heatmap
  - Save figure to output_path

Fixes from v1:
  - ConfusionMatrixDisplay.plot() called correctly (assign to variable first)
  - Both models share one figure with clear subplot layout
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from dataclasses import dataclass


@dataclass
class Metrics:
    name: str
    precision: float
    recall: float
    f1: float
    auc: float

    def __str__(self) -> str:
        return (
            f"{self.name}: "
            f"P={self.precision:.3f} R={self.recall:.3f} "
            f"F1={self.f1:.3f} AUC={self.auc:.3f}"
        )


def compute_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> Metrics:
    p  = precision_score(y_true, y_pred, zero_division=0)
    r  = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = float("nan")
    m = Metrics(name=name, precision=p, recall=r, f1=f1, auc=auc)
    print(m)
    return m


def plot_results(
    y_true: np.ndarray,
    X_test_2d: np.ndarray,
    q_pred: np.ndarray,
    q_scores: np.ndarray,
    c_pred: np.ndarray,
    c_scores: np.ndarray,
    K_train: np.ndarray,
    output_path: str = "nautilus_results.png",
) -> None:
    """
    6-panel figure:
      [0,0] ROC curves (both models)
      [0,1] Confusion matrix — quantum
      [0,2] Confusion matrix — classical
      [1,0] PCA 2D scatter — true labels
      [1,1] PCA 2D scatter — quantum predictions
      [1,2] Quantum kernel heatmap (50×50 subset)
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # ROC curves
    ax = axes[0, 0]
    for name, y_sc, color in [
        ("Quantum",   q_scores, "royalblue"),
        ("Classical", c_scores, "tomato"),
    ]:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_sc)
            auc = roc_auc_score(y_true, y_sc)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", color=color)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curves"); ax.legend()

    # Confusion matrices — fix: assign ConfusionMatrixDisplay to variable before plot()
    for ax, y_pred, title in [
        (axes[0, 1], q_pred, "Confusion — Quantum"),
        (axes[0, 2], c_pred, "Confusion — Classical RBF"),
    ]:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Non-toxic", "Toxic"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(title)

    # PCA 2D scatter — true labels
    ax = axes[1, 0]
    ax.scatter(X_test_2d[y_true == 0, 0], X_test_2d[y_true == 0, 1],
               c="steelblue", alpha=0.5, s=20, label="Non-toxic")
    ax.scatter(X_test_2d[y_true == 1, 0], X_test_2d[y_true == 1, 1],
               c="crimson", alpha=0.7, s=30, marker="x", label="Toxic")
    ax.set_title("PCA 2D — True Labels"); ax.legend()

    # PCA 2D scatter — quantum predictions
    ax = axes[1, 1]
    ax.scatter(X_test_2d[q_pred == 0, 0], X_test_2d[q_pred == 0, 1],
               c="steelblue", alpha=0.5, s=20, label="Predicted normal")
    ax.scatter(X_test_2d[q_pred == 1, 0], X_test_2d[q_pred == 1, 1],
               c="crimson", alpha=0.7, s=30, marker="x", label="Flagged outlier")
    ax.set_title("Quantum Predictions"); ax.legend()

    # Kernel heatmap — visually compelling for judges
    ax = axes[1, 2]
    n = min(50, K_train.shape[0])
    im = ax.imshow(K_train[:n, :n], cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Quantum Kernel Matrix ({n}×{n} subset)")
    ax.set_xlabel("Training sample"); ax.set_ylabel("Training sample")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")
    plt.show()
