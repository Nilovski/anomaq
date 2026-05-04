"""
Outlier-detection benchmark runner
====================================

Loads ODDS-style .npz benchmarks (X, y) and evaluates each method's anomaly
scores against ground truth.

Metrics:
  ROC-AUC          — area under ROC curve (full ranking quality)
  precision@k      — fraction of top-k scored samples that are true outliers
                     (k = number of true outliers in the dataset)

Outputs:
  - Markdown table to stdout
  - CSV of every (dataset, method) row to results/benchmark_results.csv
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from methods import (
    IsolationForestDetector,
    LOFDetector,
    OCSVMDetector,
    QuantumPCAResidual,
    QuantumAutoencoder,
)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0 or k > len(y_true):
        return float("nan")
    top = np.argsort(-scores)[:k]
    return float(y_true[top].sum() / k)


def stratified_subsample(X, y, max_n, seed=42):
    """Keep all outliers (or up to half of max_n) plus enough inliers."""
    if len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    out_idx = np.where(y == 1)[0]
    in_idx = np.where(y == 0)[0]
    n_out = min(len(out_idx), max(1, int(round(max_n * len(out_idx) / len(y)))))
    n_in = max_n - n_out
    keep_out = rng.choice(out_idx, size=n_out, replace=False)
    keep_in = rng.choice(in_idx, size=min(n_in, len(in_idx)), replace=False)
    keep = np.sort(np.concatenate([keep_out, keep_in]))
    return X[keep], y[keep]


def load_dataset(npz_path: str):
    d = np.load(npz_path)
    X = np.asarray(d["X"], dtype=float)
    y = np.asarray(d["y"], dtype=int).ravel()
    return X, y


def make_methods(n_features: int):
    """Build fresh method instances tuned to the dataset's feature count."""
    n_qubits = min(4, max(2, n_features))
    return [
        IsolationForestDetector(),
        LOFDetector(n_neighbors=20),
        OCSVMDetector(nu=0.05),
        QuantumPCAResidual(n_qubits=n_qubits, n_components=2,
                           reps=2, max_iter=80, n_restarts=2),
        QuantumAutoencoder(n_qubits=n_qubits, n_trash=1, reps=2,
                           max_iter=80, n_restarts=2,
                           max_train_samples=180),
    ]


def run_one(method, X, y):
    """Returns (auc, p_at_k, seconds, error_message)."""
    k = int(y.sum())
    t0 = time.time()
    try:
        scores = method.fit_score(X)
        elapsed = time.time() - t0
        auc = float(roc_auc_score(y, scores)) if k > 0 else float("nan")
        pak = precision_at_k(y, scores, k)
        return auc, pak, elapsed, None
    except Exception as e:
        elapsed = time.time() - t0
        return float("nan"), float("nan"), elapsed, f"{type(e).__name__}: {e}"


def fmt(v, fmt_str="{:.3f}"):
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return "  --  "
    return fmt_str.format(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default=str(Path(__file__).resolve().parents[1]
                                                / "benchmarks"))
    ap.add_argument("--max-n", type=int, default=0,
                    help="Subsample datasets larger than this (stratified). "
                         "0 = uncapped (recommended for QPCA/QAE which scale).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Subset of dataset names (without .npz). Default = all.")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Subset of method names. Default = all.")
    ap.add_argument("--out-csv", default=str(Path(__file__).resolve().parent
                                              / "results" / "benchmark_results.csv"))
    ap.add_argument("--verbose-errors", action="store_true",
                    help="Print full tracebacks when a method fails.")
    args = ap.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.bench_dir, "*.npz")))
    if args.datasets:
        wanted = set(args.datasets)
        npz_files = [p for p in npz_files
                     if Path(p).stem in wanted or Path(p).name in wanted]

    if not npz_files:
        print(f"No .npz files in {args.bench_dir}")
        return

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for npz in npz_files:
        ds_name = Path(npz).stem
        X, y = load_dataset(npz)
        n_full = len(X)
        if args.max_n and n_full > args.max_n:
            X, y = stratified_subsample(X, y, args.max_n)
        n_samples, n_features = X.shape
        n_outliers = int(y.sum())

        print(f"\n=== {ds_name}  (n={n_samples}/{n_full}, "
              f"feat={n_features}, outliers={n_outliers}) ===")
        print(f"{'method':<22} {'AUC':>7} {'P@k':>7} {'sec':>8}")
        print("-" * 50)

        methods = make_methods(n_features)
        if args.methods:
            wanted = {m.lower() for m in args.methods}
            methods = [m for m in methods if m.name.lower() in wanted
                       or m.name.lower().startswith(tuple(wanted))]

        for m in methods:
            auc, pak, secs, err = run_one(m, X, y)
            print(f"{m.name:<22} {fmt(auc):>7} {fmt(pak):>7} {secs:>7.1f}s"
                  + (f"  [{err}]" if err else ""))
            if err and args.verbose_errors:
                traceback.print_exc()
            rows.append({
                "dataset": ds_name,
                "n": n_samples,
                "n_full": n_full,
                "features": n_features,
                "outliers": n_outliers,
                "method": m.name,
                "auc": auc,
                "p_at_k": pak,
                "seconds": secs,
                "error": err or "",
            })

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {args.out_csv} ({len(rows)} rows)")

    print("\n=== AUC summary (rows = datasets, cols = methods) ===")
    datasets = sorted({r["dataset"] for r in rows})
    methods_seen = sorted({r["method"] for r in rows})
    print(f"{'dataset':<20}", " ".join(f"{m[:18]:>18}" for m in methods_seen))
    by = {(r["dataset"], r["method"]): r["auc"] for r in rows}
    for d in datasets:
        cells = " ".join(f"{fmt(by.get((d, m), float('nan'))):>18}"
                         for m in methods_seen)
        print(f"{d:<20} {cells}")


if __name__ == "__main__":
    main()
