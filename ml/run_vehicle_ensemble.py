"""
Vehicle-ensemble outlier detection — the headline hackathon demo.
=================================================================

Loads `vehicle_ensemble_data.csv`, runs all 4 quantum methods + classical
baselines, and prints which vehicle each method flags as the *physical
outlier*. Saves a per-method ranking to results/vehicle_ranking.csv.
"""

from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

from methods import (
    IsolationForestDetector,
    LOFDetector,
    OCSVMDetector,
    QuantumPCAResidual,
    QuantumAutoencoder,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(__file__).resolve().parents[1]
                                          / "vehicle_ensemble_data.csv"))
    ap.add_argument("--top", type=int, default=5,
                    help="How many top-scored vehicles to print per method.")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent
                                          / "results" / "vehicle_ranking.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    id_col = df.columns[0]
    feature_cols = [c for c in df.columns if c != id_col
                    and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].to_numpy(dtype=float)
    ids = df[id_col].astype(str).to_numpy()
    print(f"Loaded {len(df)} vehicles × {len(feature_cols)} features")
    print(f"Features: {feature_cols}")

    n_qubits = min(4, len(feature_cols))
    methods = [
        IsolationForestDetector(),
        LOFDetector(n_neighbors=min(20, len(df) - 1)),
        OCSVMDetector(nu=0.05),
        QuantumPCAResidual(n_qubits=n_qubits, n_components=2,
                           reps=2, max_iter=80, n_restarts=2),
        QuantumAutoencoder(n_qubits=n_qubits, n_trash=1, reps=2,
                           max_iter=80, n_restarts=2,
                           max_train_samples=len(df)),
    ]

    rankings = {}
    for m in methods:
        print(f"\n--- {m.name} ---")
        t0 = time.time()
        try:
            scores = m.fit_score(X)
            elapsed = time.time() - t0
            order = np.argsort(-scores)
            print(f"  trained in {elapsed:.2f}s")
            print(f"  top-{args.top} most anomalous:")
            for rank, idx in enumerate(order[:args.top], 1):
                print(f"    {rank}. {ids[idx]:<10}  score={scores[idx]:+.4f}")
            rankings[m.name] = scores
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            rankings[m.name] = np.full(len(df), np.nan)

    # Consensus: top-1 vote across methods
    print("\n=== Top-1 vote ===")
    votes: dict[str, int] = {}
    for name, scores in rankings.items():
        if np.all(np.isnan(scores)):
            continue
        winner = ids[int(np.argmax(scores))]
        votes[winner] = votes.get(winner, 0) + 1
        print(f"  {name:<22} → {winner}")
    if votes:
        winner = max(votes.items(), key=lambda kv: kv[1])
        print(f"\n  Consensus physical outlier: {winner[0]}  "
              f"({winner[1]}/{len(rankings)} methods agree)")

    # Persist ranking CSV
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        header = [id_col] + list(rankings.keys())
        w.writerow(header)
        for i in range(len(df)):
            row = [ids[i]] + [f"{rankings[m][i]:.6f}" if not np.isnan(rankings[m][i])
                              else "" for m in rankings]
            w.writerow(row)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
