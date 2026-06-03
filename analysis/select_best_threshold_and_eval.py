#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_prob, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = sens

    return {
        "threshold": float(threshold),
        "auc": float(auc),
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(
    y_true,
    y_prob,
    metric: str = "f1",
    min_threshold: float = 0.0,
    max_threshold: float = 1.0,
    step: float = 0.001,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    thresholds = np.arange(min_threshold, max_threshold + 1e-12, step)

    rows = []
    best_t = 0.5
    best_score = -1e18
    best_metrics = None

    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        rows.append(m)

        score = m[metric]
        if score > best_score:
            best_score = score
            best_t = float(t)
            best_metrics = m

    table = pd.DataFrame(rows)
    return best_t, best_metrics, table


def main():
    parser = argparse.ArgumentParser("Select threshold on val and evaluate on test")
    parser.add_argument("--val_csv", type=str, required=True,
                        help="CSV with columns: y_true, y_prob")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="CSV with columns: y_true, y_prob")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--metric", type=str, default="f1",
                        choices=["f1", "acc", "sensitivity", "specificity", "precision", "recall"])
    parser.add_argument("--step", type=float, default=0.001)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    for name, df in [("val", val_df), ("test", test_df)]:
        if "y_true" not in df.columns or "y_prob" not in df.columns:
            raise ValueError(f"{name}_csv must contain columns: y_true, y_prob")

    y_val = val_df["y_true"].values
    p_val = val_df["y_prob"].values

    best_t, best_val_metrics, sweep_df = find_best_threshold(
        y_true=y_val,
        y_prob=p_val,
        metric=args.metric,
        step=args.step,
    )

    y_test = test_df["y_true"].values
    p_test = test_df["y_prob"].values
    test_metrics = compute_metrics(y_test, p_test, threshold=best_t)

    sweep_path = os.path.join(args.out_dir, "val_threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)

    summary = {
        "selection_metric": args.metric,
        "best_threshold": best_t,
        "val_metrics_at_best_threshold": best_val_metrics,
        "test_metrics_at_best_threshold": test_metrics,
    }

    summary_path = os.path.join(args.out_dir, "threshold_selection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("========== Threshold Selection ==========")
    print(f"selection metric: {args.metric}")
    print(f"best threshold:   {best_t:.4f}")

    print("\n[Validation @ best threshold]")
    print(json.dumps(best_val_metrics, indent=2))

    print("\n[Test @ best threshold]")
    print(json.dumps(test_metrics, indent=2))

    print(f"\n[Saved] {sweep_path}")
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()