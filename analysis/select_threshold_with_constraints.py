#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Tuple, Optional

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

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = sensitivity

    return {
        "threshold": float(threshold),
        "auc": float(auc),
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold_with_constraints(
    y_true,
    y_prob,
    primary_metric: str = "f1",
    min_recall: Optional[float] = None,
    min_specificity: Optional[float] = None,
    min_precision: Optional[float] = None,
    step: float = 0.001,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    thresholds = np.arange(0.0, 1.0 + 1e-12, step)

    rows = []
    feasible_rows = []

    for t in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        rows.append(m)

        ok = True
        if min_recall is not None and m["recall"] < min_recall:
            ok = False
        if min_specificity is not None and m["specificity"] < min_specificity:
            ok = False
        if min_precision is not None and m["precision"] < min_precision:
            ok = False

        if ok:
            feasible_rows.append(m)

    all_df = pd.DataFrame(rows)
    feasible_df = pd.DataFrame(feasible_rows)

    if len(feasible_rows) == 0:
        raise ValueError(
            "No threshold satisfies the given constraints on validation set. "
            "Try relaxing min_recall / min_specificity / min_precision."
        )

    best_row = sorted(
        feasible_rows,
        key=lambda x: (
            x[primary_metric],
            x["recall"],
            x["specificity"],
            -x["threshold"],
        ),
        reverse=True,
    )[0]

    return float(best_row["threshold"]), best_row, all_df


def main():
    parser = argparse.ArgumentParser(
        "Select threshold on val under recall/spec/precision constraints, then evaluate on test"
    )
    parser.add_argument("--val_csv", type=str, required=True,
                        help="CSV with columns: y_true, y_prob")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="CSV with columns: y_true, y_prob")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--primary_metric", type=str, default="f1",
                        choices=["f1", "acc", "precision", "recall", "sensitivity", "specificity"])
    parser.add_argument("--min_recall", type=float, default=None)
    parser.add_argument("--min_specificity", type=float, default=None)
    parser.add_argument("--min_precision", type=float, default=None)
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
    y_test = test_df["y_true"].values
    p_test = test_df["y_prob"].values

    best_t, best_val_metrics, sweep_df = select_threshold_with_constraints(
        y_true=y_val,
        y_prob=p_val,
        primary_metric=args.primary_metric,
        min_recall=args.min_recall,
        min_specificity=args.min_specificity,
        min_precision=args.min_precision,
        step=args.step,
    )

    test_metrics = compute_metrics(y_test, p_test, threshold=best_t)

    sweep_path = os.path.join(args.out_dir, "val_threshold_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)

    summary = {
        "primary_metric": args.primary_metric,
        "constraints": {
            "min_recall": args.min_recall,
            "min_specificity": args.min_specificity,
            "min_precision": args.min_precision,
        },
        "best_threshold": best_t,
        "val_metrics_at_best_threshold": best_val_metrics,
        "test_metrics_at_best_threshold": test_metrics,
    }

    summary_path = os.path.join(args.out_dir, "threshold_selection_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("========== Threshold Selection with Constraints ==========")
    print(f"primary metric:   {args.primary_metric}")
    print(f"min_recall:       {args.min_recall}")
    print(f"min_specificity:  {args.min_specificity}")
    print(f"min_precision:    {args.min_precision}")
    print(f"best threshold:   {best_t:.4f}")

    print("\n[Validation @ best threshold]")
    print(json.dumps(best_val_metrics, indent=2))

    print("\n[Test @ best threshold]")
    print(json.dumps(test_metrics, indent=2))

    print(f"\n[Saved] {sweep_path}")
    print(f"[Saved] {summary_path}")


if __name__ == "__main__":
    main()