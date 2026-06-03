import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_prob, threshold=0.5):
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
    youden = sensitivity + specificity - 1.0

    return {
        "threshold": float(threshold),
        "auc": float(auc),
        "acc": float(acc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "youden": float(youden),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_best_threshold(y_true, y_prob, mode="f1", num_thresholds=1001):
    """
    mode:
      - f1
      - youden
      - sensitivity_at_least
    """
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    rows = []
    for th in thresholds:
        m = compute_metrics(y_true, y_prob, threshold=th)
        rows.append(m)

    df = pd.DataFrame(rows)

    if mode == "f1":
        best_idx = df["f1"].idxmax()

    elif mode == "youden":
        best_idx = df["youden"].idxmax()

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    best_row = df.loc[best_idx].to_dict()
    return best_row, df


def main():
    parser = argparse.ArgumentParser("Tune threshold on val predictions and evaluate on test predictions")

    parser.add_argument("--val_csv", type=str, required=True,
                        help="best_val_predictions.csv path")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="test_predictions.csv path")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="output directory")
    parser.add_argument("--mode", type=str, default="f1", choices=["f1", "youden"],
                        help="criterion for selecting threshold on val")
    parser.add_argument("--default_threshold", type=float, default=0.5,
                        help="reference threshold for comparison")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)

    required_cols = {"y_true", "y_prob"}
    if not required_cols.issubset(val_df.columns):
        raise ValueError(f"val_csv must contain columns: {required_cols}")
    if not required_cols.issubset(test_df.columns):
        raise ValueError(f"test_csv must contain columns: {required_cols}")

    val_y_true = val_df["y_true"].values
    val_y_prob = val_df["y_prob"].values

    test_y_true = test_df["y_true"].values
    test_y_prob = test_df["y_prob"].values

    # baseline at 0.5
    val_default = compute_metrics(val_y_true, val_y_prob, threshold=args.default_threshold)
    test_default = compute_metrics(test_y_true, test_y_prob, threshold=args.default_threshold)

    # best threshold on val
    best_val_row, search_df = find_best_threshold(
        val_y_true,
        val_y_prob,
        mode=args.mode,
        num_thresholds=1001
    )
    best_threshold = best_val_row["threshold"]

    # test metrics using tuned threshold
    test_tuned = compute_metrics(test_y_true, test_y_prob, threshold=best_threshold)

    # save search table
    search_path = out_dir / f"val_threshold_search_{args.mode}.csv"
    search_df.to_csv(search_path, index=False)

    # save tuned test predictions
    tuned_test_df = test_df.copy()
    tuned_test_df["threshold"] = best_threshold
    tuned_test_df["y_pred_default"] = (tuned_test_df["y_prob"] >= args.default_threshold).astype(int)
    tuned_test_df["y_pred_tuned"] = (tuned_test_df["y_prob"] >= best_threshold).astype(int)
    tuned_test_path = out_dir / "test_predictions_tuned.csv"
    tuned_test_df.to_csv(tuned_test_path, index=False)

    # save summary json
    summary = {
        "selection_mode": args.mode,
        "default_threshold": args.default_threshold,
        "best_threshold_on_val": best_threshold,
        "val_default": val_default,
        "val_best": best_val_row,
        "test_default": test_default,
        "test_tuned": test_tuned,
    }

    summary_path = out_dir / "threshold_tuning_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== Threshold Tuning Summary ==========")
    print(f"selection_mode: {args.mode}")
    print(f"default_threshold: {args.default_threshold:.4f}")
    print(f"best_threshold_on_val: {best_threshold:.4f}")

    print("\n[Val @ default]")
    for k in ["auc", "acc", "f1", "sensitivity", "specificity", "youden"]:
        print(f"{k}: {val_default[k]:.4f}")

    print("\n[Val @ best threshold]")
    for k in ["auc", "acc", "f1", "sensitivity", "specificity", "youden"]:
        print(f"{k}: {best_val_row[k]:.4f}")

    print("\n[Test @ default]")
    for k in ["auc", "acc", "f1", "sensitivity", "specificity", "youden"]:
        print(f"{k}: {test_default[k]:.4f}")

    print("\n[Test @ tuned threshold]")
    for k in ["auc", "acc", "f1", "sensitivity", "specificity", "youden"]:
        print(f"{k}: {test_tuned[k]:.4f}")

    print("\nSaved files:")
    print(search_path)
    print(tuned_test_path)
    print(summary_path)


if __name__ == "__main__":
    main()