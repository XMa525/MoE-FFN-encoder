#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import json
import pandas as pd


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default=None)
    args = parser.parse_args()

    rows = []

    exp_dirs = sorted(glob.glob(os.path.join(args.sweep_root, "*")))
    for exp_dir in exp_dirs:
        hist_path = os.path.join(exp_dir, "train_history.csv")
        final_json = os.path.join(exp_dir, "final_test_metrics.json")

        if not os.path.exists(hist_path):
            continue

        df = pd.read_csv(hist_path)
        if len(df) == 0:
            continue

        if "val_auc" in df.columns:
            best_idx = df["val_auc"].idxmax()
        else:
            best_idx = df["val_f1"].idxmax()

        best_row = df.loc[best_idx].to_dict()
        best_row["exp_dir"] = exp_dir
        best_row["exp_name"] = os.path.basename(exp_dir)

        if os.path.exists(final_json):
            with open(final_json, "r") as f:
                meta = json.load(f)
            test_metrics = meta.get("test_metrics", {})
            best_row["test_auc"] = test_metrics.get("auc", None)
            best_row["test_acc"] = test_metrics.get("acc", None)
            best_row["test_f1"] = test_metrics.get("f1", None)
            best_row["test_sens"] = test_metrics.get("sensitivity", None)
            best_row["test_spec"] = test_metrics.get("specificity", None)
            best_row["best_epoch"] = meta.get("best_epoch", None)
            best_row["best_val_score"] = meta.get("best_val_score", None)

            arg_dict = meta.get("args", {})
            best_row["att_dim"] = arg_dict.get("att_dim", None)
            best_row["lr"] = arg_dict.get("lr", None)
            best_row["weight_decay"] = arg_dict.get("weight_decay", None)
            best_row["max_instances"] = arg_dict.get("max_instances", None)
            best_row["mil_model"] = arg_dict.get("mil_model", None)

        rows.append(best_row)

    out_df = pd.DataFrame(rows)
    if len(out_df) == 0:
        print("No valid experiments found.")
        return

    sort_col = "val_auc" if "val_auc" in out_df.columns else "val_f1"
    out_df = out_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    print(out_df[
        [
            c for c in [
                "exp_name", "att_dim", "lr", "weight_decay", "max_instances",
                "epoch", "val_auc", "val_f1", "val_acc", "val_loss",
                "test_auc", "test_f1", "test_acc", "test_sens", "test_spec"
            ] if c in out_df.columns
        ]
    ].head(20))

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = os.path.join(args.sweep_root, "sweep_summary.csv")
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved summary to: {out_csv}")


if __name__ == "__main__":
    main()