from __future__ import annotations

import os
import argparse
from typing import List
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import pandas as pd
import numpy as np


def pick_existing(cols: List[str], candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assignment-csv",
        type=str,
        required=True,
        help="gland_like_subclass_assignments.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="how many top patches to export per subclass",
    )
    parser.add_argument(
        "--score-col",
        type=str,
        default=None,
        help="Optional explicit score column. If None, auto infer.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.assignment_csv)

    if "subclass_id" not in df.columns:
        raise ValueError("assignment-csv must contain subclass_id")

    cols = set(df.columns)

    score_col = args.score_col
    if score_col is None:
        for c in [
            "delta_tumor_minus_max_other",
            "score",
            "score_tumor_minus_other_max",
            "topk_score",
        ]:
            if c in cols:
                score_col = c
                break

    sim_tumor_col = "sim_tumor" if "sim_tumor" in cols else None
    sim_other_col = None
    for c in ["sim_other_max", "other_max_sim", "sim_max_other"]:
        if c in cols:
            sim_other_col = c
            break

    group_rows = []
    for sid, g in df.groupby("subclass_id"):
        row = {
            "subclass_id": int(sid),
            "count": int(len(g)),
        }

        if score_col is not None:
            row["mean_score"] = float(g[score_col].mean())
            row["std_score"] = float(g[score_col].std(ddof=0))
            row["frac_score_gt_0"] = float((g[score_col] > 0).mean())

        if sim_tumor_col is not None:
            row["mean_sim_tumor"] = float(g[sim_tumor_col].mean())
        if sim_other_col is not None:
            row["mean_sim_other_max"] = float(g[sim_other_col].mean())

        if "slide_label" in g.columns:
            for y in sorted(g["slide_label"].dropna().unique().tolist()):
                mask = g["slide_label"] == y
                row[f"frac_slide_label_{int(y)}"] = float(mask.mean())

        if "nearest_role" in g.columns:
            vc = g["nearest_role"].value_counts(normalize=True)
            for k, v in vc.items():
                row[f"frac_nearest_{k}"] = float(v)

        group_rows.append(row)

    summary_df = pd.DataFrame(group_rows).sort_values("subclass_id")
    summary_path = os.path.join(args.out_dir, "gland_like_subclass_stats.csv")
    summary_df.to_csv(summary_path, index=False)

    print("[Summary]")
    print(summary_df)

    # 导出每个 subclass 的 top patches
    sort_col = score_col if score_col is not None else None
    keep_cols = pick_existing(
        list(df.columns),
        [
            "row_idx",
            "subclass_id",
            "slide_id",
            "slide_label",
            "svs_path",
            "h5_path",
            "coord_x",
            "coord_y",
            "coord_idx",
            "patch_level",
            "patch_size",
            "pred_label",
            "pred_confidence",
            "entropy",
            "margin_top1_top2",
            "score",
            "sim_tumor",
            "sim_other_max",
            "delta_tumor_minus_max_other",
        ],
    )

    for sid, g in df.groupby("subclass_id"):
        out_csv = os.path.join(args.out_dir, f"gland_like_sub{sid}_top_patches.csv")
        if sort_col is not None and sort_col in g.columns:
            gg = g.sort_values(sort_col, ascending=False).head(args.topk)
        else:
            gg = g.head(args.topk)
        gg[keep_cols].to_csv(out_csv, index=False)
        print(f"[OK] saved {out_csv}")

    print(f"[OK] summary saved to {summary_path}")


if __name__ == "__main__":
    main()