#!/usr/bin/env python3
"""
Refine CONCH candidate subsets with a second cleaning stage.

Main goals
----------
1. Clean the overly broad immune subset by removing low-information / edge-like patches.
2. Clean the uncertainty-based ambiguous subset by keeping only genuinely mixed cases
   rather than globally low-score background-like cases.
3. Export refined CSVs for later role prototype construction.
4. Save summary statistics before/after cleaning.

Expected inputs
---------------
At minimum, the input CSVs should come from previous CONCH analysis and contain:
- patch_path
- pred_label
- pred_confidence
- entropy
- margin_top1_top2
- score_* columns for each semantic class

Optional but recommended:
- organ_name
- organ_id

Typical usage
-------------
python refine_conch_candidate_subsets.py \
  --immune-csv /path/to/candidate_core_immune_balanced_by_organ.csv \
  --ambiguous-csv /path/to/candidate_ambiguous_entropy_margin.csv \
  --outdir /path/to/refined_candidates \
  --immune-min-score 0.90 \
  --immune-max-entropy 0.20 \
  --immune-min-margin 0.80 \
  --amb-top2-min 0.20 \
  --amb-max-top1 0.85 \
  --amb-min-entropy 0.25 \
  --amb-max-margin 0.25

Outputs
-------
- immune_clean.csv
- ambiguous_clean.csv
- immune_removed.csv
- ambiguous_removed.csv
- refine_summary.json
- optional balanced-by-organ versions if requested

Cleaning logic
--------------
Immune cleaning:
- keep only high-confidence immune-like rows
- remove rows with overly high entropy or low margin
- optionally require top1=immune if the CSV is not already immune-only

Ambiguous cleaning:
- keep rows with high uncertainty, but avoid globally low-score cases
- require top2 semantic scores to both be non-trivially high
- optionally suppress cases where top1 score is too dominant
- this better captures mixed/boundary cases instead of background-like uncertainty
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_BASE_COLS = {
    "patch_path",
    "pred_label",
    "pred_confidence",
    "entropy",
    "margin_top1_top2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Second-stage cleaning for CONCH candidate subsets")
    parser.add_argument("--immune-csv", type=str, required=True)
    parser.add_argument("--ambiguous-csv", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)

    parser.add_argument("--immune-require-pred-label", action="store_true", help="Require pred_label == immune for immune_clean")
    parser.add_argument("--immune-min-score", type=float, default=0.90, help="Minimum score_immune")
    parser.add_argument("--immune-min-confidence", type=float, default=0.90, help="Minimum pred_confidence")
    parser.add_argument("--immune-max-entropy", type=float, default=0.20, help="Maximum entropy")
    parser.add_argument("--immune-min-margin", type=float, default=0.80, help="Minimum top1-top2 margin")
    parser.add_argument("--immune-top2-max", type=float, default=0.10, help="Maximum second-best score to avoid broad/edge-like immune calls")

    parser.add_argument("--amb-min-entropy", type=float, default=0.25, help="Minimum entropy for ambiguous_clean")
    parser.add_argument("--amb-max-margin", type=float, default=0.25, help="Maximum top1-top2 margin")
    parser.add_argument("--amb-top2-min", type=float, default=0.20, help="Minimum second-best score; both top classes should have non-trivial support")
    parser.add_argument("--amb-top1-min", type=float, default=0.25, help="Minimum top1 score; avoids globally low-response cases")
    parser.add_argument("--amb-max-top1", type=float, default=0.85, help="Maximum top1 score; avoids nearly pure confident cases")
    parser.add_argument("--amb-min-score-sum-top2", type=float, default=0.60, help="Minimum sum of top2 scores")

    parser.add_argument("--per-organ-balance", action="store_true")
    parser.add_argument("--max-per-organ-immune", type=int, default=3000)
    parser.add_argument("--max-per-organ-ambiguous", type=int, default=3000)

    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_BASE_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def get_score_cols(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("score_")])


def add_topk_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    score_cols = get_score_cols(df)
    if not score_cols:
        raise ValueError("No score_* columns found; cannot refine subsets.")

    score_mat = df[score_cols].to_numpy(dtype=np.float32)
    order = np.argsort(score_mat, axis=1)
    top1_idx = order[:, -1]
    top2_idx = order[:, -2] if score_mat.shape[1] >= 2 else order[:, -1]

    top1_score = score_mat[np.arange(len(df)), top1_idx]
    top2_score = score_mat[np.arange(len(df)), top2_idx]

    top1_name = [score_cols[i].replace("score_", "") for i in top1_idx]
    top2_name = [score_cols[i].replace("score_", "") for i in top2_idx]

    out = df.copy()
    out["top1_score_from_scores"] = top1_score
    out["top2_score"] = top2_score
    out["top1_name_from_scores"] = top1_name
    out["top2_name"] = top2_name
    out["top2_sum"] = top1_score + top2_score
    out["top1_minus_top2_from_scores"] = top1_score - top2_score
    return out


def summarize_df(df: pd.DataFrame, name: str) -> Dict[str, object]:
    out = {
        "name": name,
        "count": int(len(df)),
    }
    if len(df) == 0:
        return out
    out.update({
        "pred_label_counts": df["pred_label"].value_counts().to_dict(),
        "pred_confidence_mean": float(df["pred_confidence"].mean()),
        "entropy_mean": float(df["entropy"].mean()),
        "margin_mean": float(df["margin_top1_top2"].mean()),
    })
    if "organ_name" in df.columns:
        out["organ_counts"] = df["organ_name"].value_counts().to_dict()
    if "top2_name" in df.columns:
        pair = (df["top1_name_from_scores"] + "__" + df["top2_name"]).value_counts().head(20).to_dict()
        out["top1_top2_pairs_top20"] = pair
    return out


def balance_by_organ(df: pd.DataFrame, max_per_organ: int) -> pd.DataFrame:
    if len(df) == 0 or "organ_name" not in df.columns:
        return df.copy()
    parts = []
    for organ, sub in df.groupby("organ_name"):
        sub = sub.sort_values(["pred_confidence", "margin_top1_top2"], ascending=[False, False])
        parts.append(sub.head(min(max_per_organ, len(sub))))
    if not parts:
        return df.iloc[:0].copy()
    return pd.concat(parts, axis=0).reset_index(drop=True)


def refine_immune(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = add_topk_score_columns(df)

    mask = pd.Series(True, index=df.index)
    if args.immune_require_pred_label:
        mask &= (df["pred_label"] == "immune")

    if "score_immune" in df.columns:
        mask &= (df["score_immune"] >= args.immune_min_score)

    mask &= (df["pred_confidence"] >= args.immune_min_confidence)
    mask &= (df["entropy"] <= args.immune_max_entropy)
    mask &= (df["margin_top1_top2"] >= args.immune_min_margin)
    mask &= (df["top2_score"] <= args.immune_top2_max)

    keep = df[mask].copy()
    removed = df[~mask].copy()

    keep = keep.sort_values(
        ["score_immune" if "score_immune" in keep.columns else "pred_confidence", "margin_top1_top2", "entropy"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    keep["clean_rank"] = np.arange(1, len(keep) + 1)

    return keep, removed


def refine_ambiguous(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = add_topk_score_columns(df)

    mask = pd.Series(True, index=df.index)
    mask &= (df["entropy"] >= args.amb_min_entropy)
    mask &= (df["margin_top1_top2"] <= args.amb_max_margin)
    mask &= (df["top2_score"] >= args.amb_top2_min)
    mask &= (df["top1_score_from_scores"] >= args.amb_top1_min)
    mask &= (df["top1_score_from_scores"] <= args.amb_max_top1)
    mask &= (df["top2_sum"] >= args.amb_min_score_sum_top2)

    keep = df[mask].copy()
    removed = df[~mask].copy()

    keep = keep.sort_values(
        ["entropy", "top2_score", "top2_sum", "margin_top1_top2"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    keep["clean_rank"] = np.arange(1, len(keep) + 1)

    return keep, removed


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    immune_df = load_df(args.immune_csv)
    ambiguous_df = load_df(args.ambiguous_csv)

    immune_clean, immune_removed = refine_immune(immune_df, args)
    ambiguous_clean, ambiguous_removed = refine_ambiguous(ambiguous_df, args)

    immune_clean.to_csv(os.path.join(args.outdir, "immune_clean.csv"), index=False)
    immune_removed.to_csv(os.path.join(args.outdir, "immune_removed.csv"), index=False)
    ambiguous_clean.to_csv(os.path.join(args.outdir, "ambiguous_clean.csv"), index=False)
    ambiguous_removed.to_csv(os.path.join(args.outdir, "ambiguous_removed.csv"), index=False)

    if args.per_organ_balance:
        immune_bal = balance_by_organ(immune_clean, args.max_per_organ_immune)
        amb_bal = balance_by_organ(ambiguous_clean, args.max_per_organ_ambiguous)
        immune_bal.to_csv(os.path.join(args.outdir, "immune_clean_balanced_by_organ.csv"), index=False)
        amb_bal.to_csv(os.path.join(args.outdir, "ambiguous_clean_balanced_by_organ.csv"), index=False)
    else:
        immune_bal = None
        amb_bal = None

    summary = {
        "settings": vars(args),
        "immune_before": summarize_df(immune_df, "immune_before"),
        "immune_clean": summarize_df(immune_clean, "immune_clean"),
        "immune_removed": summarize_df(immune_removed, "immune_removed"),
        "ambiguous_before": summarize_df(ambiguous_df, "ambiguous_before"),
        "ambiguous_clean": summarize_df(ambiguous_clean, "ambiguous_clean"),
        "ambiguous_removed": summarize_df(ambiguous_removed, "ambiguous_removed"),
    }
    if immune_bal is not None:
        summary["immune_clean_balanced_by_organ"] = summarize_df(immune_bal, "immune_clean_balanced_by_organ")
    if amb_bal is not None:
        summary["ambiguous_clean_balanced_by_organ"] = summarize_df(amb_bal, "ambiguous_clean_balanced_by_organ")

    with open(os.path.join(args.outdir, "refine_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Saved refined subsets to: {args.outdir}")
    print(f"immune_clean: {len(immune_clean)}")
    print(f"ambiguous_clean: {len(ambiguous_clean)}")
    if immune_bal is not None:
        print(f"immune_clean_balanced_by_organ: {len(immune_bal)}")
    if amb_bal is not None:
        print(f"ambiguous_clean_balanced_by_organ: {len(amb_bal)}")


if __name__ == "__main__":
    main()
