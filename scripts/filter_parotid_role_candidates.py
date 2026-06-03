#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_str_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def validate_prediction_csv(df: pd.DataFrame):
    required = [
        "project",
        "slide_id",
        "pred_label",
        "svs_path",
        "coord_x",
        "coord_y",
        "patch_level",
        "patch_size",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"prediction csv missing required columns: {missing}")


def validate_shortlist_csv(df: pd.DataFrame):
    required = ["role", "project", "slide_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"shortlist csv missing required columns: {missing}")


def filter_prediction_by_shortlist(
    pred_df: pd.DataFrame,
    shortlist_df: pd.DataFrame,
    role_name: str,
) -> pd.DataFrame:
    pred_df = pred_df.copy()
    shortlist_df = shortlist_df.copy()

    for c in ["project", "slide_id", "pred_label"]:
        pred_df = normalize_str_col(pred_df, c)
    for c in ["role", "project", "slide_id"]:
        shortlist_df = normalize_str_col(shortlist_df, c)

    shortlist_df = shortlist_df[shortlist_df["role"] == role_name].copy()
    if len(shortlist_df) == 0:
        return pred_df.iloc[:0].copy()

    keep_keys = shortlist_df[["project", "slide_id"]].drop_duplicates()
    out = pred_df.merge(keep_keys, on=["project", "slide_id"], how="inner")

    out = out[out["pred_label"] == role_name].copy()
    return out.reset_index(drop=True)


def apply_optional_filters(
    df: pd.DataFrame,
    role_name: str,
    min_confidence: float | None,
    max_entropy: float | None,
    min_margin: float | None,
    max_background_score: float | None,
) -> pd.DataFrame:
    df = df.copy()

    if min_confidence is not None and "pred_confidence" in df.columns:
        df = df[df["pred_confidence"] >= min_confidence].copy()

    if max_entropy is not None and "entropy" in df.columns:
        df = df[df["entropy"] <= max_entropy].copy()

    if min_margin is not None and "margin_top1_top2" in df.columns:
        df = df[df["margin_top1_top2"] >= min_margin].copy()

    if max_background_score is not None and "score_background" in df.columns:
        df = df[df["score_background"] <= max_background_score].copy()

    score_col = f"score_{role_name}"
    if score_col in df.columns:
        df = df[df[score_col].notna()].copy()

    return df.reset_index(drop=True)


def select_topk_per_slide(
    df: pd.DataFrame,
    role_name: str,
    topk_per_slide: int,
) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    score_col = f"score_{role_name}"

    sort_cols: List[str] = []
    ascending: List[bool] = []

    if score_col in df.columns:
        sort_cols.append(score_col)
        ascending.append(False)

    if "pred_confidence" in df.columns:
        sort_cols.append("pred_confidence")
        ascending.append(False)

    if "margin_top1_top2" in df.columns:
        sort_cols.append("margin_top1_top2")
        ascending.append(False)

    if "entropy" in df.columns:
        sort_cols.append("entropy")
        ascending.append(True)

    if len(sort_cols) == 0:
        raise ValueError("No ranking columns available.")

    parts = []
    for (project, slide_id), sub in df.groupby(["project", "slide_id"], dropna=False):
        sub = sub.sort_values(sort_cols, ascending=ascending).head(topk_per_slide).copy()
        sub["role_rank_within_slide"] = range(1, len(sub) + 1)
        parts.append(sub)

    return pd.concat(parts, axis=0).reset_index(drop=True)


def save_summary(df: pd.DataFrame, out_csv: str):
    if len(df) == 0:
        pd.DataFrame(columns=["project", "slide_id", "num_selected"]).to_csv(out_csv, index=False)
        return

    summary = (
        df.groupby(["project", "slide_id"], dropna=False)
        .size()
        .reset_index(name="num_selected")
        .sort_values(["project", "slide_id"])
        .reset_index(drop=True)
    )
    summary.to_csv(out_csv, index=False)


def process_one_role(
    pred_df: pd.DataFrame,
    shortlist_csv: str,
    role_name: str,
    out_csv: str,
    out_summary_csv: str,
    topk_per_slide: int,
    min_confidence: float | None,
    max_entropy: float | None,
    min_margin: float | None,
    max_background_score: float | None,
):
    shortlist_df = load_csv(shortlist_csv)
    validate_shortlist_csv(shortlist_df)

    filtered = filter_prediction_by_shortlist(
        pred_df=pred_df,
        shortlist_df=shortlist_df,
        role_name=role_name,
    )
    print(f"[{role_name}] after shortlist + pred_label filter = {len(filtered)}")

    filtered = apply_optional_filters(
        df=filtered,
        role_name=role_name,
        min_confidence=min_confidence,
        max_entropy=max_entropy,
        min_margin=min_margin,
        max_background_score=max_background_score,
    )
    print(f"[{role_name}] after metric filters = {len(filtered)}")

    selected = select_topk_per_slide(
        df=filtered,
        role_name=role_name,
        topk_per_slide=topk_per_slide,
    )
    print(f"[{role_name}] after topk_per_slide = {len(selected)}")

    selected.to_csv(out_csv, index=False)
    save_summary(selected, out_summary_csv)
    print(f"[{role_name}] saved -> {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        "Filter parotid role candidates directly from patch_semantic_predictions.csv"
    )

    parser.add_argument("--prediction_csv", type=str, required=True,
                        help="Merged patch_semantic_predictions.csv")
    parser.add_argument("--tumor_shortlist", type=str, required=True)
    parser.add_argument("--stroma_shortlist", type=str, required=True)
    parser.add_argument("--normal_shortlist", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--tumor_topk", type=int, default=20)
    parser.add_argument("--stroma_topk", type=int, default=20)
    parser.add_argument("--normal_topk", type=int, default=20)

    parser.add_argument("--tumor_min_confidence", type=float, default=None)
    parser.add_argument("--tumor_max_entropy", type=float, default=None)
    parser.add_argument("--tumor_min_margin", type=float, default=None)
    parser.add_argument("--tumor_max_background_score", type=float, default=None)

    parser.add_argument("--stroma_min_confidence", type=float, default=None)
    parser.add_argument("--stroma_max_entropy", type=float, default=None)
    parser.add_argument("--stroma_min_margin", type=float, default=None)
    parser.add_argument("--stroma_max_background_score", type=float, default=None)

    parser.add_argument("--normal_min_confidence", type=float, default=None)
    parser.add_argument("--normal_max_entropy", type=float, default=None)
    parser.add_argument("--normal_min_margin", type=float, default=None)
    parser.add_argument("--normal_max_background_score", type=float, default=None)

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    pred_df = load_csv(args.prediction_csv)
    validate_prediction_csv(pred_df)

    for c in ["project", "slide_id", "pred_label"]:
        pred_df = normalize_str_col(pred_df, c)

    process_one_role(
        pred_df=pred_df,
        shortlist_csv=args.tumor_shortlist,
        role_name="tumor",
        out_csv=os.path.join(args.out_dir, "tumor_proto_candidates.csv"),
        out_summary_csv=os.path.join(args.out_dir, "tumor_proto_candidates_summary.csv"),
        topk_per_slide=args.tumor_topk,
        min_confidence=args.tumor_min_confidence,
        max_entropy=args.tumor_max_entropy,
        min_margin=args.tumor_min_margin,
        max_background_score=args.tumor_max_background_score,
    )

    process_one_role(
        pred_df=pred_df,
        shortlist_csv=args.stroma_shortlist,
        role_name="stroma",
        out_csv=os.path.join(args.out_dir, "stroma_proto_candidates.csv"),
        out_summary_csv=os.path.join(args.out_dir, "stroma_proto_candidates_summary.csv"),
        topk_per_slide=args.stroma_topk,
        min_confidence=args.stroma_min_confidence,
        max_entropy=args.stroma_max_entropy,
        min_margin=args.stroma_min_margin,
        max_background_score=args.stroma_max_background_score,
    )

    process_one_role(
        pred_df=pred_df,
        shortlist_csv=args.normal_shortlist,
        role_name="normal_epithelium",
        out_csv=os.path.join(args.out_dir, "normal_proto_candidates.csv"),
        out_summary_csv=os.path.join(args.out_dir, "normal_proto_candidates_summary.csv"),
        topk_per_slide=args.normal_topk,
        min_confidence=args.normal_min_confidence,
        max_entropy=args.normal_max_entropy,
        min_margin=args.normal_min_margin,
        max_background_score=args.normal_max_background_score,
    )

    print("[Done]")


if __name__ == "__main__":
    main()