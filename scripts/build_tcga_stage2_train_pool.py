#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd


LOCATOR_COLS = [
    "project",
    "slide_id",
    "svs_path",
    "h5_path",
    "coord_x",
    "coord_y",
    "coord_idx",
    "patch_level",
    "patch_size",
]

REQUIRED_COLS = [
    "svs_path",
    "coord_x",
    "coord_y",
    "patch_level",
    "patch_size",
    "pred_label",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def make_patch_key(df: pd.DataFrame) -> pd.Series:
    svs = df["svs_path"].astype(str).map(canonicalize_path)
    x = df["coord_x"].astype(int).astype(str)
    y = df["coord_y"].astype(int).astype(str)
    level = df["patch_level"].astype(int).astype(str)
    size = df["patch_size"].astype(int).astype(str)
    return svs + "||" + x + "||" + y + "||" + level + "||" + size


def load_background_like_keys(csv_path: Optional[str]) -> Set[str]:
    if csv_path is None:
        return set()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"background-like csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    needed = ["svs_path", "coord_x", "coord_y", "patch_level", "patch_size"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"background-like csv missing columns: {missing}")

    keys = set(make_patch_key(df).tolist())
    print(f"[Info] Loaded {len(keys)} background-like keys from: {csv_path}")
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage2 TCGA training pool CSV from patch_semantic_predictions.csv")

    parser.add_argument("--input-csv", type=str, required=True,
                        help="Merged patch_semantic_predictions.csv")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Output filtered TCGA stage2 train pool CSV")
    parser.add_argument("--background-like-csv", type=str, default=None,
                        help="Optional candidate_background_like.csv for extra filtering")

    parser.add_argument("--drop-background-label", action="store_true",
                        help="Drop rows with pred_label == background")
    parser.add_argument("--drop-prefilter-white", action="store_true",
                        help="Drop rows with prefilter_white == 1 if that column exists")
    parser.add_argument("--drop-background-like", action="store_true",
                        help="Drop rows matched by background-like csv")

    parser.add_argument("--filter-high-entropy", action="store_true",
                        help="Drop rows above entropy upper quantile")
    parser.add_argument("--entropy-quantile", type=float, default=0.995,
                        help="Upper quantile used when --filter-high-entropy is enabled")

    parser.add_argument("--filter-low-margin", action="store_true",
                        help="Drop rows below margin lower quantile")
    parser.add_argument("--margin-quantile", type=float, default=0.005,
                        help="Lower quantile used when --filter-low-margin is enabled")

    parser.add_argument("--min-confidence", type=float, default=None,
                        help="Optional global minimum pred_confidence threshold")

    parser.add_argument("--keep-labels", nargs="+", default=None,
                        help="Optional explicit keep list, e.g. tumor stroma necrosis immune ambiguous normal_epithelium")

    parser.add_argument("--max-rows", type=int, default=None,
                        help="Optional cap after filtering, mainly for debugging")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def validate_input(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input csv missing required columns: {missing}")


def summarize(df: pd.DataFrame, tag: str) -> dict:
    out = {
        "tag": tag,
        "num_rows": int(len(df)),
    }
    if "pred_label" in df.columns:
        out["label_counts"] = df["pred_label"].value_counts().to_dict()
    if "project" in df.columns:
        out["project_counts"] = df["project"].value_counts().to_dict()
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    outdir = os.path.dirname(args.output_csv)
    if outdir:
        ensure_dir(outdir)

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"input csv not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    validate_input(df)

    if "svs_path" in df.columns:
        df["svs_path"] = df["svs_path"].astype(str).map(canonicalize_path)

    print(f"[Info] Loaded input rows: {len(df)}")
    if "pred_label" in df.columns:
        print("[Info] Input label counts:")
        print(df["pred_label"].value_counts())

    summary_steps = []
    summary_steps.append(summarize(df, "loaded"))

    # Optional keep label list
    if args.keep_labels is not None:
        keep_labels = set(args.keep_labels)
        old_n = len(df)
        df = df[df["pred_label"].isin(keep_labels)].copy()
        print(f"[Filter] keep_labels: {old_n} -> {len(df)}")
        summary_steps.append(summarize(df, "after_keep_labels"))

    # Drop background label
    if args.drop_background_label:
        old_n = len(df)
        df = df[df["pred_label"] != "background"].copy()
        print(f"[Filter] drop background label: {old_n} -> {len(df)}")
        summary_steps.append(summarize(df, "after_drop_background_label"))

    # Drop prefilter white
    if args.drop_prefilter_white and "prefilter_white" in df.columns:
        old_n = len(df)
        df = df[df["prefilter_white"].fillna(0).astype(int) == 0].copy()
        print(f"[Filter] drop prefilter_white==1: {old_n} -> {len(df)}")
        summary_steps.append(summarize(df, "after_drop_prefilter_white"))

    # Drop background-like
    if args.drop_background_like:
        bg_keys = load_background_like_keys(args.background_like_csv)
        if len(bg_keys) > 0:
            old_n = len(df)
            keys = make_patch_key(df)
            keep_mask = ~keys.isin(bg_keys)
            df = df[keep_mask].copy()
            print(f"[Filter] drop background-like matches: {old_n} -> {len(df)}")
            summary_steps.append(summarize(df, "after_drop_background_like"))

    # Min confidence
    if args.min_confidence is not None:
        if "pred_confidence" not in df.columns:
            raise ValueError("--min-confidence requires pred_confidence column in input csv")
        old_n = len(df)
        df = df[df["pred_confidence"] >= float(args.min_confidence)].copy()
        print(f"[Filter] min_confidence={args.min_confidence}: {old_n} -> {len(df)}")
        summary_steps.append(summarize(df, "after_min_confidence"))

    # High entropy tail
    if args.filter_high_entropy:
        if "entropy" not in df.columns:
            raise ValueError("--filter-high-entropy requires entropy column")
        thr = float(df["entropy"].quantile(args.entropy_quantile))
        old_n = len(df)
        df = df[df["entropy"] <= thr].copy()
        print(f"[Filter] entropy <= q{args.entropy_quantile:.3f} ({thr:.6f}): {old_n} -> {len(df)}")
        summary_steps.append(summarize(df, "after_filter_high_entropy"))

    # Low margin tail
    if args.filter_low_margin:
        if "margin_top1_top2" not in df.columns:
            raise ValueError("--filter-low-margin requires margin_top1_top2 column")
        thr = float(df["margin_top1_top2"].quantile(args.margin_quantile))
        old_n = len(df)
        df = df[df["margin_top1_top2"] >= thr].copy()
        print(f"[Filter] margin >= q{args.margin_quantile:.3f} ({thr:.6f}): {old_n} -> {len(df)}")
        summary_steps.append(summarize(df, "after_filter_low_margin"))

    # Optional max rows
    if args.max_rows is not None and len(df) > args.max_rows:
        old_n = len(df)
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        print(f"[Filter] max_rows={args.max_rows}: {old_n} -> {len(df)}")
        summary_steps.append(summarize(df, "after_max_rows"))

    # Keep a stable column order: locator + key metadata + rest
    preferred_cols = LOCATOR_COLS + [
        "pred_label",
        "pred_confidence",
        "entropy",
        "margin_top1_top2",
        "prefilter_white",
    ]
    score_cols = [c for c in df.columns if c.startswith("score_")]

    keep_cols = []
    for c in preferred_cols + score_cols:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)
    for c in df.columns:
        if c not in keep_cols:
            keep_cols.append(c)

    df = df[keep_cols].reset_index(drop=True)
    df.to_csv(args.output_csv, index=False)

    final_summary = summarize(df, "final")
    summary = {
        "settings": vars(args),
        "steps": summary_steps,
        "final": final_summary,
    }

    summary_path = os.path.splitext(args.output_csv)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[Done] Saved stage2 pool to:")
    print(args.output_csv)
    print("\n[Final] label counts:")
    if "pred_label" in df.columns:
        print(df["pred_label"].value_counts())
    if "project" in df.columns:
        print("\n[Final] project x label:")
        print(pd.crosstab(df["project"], df["pred_label"]))


if __name__ == "__main__":
    main()