#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLS = {
    "pred_label",
    "pred_confidence",
    "entropy",
    "margin_top1_top2",
}

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
# CORE_LABELS = [
#     "tumor",
#     "stroma",
#     "immune",
#     "necrosis",
#     "normal_epithelium",
# ]
# CORE_LABELS = [
#     "tumor",
#     "fibrovascular_stroma",
#     "normal_kidney_parenchyma",
#     "vascular_hemorrhage",
#     "background_artifact",
#     "ambiguous_mixed"
# ]
# CORE_LABELS = [
#     "tumor_metastasis",
#     "adipose_stroma",
#     "lymphoid_tissue",
#     "background_artifact",
#     "ambiguous_mixed"
# ]
CORE_LABELS = [
    "atypical_epithelial_lesion",
    "fibrocollagenous_stroma",
    "normal_breast_epithelium",
    "benign_proliferative_epithelium",
    "background_artifact",
    "adipose_tissue",
    "ambiguous_mixed"
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TCGA CONCH semantic predictions (svs+h5 coords)")
    parser.add_argument("--csv", type=str, required=True, help="Path to patch_semantic_predictions.csv")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save figures/tables")
    parser.add_argument("--top-frac", type=float, default=0.10)
    parser.add_argument("--high-entropy-frac", type=float, default=0.10)
    parser.add_argument("--low-margin-frac", type=float, default=0.10)
    parser.add_argument("--min-class-core", type=int, default=500)
    parser.add_argument("--max-class-core", type=int, default=50000)
    parser.add_argument("--per-organ-balance", action="store_true")
    parser.add_argument("--max-per-organ-per-label", type=int, default=5000)
    parser.add_argument("--organ-col", type=str, default="project")
    parser.add_argument(
        "--max-per-slide-per-label",
        type=int,
        default=0,
        help="Limit maximum core candidates per slide for each label. 0 means no slide-level cap."
    )
    parser.add_argument(
        "--slide-col",
        type=str,
        default="slide_id",
        help="Column name for slide-level balancing. Default: slide_id"
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_csv(path: str, organ_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if "organ_name" not in df.columns:
        if organ_col in df.columns:
            df["organ_name"] = df[organ_col].astype(str)
        elif "project" in df.columns:
            df["organ_name"] = df["project"].astype(str)
        else:
            df["organ_name"] = "all"

    return df


def export_candidate_csv(df: pd.DataFrame, out_path: str) -> None:
    keep_cols = []

    preferred = LOCATOR_COLS + [
        "organ_name",
        "pred_label",
        "pred_confidence",
        "entropy",
        "margin_top1_top2",
        "prefilter_white",
        "core_rank",
        "core_label",
        "ambiguity_rank",
    ]
    score_cols = [c for c in df.columns if c.startswith("score_")]

    for c in preferred + score_cols:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    # 其余列也保留，避免丢信息
    for c in df.columns:
        if c not in keep_cols:
            keep_cols.append(c)

    df[keep_cols].to_csv(out_path, index=False)


def plot_bar_counts(df: pd.DataFrame, outdir: str) -> None:
    counts = df["pred_label"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")
    plt.ylabel("Count")
    plt.title("Overall predicted label counts")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "overall_label_counts.png"), dpi=200)
    plt.close()


def plot_organ_label_tables(df: pd.DataFrame, outdir: str) -> None:
    ct = pd.crosstab(df["organ_name"], df["pred_label"])
    ct_frac = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    ct.to_csv(os.path.join(outdir, "summary_by_organ_label.csv"))
    ct_frac.to_csv(os.path.join(outdir, "summary_by_organ_label_fraction.csv"))

    plt.figure(figsize=(10, 6))
    ct.plot(kind="bar", stacked=True)
    plt.ylabel("Count")
    plt.title("Organ by predicted label counts")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "organ_by_label_counts.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    ct_frac.plot(kind="bar", stacked=True)
    plt.ylabel("Fraction")
    plt.title("Organ by predicted label fraction")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "organ_by_label_fraction.png"), dpi=200)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(ct.values, aspect="auto")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right")
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels(ct.index)
    ax.set_title("Organ × label heatmap (counts)")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "organ_label_heatmap_counts.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = max(1e-6, float(ct_frac.values.max()))
    im = ax.imshow(ct_frac.values, aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(ct_frac.shape[1]))
    ax.set_xticklabels(ct_frac.columns, rotation=45, ha="right")
    ax.set_yticks(range(ct_frac.shape[0]))
    ax.set_yticklabels(ct_frac.index)
    ax.set_title("Organ × label heatmap (row fraction)")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "organ_label_heatmap_fraction.png"), dpi=200)
    plt.close(fig)


def boxplot_metric_by_label(df: pd.DataFrame, metric: str, outdir: str) -> None:
    labels = sorted(df["pred_label"].unique())
    data = [df.loc[df["pred_label"] == label, metric].dropna().values for label in labels]
    plt.figure(figsize=(9, 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel(metric)
    plt.title(f"{metric} by predicted label")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{metric}_by_label.png"), dpi=200)
    plt.close()


def histogram_score_cols(df: pd.DataFrame, outdir: str) -> None:
    score_cols = [c for c in df.columns if c.startswith("score_")]
    for col in score_cols:
        plt.figure(figsize=(8, 5))
        plt.hist(df[col].dropna().values, bins=80)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{col}_hist.png"), dpi=200)
        plt.close()


def summarize_by_label(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = sorted(df["pred_label"].unique())
    for label in labels:
        sub = df[df["pred_label"] == label]
        row = {
            "label": label,
            "count": int(len(sub)),
            "fraction": float(len(sub) / max(len(df), 1)),
            "pred_confidence_mean": float(sub["pred_confidence"].mean()),
            "pred_confidence_median": float(sub["pred_confidence"].median()),
            "entropy_mean": float(sub["entropy"].mean()),
            "entropy_median": float(sub["entropy"].median()),
            "margin_mean": float(sub["margin_top1_top2"].mean()),
            "margin_median": float(sub["margin_top1_top2"].median()),
        }
        for c in [x for x in df.columns if x.startswith("score_")]:
            row[f"mean_{c}"] = float(sub[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def select_core_subset_for_label(
    df: pd.DataFrame,
    label: str,
    top_frac: float,
    min_class_core: int,
    max_class_core: int,
) -> pd.DataFrame:
    sub = df[df["pred_label"] == label].copy()
    if sub.empty:
        return sub

    score_col = f"score_{label}"
    if score_col in sub.columns:
        sort_cols = [score_col, "pred_confidence", "margin_top1_top2"]
        sort_ascending = [False, False, False]
    else:
        sort_cols = ["pred_confidence", "margin_top1_top2"]
        sort_ascending = [False, False]

    sub = sub.sort_values(sort_cols, ascending=sort_ascending)
    k = int(math.ceil(len(sub) * top_frac))
    k = max(k, min_class_core)
    k = min(k, max_class_core, len(sub))
    core = sub.head(k).copy()
    core["core_rank"] = np.arange(1, len(core) + 1)
    core["core_label"] = label
    return core


def select_balanced_core_subset_by_organ(core_df: pd.DataFrame, max_per_organ: int) -> pd.DataFrame:
    if core_df.empty or "organ_name" not in core_df.columns:
        return core_df
    parts = []
    for organ, sub in core_df.groupby("organ_name"):
        parts.append(sub.head(min(max_per_organ, len(sub))))
    if not parts:
        return core_df.iloc[:0].copy()
    return pd.concat(parts, axis=0).reset_index(drop=True)

def select_balanced_core_subset_by_slide(
    core_df: pd.DataFrame,
    slide_col: str,
    max_per_slide: int,
) -> pd.DataFrame:
    if core_df.empty or max_per_slide <= 0:
        return core_df

    if slide_col not in core_df.columns:
        print(f"[WARN] slide_col={slide_col} not found. Skip slide-level balancing.")
        return core_df

    parts = []
    for slide_id, sub in core_df.groupby(slide_col, sort=False):
        parts.append(sub.head(min(max_per_slide, len(sub))))

    if not parts:
        return core_df.iloc[:0].copy()

    out = pd.concat(parts, axis=0).reset_index(drop=True)
    out["slide_balanced_rank"] = np.arange(1, len(out) + 1)
    return out

def select_ambiguous_subset(df: pd.DataFrame, high_entropy_frac: float, low_margin_frac: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    n = len(df)
    k_entropy = max(1, int(math.ceil(n * high_entropy_frac)))
    k_margin = max(1, int(math.ceil(n * low_margin_frac)))

    high_entropy_idx = set(df.nlargest(k_entropy, "entropy").index.tolist())
    low_margin_idx = set(df.nsmallest(k_margin, "margin_top1_top2").index.tolist())
    chosen = sorted(high_entropy_idx.intersection(low_margin_idx))

    amb = df.loc[chosen].copy()
    amb = amb.sort_values(["entropy", "margin_top1_top2"], ascending=[False, True])
    amb["ambiguity_rank"] = np.arange(1, len(amb) + 1)
    return amb


def select_background_like_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    conf_thr = df["pred_confidence"].quantile(0.10)
    ent_thr = df["entropy"].quantile(0.90)
    margin_thr = df["margin_top1_top2"].quantile(0.10)
    bg = df[
        (df["pred_confidence"] <= conf_thr)
        & (df["entropy"] >= ent_thr)
        & (df["margin_top1_top2"] <= margin_thr)
    ].copy()
    bg = bg.sort_values(["pred_confidence", "entropy", "margin_top1_top2"], ascending=[True, False, True])
    return bg


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    df = load_csv(args.csv, args.organ_col)

    overall = {
        "num_rows": int(len(df)),
        "labels": sorted(df["pred_label"].unique().tolist()),
        "organs": sorted(df["organ_name"].astype(str).unique().tolist()),
        "mean_pred_confidence": float(df["pred_confidence"].mean()),
        "mean_entropy": float(df["entropy"].mean()),
        "mean_margin_top1_top2": float(df["margin_top1_top2"].mean()),
    }
    save_json(overall, os.path.join(args.outdir, "summary_overall.json"))

    summary_by_label = summarize_by_label(df)
    summary_by_label.to_csv(os.path.join(args.outdir, "summary_by_label.csv"), index=False)

    plot_bar_counts(df, args.outdir)
    plot_organ_label_tables(df, args.outdir)
    boxplot_metric_by_label(df, "pred_confidence", args.outdir)
    boxplot_metric_by_label(df, "entropy", args.outdir)
    boxplot_metric_by_label(df, "margin_top1_top2", args.outdir)
    histogram_score_cols(df, args.outdir)

    candidate_counts = {}
    present_labels = set(df["pred_label"].unique().tolist())

    for label in CORE_LABELS:
        if label not in present_labels:
            print(f"[Skip core label not present] {label}")
            candidate_counts[f"core_{label}"] = 0
            if args.per_organ_balance:
                candidate_counts[f"core_{label}_balanced_by_organ"] = 0
            continue

        core = select_core_subset_for_label(
            df=df,
            label=label,
            top_frac=args.top_frac,
            min_class_core=args.min_class_core,
            max_class_core=args.max_class_core,
        )

        export_candidate_csv(core, os.path.join(args.outdir, f"candidate_core_{label}.csv"))
        candidate_counts[f"core_{label}"] = int(len(core))

        if args.max_per_slide_per_label > 0:
            slide_balanced = select_balanced_core_subset_by_slide(
                core_df=core,
                slide_col=args.slide_col,
                max_per_slide=args.max_per_slide_per_label,
            )
            export_candidate_csv(
                slide_balanced,
                os.path.join(args.outdir, f"candidate_core_{label}_balanced_by_slide.csv")
            )
            candidate_counts[f"core_{label}_balanced_by_slide"] = int(len(slide_balanced))

        if args.per_organ_balance:
            balanced = select_balanced_core_subset_by_organ(core, args.max_per_organ_per_label)
            export_candidate_csv(
                balanced,
                os.path.join(args.outdir, f"candidate_core_{label}_balanced_by_organ.csv")
            )
            candidate_counts[f"core_{label}_balanced_by_organ"] = int(len(balanced))

    ambiguous = select_ambiguous_subset(df, args.high_entropy_frac, args.low_margin_frac)
    export_candidate_csv(ambiguous, os.path.join(args.outdir, "candidate_ambiguous.csv"))
    candidate_counts["ambiguous_clean"] = int(len(ambiguous))

    background_like = select_background_like_subset(df)
    export_candidate_csv(background_like, os.path.join(args.outdir, "candidate_background_like.csv"))
    candidate_counts["background_like"] = int(len(background_like))

    save_json(candidate_counts, os.path.join(args.outdir, "candidate_counts.json"))

    print("Done.")
    print(f"Saved analysis to: {args.outdir}")
    print("Candidate counts:")
    for k, v in candidate_counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()