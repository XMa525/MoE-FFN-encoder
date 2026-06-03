#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd


ROLE_RULES = {
    "atypical_epithelial_lesion": {
        "source_labels": ["atypical_epithelial_lesion"],
        "score_col": "score_atypical_epithelial_lesion",
        "max_per_slide": 120,
        "max_total": 12000,
    },
    "benign_or_normal_epithelium": {
        "source_labels": ["benign_proliferative_epithelium"],
        "score_col": "score_benign_proliferative_epithelium",
        "max_per_slide": 120,
        "max_total": 12000,
    },
    "fibro_adipose_stroma": {
        "source_labels": ["fibrocollagenous_stroma"],
        "score_col": "score_fibrocollagenous_stroma",
        "max_per_slide": 120,
        "max_total": 12000,
    },
}


def select_role(
    df,
    role_name,
    source_labels,
    score_col,
    max_per_slide,
    max_total,
    min_conf,
    min_margin,
    max_entropy,
):
    sub = df[df["pred_label"].isin(source_labels)].copy()

    if score_col in sub.columns:
        sub["role_score"] = sub[score_col].astype(float)
    else:
        sub["role_score"] = sub["pred_confidence"].astype(float)

    # 基础质量过滤
    sub = sub[
        (sub["pred_confidence"] >= min_conf)
        & (sub["margin_top1_top2"] >= min_margin)
        & (sub["entropy"] <= max_entropy)
    ].copy()

    # 排除明显 background / ambiguous / artifact 竞争分数高的 patch
    for bad_col in [
        "score_background_artifact",
        "score_ambiguous_mixed",
        "score_blood_debris_artifact",
    ]:
        if bad_col in sub.columns:
            sub = sub[sub[bad_col] <= 0.20].copy()

    # 如果有 near-white 预过滤标记，排除
    if "prefilter_white" in sub.columns:
        sub = sub[sub["prefilter_white"].fillna(0).astype(int) == 0].copy()

    # 排序：更像该 role、更确定、更大 margin
    sub = sub.sort_values(
        ["role_score", "pred_confidence", "margin_top1_top2", "entropy"],
        ascending=[False, False, False, True],
    )

    # slide-level balance
    parts = []
    if "slide_id" in sub.columns and max_per_slide > 0:
        for slide_id, g in sub.groupby("slide_id", sort=False):
            parts.append(g.head(max_per_slide))
        sub = pd.concat(parts, axis=0) if parts else sub.iloc[:0].copy()

    sub = sub.sort_values(
        ["role_score", "pred_confidence", "margin_top1_top2", "entropy"],
        ascending=[False, False, False, True],
    ).head(max_total)

    sub["core_label"] = role_name
    sub["core_rank"] = np.arange(1, len(sub) + 1)
    return sub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--min-conf", type=float, default=0.97)
    parser.add_argument("--min-margin", type=float, default=0.30)
    parser.add_argument("--max-entropy", type=float, default=0.20)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)

    all_parts = []
    counts = {}

    for role_name, cfg in ROLE_RULES.items():
        role_df = select_role(
            df=df,
            role_name=role_name,
            source_labels=cfg["source_labels"],
            score_col=cfg["score_col"],
            max_per_slide=cfg["max_per_slide"],
            max_total=cfg["max_total"],
            min_conf=args.min_conf,
            min_margin=args.min_margin,
            max_entropy=args.max_entropy,
        )

        out_path = os.path.join(args.outdir, f"candidate_core_{role_name}.csv")
        role_df.to_csv(out_path, index=False)

        counts[role_name] = len(role_df)
        all_parts.append(role_df)

    merged = pd.concat(all_parts, axis=0, ignore_index=True)
    merged.to_csv(os.path.join(args.outdir, "candidate_core_3role_merged.csv"), index=False)

    pd.Series(counts).to_csv(os.path.join(args.outdir, "candidate_counts_3role.csv"))

    print("Candidate counts:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print("Saved to:", args.outdir)


if __name__ == "__main__":
    main()