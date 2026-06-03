#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze one expert across all clusters.

Functions:
1. Count how many tokens of a target expert fall into each cluster
2. Export expert-cluster summary csv
3. Select top-K clusters by token count
4. Automatically call visualize_cluster_tokens_from_csv.py for each cluster

Example:
python visualization/analyze_expert_clusters.py \
    --csv analysis_outputs/token_meta.csv \
    --cluster_col cluster_id_moe \
    --target_expert 0 \
    --outdir analysis_outputs/expert0_summary \
    --topk_clusters 5 \
    --min_tokens 30 \
    --grid_size 16 \
    --topk_patches 16
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def summarize_expert_clusters(df: pd.DataFrame, cluster_col: str, target_expert: int):
    """
    Count target expert's tokens in each cluster.
    """
    df_exp = df[df["expert_id"] == target_expert].copy()

    summary = (
        df_exp.groupby(cluster_col)
        .agg(
            token_count=("token_idx", "count"),
            unique_patch_count=("patch_path", "nunique"),
            unique_sample_count=("sample_category", "nunique"),
        )
        .reset_index()
        .sort_values("token_count", ascending=False)
    )

    return df_exp, summary


def plot_cluster_distribution(summary: pd.DataFrame, cluster_col: str, save_path: str, topn: int = 20):
    sub = summary.head(topn).copy()
    if len(sub) == 0:
        return

    plt.figure(figsize=(10, 5))
    plt.bar(sub[cluster_col].astype(str), sub["token_count"])
    plt.xlabel("Cluster ID")
    plt.ylabel("Token count")
    plt.title(f"Top-{topn} clusters for target expert")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def export_patch_level_table(df_exp: pd.DataFrame, cluster_col: str, save_path: str):
    """
    Export patch-level cluster usage for target expert.
    """
    patch_table = (
        df_exp.groupby([cluster_col, "patch_path", "patch_name", "sample_category"])
        .agg(
            token_count=("token_idx", "count"),
            unique_token_count=("token_idx", "nunique"),
        )
        .reset_index()
        .sort_values([cluster_col, "token_count"], ascending=[True, False])
    )
    patch_table.to_csv(save_path, index=False, encoding="utf-8-sig")
    return patch_table


def run_visualizer_for_cluster(
    visualizer_script: str,
    csv_path: str,
    cluster_col: str,
    cluster_id: int,
    target_expert: int,
    outdir: str,
    grid_size: int,
    topk_patches: int,
):
    cmd = [
        sys.executable,
        visualizer_script,
        "--csv", csv_path,
        "--cluster_col", cluster_col,
        "--target_cluster", str(cluster_id),
        "--target_expert", str(target_expert),
        "--outdir", outdir,
        "--grid_size", str(grid_size),
        "--topk_patches", str(topk_patches),
    ]

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to token_meta.csv")
    parser.add_argument("--cluster_col", type=str, required=True,
                        choices=["cluster_id_moe", "cluster_id_final"])
    parser.add_argument("--target_expert", type=int, required=True, help="Expert ID to analyze")
    parser.add_argument("--outdir", type=str, required=True, help="Output root directory")
    parser.add_argument("--topk_clusters", type=int, default=5, help="How many top clusters to visualize")
    parser.add_argument("--min_tokens", type=int, default=30, help="Minimum token count to keep a cluster")
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--topk_patches", type=int, default=16)
    parser.add_argument(
        "--visualizer_script",
        type=str,
        default="visualization/visualize_cluster_tokens_from_csv.py",
        help="Path to visualize_cluster_tokens_from_csv.py"
    )
    args = parser.parse_args()

    ensure_dir(args.outdir)

    df = pd.read_csv(args.csv)

    required_cols = [
        "patch_path", "patch_name", "sample_category",
        "token_idx", "expert_id", args.cluster_col
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # basic type cast
    df["token_idx"] = df["token_idx"].astype(int)
    df["expert_id"] = df["expert_id"].astype(int)
    df[args.cluster_col] = df[args.cluster_col].astype(int)

    # keep only target expert
    df_exp, summary = summarize_expert_clusters(
        df=df,
        cluster_col=args.cluster_col,
        target_expert=args.target_expert,
    )

    # export summary
    summary_csv = os.path.join(args.outdir, f"expert{args.target_expert}_cluster_summary.csv")
    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # export patch-level table
    patch_csv = os.path.join(args.outdir, f"expert{args.target_expert}_patch_cluster_table.csv")
    export_patch_level_table(df_exp, cluster_col=args.cluster_col, save_path=patch_csv)

    # plot cluster distribution
    plot_cluster_distribution(
        summary=summary,
        cluster_col=args.cluster_col,
        save_path=os.path.join(args.outdir, f"expert{args.target_expert}_cluster_distribution.png"),
        topn=min(20, len(summary)),
    )

    # text summary
    txt_path = os.path.join(args.outdir, f"expert{args.target_expert}_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"csv = {args.csv}\n")
        f.write(f"cluster_col = {args.cluster_col}\n")
        f.write(f"target_expert = {args.target_expert}\n")
        f.write(f"total_tokens_for_expert = {len(df_exp)}\n")
        f.write(f"num_clusters_for_expert = {len(summary)}\n\n")
        f.write("[Top clusters]\n")
        for _, row in summary.head(20).iterrows():
            f.write(
                f"{args.cluster_col}={int(row[args.cluster_col])}, "
                f"token_count={int(row['token_count'])}, "
                f"unique_patch_count={int(row['unique_patch_count'])}, "
                f"unique_sample_count={int(row['unique_sample_count'])}\n"
            )

    print(f"[Saved] summary csv -> {summary_csv}")
    print(f"[Saved] patch table  -> {patch_csv}")
    print(f"[Saved] txt summary  -> {txt_path}")

    # filter top clusters
    selected = summary[summary["token_count"] >= args.min_tokens].copy()
    selected = selected.head(args.topk_clusters)

    if len(selected) == 0:
        print("[WARN] No clusters satisfy min_tokens threshold.")
        return

    selected_csv = os.path.join(args.outdir, f"expert{args.target_expert}_selected_clusters.csv")
    selected.to_csv(selected_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] selected clusters -> {selected_csv}")

    # auto visualize each selected cluster
    for _, row in selected.iterrows():
        cluster_id = int(row[args.cluster_col])
        token_count = int(row["token_count"])

        cluster_outdir = os.path.join(
            args.outdir,
            f"{args.cluster_col}_cluster{cluster_id}_expert{args.target_expert}_tokens{token_count}"
        )
        ensure_dir(cluster_outdir)

        run_visualizer_for_cluster(
            visualizer_script=args.visualizer_script,
            csv_path=args.csv,
            cluster_col=args.cluster_col,
            cluster_id=cluster_id,
            target_expert=args.target_expert,
            outdir=cluster_outdir,
            grid_size=args.grid_size,
            topk_patches=args.topk_patches,
        )

    print(f"[DONE] All outputs saved under: {args.outdir}")


if __name__ == "__main__":
    main()