#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reuse visualize_cluster_tokens_from_csv.py to visualize noise subclusters.

Input:
    noise_all_experts_token_meta.csv
which contains:
    ... expert_id, subcluster_id, patch_path, token_row, token_col ...

This script:
1. loads the noise token csv
2. converts subcluster_id -> cluster_id_moe_temp
3. optionally filters one expert
4. calls your existing visualize_cluster_tokens_from_csv.py

Example:
python visualization/visualize_noise_subclusters.py \
    --noise_csv analysis_outputs/noise_split_all/noise_all_experts_token_meta.csv \
    --target_subcluster 0 \
    --outdir analysis_outputs/noise_split_all/vis_subcluster0 \
    --target_expert all \
    --grid_size 16 \
    --topk_patches 16

python visualization/visualize_noise_subclusters.py \
    --noise_csv analysis_outputs/noise_split_all/noise_all_experts_token_meta.csv \
    --target_subcluster 1 \
    --outdir analysis_outputs/noise_split_all/vis_subcluster1_exp0 \
    --target_expert 0 \
    --grid_size 16 \
    --topk_patches 16
"""

import os
import sys
import argparse
import subprocess
import pandas as pd


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_csv", type=str, required=True,
                        help="Path to noise_all_experts_token_meta.csv")
    parser.add_argument("--target_subcluster", type=int, required=True,
                        help="Which subcluster_id to visualize")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--target_expert", type=str, default="all",
                        help='Use "all" or expert id like 0')
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--topk_patches", type=int, default=16)
    parser.add_argument("--visualizer_script", type=str,
                        default="visualization/visualize_cluster_tokens_from_csv.py")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    df = pd.read_csv(args.noise_csv)

    if "subcluster_id" not in df.columns:
        raise ValueError("subcluster_id column not found in noise csv.")

    # 只保留目标 subcluster
    df = df[df["subcluster_id"] == args.target_subcluster].copy()

    # 可选：只看某个 expert
    if args.target_expert != "all":
        target_expert = int(args.target_expert)
        df = df[df["expert_id"] == target_expert].copy()

    if len(df) == 0:
        raise ValueError("No rows found for this subcluster / expert.")

    # 为了兼容已有脚本，复制一列临时 cluster 列
    df["cluster_id_moe"] = df["subcluster_id"].astype(int)

    temp_csv = os.path.join(args.outdir, "temp_subcluster_tokens.csv")
    df.to_csv(temp_csv, index=False, encoding="utf-8-sig")

    cmd = [
        sys.executable,
        args.visualizer_script,
        "--csv", temp_csv,
        "--cluster_col", "cluster_id_moe",
        "--target_cluster", str(args.target_subcluster),
        "--outdir", args.outdir,
        "--grid_size", str(args.grid_size),
        "--topk_patches", str(args.topk_patches),
    ]

    if args.target_expert != "all":
        cmd.extend(["--target_expert", str(target_expert)])

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"[DONE] Saved to: {args.outdir}")


if __name__ == "__main__":
    main()