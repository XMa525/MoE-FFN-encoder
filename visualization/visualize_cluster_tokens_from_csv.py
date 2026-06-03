#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize token-level clusters from exported token_meta.csv

Supported columns in CSV:
    image_index, patch_path, patch_name, sample_category,
    token_idx, token_row, token_col, expert_id,
    cluster_id_moe, cluster_id_final

Main functions:
    1) top prototype patches
    2) token overlay on patch
    3) average token position heatmap
    4) patch/sample statistics
    5) optional expert purity summary inside target cluster

Example:
python visualize_cluster_tokens_from_csv.py \
    --csv analysis_outputs/token_meta.csv \
    --cluster_col cluster_id_moe \
    --target_cluster 3 \
    --target_expert 2 \
    --outdir analysis_outputs/vis_moe_cluster3_exp2 \
    --grid_size 16 \
    --topk_patches 16
"""

import os
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Cannot open image: {path} | {e}")
        return None


def make_image_grid(images, titles=None, ncols=4, figsize_scale=4, save_path=None, suptitle=None):
    n = len(images)
    if n == 0:
        print("[WARN] No images to draw.")
        return

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figsize_scale, nrows * figsize_scale))

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    axes = axes.reshape(nrows, ncols)

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            if titles is not None:
                ax.set_title(titles[i], fontsize=9)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=15)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def overlay_tokens_on_patch(
    image: Image.Image,
    token_rows,
    token_cols,
    grid_size: int,
    alpha: float = 0.35,
    outline_width: int = 2,
):
    img = image.convert("RGB")
    w, h = img.size
    cell_w = w / grid_size
    cell_h = h / grid_size

    overlay = img.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")

    for r, c in zip(token_rows, token_cols):
        x0 = int(c * cell_w)
        y0 = int(r * cell_h)
        x1 = int((c + 1) * cell_w)
        y1 = int((r + 1) * cell_h)

        draw.rectangle(
            [x0, y0, x1, y1],
            fill=(255, 0, 0, int(255 * alpha)),
            outline=(255, 255, 0, 255),
            width=outline_width,
        )

    return overlay


def build_avg_position_heatmap(df_target: pd.DataFrame, grid_size: int):
    heat = np.zeros((grid_size, grid_size), dtype=np.float32)

    for _, row in df_target.iterrows():
        r = int(row["token_row"])
        c = int(row["token_col"])
        if 0 <= r < grid_size and 0 <= c < grid_size:
            heat[r, c] += 1.0

    if heat.sum() > 0:
        heat = heat / heat.sum()
    return heat


def plot_heatmap(heatmap, save_path, title="Average token position heatmap"):
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Token column")
    plt.ylabel("Token row")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def summarize_patch_counts(df_target: pd.DataFrame):
    patch_counts = (
        df_target.groupby(["patch_path", "patch_name", "sample_category"])
        .agg(
            cluster_token_count=("token_idx", "count"),
            unique_token_count=("token_idx", "nunique"),
            dominant_expert=("expert_id", lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
        .sort_values(["cluster_token_count", "unique_token_count"], ascending=False)
    )
    return patch_counts


def summarize_sample_counts(df_target: pd.DataFrame):
    sample_counts = (
        df_target.groupby("sample_category")
        .agg(
            cluster_token_count=("token_idx", "count"),
            unique_patch_count=("patch_path", "nunique"),
        )
        .reset_index()
        .sort_values(["cluster_token_count", "unique_patch_count"], ascending=False)
    )
    return sample_counts


def summarize_expert_purity(df_target: pd.DataFrame):
    expert_counts = df_target["expert_id"].value_counts().sort_index()
    total = expert_counts.sum()
    summary = pd.DataFrame({
        "expert_id": expert_counts.index,
        "count": expert_counts.values,
        "ratio": expert_counts.values / total
    })
    summary = summary.sort_values("count", ascending=False)
    return summary


def filter_target(df: pd.DataFrame, cluster_col: str, target_cluster: int, target_expert=None):
    df_target = df[df[cluster_col] == target_cluster].copy()
    if target_expert is not None:
        df_target = df_target[df_target["expert_id"] == target_expert].copy()
    return df_target


def visualize_top_patches(
    df_target: pd.DataFrame,
    patch_counts: pd.DataFrame,
    outdir: str,
    topk_patches: int,
    grid_size: int,
    cluster_col: str,
    target_cluster: int,
    target_expert=None,
):
    top_patch_paths = patch_counts["patch_path"].head(topk_patches).tolist()

    raw_imgs = []
    raw_titles = []
    overlay_imgs = []
    overlay_titles = []

    for patch_path in top_patch_paths:
        img = safe_open_image(patch_path)
        if img is None:
            continue

        sub = df_target[df_target["patch_path"] == patch_path].copy()
        sub = sub.sort_values(["token_row", "token_col"])

        token_rows = sub["token_row"].astype(int).tolist()
        token_cols = sub["token_col"].astype(int).tolist()
        n_tokens = len(sub)
        dominant_expert = int(sub["expert_id"].mode().iloc[0]) if len(sub) > 0 else -1

        raw_imgs.append(img)
        raw_titles.append(
            f"{Path(patch_path).name}\n#tokens={n_tokens} | exp={dominant_expert}"
        )

        overlay = overlay_tokens_on_patch(
            image=img,
            token_rows=token_rows,
            token_cols=token_cols,
            grid_size=grid_size,
            alpha=0.35,
            outline_width=2,
        )
        overlay_imgs.append(overlay)
        overlay_titles.append(
            f"{Path(patch_path).name}\n#tokens={n_tokens} | exp={dominant_expert}"
        )

    suffix = f"{cluster_col}= {target_cluster}"
    if target_expert is not None:
        suffix += f", expert={target_expert}"

    make_image_grid(
        raw_imgs,
        titles=raw_titles,
        ncols=4,
        figsize_scale=4,
        save_path=os.path.join(outdir, "top_patches_raw.png"),
        suptitle=f"Top prototype patches ({suffix})"
    )

    make_image_grid(
        overlay_imgs,
        titles=overlay_titles,
        ncols=4,
        figsize_scale=4,
        save_path=os.path.join(outdir, "top_patches_overlay.png"),
        suptitle=f"Top prototype patches with token overlays ({suffix})"
    )


def plot_patch_token_histogram(patch_counts: pd.DataFrame, save_path: str, topn: int = 30):
    sub = patch_counts.head(topn).copy()
    if len(sub) == 0:
        return

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(sub)), sub["cluster_token_count"].values)
    plt.xticks(
        range(len(sub)),
        [x if len(x) <= 20 else x[:18] + ".." for x in sub["patch_name"]],
        rotation=75,
        fontsize=8
    )
    plt.ylabel("Cluster token count")
    plt.title(f"Top-{topn} patches ranked by cluster token count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_sample_distribution(sample_counts: pd.DataFrame, save_path: str):
    if len(sample_counts) == 0:
        return

    plt.figure(figsize=(8, 5))
    plt.bar(sample_counts["sample_category"], sample_counts["cluster_token_count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cluster token count")
    plt.title("Cluster distribution across sample categories")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--cluster_col", type=str, required=True,
                        choices=["cluster_id_moe", "cluster_id_final"])
    parser.add_argument("--target_cluster", type=int, required=True)
    parser.add_argument("--target_expert", type=int, default=None)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--topk_patches", type=int, default=16)
    args = parser.parse_args()

    ensure_dir(args.outdir)

    df = pd.read_csv(args.csv)

    required_cols = [
        "patch_path", "patch_name", "sample_category",
        "token_idx", "token_row", "token_col",
        "expert_id", args.cluster_col
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.dropna(subset=required_cols).copy()

    int_cols = ["token_idx", "token_row", "token_col", "expert_id", args.cluster_col]
    for col in int_cols:
        df[col] = df[col].astype(int)

    df_target = filter_target(
        df=df,
        cluster_col=args.cluster_col,
        target_cluster=args.target_cluster,
        target_expert=args.target_expert,
    )

    if len(df_target) == 0:
        raise ValueError("No rows found for the given target cluster / expert.")

    print(f"[INFO] cluster_col = {args.cluster_col}")
    print(f"[INFO] target_cluster = {args.target_cluster}")
    print(f"[INFO] target_expert = {args.target_expert}")
    print(f"[INFO] total target tokens = {len(df_target)}")
    print(f"[INFO] unique patches = {df_target['patch_path'].nunique()}")

    # save filtered rows
    df_target.to_csv(os.path.join(args.outdir, "filtered_tokens.csv"), index=False)

    # expert purity inside target cluster
    expert_summary = summarize_expert_purity(df_target)
    expert_summary.to_csv(os.path.join(args.outdir, "expert_purity_summary.csv"), index=False)

    # patch counts
    patch_counts = summarize_patch_counts(df_target)
    patch_counts.to_csv(os.path.join(args.outdir, "patch_token_count.csv"), index=False)

    # sample counts
    sample_counts = summarize_sample_counts(df_target)
    sample_counts.to_csv(os.path.join(args.outdir, "sample_token_count.csv"), index=False)

    # avg token position heatmap
    heat = build_avg_position_heatmap(df_target, grid_size=args.grid_size)
    plot_heatmap(
        heat,
        save_path=os.path.join(args.outdir, "avg_token_position_heatmap.png"),
        title=f"Average token position heatmap | {args.cluster_col}={args.target_cluster}"
              + (f", expert={args.target_expert}" if args.target_expert is not None else "")
    )

    # patch raw/overlay
    visualize_top_patches(
        df_target=df_target,
        patch_counts=patch_counts,
        outdir=args.outdir,
        topk_patches=args.topk_patches,
        grid_size=args.grid_size,
        cluster_col=args.cluster_col,
        target_cluster=args.target_cluster,
        target_expert=args.target_expert,
    )

    # histograms
    plot_patch_token_histogram(
        patch_counts=patch_counts,
        save_path=os.path.join(args.outdir, "top_patch_token_histogram.png"),
        topn=min(30, len(patch_counts)),
    )

    plot_sample_distribution(
        sample_counts=sample_counts,
        save_path=os.path.join(args.outdir, "sample_distribution.png"),
    )

    # text summary
    with open(os.path.join(args.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"csv = {args.csv}\n")
        f.write(f"cluster_col = {args.cluster_col}\n")
        f.write(f"target_cluster = {args.target_cluster}\n")
        f.write(f"target_expert = {args.target_expert}\n")
        f.write(f"total_target_tokens = {len(df_target)}\n")
        f.write(f"unique_patches = {df_target['patch_path'].nunique()}\n\n")

        f.write("[Expert purity]\n")
        for _, row in expert_summary.iterrows():
            f.write(f"expert {int(row['expert_id'])}: count={int(row['count'])}, ratio={row['ratio']:.4f}\n")

    print(f"[DONE] Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()