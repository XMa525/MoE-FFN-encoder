#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import openslide
import pandas as pd


REQUIRED_COLS = {"svs_path", "coord_x", "coord_y", "patch_level", "patch_size"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize TCGA candidate patches from svs+h5 locator CSVs")
    parser.add_argument("--csv", type=str, action="append", required=True, help="Input candidate CSV; can be repeated")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--group-col", type=str, default="organ_name")
    parser.add_argument("--samples-per-group", type=int, default=36)
    parser.add_argument("--overview-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--sort-groups-by-size", action="store_true")
    parser.add_argument("--title-col", type=str, default=None)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str) -> str:
    bad = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
    out = str(name)
    for ch in bad:
        out = out.replace(ch, "_")
    return out


def sample_rows(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy().reset_index(drop=True)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def make_auto_caption(row: pd.Series) -> str:
    parts = []
    if "organ_name" in row and pd.notna(row["organ_name"]):
        parts.append(str(row["organ_name"]))
    elif "project" in row and pd.notna(row["project"]):
        parts.append(str(row["project"]))

    if "pred_label" in row and pd.notna(row["pred_label"]):
        parts.append(str(row["pred_label"]))

    metrics = []
    if "pred_confidence" in row and pd.notna(row["pred_confidence"]):
        metrics.append(f"c={float(row['pred_confidence']):.3f}")
    if "entropy" in row and pd.notna(row["entropy"]):
        metrics.append(f"e={float(row['entropy']):.3f}")
    if "margin_top1_top2" in row and pd.notna(row["margin_top1_top2"]):
        metrics.append(f"m={float(row['margin_top1_top2']):.3f}")

    if metrics:
        parts.append(" ".join(metrics))
    return "\n".join(parts)


def read_patch(row: pd.Series, slide_cache: dict):
    svs_path = str(row["svs_path"])
    if svs_path not in slide_cache:
        slide_cache[svs_path] = openslide.OpenSlide(svs_path)
    slide = slide_cache[svs_path]

    x = int(row["coord_x"])
    y = int(row["coord_y"])
    level = int(row["patch_level"])
    patch_size = int(row["patch_size"])

    return slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")


def close_cache(slide_cache: dict):
    for slide in slide_cache.values():
        try:
            slide.close()
        except Exception:
            pass
    slide_cache.clear()


def plot_montage(df: pd.DataFrame, outpath: str, title: str, title_col: Optional[str] = None) -> None:
    n = len(df)
    if n == 0:
        return

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.6))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes:
        ax.axis("off")

    slide_cache = {}
    try:
        for ax, (_, row) in zip(axes, df.iterrows()):
            try:
                img = read_patch(row, slide_cache)
                ax.imshow(img)
                if title_col is not None and title_col in row and pd.notna(row[title_col]):
                    caption = str(row[title_col])
                else:
                    caption = make_auto_caption(row)
                ax.set_title(caption, fontsize=7)
            except Exception as e:
                fallback = str(row.get("slide_id", Path(str(row["svs_path"])).stem))
                ax.text(0.5, 0.5, f"Failed\n{fallback}", ha="center", va="center", fontsize=7)
                ax.set_title(str(e)[:80], fontsize=6)

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close(fig)
    finally:
        close_cache(slide_cache)


def process_one_csv(csv_path: str, outdir: str, args: argparse.Namespace) -> None:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    if "organ_name" not in df.columns:
        if "project" in df.columns:
            df["organ_name"] = df["project"].astype(str)
        else:
            df["organ_name"] = "all"

    stem = Path(csv_path).stem
    local_outdir = os.path.join(outdir, stem)
    ensure_dir(local_outdir)

    sampled_rows_all = []

    overview_df = sample_rows(df, n=args.overview_samples, seed=args.seed)
    overview_df = overview_df.copy()
    overview_df["_sample_source"] = "overview"
    sampled_rows_all.append(overview_df)

    plot_montage(
        overview_df,
        outpath=os.path.join(local_outdir, f"{stem}__overview.png"),
        title=f"{stem} | overview | n={len(overview_df)}",
        title_col=args.title_col,
    )

    if args.group_col in df.columns:
        grouped = list(df.groupby(args.group_col))
        if args.sort_groups_by_size:
            grouped = sorted(grouped, key=lambda x: len(x[1]), reverse=True)
        else:
            grouped = sorted(grouped, key=lambda x: str(x[0]))

        if args.max_groups is not None:
            grouped = grouped[:args.max_groups]

        for group_name, sub in grouped:
            sampled = sample_rows(sub, n=args.samples_per_group, seed=args.seed)
            sampled = sampled.copy()
            sampled["_sample_source"] = f"group:{group_name}"
            sampled_rows_all.append(sampled)

            outpath = os.path.join(local_outdir, f"{stem}__group_{sanitize_filename(group_name)}.png")
            title = f"{stem} | {args.group_col}={group_name} | n={len(sampled)} / {len(sub)}"
            plot_montage(sampled, outpath=outpath, title=title, title_col=args.title_col)

    sampled_rows = pd.concat(sampled_rows_all, axis=0).reset_index(drop=True)
    sampled_rows.to_csv(os.path.join(local_outdir, f"{stem}__sampled_rows.csv"), index=False)


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    for csv_path in args.csv:
        process_one_csv(csv_path, args.outdir, args)

    print("Done.")
    print(f"Saved visualizations to: {args.outdir}")


if __name__ == "__main__":
    main()