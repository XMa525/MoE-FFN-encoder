#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import openslide
from PIL import Image


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def compute_tissue_ratio(pil_img, s_thresh: float = 20, v_thresh: float = 235) -> float:
    """
    Return approximate tissue occupancy ratio using HSV-like cues.

    A pixel is counted as tissue-like if:
      - saturation is above s_thresh
      OR
      - value/brightness is below v_thresh

    This is more robust than simple non-white filtering for pathology patches
    with pale stroma / glandular lumen / weak staining.
    """
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.uint8).astype(np.float32)

    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    maxc = np.max(arr, axis=2)
    minc = np.min(arr, axis=2)

    # approximate HSV saturation in [0,255]
    sat = np.zeros_like(maxc, dtype=np.float32)
    nonzero = maxc > 0
    sat[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero] * 255.0

    val = maxc  # HSV value approx in [0,255]

    tissue_mask = (sat > s_thresh) | (val < v_thresh)
    return float(tissue_mask.mean())


def read_patch(
    svs_path: str,
    coord_x: int,
    coord_y: int,
    patch_level: int,
    patch_size: int,
    slide_cache: dict,
) -> Image.Image:
    if svs_path not in slide_cache:
        slide_cache[svs_path] = openslide.OpenSlide(svs_path)
    slide = slide_cache[svs_path]
    return slide.read_region((coord_x, coord_y), patch_level, (patch_size, patch_size)).convert("RGB")


def close_slide_cache(slide_cache: dict):
    for slide in slide_cache.values():
        try:
            slide.close()
        except Exception:
            pass
    slide_cache.clear()


def add_tissue_ratio(df: pd.DataFrame, s_thresh: float = 20, v_thresh: float = 235, verbose: bool = True) -> pd.DataFrame:
    need_cols = ["svs_path", "coord_x", "coord_y", "patch_level", "patch_size"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required locator columns: {miss}")

    df = df.copy()
    ratios = []
    slide_cache = {}

    try:
        for i, row in df.iterrows():
            try:
                img = read_patch(
                    svs_path=str(row["svs_path"]),
                    coord_x=int(row["coord_x"]),
                    coord_y=int(row["coord_y"]),
                    patch_level=int(row["patch_level"]),
                    patch_size=int(row["patch_size"]),
                    slide_cache=slide_cache,
                )
                ratio = compute_tissue_ratio(img, s_thresh=s_thresh, v_thresh=v_thresh)
            except Exception:
                ratio = np.nan
            ratios.append(ratio)

            if verbose and (i + 1) % 200 == 0:
                print(f"[tissue_ratio] processed {i+1}/{len(df)}")
    finally:
        close_slide_cache(slide_cache)

    df["tissue_ratio"] = ratios
    return df

def role_specific_filter(
    df: pd.DataFrame,
    role_name: str,
    min_confidence: Optional[float],
    max_entropy: Optional[float],
    min_margin: Optional[float],
    max_background_score: Optional[float],
    min_tissue_ratio: Optional[float],
) -> pd.DataFrame:
    df = df.copy()

    if "pred_label" in df.columns:
        df = df[df["pred_label"] == role_name].copy()

    if min_confidence is not None and "pred_confidence" in df.columns:
        df = df[df["pred_confidence"] >= min_confidence].copy()

    if max_entropy is not None and "entropy" in df.columns:
        df = df[df["entropy"] <= max_entropy].copy()

    if min_margin is not None and "margin_top1_top2" in df.columns:
        df = df[df["margin_top1_top2"] >= min_margin].copy()

    if max_background_score is not None and "score_background" in df.columns:
        df = df[df["score_background"] <= max_background_score].copy()

    if min_tissue_ratio is not None and "tissue_ratio" in df.columns:
        df = df[df["tissue_ratio"] >= min_tissue_ratio].copy()

    return df.reset_index(drop=True)


def topk_per_slide(
    df: pd.DataFrame,
    role_name: str,
    topk_per_slide: int,
) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    score_col = f"score_{role_name}"
    sort_cols = []
    ascending = []

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
    if "tissue_ratio" in df.columns:
        sort_cols.append("tissue_ratio")
        ascending.append(False)

    if len(sort_cols) == 0:
        raise ValueError("No ranking columns available for topk selection.")

    parts = []
    for (project, slide_id), sub in df.groupby(["project", "slide_id"], dropna=False):
        sub = sub.sort_values(sort_cols, ascending=ascending).head(topk_per_slide).copy()
        sub["final_rank_within_slide"] = range(1, len(sub) + 1)
        parts.append(sub)

    out = pd.concat(parts, axis=0).reset_index(drop=True)
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["project", "slide_id", "num_selected"])

    return (
        df.groupby(["project", "slide_id"], dropna=False)
        .size()
        .reset_index(name="num_selected")
        .sort_values(["project", "slide_id"])
        .reset_index(drop=True)
    )


def process_one_role(
    csv_path: str,
    role_name: str,
    out_csv: str,
    out_summary_csv: str,
    white_threshold: float,
    min_confidence: Optional[float],
    max_entropy: Optional[float],
    min_margin: Optional[float],
    max_background_score: Optional[float],
    min_tissue_ratio: Optional[float],
    topk_per_slide_n: int,
):
    df = pd.read_csv(csv_path)
    print(f"[{role_name}] input rows = {len(df)}")

    # 只有当需要非白比例过滤时才真的去读图
    if min_tissue_ratio is not None:
        df = add_tissue_ratio(df, s_thresh=20, v_thresh=235, verbose=True)

    df = role_specific_filter(
        df=df,
        role_name=role_name,
        min_confidence=min_confidence,
        max_entropy=max_entropy,
        min_margin=min_margin,
        max_background_score=max_background_score,
        min_tissue_ratio=min_tissue_ratio,
    )
    print(f"[{role_name}] after filter = {len(df)}")

    df = topk_per_slide(df, role_name=role_name, topk_per_slide=topk_per_slide_n)
    print(f"[{role_name}] after topk/slide = {len(df)}")

    df.to_csv(out_csv, index=False)
    summarize(df).to_csv(out_summary_csv, index=False)
    print(f"[{role_name}] saved -> {out_csv}")


def main():
    parser = argparse.ArgumentParser("Final light filtering for proto candidates")

    parser.add_argument("--tumor_csv", type=str, required=True)
    parser.add_argument("--stroma_csv", type=str, required=True)
    parser.add_argument("--normal_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--white_threshold", type=float, default=235.0)

    parser.add_argument("--tumor_topk", type=int, default=20)
    parser.add_argument("--stroma_topk", type=int, default=20)
    parser.add_argument("--normal_topk", type=int, default=20)

    parser.add_argument("--tumor_min_confidence", type=float, default=0.95)
    parser.add_argument("--tumor_max_entropy", type=float, default=0.25)
    parser.add_argument("--tumor_min_margin", type=float, default=0.80)
    parser.add_argument("--tumor_max_background_score", type=float, default=0.05)
    parser.add_argument("--tumor_min_tissue_ratio", type=float, default=None)

    parser.add_argument("--stroma_min_confidence", type=float, default=0.95)
    parser.add_argument("--stroma_max_entropy", type=float, default=0.20)
    parser.add_argument("--stroma_min_margin", type=float, default=0.85)
    parser.add_argument("--stroma_max_background_score", type=float, default=0.10)
    parser.add_argument("--stroma_min_tissue_ratio", type=float, default=0.35)

    parser.add_argument("--normal_min_confidence", type=float, default=0.95)
    parser.add_argument("--normal_max_entropy", type=float, default=0.20)
    parser.add_argument("--normal_min_margin", type=float, default=0.85)
    parser.add_argument("--normal_max_background_score", type=float, default=0.10)
    parser.add_argument("--normal_min_tissue_ratio", type=float, default=0.45)

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    process_one_role(
        csv_path=args.tumor_csv,
        role_name="tumor",
        out_csv=os.path.join(args.out_dir, "tumor_proto_candidates_final.csv"),
        out_summary_csv=os.path.join(args.out_dir, "tumor_proto_candidates_final_summary.csv"),
        white_threshold=args.white_threshold,
        min_confidence=args.tumor_min_confidence,
        max_entropy=args.tumor_max_entropy,
        min_margin=args.tumor_min_margin,
        max_background_score=args.tumor_max_background_score,
        min_tissue_ratio=args.tumor_min_tissue_ratio,
        topk_per_slide_n=args.tumor_topk,
    )

    process_one_role(
        csv_path=args.stroma_csv,
        role_name="stroma",
        out_csv=os.path.join(args.out_dir, "stroma_proto_candidates_final.csv"),
        out_summary_csv=os.path.join(args.out_dir, "stroma_proto_candidates_final_summary.csv"),
        white_threshold=args.white_threshold,
        min_confidence=args.stroma_min_confidence,
        max_entropy=args.stroma_max_entropy,
        min_margin=args.stroma_min_margin,
        max_background_score=args.stroma_max_background_score,
        min_tissue_ratio=args.stroma_min_tissue_ratio,
        topk_per_slide_n=args.stroma_topk,
    )

    process_one_role(
        csv_path=args.normal_csv,
        role_name="normal_epithelium",
        out_csv=os.path.join(args.out_dir, "normal_proto_candidates_final.csv"),
        out_summary_csv=os.path.join(args.out_dir, "normal_proto_candidates_final_summary.csv"),
        white_threshold=args.white_threshold,
        min_confidence=args.normal_min_confidence,
        max_entropy=args.normal_max_entropy,
        min_margin=args.normal_min_margin,
        max_background_score=args.normal_max_background_score,
        min_tissue_ratio=args.normal_min_tissue_ratio,
        topk_per_slide_n=args.normal_topk,
    )

    print("[Done]")


if __name__ == "__main__":
    main()