#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def ensure_required_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def maybe_fill_slide_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "slide_id" not in df.columns:
        if "svs_path" not in df.columns:
            raise ValueError("Need either 'slide_id' or 'svs_path' in input csv.")
        df["slide_id"] = df["svs_path"].astype(str).map(canonicalize_path)
    else:
        df["slide_id"] = df["slide_id"].astype(str)
    return df


def maybe_fill_coord_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    如果没有 coord_idx，则在每个 slide 内按 (coord_y, coord_x) 排序后补一个局部索引。
    注意：这只适合建 bank + 立刻配套使用。
    若你的训练 csv 本身已有 coord_idx，务必保持一致，不要重建。
    """
    df = df.copy()

    if "coord_idx" in df.columns:
        # 强制转 int，尽量兼容 NaN
        bad_mask = df["coord_idx"].isna()
        if bad_mask.any():
            raise ValueError("coord_idx column exists but contains NaN. Please fix it first.")
        df["coord_idx"] = df["coord_idx"].astype(int)
        return df

    print("[Warn] 'coord_idx' not found. Building coord_idx within each slide by sorted (coord_y, coord_x).")
    out_parts = []
    for slide_id, sdf in df.groupby("slide_id", sort=False):
        sdf = sdf.sort_values(["coord_y", "coord_x"]).copy().reset_index(drop=True)
        sdf["coord_idx"] = np.arange(len(sdf), dtype=np.int64)
        out_parts.append(sdf)

    return pd.concat(out_parts, axis=0, ignore_index=True)


def filter_negative_slides(df: pd.DataFrame, slide_label_col: str = "slide_label") -> pd.DataFrame:
    if slide_label_col not in df.columns:
        raise ValueError(f"slide label column '{slide_label_col}' not found.")

    df = df.copy()
    bad = df[df[slide_label_col].isna()]
    if len(bad) > 0:
        print(f"[Warn] Found {len(bad)} rows with NaN {slide_label_col}; they will be dropped.")
        df = df[df[slide_label_col].notna()].copy()

    df[slide_label_col] = df[slide_label_col].astype(int)
    df = df[df[slide_label_col] == 0].copy().reset_index(drop=True)
    return df


def maybe_filter_prefilter_white(df: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if enabled and "prefilter_white" in df.columns:
        df = df[df["prefilter_white"].fillna(0).astype(int) == 0].copy().reset_index(drop=True)
    return df


def build_neighbors_for_one_slide(
    sdf: pd.DataFrame,
    k: int,
    distance: str = "euclidean",
    max_radius: float = -1.0,
    min_neighbors: int = 1,
):
    """
    对单个 slide 内的所有 patch 构建邻域。
    返回 list[dict]
    """
    coords = sdf[["coord_x", "coord_y"]].to_numpy(dtype=np.float32)   # [N, 2]
    coord_idx_arr = sdf["coord_idx"].to_numpy(dtype=np.int64)
    svs_path_arr = sdf["svs_path"].astype(str).to_numpy() if "svs_path" in sdf.columns else np.array([""] * len(sdf))
    slide_label_arr = sdf["slide_label"].to_numpy(dtype=np.int64)

    if "subclass_id" in sdf.columns:
        subclass_arr = sdf["subclass_id"].fillna(-1).astype(int).to_numpy()
    else:
        subclass_arr = np.full(len(sdf), -1, dtype=np.int64)

    n = len(sdf)
    if n <= 1:
        return []

    rows = []

    # 简单直接版：每张 slide 内做一次全 pair 距离
    # N 很大时会占内存，但先做主线验证通常够用
    dx = coords[:, None, 0] - coords[None, :, 0]
    dy = coords[:, None, 1] - coords[None, :, 1]

    if distance == "euclidean":
        dist_mat = np.sqrt(dx * dx + dy * dy)
    elif distance == "manhattan":
        dist_mat = np.abs(dx) + np.abs(dy)
    else:
        raise ValueError(f"Unsupported distance: {distance}")

    for i in range(n):
        d = dist_mat[i].copy()
        d[i] = np.inf  # 排除自己

        order = np.argsort(d)
        if max_radius is not None and max_radius > 0:
            order = order[d[order] <= max_radius]

        order = order[:k]
        if len(order) < min_neighbors:
            continue

        neighbor_coord_indices = coord_idx_arr[order].tolist()

        rows.append({
            "slide_id": str(sdf.iloc[i]["slide_id"]),
            "svs_path": str(svs_path_arr[i]),
            "coord_idx": int(coord_idx_arr[i]),
            "coord_x": int(sdf.iloc[i]["coord_x"]),
            "coord_y": int(sdf.iloc[i]["coord_y"]),
            "slide_label": int(slide_label_arr[i]),
            "subclass_id": int(subclass_arr[i]),
            "neighbor_coord_indices": ";".join(map(str, neighbor_coord_indices)),
            "num_neighbors": int(len(neighbor_coord_indices)),
        })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)

    parser.add_argument("--slide-label-col", type=str, default="slide_label")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--distance", type=str, default="euclidean", choices=["euclidean", "manhattan"])
    parser.add_argument("--max-radius", type=float, default=-1.0)
    parser.add_argument("--min-neighbors", type=int, default=1)

    parser.add_argument("--filter-prefilter-white", action="store_true")
    parser.add_argument("--min-patches-per-slide", type=int, default=2)

    args = parser.parse_args()

    print(f"[Input] {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if "svs_path" in df.columns:
        df["svs_path"] = df["svs_path"].astype(str).map(canonicalize_path)

    ensure_required_columns(
        df,
        required_cols=["coord_x", "coord_y", args.slide_label_col],
    )

    df = maybe_fill_slide_id(df)
    df = maybe_fill_coord_idx(df)
    df = maybe_filter_prefilter_white(df, enabled=args.filter_prefilter_white)
    df = filter_negative_slides(df, slide_label_col=args.slide_label_col)

    # 统一命名为 slide_label，方便下游
    if args.slide_label_col != "slide_label":
        df = df.rename(columns={args.slide_label_col: "slide_label"}).copy()

    print(f"[After negative filter] rows = {len(df)}")
    if len(df) == 0:
        raise ValueError("No negative rows found after filtering.")

    slide_sizes = df.groupby("slide_id").size()
    valid_slide_ids = slide_sizes[slide_sizes >= args.min_patches_per_slide].index.tolist()
    df = df[df["slide_id"].isin(valid_slide_ids)].copy().reset_index(drop=True)

    print(f"[Valid negative slides] {len(valid_slide_ids)}")
    print(f"[Rows kept] {len(df)}")

    all_rows = []
    grouped = df.groupby("slide_id", sort=False)

    for slide_id, sdf in tqdm(grouped, total=len(grouped), desc="Building dense neighbor bank"):
        rows = build_neighbors_for_one_slide(
            sdf=sdf.reset_index(drop=True),
            k=args.k,
            distance=args.distance,
            max_radius=args.max_radius,
            min_neighbors=args.min_neighbors,
        )
        all_rows.extend(rows)

    out_df = pd.DataFrame(all_rows)

    if len(out_df) == 0:
        raise ValueError("No bank rows generated. Check k / radius / filtering settings.")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"[Done] output rows = {len(out_df)}")
    print(f"[Saved] {args.output_csv}")

    # 简单统计
    print("[Summary]")
    print(f"  unique slides: {out_df['slide_id'].nunique()}")
    print(f"  mean num_neighbors: {out_df['num_neighbors'].mean():.2f}")
    print(f"  min  num_neighbors: {out_df['num_neighbors'].min()}")
    print(f"  max  num_neighbors: {out_df['num_neighbors'].max()}")


if __name__ == "__main__":
    main()