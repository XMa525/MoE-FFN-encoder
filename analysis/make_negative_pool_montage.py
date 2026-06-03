#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import h5py
import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# =========================================================
# utils
# =========================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def load_patch_meta_from_h5(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        attrs = dict(f["coords"].attrs.items())

    patch_size = int(attrs.get("patch_size", 256))
    patch_level = int(attrs.get("patch_level", 0))
    return patch_size, patch_level


def read_patch(
    slide: openslide.OpenSlide,
    x: int,
    y: int,
    patch_size: int,
    patch_level: int,
) -> Image.Image:
    return slide.read_region(
        (int(x), int(y)),
        int(patch_level),
        (int(patch_size), int(patch_size)),
    ).convert("RGB")


def draw_text_with_bg(
    draw: ImageDraw.ImageDraw,
    xy,
    text: str,
    font,
    bg=(255, 255, 255),
    fg=(0, 0, 0),
    pad: int = 3,
):
    x, y = xy
    try:
        bbox = draw.multiline_textbbox((x, y), text, font=font, spacing=2)
        x0, y0, x1, y1 = bbox
    except Exception:
        # fallback for old PIL
        w, h = draw.multiline_textsize(text, font=font, spacing=2)
        x0, y0, x1, y1 = x, y, x + w, y + h

    draw.rectangle(
        [x0 - pad, y0 - pad, x1 + pad, y1 + pad],
        fill=bg,
    )
    draw.multiline_text((x, y), text, fill=fg, font=font, spacing=2)


def get_font(size: int = 14):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


# =========================================================
# dataframe filtering
# =========================================================
def load_and_filter_pool(
    csv_path: str,
    split: Optional[str],
    label: Optional[int],
    max_slides: Optional[int],
    seed: int,
) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    required = {"slide_id", "coord_x", "coord_y", "svs_path", "h5_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df["slide_id"] = df["slide_id"].astype(str)

    if split is not None and "split" in df.columns:
        df = df[df["split"].astype(str) == str(split)].copy()

    if label is not None and "label" in df.columns:
        df = df[df["label"].astype(int) == int(label)].copy()

    if len(df) == 0:
        raise ValueError("No rows left after filtering.")

    if max_slides is not None and df["slide_id"].nunique() > max_slides:
        rng = np.random.default_rng(seed)
        slide_ids = sorted(df["slide_id"].unique().tolist())
        keep_ids = rng.choice(slide_ids, size=max_slides, replace=False)
        df = df[df["slide_id"].isin(set(keep_ids))].copy()

    return df.reset_index(drop=True)


def select_top_candidates(
    df: pd.DataFrame,
    score_col: str,
    topk_total: int,
    topk_per_slide: Optional[int],
    ascending: bool = False,
) -> pd.DataFrame:
    if score_col not in df.columns:
        raise ValueError(
            f"score_col={score_col} not found. Available columns include: {list(df.columns)[:50]}"
        )

    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col]).copy()

    if len(df) == 0:
        raise ValueError(f"No valid numeric rows for score_col={score_col}")

    if topk_per_slide is not None and topk_per_slide > 0:
        parts = []
        for slide_id, sub in df.groupby("slide_id"):
            sub = sub.sort_values(score_col, ascending=ascending).head(topk_per_slide)
            parts.append(sub)
        df = pd.concat(parts, axis=0).reset_index(drop=True)

    df = df.sort_values(score_col, ascending=ascending).head(topk_total).reset_index(drop=True)
    return df


# =========================================================
# montage
# =========================================================
def save_montage(
    df: pd.DataFrame,
    out_path: str,
    tile_size: int = 224,
    cols: int = 6,
    title: Optional[str] = None,
    score_col: str = "tumor_gap",
    extra_cols: Optional[List[str]] = None,
):
    if len(df) == 0:
        raise ValueError("Cannot save montage with empty df.")

    extra_cols = extra_cols or []

    rows = math.ceil(len(df) / cols)
    title_h = 44 if title else 0
    label_h = 58

    canvas_w = cols * tile_size
    canvas_h = rows * (tile_size + label_h) + title_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font_title = get_font(20)
    font_label = get_font(12)

    if title:
        draw_text_with_bg(
            draw,
            (8, 8),
            title,
            font=font_title,
            bg=(255, 255, 255),
            fg=(0, 0, 0),
            pad=2,
        )

    slide_cache: Dict[str, openslide.OpenSlide] = {}
    meta_cache: Dict[str, tuple] = {}

    for idx, row in tqdm(
        list(enumerate(df.itertuples(index=False))),
        total=len(df),
        desc=f"Saving montage {Path(out_path).name}",
    ):
        row_dict = row._asdict()

        svs_path = str(row_dict["svs_path"])
        h5_path = str(row_dict["h5_path"])
        slide_id = str(row_dict["slide_id"])

        x = int(row_dict["coord_x"])
        y = int(row_dict["coord_y"])

        if h5_path not in meta_cache:
            meta_cache[h5_path] = load_patch_meta_from_h5(h5_path)
        patch_size, patch_level = meta_cache[h5_path]

        if svs_path not in slide_cache:
            if not os.path.exists(svs_path):
                raise FileNotFoundError(f"svs_path not found: {svs_path}")
            slide_cache[svs_path] = openslide.OpenSlide(svs_path)

        slide = slide_cache[svs_path]
        patch = read_patch(
            slide=slide,
            x=x,
            y=y,
            patch_size=patch_size,
            patch_level=patch_level,
        )
        patch = patch.resize((tile_size, tile_size), resample=Image.BICUBIC)

        rr = idx // cols
        cc = idx % cols

        ox = cc * tile_size
        oy = title_h + rr * (tile_size + label_h)

        canvas.paste(patch, (ox, oy + label_h))

        rank_txt = f"{idx + 1:04d}"
        score_txt = f"{score_col}={safe_float(row_dict.get(score_col)):.4f}"

        info_lines = [
            rank_txt,
            f"{slide_id[:26]}",
            score_txt,
        ]

        if "tumor_prob" in row_dict:
            info_lines.append(f"prob={safe_float(row_dict.get('tumor_prob')):.4f}")
        if "neighbor_gap_mean" in row_dict:
            info_lines.append(f"nb_mean={safe_float(row_dict.get('neighbor_gap_mean')):.4f}")
        if "neighbor_gap_max" in row_dict:
            info_lines.append(f"nb_max={safe_float(row_dict.get('neighbor_gap_max')):.4f}")

        for c in extra_cols:
            if c in row_dict:
                v = row_dict.get(c)
                if isinstance(v, (float, int, np.floating, np.integer)):
                    info_lines.append(f"{c}={float(v):.4f}")
                else:
                    info_lines.append(f"{c}={str(v)[:20]}")

        label_txt = "\n".join(info_lines[:5])

        draw_text_with_bg(
            draw,
            (ox + 3, oy + 2),
            label_txt,
            font=font_label,
            bg=(255, 255, 255),
            fg=(0, 0, 0),
            pad=2,
        )

    for slide in slide_cache.values():
        slide.close()

    canvas.save(out_path)
    print(f"[Saved] {out_path}")


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        "Quick diagnostic montage for tumor-like negative candidates"
    )
    parser.add_argument("--negative_pool_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--label", type=int, default=0)

    parser.add_argument("--score_col", type=str, default="tumor_gap",
                        help="Column used for ranking candidates, e.g. tumor_gap, tumor_prob, proposal_score, neg_context_score.")
    parser.add_argument("--topk_total", type=int, default=120)
    parser.add_argument("--topk_per_slide", type=int, default=12)
    parser.add_argument("--ascending", action="store_true",
                        help="Use lower score as top. Default selects highest score.")

    parser.add_argument("--max_slides", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--cols", type=int, default=6)

    parser.add_argument("--save_selected_csv", action="store_true")
    parser.add_argument("--prefix", type=str, default="negative_top_tumor_like")

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df = load_and_filter_pool(
        csv_path=args.negative_pool_csv,
        split=args.split,
        label=args.label,
        max_slides=args.max_slides,
        seed=args.seed,
    )

    print("=" * 80)
    print(f"[Loaded] rows={len(df)}, slides={df['slide_id'].nunique()}")
    print(f"[CSV] {args.negative_pool_csv}")
    print(f"[Score] {args.score_col}, ascending={args.ascending}")
    print("=" * 80)

    selected = select_top_candidates(
        df=df,
        score_col=args.score_col,
        topk_total=args.topk_total,
        topk_per_slide=args.topk_per_slide,
        ascending=args.ascending,
    )

    print("[Selected]")
    print(f"rows={len(selected)}, slides={selected['slide_id'].nunique()}")
    if args.score_col in selected.columns:
        print(selected[args.score_col].describe())

    selected_csv = os.path.join(
        args.out_dir,
        f"{args.prefix}_by_{args.score_col}_top{args.topk_total}.csv",
    )
    if args.save_selected_csv:
        selected.to_csv(selected_csv, index=False)
        print(f"[Saved] {selected_csv}")

    montage_path = os.path.join(
        args.out_dir,
        f"{args.prefix}_by_{args.score_col}_top{args.topk_total}.png",
    )

    title = (
        f"Negative candidates ranked by {args.score_col} | "
        f"top={args.topk_total}, per_slide={args.topk_per_slide}"
    )

    save_montage(
        df=selected,
        out_path=montage_path,
        tile_size=args.tile_size,
        cols=args.cols,
        title=title,
        score_col=args.score_col,
        extra_cols=[
            "candidate_type",
            "pool_source",
            "proposal_score",
            "neg_context_score",
            "top1_gap",
        ],
    )

    print("[Done]")


if __name__ == "__main__":
    main()