#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_font(size: int = 18):
    # 尽量使用系统默认字体，失败则用 PIL 默认字体
    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for f in font_candidates:
        if os.path.exists(f):
            return ImageFont.truetype(f, size=size)
    return ImageFont.load_default()


def open_patch_from_image_path(row: pd.Series, image_col: str) -> Image.Image:
    img_path = row[image_col]
    img = Image.open(img_path).convert("RGB")
    return img


def open_patch_from_svs(
    row: pd.Series,
    svs_col: str,
    x_col: str,
    y_col: str,
    patch_size: int,
) -> Image.Image:
    import openslide

    svs_path = row[svs_col]
    x = int(row[x_col])
    y = int(row[y_col])

    slide = openslide.OpenSlide(str(svs_path))
    img = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
    slide.close()
    return img


def draw_tile(
    img: Image.Image,
    idx_text: str,
    label_text: str,
    score_text: str,
    tile_size: int,
    header_h: int,
    font_big,
    font_small,
) -> Image.Image:
    img = img.convert("RGB")
    img = img.resize((tile_size, tile_size))

    canvas = Image.new("RGB", (tile_size, tile_size + header_h), "white")
    canvas.paste(img, (0, header_h))

    draw = ImageDraw.Draw(canvas)

    # 编号放左上角，尽量醒目
    draw.rectangle([0, 0, tile_size, header_h], fill="white")
    draw.text((4, 2), idx_text, fill="black", font=font_big)

    # label 和 score 放下面两行
    draw.text((4, 24), label_text[:28], fill="black", font=font_small)
    if score_text:
        draw.text((4, 42), score_text[:36], fill="black", font=font_small)

    return canvas


def build_score_text(row: pd.Series) -> str:
    parts = []

    for col, short in [
        ("confidence", "c"),
        ("entropy", "e"),
        ("margin", "m"),
        ("pred_confidence", "c"),
        ("pred_entropy", "e"),
        ("pred_margin", "m"),
    ]:
        if col in row.index:
            try:
                parts.append(f"{short}={float(row[col]):.3f}")
            except Exception:
                pass

    return " ".join(parts)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate numbered contact sheets and a review manifest for manual patch QC."
    )

    parser.add_argument("--csv", required=True, help="Input CSV, e.g. patch_semantic_predictions.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for review sheets and manifest")

    parser.add_argument(
        "--label",
        required=True,
        help="Which pred_label to review, e.g. fibrovascular_stroma or tumor"
    )
    parser.add_argument(
        "--label-col",
        default="pred_label",
        help="Column containing predicted label. Default: pred_label"
    )

    parser.add_argument(
        "--sort-by",
        default=None,
        help="Column to sort by descending. If not given, uses confidence if available."
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=500,
        help="Maximum candidates to export for review."
    )
    parser.add_argument(
        "--per-slide-max",
        type=int,
        default=0,
        help="Optional max candidates per slide. 0 means no limit."
    )

    parser.add_argument("--cols", type=int, default=6, help="Number of columns per contact sheet.")
    parser.add_argument("--rows", type=int, default=6, help="Number of rows per contact sheet.")
    parser.add_argument("--tile-size", type=int, default=224, help="Displayed tile size.")
    parser.add_argument("--header-h", type=int, default=64, help="Header height above each tile.")
    parser.add_argument("--patch-size", type=int, default=224, help="Patch size to read from WSI if no image_path exists.")

    parser.add_argument(
        "--image-col",
        default=None,
        help="Optional column containing existing patch image path. If absent, script tries to read from svs_path + x/y."
    )
    parser.add_argument(
        "--svs-col",
        default=None,
        help="Optional WSI path column. Auto-detects svs_path/slide_path/source_path if not set."
    )
    parser.add_argument(
        "--x-col",
        default=None,
        help="Optional x coordinate column. Auto-detects x/coord_x/patch_x if not set."
    )
    parser.add_argument(
        "--y-col",
        default=None,
        help="Optional y coordinate column. Auto-detects y/coord_y/patch_y if not set."
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    sheet_dir = outdir / f"review_sheets_{args.label}"
    ensure_dir(sheet_dir)

    df = pd.read_csv(csv_path)

    if args.label_col not in df.columns:
        raise ValueError(f"Cannot find label column: {args.label_col}. Available columns: {df.columns.tolist()}")

    df = df[df[args.label_col].astype(str) == args.label].copy()

    if len(df) == 0:
        raise ValueError(f"No rows found for label={args.label}")

    # slide_id column
    slide_col = find_col(df, ["slide_id", "case_id", "wsi_id", "svs_id"])

    # sorting
    sort_by = args.sort_by
    if sort_by is None:
        sort_by = find_col(df, ["confidence", "pred_confidence", "margin", "pred_margin"])

    if sort_by is not None and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    # per-slide limit
    if args.per_slide_max and args.per_slide_max > 0 and slide_col is not None:
        df = df.groupby(slide_col, group_keys=False).head(args.per_slide_max)

    df = df.head(args.max_candidates).copy()
    df = df.reset_index(drop=True)

    # Detect image or WSI columns
    image_col = args.image_col
    if image_col is None:
        image_col = find_col(df, ["image_path", "patch_path", "png_path", "jpg_path"])

    svs_col = args.svs_col or find_col(df, ["svs_path", "slide_path", "source_path", "wsi_path"])
    x_col = args.x_col or find_col(df, ["x", "coord_x", "patch_x", "x_coord"])
    y_col = args.y_col or find_col(df, ["y", "coord_y", "patch_y", "y_coord"])

    use_image_path = image_col is not None and image_col in df.columns

    if not use_image_path:
        if svs_col is None or x_col is None or y_col is None:
            raise ValueError(
                "Cannot locate patch image source.\n"
                f"Available columns: {df.columns.tolist()}\n"
                "Either provide --image-col, or make sure CSV has svs_path/slide_path/source_path and x/y coordinates."
            )

    # Add review columns
    df.insert(0, "review_idx", [f"{i+1:04d}" for i in range(len(df))])
    if "keep" not in df.columns:
        df["keep"] = ""
    if "comment" not in df.columns:
        df["comment"] = ""

    # Save manifest first
    manifest_path = outdir / f"{args.label}_review_manifest.csv"

    # Build sheets
    font_big = load_font(20)
    font_small = load_font(12)

    n_per_sheet = args.cols * args.rows
    n_sheets = math.ceil(len(df) / n_per_sheet)

    sheet_paths = []

    for s in range(n_sheets):
        start = s * n_per_sheet
        end = min((s + 1) * n_per_sheet, len(df))
        sub = df.iloc[start:end]

        sheet_w = args.cols * args.tile_size
        sheet_h = args.rows * (args.tile_size + args.header_h)

        sheet = Image.new("RGB", (sheet_w, sheet_h), "white")

        for j, (_, row) in enumerate(sub.iterrows()):
            r = j // args.cols
            c = j % args.cols

            try:
                if use_image_path:
                    img = open_patch_from_image_path(row, image_col)
                else:
                    img = open_patch_from_svs(row, svs_col, x_col, y_col, args.patch_size)
            except Exception as e:
                print(f"[WARN] failed to open patch {row['review_idx']}: {e}")
                img = Image.new("RGB", (args.patch_size, args.patch_size), "white")

            label_text = str(row[args.label_col])
            score_text = build_score_text(row)

            tile = draw_tile(
                img=img,
                idx_text=str(row["review_idx"]),
                label_text=label_text,
                score_text=score_text,
                tile_size=args.tile_size,
                header_h=args.header_h,
                font_big=font_big,
                font_small=font_small,
            )

            x0 = c * args.tile_size
            y0 = r * (args.tile_size + args.header_h)
            sheet.paste(tile, (x0, y0))

        sheet_path = sheet_dir / f"{args.label}_review_sheet_{s+1:03d}.jpg"
        sheet.save(sheet_path, quality=95)
        sheet_paths.append(str(sheet_path))

    # Add sheet path info to manifest
    df["review_sheet"] = ""
    df["position_in_sheet"] = ""

    for i in range(len(df)):
        s = i // n_per_sheet
        j = i % n_per_sheet
        df.loc[i, "review_sheet"] = sheet_paths[s]
        df.loc[i, "position_in_sheet"] = j + 1

    df.to_csv(manifest_path, index=False)

    print("=" * 80)
    print(f"Done.")
    print(f"Label: {args.label}")
    print(f"Candidates: {len(df)}")
    print(f"Review sheets: {sheet_dir}")
    print(f"Manifest: {manifest_path}")
    print()
    print("Manual review:")
    print(f"  Open the JPG sheets in: {sheet_dir}")
    print(f"  Fill keep=1 or keep=0 in: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()