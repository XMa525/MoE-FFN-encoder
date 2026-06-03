import os
import math
import argparse

import numpy as np
import pandas as pd
import openslide

from PIL import Image, ImageDraw, ImageFont, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def load_patch_image(svs_path, x, y, patch_level, patch_size, out_size=224):
    slide = openslide.OpenSlide(svs_path)
    try:
        img = slide.read_region(
            (int(x), int(y)),
            int(patch_level),
            (int(patch_size), int(patch_size)),
        ).convert("RGB")
    finally:
        slide.close()

    if out_size is not None and img.size != (out_size, out_size):
        img = img.resize((out_size, out_size))
    return img


def try_get_font(size=16):
    # 尽量用系统字体；失败则回退默认
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_patch_tile(
    img: Image.Image,
    row: pd.Series,
    tile_w: int = 224,
    header_h: int = 92,
):
    canvas = Image.new("RGB", (tile_w, tile_w + header_h), color=(255, 255, 255))
    canvas.paste(img, (0, header_h))

    draw = ImageDraw.Draw(canvas)
    font_title = try_get_font(16)
    font_text = try_get_font(14)

    slide_label = safe_int(row.get("slide_label", -1), -1)
    label_name = "pos" if slide_label == 1 else ("neg" if slide_label == 0 else "unk")
    nearest_hn_class = str(row.get("nearest_hn_class", "NA"))
    slide_id = str(row.get("slide_id", "NA"))

    score = safe_float(row.get("tumor_minus_max_other", 0.0))
    nearest_hn_sim = safe_float(row.get("nearest_hn_sim", 0.0))
    sim_g = safe_float(row.get("sim_to_gland_like_sub3", row.get("sim_to_gland_like", 0.0)))
    sim_f = safe_float(row.get("sim_to_fibrous_dense", 0.0))
    rank = safe_float(row.get("rank_within_slide_by_tumor_evidence", -1))

    lines = [
        f"{label_name} | {nearest_hn_class}",
        f"score={score:.3f}  hn={nearest_hn_sim:.3f}",
        f"g={sim_g:.3f}  f={sim_f:.3f}  rank={rank:.0f}",
        f"{slide_id[:28]}",
    ]

    y = 6
    for i, line in enumerate(lines):
        font = font_title if i == 0 else font_text
        draw.text((6, y), line, fill=(0, 0, 0), font=font)
        y += 20

    # 边框：正负不同色，方便肉眼区分
    if slide_label == 1:
        border_color = (30, 144, 255)   # 蓝
    elif slide_label == 0:
        border_color = (220, 20, 60)    # 红
    else:
        border_color = (120, 120, 120)

    draw.rectangle([0, 0, tile_w - 1, tile_w + header_h - 1], outline=border_color, width=4)
    return canvas


def make_grid(tiles, ncols=4, pad=12, bg=(240, 240, 240)):
    if len(tiles) == 0:
        raise ValueError("No tiles to make grid.")

    tile_w, tile_h = tiles[0].size
    n = len(tiles)
    nrows = math.ceil(n / ncols)

    grid_w = ncols * tile_w + (ncols + 1) * pad
    grid_h = nrows * tile_h + (nrows + 1) * pad

    canvas = Image.new("RGB", (grid_w, grid_h), color=bg)

    for idx, tile in enumerate(tiles):
        r = idx // ncols
        c = idx % ncols
        x = pad + c * (tile_w + pad)
        y = pad + r * (tile_h + pad)
        canvas.paste(tile, (x, y))

    return canvas


def sample_group(
    df: pd.DataFrame,
    label_val: int,
    hn_class: str,
    num_samples: int,
    sort_by: str = "tumor_minus_max_other",
    ascending: bool = False,
    random_sample: bool = False,
    min_score: float = None,
):
    sdf = df.copy()
    sdf = sdf[sdf["slide_label"] == label_val]
    sdf = sdf[sdf["nearest_hn_class"] == hn_class]

    if min_score is not None:
        sdf = sdf[sdf["tumor_minus_max_other"] >= min_score]

    if len(sdf) == 0:
        return sdf

    if random_sample:
        n = min(num_samples, len(sdf))
        return sdf.sample(n=n, random_state=42).reset_index(drop=True)

    if sort_by in sdf.columns:
        sdf = sdf.sort_values(sort_by, ascending=ascending)

    return sdf.head(num_samples).reset_index(drop=True)


def visualize_group(
    df: pd.DataFrame,
    out_path: str,
    num_samples: int = 16,
    label_val: int = 0,
    hn_class: str = "fibrous_dense",
    patch_out_size: int = 224,
    grid_cols: int = 4,
    sort_by: str = "tumor_minus_max_other",
    ascending: bool = False,
    random_sample: bool = False,
    min_score: float = None,
):
    sdf = sample_group(
        df=df,
        label_val=label_val,
        hn_class=hn_class,
        num_samples=num_samples,
        sort_by=sort_by,
        ascending=ascending,
        random_sample=random_sample,
        min_score=min_score,
    )

    if len(sdf) == 0:
        print(f"[Skip] no samples for label={label_val}, hn_class={hn_class}")
        return

    tiles = []
    for _, row in tqdm(sdf.iterrows(), total=len(sdf), desc=f"Visualize {hn_class} label={label_val}"):
        try:
            img = load_patch_image(
                svs_path=row["svs_path"],
                x=row["coord_x"],
                y=row["coord_y"],
                patch_level=row["patch_level"],
                patch_size=row["patch_size"],
                out_size=patch_out_size,
            )
            tile = draw_patch_tile(img, row, tile_w=patch_out_size)
            tiles.append(tile)
        except Exception as e:
            print(f"[Warn] failed to load patch: {e}")
            continue

    if len(tiles) == 0:
        print(f"[Skip] all samples failed for label={label_val}, hn_class={hn_class}")
        return

    grid = make_grid(tiles, ncols=grid_cols)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
    print(f"[Saved] {out_path}")


def visualize_four_way_comparison(
    df: pd.DataFrame,
    out_dir: str,
    num_samples_each: int = 12,
    patch_out_size: int = 224,
    grid_cols: int = 4,
    sort_by: str = "tumor_minus_max_other",
    ascending: bool = False,
    min_score: float = None,
):
    """
    生成 4 张图：
      1) neg + fibrous_dense
      2) pos + fibrous_dense
      3) neg + gland_like_sub3
      4) pos + gland_like_sub3
    """
    groups = [
        (0, "fibrous_dense", "neg_fibrous_dense.png"),
        (1, "fibrous_dense", "pos_fibrous_dense.png"),
        (0, "gland_like_sub3", "neg_gland_like_sub3.png"),
        (1, "gland_like_sub3", "pos_gland_like_sub3.png"),
    ]

    for label_val, hn_class, fname in groups:
        out_path = os.path.join(out_dir, fname)
        visualize_group(
            df=df,
            out_path=out_path,
            num_samples=num_samples_each,
            label_val=label_val,
            hn_class=hn_class,
            patch_out_size=patch_out_size,
            grid_cols=grid_cols,
            sort_by=sort_by,
            ascending=ascending,
            random_sample=False,
            min_score=min_score,
        )


def build_slide_level_contact_sheet(
    df: pd.DataFrame,
    out_path: str,
    slide_id: str,
    topk: int = 12,
    patch_out_size: int = 224,
    grid_cols: int = 4,
):
    sdf = df[df["slide_id"] == slide_id].copy()
    if len(sdf) == 0:
        print(f"[Skip] no rows for slide_id={slide_id}")
        return

    sdf = sdf.sort_values("tumor_minus_max_other", ascending=False).head(topk).reset_index(drop=True)

    tiles = []
    for _, row in tqdm(sdf.iterrows(), total=len(sdf), desc=f"Slide {slide_id}"):
        try:
            img = load_patch_image(
                svs_path=row["svs_path"],
                x=row["coord_x"],
                y=row["coord_y"],
                patch_level=row["patch_level"],
                patch_size=row["patch_size"],
                out_size=patch_out_size,
            )
            tile = draw_patch_tile(img, row, tile_w=patch_out_size)
            tiles.append(tile)
        except Exception as e:
            print(f"[Warn] failed to load patch: {e}")

    if len(tiles) == 0:
        print(f"[Skip] all slide patches failed for {slide_id}")
        return

    grid = make_grid(tiles, ncols=grid_cols)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path)
    print(f"[Saved] {out_path}")


def save_summary_tables(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # group summary
    rows = []
    for label_val, label_name in [(0, "negative"), (1, "positive")]:
        sdf = df[df["slide_label"] == label_val]
        if len(sdf) == 0:
            continue
        for hn_class in sorted(df["nearest_hn_class"].dropna().unique().tolist()):
            gdf = sdf[sdf["nearest_hn_class"] == hn_class]
            if len(gdf) == 0:
                continue
            rows.append({
                "group": label_name,
                "nearest_hn_class": hn_class,
                "count": int(len(gdf)),
                "mean_tumor_minus_max_other": float(gdf["tumor_minus_max_other"].mean()),
                "mean_nearest_hn_sim": float(gdf["nearest_hn_sim"].mean()),
            })

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "visualization_group_summary.csv"),
        index=False
    )

    # representative slides
    if "is_topk_patch_within_slide" in df.columns:
        top_df = df[df["is_topk_patch_within_slide"] == 1].copy()
    else:
        top_df = df.copy()

    slide_df = (
        top_df.groupby(["slide_id", "slide_label"])
        .agg(
            num_top_patches=("slide_id", "size"),
            max_score=("tumor_minus_max_other", "max"),
            mean_score=("tumor_minus_max_other", "mean"),
            mean_hn_sim=("nearest_hn_sim", "mean"),
        )
        .reset_index()
        .sort_values(["slide_label", "max_score"], ascending=[True, False])
    )
    slide_df.to_csv(os.path.join(out_dir, "representative_slides.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="patch_hn_overlap.csv")
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--num-samples-each", type=int, default=12)
    parser.add_argument("--patch-out-size", type=int, default=224)
    parser.add_argument("--grid-cols", type=int, default=4)

    parser.add_argument("--sort-by", type=str, default="tumor_minus_max_other")
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--min-score", type=float, default=None)

    parser.add_argument("--make-four-way", action="store_true")
    parser.add_argument("--slide-id", type=str, default=None)
    parser.add_argument("--slide-topk", type=int, default=12)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["svs_path"] = df["svs_path"].map(canonicalize_path)

    required_cols = [
        "slide_id", "slide_label", "svs_path",
        "coord_x", "coord_y", "patch_level", "patch_size",
        "tumor_minus_max_other", "nearest_hn_class", "nearest_hn_sim"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in csv: {missing}")

    save_summary_tables(df, args.out_dir)

    if args.make_four_way:
        visualize_four_way_comparison(
            df=df,
            out_dir=args.out_dir,
            num_samples_each=args.num_samples_each,
            patch_out_size=args.patch_out_size,
            grid_cols=args.grid_cols,
            sort_by=args.sort_by,
            ascending=args.ascending,
            min_score=args.min_score,
        )

    if args.slide_id is not None:
        build_slide_level_contact_sheet(
            df=df,
            out_path=os.path.join(args.out_dir, f"slide_{args.slide_id}_top{args.slide_topk}.png"),
            slide_id=args.slide_id,
            topk=args.slide_topk,
            patch_out_size=args.patch_out_size,
            grid_cols=args.grid_cols,
        )


if __name__ == "__main__":
    main()