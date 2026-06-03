import os
import math
import argparse
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import pandas as pd
import openslide

from PIL import Image, ImageDraw, ImageFont, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# utils
# =========================================================
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


def try_get_font(size=16):
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


# =========================================================
# merge baseline & spatial
# =========================================================
PATCH_KEY_COLS = [
    "slide_id",
    "svs_path",
    "coord_x",
    "coord_y",
    "patch_level",
    "patch_size",
]

OPTIONAL_KEY_COLS = [
    "coord_idx",
]


def prepare_df(df: pd.DataFrame):
    df = df.copy()
    if "svs_path" in df.columns:
        df["svs_path"] = df["svs_path"].map(canonicalize_path)
    return df


def merge_patch_csvs(
    baseline_csv: str,
    spatial_csv: str,
):
    df_base = pd.read_csv(baseline_csv)
    df_sp = pd.read_csv(spatial_csv)

    df_base = prepare_df(df_base)
    df_sp = prepare_df(df_sp)

    missing_base = [c for c in PATCH_KEY_COLS if c not in df_base.columns]
    missing_sp = [c for c in PATCH_KEY_COLS if c not in df_sp.columns]
    if missing_base:
        raise ValueError(f"Baseline csv missing required cols: {missing_base}")
    if missing_sp:
        raise ValueError(f"Spatial csv missing required cols: {missing_sp}")

    join_cols = list(PATCH_KEY_COLS)
    for c in OPTIONAL_KEY_COLS:
        if c in df_base.columns and c in df_sp.columns:
            join_cols.append(c)

    merged = df_base.merge(
        df_sp,
        on=join_cols,
        how="inner",
        suffixes=("_base", "_sp"),
    )

    if len(merged) == 0:
        raise ValueError("No matched patches found between baseline and spatial csv.")

    # slide_label
    if "slide_label_base" in merged.columns:
        merged["slide_label"] = merged["slide_label_base"]
    elif "slide_label_sp" in merged.columns:
        merged["slide_label"] = merged["slide_label_sp"]
    else:
        raise ValueError("No slide_label found after merge.")

    # key delta metrics
    delta_specs = [
        ("tumor_minus_max_other", "delta_tumor_minus_max_other"),
        ("sim_tumor", "delta_sim_tumor"),
        ("sim_other_max", "delta_sim_other_max"),
        ("nearest_hn_sim", "delta_nearest_hn_sim"),
    ]
    for src, dst in delta_specs:
        c0 = f"{src}_base"
        c1 = f"{src}_sp"
        if c0 in merged.columns and c1 in merged.columns:
            merged[dst] = merged[c1] - merged[c0]

    # top-k status
    if "is_topk_patch_within_slide_base" in merged.columns:
        merged["is_topk_base"] = merged["is_topk_patch_within_slide_base"].astype(int)
    else:
        merged["is_topk_base"] = 0

    if "is_topk_patch_within_slide_sp" in merged.columns:
        merged["is_topk_sp"] = merged["is_topk_patch_within_slide_sp"].astype(int)
    else:
        merged["is_topk_sp"] = 0

    merged["is_topk_either"] = (
        (merged["is_topk_base"] > 0) | (merged["is_topk_sp"] > 0)
    ).astype(int)

    return merged


# =========================================================
# grouping / structure label
# =========================================================
def infer_group_col(df: pd.DataFrame):
    candidates = [
        "nearest_hn_class_base",
        "nearest_hn_class_sp",
        "nearest_hn_class",
        "subclass_name",
        "subclass_id",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


# =========================================================
# filtering / ranking
# =========================================================
def select_most_suppressed(
    merged: pd.DataFrame,
    slide_label: int,
    top_n: int = 24,
    only_topk_either: bool = True,
    baseline_min_score: float = None,
    spatial_min_score: float = None,
):
    df = merged.copy()
    df = df[df["slide_label"] == slide_label]

    if only_topk_either:
        df = df[df["is_topk_either"] == 1]

    if baseline_min_score is not None and "tumor_minus_max_other_base" in df.columns:
        df = df[df["tumor_minus_max_other_base"] >= baseline_min_score]

    if spatial_min_score is not None and "tumor_minus_max_other_sp" in df.columns:
        df = df[df["tumor_minus_max_other_sp"] >= spatial_min_score]

    if "delta_tumor_minus_max_other" not in df.columns:
        raise ValueError("delta_tumor_minus_max_other not found.")

    df = df.sort_values("delta_tumor_minus_max_other", ascending=True).reset_index(drop=True)
    return df.head(top_n).reset_index(drop=True)


def select_suppressed_pool(
    merged: pd.DataFrame,
    slide_label: int,
    only_topk_either: bool = True,
    baseline_min_score: float = None,
    spatial_min_score: float = None,
    delta_score_thresh: float = None,
    bottom_frac: float = None,
):
    df = merged.copy()
    df = df[df["slide_label"] == slide_label]

    if only_topk_either:
        df = df[df["is_topk_either"] == 1]

    if baseline_min_score is not None and "tumor_minus_max_other_base" in df.columns:
        df = df[df["tumor_minus_max_other_base"] >= baseline_min_score]

    if spatial_min_score is not None and "tumor_minus_max_other_sp" in df.columns:
        df = df[df["tumor_minus_max_other_sp"] >= spatial_min_score]

    if "delta_tumor_minus_max_other" not in df.columns:
        raise ValueError("delta_tumor_minus_max_other not found.")

    if delta_score_thresh is not None:
        df = df[df["delta_tumor_minus_max_other"] <= delta_score_thresh]

    if bottom_frac is not None:
        if not (0.0 < bottom_frac < 1.0):
            raise ValueError("bottom_frac must be in (0, 1).")
        q = df["delta_tumor_minus_max_other"].quantile(bottom_frac)
        df = df[df["delta_tumor_minus_max_other"] <= q]

    return df.reset_index(drop=True)


# =========================================================
# stats
# =========================================================
def summarize_suppressed_by_class(df: pd.DataFrame, group_col: str):
    if group_col is None or group_col not in df.columns or len(df) == 0:
        return pd.DataFrame()

    sdf = df.copy()
    sdf[group_col] = sdf[group_col].fillna("NA").astype(str)

    rows = []
    total = len(sdf)

    for cls_name, g in sdf.groupby(group_col):
        row = {
            "class_name": cls_name,
            "count": int(len(g)),
            "fraction": float(len(g) / max(total, 1)),
        }

        for col in [
            "tumor_minus_max_other_base",
            "tumor_minus_max_other_sp",
            "delta_tumor_minus_max_other",
            "nearest_hn_sim_base",
            "nearest_hn_sim_sp",
            "delta_nearest_hn_sim",
            "sim_tumor_base",
            "sim_tumor_sp",
            "sim_other_max_base",
            "sim_other_max_sp",
            "rank_within_slide_by_tumor_evidence_base",
            "rank_within_slide_by_tumor_evidence_sp",
        ]:
            if col in g.columns:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std())

        rows.append(row)

    out = pd.DataFrame(rows)
    if "delta_tumor_minus_max_other_mean" in out.columns:
        out = out.sort_values(
            ["count", "delta_tumor_minus_max_other_mean"],
            ascending=[False, True]
        ).reset_index(drop=True)
    else:
        out = out.sort_values("count", ascending=False).reset_index(drop=True)

    return out


def compare_pos_neg_class_stats(
    neg_stats: pd.DataFrame,
    pos_stats: pd.DataFrame,
):
    if len(neg_stats) == 0 and len(pos_stats) == 0:
        return pd.DataFrame()

    neg_stats = neg_stats.copy()
    pos_stats = pos_stats.copy()

    neg_stats = neg_stats.rename(columns={
        "count": "neg_count",
        "fraction": "neg_fraction",
        "delta_tumor_minus_max_other_mean": "neg_mean_delta_score",
        "tumor_minus_max_other_base_mean": "neg_mean_base_score",
        "tumor_minus_max_other_sp_mean": "neg_mean_sp_score",
    })

    pos_stats = pos_stats.rename(columns={
        "count": "pos_count",
        "fraction": "pos_fraction",
        "delta_tumor_minus_max_other_mean": "pos_mean_delta_score",
        "tumor_minus_max_other_base_mean": "pos_mean_base_score",
        "tumor_minus_max_other_sp_mean": "pos_mean_sp_score",
    })

    keep_neg = [
        "class_name",
        "neg_count",
        "neg_fraction",
        "neg_mean_delta_score",
        "neg_mean_base_score",
        "neg_mean_sp_score",
    ]
    keep_pos = [
        "class_name",
        "pos_count",
        "pos_fraction",
        "pos_mean_delta_score",
        "pos_mean_base_score",
        "pos_mean_sp_score",
    ]

    neg_stats = neg_stats[[c for c in keep_neg if c in neg_stats.columns]]
    pos_stats = pos_stats[[c for c in keep_pos if c in pos_stats.columns]]

    merged = neg_stats.merge(pos_stats, on="class_name", how="outer").fillna(0.0)

    if "neg_fraction" in merged.columns and "pos_fraction" in merged.columns:
        merged["neg_minus_pos_fraction"] = merged["neg_fraction"] - merged["pos_fraction"]

    if "neg_mean_delta_score" in merged.columns and "pos_mean_delta_score" in merged.columns:
        merged["neg_minus_pos_delta_score"] = (
            merged["neg_mean_delta_score"] - merged["pos_mean_delta_score"]
        )

    merged = merged.sort_values(
        ["neg_minus_pos_fraction", "neg_count"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return merged


# =========================================================
# visualization
# =========================================================
def draw_single_tile(img: Image.Image, row: pd.Series, mode: str, tile_w=224, header_h=132):
    suffix = "_base" if mode == "base" else "_sp"

    canvas = Image.new("RGB", (tile_w, tile_w + header_h), color=(255, 255, 255))
    canvas.paste(img, (0, header_h))

    draw = ImageDraw.Draw(canvas)
    font_title = try_get_font(16)
    font_text = try_get_font(14)

    slide_label = safe_int(row.get("slide_label", -1), -1)
    label_name = "pos" if slide_label == 1 else ("neg" if slide_label == 0 else "unk")

    nearest_hn_class = str(row.get(f"nearest_hn_class{suffix}", "NA"))
    score = safe_float(row.get(f"tumor_minus_max_other{suffix}", 0.0))
    hn = safe_float(row.get(f"nearest_hn_sim{suffix}", 0.0))
    sim_t = safe_float(row.get(f"sim_tumor{suffix}", 0.0))
    sim_o = safe_float(row.get(f"sim_other_max{suffix}", 0.0))
    rank = safe_float(row.get(f"rank_within_slide_by_tumor_evidence{suffix}", -1))
    topk = safe_int(row.get("is_topk_base" if mode == "base" else "is_topk_sp", 0), 0)
    slide_id = str(row.get("slide_id", "NA"))

    title = f"{mode} | {label_name} | {nearest_hn_class}"
    lines = [
        title,
        f"score={score:.3f} hn={hn:.3f}",
        f"tumor={sim_t:.3f} other={sim_o:.3f}",
        f"rank={rank:.0f} topk={topk}",
        f"{slide_id[:30]}",
    ]

    y = 6
    for i, line in enumerate(lines):
        font = font_title if i == 0 else font_text
        draw.text((6, y), line, fill=(0, 0, 0), font=font)
        y += 22

    border_color = (90, 90, 90) if mode == "base" else (0, 128, 0)
    draw.rectangle([0, 0, tile_w - 1, tile_w + header_h - 1], outline=border_color, width=4)
    return canvas


def draw_comparison_tile(
    img: Image.Image,
    row: pd.Series,
    tile_w=224,
    header_h=164,
):
    canvas = Image.new("RGB", (tile_w, tile_w + header_h), color=(255, 255, 255))
    canvas.paste(img, (0, header_h))

    draw = ImageDraw.Draw(canvas)
    font_title = try_get_font(16)
    font_text = try_get_font(14)

    slide_label = safe_int(row.get("slide_label", -1), -1)
    label_name = "pos" if slide_label == 1 else ("neg" if slide_label == 0 else "unk")
    slide_id = str(row.get("slide_id", "NA"))

    score_b = safe_float(row.get("tumor_minus_max_other_base", 0.0))
    score_s = safe_float(row.get("tumor_minus_max_other_sp", 0.0))
    d_score = safe_float(row.get("delta_tumor_minus_max_other", 0.0))

    hn_b = safe_float(row.get("nearest_hn_sim_base", 0.0))
    hn_s = safe_float(row.get("nearest_hn_sim_sp", 0.0))
    d_hn = safe_float(row.get("delta_nearest_hn_sim", 0.0))

    r_b = safe_float(row.get("rank_within_slide_by_tumor_evidence_base", -1))
    r_s = safe_float(row.get("rank_within_slide_by_tumor_evidence_sp", -1))

    topk_b = safe_int(row.get("is_topk_base", 0), 0)
    topk_s = safe_int(row.get("is_topk_sp", 0), 0)

    hn_class_b = str(row.get("nearest_hn_class_base", "NA"))
    hn_class_s = str(row.get("nearest_hn_class_sp", "NA"))

    lines = [
        f"compare | {label_name}",
        f"d_score={d_score:.3f}  d_hn={d_hn:.3f}",
        f"base: score={score_b:.3f} hn={hn_b:.3f}",
        f"sp:   score={score_s:.3f} hn={hn_s:.3f}",
        f"rank: {r_b:.0f}->{r_s:.0f}  topk: {topk_b}->{topk_s}",
        f"hn: {hn_class_b} -> {hn_class_s}",
        f"{slide_id[:30]}",
    ]

    y = 6
    for i, line in enumerate(lines):
        font = font_title if i == 0 else font_text
        draw.text((6, y), line, fill=(0, 0, 0), font=font)
        y += 22

    border_color = (255, 140, 0) if slide_label == 1 else (220, 20, 60)
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


def save_patch_triplets(
    df: pd.DataFrame,
    out_dir: str,
    prefix: str,
    patch_out_size: int = 224,
    grid_cols: int = 4,
):
    os.makedirs(out_dir, exist_ok=True)

    base_tiles = []
    sp_tiles = []
    cmp_tiles = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Render {prefix}"):
        try:
            img = load_patch_image(
                svs_path=row["svs_path"],
                x=row["coord_x"],
                y=row["coord_y"],
                patch_level=row["patch_level"],
                patch_size=row["patch_size"],
                out_size=patch_out_size,
            )
        except Exception as e:
            print(f"[Warn] failed to load patch: {e}")
            continue

        base_tiles.append(draw_single_tile(img, row, mode="base", tile_w=patch_out_size))
        sp_tiles.append(draw_single_tile(img, row, mode="sp", tile_w=patch_out_size))
        cmp_tiles.append(draw_comparison_tile(img, row, tile_w=patch_out_size))

    if len(base_tiles) == 0:
        print(f"[Skip] no tiles rendered for {prefix}")
        return

    base_grid = make_grid(base_tiles, ncols=grid_cols)
    sp_grid = make_grid(sp_tiles, ncols=grid_cols)
    cmp_grid = make_grid(cmp_tiles, ncols=grid_cols)

    base_path = os.path.join(out_dir, f"{prefix}_baseline.png")
    sp_path = os.path.join(out_dir, f"{prefix}_spatial.png")
    cmp_path = os.path.join(out_dir, f"{prefix}_comparison.png")

    base_grid.save(base_path)
    sp_grid.save(sp_path)
    cmp_grid.save(cmp_path)

    print(f"[Saved] {base_path}")
    print(f"[Saved] {sp_path}")
    print(f"[Saved] {cmp_path}")


# =========================================================
# export tables
# =========================================================
def save_summary_tables(
    merged: pd.DataFrame,
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    out_dir: str,
    suppressed_bottom_frac: float = 0.10,
    suppressed_delta_thresh: float = None,
    only_topk_either: bool = True,
    baseline_min_score: float = None,
    spatial_min_score: float = None,
):
    os.makedirs(out_dir, exist_ok=True)

    merged.to_csv(os.path.join(out_dir, "merged_patch_delta.csv"), index=False)
    pos_df.to_csv(os.path.join(out_dir, "most_suppressed_positive.csv"), index=False)
    neg_df.to_csv(os.path.join(out_dir, "most_suppressed_negative.csv"), index=False)

    summary_rows = []
    for label_val, label_name in [(0, "negative"), (1, "positive")]:
        sdf = merged[merged["slide_label"] == label_val]
        if len(sdf) == 0:
            continue

        row = {
            "group": label_name,
            "count": int(len(sdf)),
        }

        for col in [
            "delta_tumor_minus_max_other",
            "delta_sim_tumor",
            "delta_sim_other_max",
            "delta_nearest_hn_sim",
        ]:
            if col in sdf.columns:
                row[f"{col}_mean"] = float(sdf[col].mean())
                row[f"{col}_std"] = float(sdf[col].std())

        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_dir, "delta_summary.csv"),
        index=False,
    )

    # ===== new: suppressed pool / class-aware stats =====
    group_col = infer_group_col(merged)

    neg_pool = select_suppressed_pool(
        merged=merged,
        slide_label=0,
        only_topk_either=only_topk_either,
        baseline_min_score=baseline_min_score,
        spatial_min_score=spatial_min_score,
        delta_score_thresh=suppressed_delta_thresh,
        bottom_frac=suppressed_bottom_frac,
    )
    pos_pool = select_suppressed_pool(
        merged=merged,
        slide_label=1,
        only_topk_either=only_topk_either,
        baseline_min_score=baseline_min_score,
        spatial_min_score=spatial_min_score,
        delta_score_thresh=suppressed_delta_thresh,
        bottom_frac=suppressed_bottom_frac,
    )

    neg_pool.to_csv(os.path.join(out_dir, "negative_suppressed_pool.csv"), index=False)
    pos_pool.to_csv(os.path.join(out_dir, "positive_suppressed_pool.csv"), index=False)

    neg_stats = summarize_suppressed_by_class(neg_pool, group_col)
    pos_stats = summarize_suppressed_by_class(pos_pool, group_col)
    cmp_stats = compare_pos_neg_class_stats(neg_stats, pos_stats)

    neg_stats.to_csv(os.path.join(out_dir, "negative_suppressed_class_stats.csv"), index=False)
    pos_stats.to_csv(os.path.join(out_dir, "positive_suppressed_class_stats.csv"), index=False)
    cmp_stats.to_csv(os.path.join(out_dir, "pos_neg_class_comparison.csv"), index=False)

    print(f"[Saved] {os.path.join(out_dir, 'delta_summary.csv')}")
    print(f"[Saved] {os.path.join(out_dir, 'negative_suppressed_pool.csv')}")
    print(f"[Saved] {os.path.join(out_dir, 'positive_suppressed_pool.csv')}")
    print(f"[Saved] {os.path.join(out_dir, 'negative_suppressed_class_stats.csv')}")
    print(f"[Saved] {os.path.join(out_dir, 'positive_suppressed_class_stats.csv')}")
    print(f"[Saved] {os.path.join(out_dir, 'pos_neg_class_comparison.csv')}")


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--baseline-csv", type=str, required=True)
    parser.add_argument("--spatial-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--top-n-pos", type=int, default=24)
    parser.add_argument("--top-n-neg", type=int, default=24)

    parser.add_argument("--only-topk-either", action="store_true")
    parser.add_argument("--baseline-min-score", type=float, default=None)
    parser.add_argument("--spatial-min-score", type=float, default=None)

    parser.add_argument("--suppressed-bottom-frac", type=float, default=0.10)
    parser.add_argument("--suppressed-delta-thresh", type=float, default=None)

    parser.add_argument("--patch-out-size", type=int, default=224)
    parser.add_argument("--grid-cols", type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    merged = merge_patch_csvs(
        baseline_csv=args.baseline_csv,
        spatial_csv=args.spatial_csv,
    )

    pos_df = select_most_suppressed(
        merged=merged,
        slide_label=1,
        top_n=args.top_n_pos,
        only_topk_either=args.only_topk_either,
        baseline_min_score=args.baseline_min_score,
        spatial_min_score=args.spatial_min_score,
    )

    neg_df = select_most_suppressed(
        merged=merged,
        slide_label=0,
        top_n=args.top_n_neg,
        only_topk_either=args.only_topk_either,
        baseline_min_score=args.baseline_min_score,
        spatial_min_score=args.spatial_min_score,
    )

    save_summary_tables(
        merged=merged,
        pos_df=pos_df,
        neg_df=neg_df,
        out_dir=args.out_dir,
        suppressed_bottom_frac=args.suppressed_bottom_frac,
        suppressed_delta_thresh=args.suppressed_delta_thresh,
        only_topk_either=args.only_topk_either,
        baseline_min_score=args.baseline_min_score,
        spatial_min_score=args.spatial_min_score,
    )

    save_patch_triplets(
        df=pos_df,
        out_dir=args.out_dir,
        prefix="most_suppressed_positive",
        patch_out_size=args.patch_out_size,
        grid_cols=args.grid_cols,
    )

    save_patch_triplets(
        df=neg_df,
        out_dir=args.out_dir,
        prefix="most_suppressed_negative",
        patch_out_size=args.patch_out_size,
        grid_cols=args.grid_cols,
    )


if __name__ == "__main__":
    main()