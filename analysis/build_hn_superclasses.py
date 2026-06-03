import os
import json
import math
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageDraw, ImageFont


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def get_font(size=18):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def read_patch(svs_path, x, y, patch_level, patch_size, out_size=224):
    slide = openslide.OpenSlide(svs_path)
    try:
        patch = slide.read_region(
            (int(x), int(y)),
            int(patch_level),
            (int(patch_size), int(patch_size)),
        ).convert("RGB")
    finally:
        slide.close()

    if out_size is not None and out_size > 0:
        patch = patch.resize((out_size, out_size))
    return patch


def draw_patch_with_text(img, lines, text_h=48, border=(60, 90, 170)):
    w, h = img.size
    canvas = Image.new("RGB", (w, h + text_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = get_font(16)

    canvas.paste(img, (0, text_h))
    draw.rectangle([0, text_h, w - 1, text_h + h - 1], outline=border, width=3)

    y = 4
    for line in lines:
        draw.text((6, y), line, fill=(10, 10, 10), font=font)
        y += 18
    return canvas


def save_montage(df, out_path, title, n_show=16, patch_size=224, cols=4, sort_col="tumor_minus_max_other"):
    rows = df.sort_values(sort_col, ascending=False).head(n_show).to_dict("records")
    if len(rows) == 0:
        return

    panels = []
    for row in rows:
        patch = read_patch(
            row["svs_path"],
            row["coord_x"],
            row["coord_y"],
            row["patch_level"],
            row["patch_size"],
            out_size=patch_size,
        )
        panel = draw_patch_with_text(
            patch,
            [
                f"score={row['tumor_minus_max_other']:.4f}",
                f"cluster={int(row['cluster_id'])}",
            ],
        )
        panels.append(panel)

    cols = min(cols, len(panels))
    nrows = math.ceil(len(panels) / cols)
    pw, ph = panels[0].size
    gap = 12
    title_h = 40

    W = cols * pw + (cols - 1) * gap
    H = title_h + nrows * ph + (nrows - 1) * gap
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill=(10, 10, 10), font=get_font(24))

    for i, p in enumerate(panels):
        r = i // cols
        c = i % cols
        x = c * (pw + gap)
        y = title_h + r * (ph + gap)
        canvas.paste(p, (x, y))

    canvas.save(out_path)


def default_mapping() -> Dict[int, str]:
    """
    第一版保守映射：
    0 -> gland_like
    4 -> fibrous_dense
    2 -> blank_boundary
    1/3/5 -> mixed
    """
    return {
        0: "gland_like",
        1: "mixed",
        2: "blank_boundary",
        3: "mixed",
        4: "fibrous_dense",
        5: "mixed",
    }


def load_mapping(mapping_json: str) -> Dict[int, str]:
    if mapping_json is None:
        return default_mapping()

    with open(mapping_json, "r", encoding="utf-8") as f:
        obj = json.load(f)

    mapping = {}
    for k, v in obj.items():
        mapping[int(k)] = str(v)
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-csv", type=str, required=True)
    parser.add_argument("--features-npy", type=str, required=False, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--mapping-json", type=str, default=None)

    parser.add_argument("--n-show-per-class", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--ignore-label", type=str, default="mixed")
    args = parser.parse_args()

    safe_mkdir(args.out_dir)

    df = pd.read_csv(args.cluster_csv)
    df["svs_path"] = df["svs_path"].map(canonicalize_path)

    mapping = load_mapping(args.mapping_json)

    if "cluster_id" not in df.columns:
        raise ValueError("cluster_id column not found in cluster csv")

    df["superclass_name"] = df["cluster_id"].map(mapping).fillna("unmapped")

    labeled_csv = os.path.join(args.out_dir, "hard_negative_superclasses.csv")
    df.to_csv(labeled_csv, index=False)
    print("[Saved]", labeled_csv)

    # superclass summary
    rows: List[dict] = []
    for sname in sorted(df["superclass_name"].unique().tolist()):
        sdf = df[df["superclass_name"] == sname].copy()
        row = {
            "superclass_name": sname,
            "count": int(len(sdf)),
            "num_clusters": int(sdf["cluster_id"].nunique()),
            "mean_score": float(sdf["tumor_minus_max_other"].mean()),
            "std_score": float(sdf["tumor_minus_max_other"].std()),
            "mean_sim_tumor": float(sdf["sim_tumor"].mean()),
            "mean_sim_other_max": float(sdf["sim_other_max"].mean()),
        }

        for cid, frac in sdf["cluster_id"].value_counts(normalize=True).sort_index().to_dict().items():
            row[f"frac_cluster_{cid}"] = float(frac)

        if "nearest_role_name" in sdf.columns:
            for role_name, frac in sdf["nearest_role_name"].value_counts(normalize=True).to_dict().items():
                row[f"frac_nearest_{role_name}"] = float(frac)

        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values("count", ascending=False)
    summary_csv = os.path.join(args.out_dir, "superclass_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("[Saved]", summary_csv)
    print(summary_df)

    # keep subset for first-pass training
    keep_df = df[df["superclass_name"] != args.ignore_label].copy().reset_index(drop=True)
    keep_csv = os.path.join(args.out_dir, "hard_negative_superclasses_keep.csv")
    keep_df.to_csv(keep_csv, index=False)
    print("[Saved]", keep_csv)

    # montage per superclass
    montage_dir = os.path.join(args.out_dir, "superclass_montages")
    safe_mkdir(montage_dir)

    for sname in sorted(df["superclass_name"].unique().tolist()):
        sdf = df[df["superclass_name"] == sname].copy()
        out_path = os.path.join(montage_dir, f"{sname}.png")
        title = f"{sname} | n={len(sdf)} | mean_score={sdf['tumor_minus_max_other'].mean():.4f}"
        save_montage(
            sdf,
            out_path=out_path,
            title=title,
            n_show=args.n_show_per_class,
            patch_size=args.patch_size,
            cols=4,
        )
        print("[Saved]", out_path)

    # optional: align features with keep_df
    if args.features_npy is not None:
        feats = np.load(args.features_npy)
        if len(feats) != len(df):
            raise ValueError(f"feature rows ({len(feats)}) != csv rows ({len(df)})")

        keep_mask = (df["superclass_name"] != args.ignore_label).to_numpy()
        keep_feats = feats[keep_mask]
        keep_feat_path = os.path.join(args.out_dir, "hard_negative_superclasses_keep_features.npy")
        np.save(keep_feat_path, keep_feats)
        print("[Saved]", keep_feat_path)

    mapping_path = os.path.join(args.out_dir, "superclass_mapping_used.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=2, ensure_ascii=False)
    print("[Saved]", mapping_path)


if __name__ == "__main__":
    main()