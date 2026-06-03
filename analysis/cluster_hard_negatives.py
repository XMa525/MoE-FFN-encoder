import os
import json
import math
import argparse
from typing import List

import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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


def save_cluster_montage(cluster_df, out_path, title, n_show=16, patch_size=224, cols=4):
    rows = cluster_df.head(n_show).to_dict("records")
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
                f"{row['nearest_role_name']}",
            ],
        )
        panels.append(panel)

    if len(panels) == 0:
        return

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--n-clusters", type=int, default=6)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--n-show-per-cluster", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=224)
    args = parser.parse_args()

    safe_mkdir(args.out_dir)

    df = pd.read_csv(args.csv)
    df["svs_path"] = df["svs_path"].map(canonicalize_path)
    feats = np.load(args.features)

    if len(df) != len(feats):
        raise ValueError(f"csv rows ({len(df)}) != features rows ({len(feats)})")

    print("[Loaded] rows =", len(df), "feat_shape =", feats.shape)

    # PCA
    pca_dim = min(args.pca_dim, feats.shape[1], max(2, len(df) - 1))
    pca = PCA(n_components=pca_dim, random_state=args.random_state)
    feat_pca = pca.fit_transform(feats)

    # KMeans
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state, n_init=10)
    cluster_id = kmeans.fit_predict(feat_pca)
    df["cluster_id"] = cluster_id

    # t-SNE for visualization
    tsne = TSNE(
        n_components=2,
        perplexity=min(args.tsne_perplexity, max(5.0, len(df) / 5.0)),
        random_state=args.random_state,
        init="pca",
        learning_rate="auto",
    )
    emb2d = tsne.fit_transform(feat_pca)
    df["tsne_x"] = emb2d[:, 0]
    df["tsne_y"] = emb2d[:, 1]

    # save labeled csv
    labeled_csv = os.path.join(args.out_dir, "hard_negative_clusters.csv")
    df.to_csv(labeled_csv, index=False)
    print("[Saved]", labeled_csv)

    # cluster summary
    summary_rows = []
    for cid in sorted(df["cluster_id"].unique().tolist()):
        sdf = df[df["cluster_id"] == cid].copy()
        row = {
            "cluster_id": int(cid),
            "count": int(len(sdf)),
            "mean_score": float(sdf["tumor_minus_max_other"].mean()),
            "std_score": float(sdf["tumor_minus_max_other"].std()),
            "mean_sim_tumor": float(sdf["sim_tumor"].mean()),
            "mean_sim_other_max": float(sdf["sim_other_max"].mean()),
        }
        for role_name, frac in sdf["nearest_role_name"].value_counts(normalize=True).to_dict().items():
            row[f"frac_nearest_{role_name}"] = float(frac)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("count", ascending=False)
    summary_csv = os.path.join(args.out_dir, "cluster_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("[Saved]", summary_csv)

    # scatter by cluster
    plt.figure(figsize=(8, 6))
    for cid in sorted(df["cluster_id"].unique().tolist()):
        sdf = df[df["cluster_id"] == cid]
        plt.scatter(sdf["tsne_x"], sdf["tsne_y"], s=8, alpha=0.7, label=f"C{cid}")
    plt.legend(markerscale=2, fontsize=8)
    plt.title("Hard Negative Clusters (t-SNE)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "hard_negative_tsne_by_cluster.png"), dpi=200)
    plt.close()

    # scatter by nearest role
    plt.figure(figsize=(8, 6))
    for role_name in sorted(df["nearest_role_name"].unique().tolist()):
        sdf = df[df["nearest_role_name"] == role_name]
        plt.scatter(sdf["tsne_x"], sdf["tsne_y"], s=8, alpha=0.7, label=role_name)
    plt.legend(markerscale=2, fontsize=8)
    plt.title("Hard Negatives colored by nearest role")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "hard_negative_tsne_by_role.png"), dpi=200)
    plt.close()

    # scatter by score
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df["tsne_x"], df["tsne_y"], c=df["tumor_minus_max_other"], s=8, alpha=0.8)
    plt.colorbar(sc)
    plt.title("Hard Negatives colored by tumor-minus-max-other")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "hard_negative_tsne_by_score.png"), dpi=200)
    plt.close()

    # per-cluster montage
    cluster_dir = os.path.join(args.out_dir, "cluster_montages")
    safe_mkdir(cluster_dir)

    for cid in sorted(df["cluster_id"].unique().tolist()):
        sdf = (
            df[df["cluster_id"] == cid]
            .sort_values("tumor_minus_max_other", ascending=False)
            .reset_index(drop=True)
        )
        title = f"Cluster {cid} | n={len(sdf)} | mean_score={sdf['tumor_minus_max_other'].mean():.4f}"
        out_path = os.path.join(cluster_dir, f"cluster_{cid}.png")
        save_cluster_montage(
            sdf,
            out_path=out_path,
            title=title,
            n_show=args.n_show_per_cluster,
            patch_size=args.patch_size,
            cols=4,
        )
        print("[Saved]", out_path)

    meta = {
        "n_rows": int(len(df)),
        "feature_dim": int(feats.shape[1]),
        "n_clusters": int(args.n_clusters),
        "pca_dim": int(pca_dim),
        "tsne_perplexity": float(min(args.tsne_perplexity, max(5.0, len(df) / 5.0))),
    }
    with open(os.path.join(args.out_dir, "cluster_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(summary_df)


if __name__ == "__main__":
    main()