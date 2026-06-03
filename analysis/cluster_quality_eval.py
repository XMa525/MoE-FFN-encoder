import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
from tqdm import tqdm


# =========================
# Utility
# =========================

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_image_rgb(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception:
        return None


def resize_for_grid(img: Image.Image, size: int) -> Image.Image:
    return ImageOps.fit(img, (size, size), method=Image.Resampling.BILINEAR)


def make_montage(
    image_paths: List[str],
    save_path: str,
    title: str,
    thumb_size: int = 128,
    ncols: int = 6,
    max_items: int = 36,
    annotate: Optional[List[str]] = None,
) -> None:
    image_paths = image_paths[:max_items]
    if annotate is not None:
        annotate = annotate[:max_items]

    n = len(image_paths)
    if n == 0:
        return

    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    top_margin = 40
    pad = 8

    canvas_w = ncols * thumb_size + (ncols + 1) * pad
    canvas_h = nrows * thumb_size + (nrows + 1) * pad + top_margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    draw.text((10, 10), title, fill=(20, 20, 20), font=font)

    for i, p in enumerate(image_paths):
        r = i // ncols
        c = i % ncols
        x = pad + c * (thumb_size + pad)
        y = top_margin + pad + r * (thumb_size + pad)

        try:
            img = Image.open(p).convert("RGB")
            img = resize_for_grid(img, thumb_size)
        except Exception:
            img = Image.new("RGB", (thumb_size, thumb_size), (220, 220, 220))
            d = ImageDraw.Draw(img)
            d.text((10, 10), "ERR", fill=(255, 0, 0), font=font)

        canvas.paste(img, (x, y))

        if annotate is not None and i < len(annotate):
            draw.rectangle([x, y, x + thumb_size - 1, y + 16], fill=(255, 255, 255))
            draw.text((x + 2, y + 2), annotate[i], fill=(0, 0, 0), font=font)

    canvas.save(save_path)


# =========================
# Feature extraction
# =========================

@dataclass
class PatchStats:
    path: str
    cluster_id: int
    exists: int
    width: int
    height: int

    dominant_ratio: float
    non_bg_ratio: float
    num_unique_clusters: float

    foreground_ratio: float
    white_ratio: float
    purple_ratio: float
    saturation_mean: float
    brightness_mean: float
    grayscale_std: float
    laplacian_var: float
    edge_density: float

    hematoxylin_mean: float
    hematoxylin_std: float

    score: Optional[float] = None


def rgb_to_hed_approx(img_rgb: np.ndarray) -> np.ndarray:
    """
    Approximate HED conversion using Ruifrok-like OD transform.
    We mainly use H channel as a coarse hematoxylin proxy.
    """
    img = img_rgb.astype(np.float32) + 1.0
    img = img / 255.0
    od = -np.log(img)

    stain_matrix = np.array([
        [0.650, 0.704, 0.286],  # Hematoxylin
        [0.072, 0.990, 0.105],  # Eosin
        [0.268, 0.570, 0.776],  # DAB-ish / residual
    ], dtype=np.float32)

    try:
        inv = np.linalg.pinv(stain_matrix.T)
        hed = od.reshape(-1, 3) @ inv.T
        hed = hed.reshape(img_rgb.shape[0], img_rgb.shape[1], 3)
    except np.linalg.LinAlgError:
        hed = np.zeros((*img_rgb.shape[:2], 3), dtype=np.float32)

    return hed


def compute_patch_stats(
    path: str,
    cluster_id: int,
    dominant_ratio: float = 1.0,
    non_bg_ratio: float = 1.0,
    num_unique_clusters: float = 1.0,
    score: Optional[float] = None,
    white_thresh: int = 220,
    sat_foreground_thresh: int = 15,
    purple_h_low: int = 110,
    purple_h_high: int = 170,
    purple_s_low: int = 20,
) -> PatchStats:
    img_rgb = read_image_rgb(path)
    if img_rgb is None:
        return PatchStats(
            path=path,
            cluster_id=int(cluster_id),
            exists=0,
            width=0,
            height=0,
            dominant_ratio=float(dominant_ratio),
            non_bg_ratio=float(non_bg_ratio),
            num_unique_clusters=float(num_unique_clusters),
            foreground_ratio=0.0,
            white_ratio=1.0,
            purple_ratio=0.0,
            saturation_mean=0.0,
            brightness_mean=0.0,
            grayscale_std=0.0,
            laplacian_var=0.0,
            edge_density=0.0,
            hematoxylin_mean=0.0,
            hematoxylin_std=0.0,
            score=score,
        )

    h, w = img_rgb.shape[:2]

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    white_mask = (gray >= white_thresh) & (S <= 20)
    white_ratio = float(white_mask.mean())

    foreground_mask = (~white_mask) & (S >= sat_foreground_thresh)
    foreground_ratio = float(foreground_mask.mean())

    purple_mask = (H >= purple_h_low) & (H <= purple_h_high) & (S >= purple_s_low)
    purple_ratio = float(purple_mask.mean())

    saturation_mean = float(S.mean())
    brightness_mean = float(V.mean())
    grayscale_std = float(gray.std())

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    laplacian_var = float(lap.var())

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())

    hed = rgb_to_hed_approx(img_rgb)
    hematoxylin = hed[:, :, 0]
    hematoxylin_mean = float(np.mean(hematoxylin))
    hematoxylin_std = float(np.std(hematoxylin))

    return PatchStats(
        path=path,
        cluster_id=int(cluster_id),
        exists=1,
        width=w,
        height=h,
        dominant_ratio=float(dominant_ratio),
        non_bg_ratio=float(non_bg_ratio),
        num_unique_clusters=float(num_unique_clusters),
        foreground_ratio=foreground_ratio,
        white_ratio=white_ratio,
        purple_ratio=purple_ratio,
        saturation_mean=saturation_mean,
        brightness_mean=brightness_mean,
        grayscale_std=grayscale_std,
        laplacian_var=laplacian_var,
        edge_density=edge_density,
        hematoxylin_mean=hematoxylin_mean,
        hematoxylin_std=hematoxylin_std,
        score=score,
    )


# =========================
# Cluster scoring
# =========================

def robust_mean_std(x: pd.Series) -> Tuple[float, float]:
    if len(x) == 0:
        return 0.0, 0.0
    return float(x.mean()), float(x.std(ddof=0))


def compute_cluster_prototype_consistency(df_cluster: pd.DataFrame) -> float:
    """
    A simple proxy:
    lower std on a few visual stats => higher consistency.
    Higher returned score = more consistent.
    """
    if len(df_cluster) < 5:
        return 0.0

    stds = []
    for col in [
        "foreground_ratio",
        "purple_ratio",
        "saturation_mean",
        "grayscale_std",
        "laplacian_var",
        "hematoxylin_mean",
        "dominant_ratio",
    ]:
        vals = df_cluster[col].dropna()
        if len(vals) > 0:
            stds.append(float(vals.std(ddof=0)))

    if len(stds) == 0:
        return 0.0

    avg_std = np.mean(stds)
    score = 1.0 / (1.0 + avg_std)
    return float(score)


def suggest_cluster_label(row: Dict) -> Tuple[str, str]:
    fg = row["foreground_ratio_mean"]
    white = row["white_ratio_mean"]
    purple = row["purple_ratio_mean"]
    tex = row["laplacian_var_mean"]
    edge = row["edge_density_mean"]
    hmean = row["hematoxylin_mean_mean"]
    consistency = row["prototype_consistency"]
    dom_ratio = row["dominant_ratio_mean"]
    non_bg = row["non_bg_ratio_mean"]

    reasons = []

    if white > 0.75 or (fg < 0.18 and purple < 0.03 and edge < 0.03 and non_bg < 0.35):
        reasons.append("high white / low tissue / low structure / background-heavy")
        return "low_info", "; ".join(reasons)

    if (
        fg > 0.35
        and (purple > 0.05 or hmean > 0.15)
        and tex > 50
        and consistency > 0.35
        and dom_ratio > 0.65
        and non_bg > 0.5
    ):
        reasons.append("good tissue occupancy")
        reasons.append("sufficient chromatin / purple signal")
        reasons.append("reasonable texture")
        reasons.append("dominant cluster is relatively pure")
        return "high_info", "; ".join(reasons)

    reasons.append("intermediate or mixed visual statistics")
    return "mixed", "; ".join(reasons)


def summarize_clusters(stats_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cid, dfc in stats_df.groupby("cluster_id"):
        n_total = len(dfc)
        dfv = dfc[dfc["exists"] == 1].copy()
        n_valid = len(dfv)
        valid_ratio = float(n_valid / max(n_total, 1))

        summary = {
            "cluster_id": int(cid),
            "n_total": int(n_total),
            "n_valid": int(n_valid),
            "valid_ratio": valid_ratio,
        }

        cols = [
            "dominant_ratio",
            "non_bg_ratio",
            "num_unique_clusters",
            "foreground_ratio",
            "white_ratio",
            "purple_ratio",
            "saturation_mean",
            "brightness_mean",
            "grayscale_std",
            "laplacian_var",
            "edge_density",
            "hematoxylin_mean",
            "hematoxylin_std",
        ]

        for col in cols:
            m, s = robust_mean_std(dfv[col]) if n_valid > 0 else (0.0, 0.0)
            summary[f"{col}_mean"] = m
            if col in ["dominant_ratio", "foreground_ratio", "non_bg_ratio"]:
                summary[f"{col}_std"] = s

        consistency = compute_cluster_prototype_consistency(dfv)
        summary["prototype_consistency"] = consistency

        label, reason = suggest_cluster_label(summary)
        summary["suggested_label"] = label
        summary["reason"] = reason

        rows.append(summary)

    out = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)
    return out


# =========================
# Sampling for montages
# =========================

def select_prototypes(
    df_cluster: pd.DataFrame,
    top_k: int = 36,
    use_score: bool = True,
) -> pd.DataFrame:
    dfv = df_cluster[df_cluster["exists"] == 1].copy()
    if len(dfv) == 0:
        return dfv

    if use_score and "score" in dfv.columns and dfv["score"].notna().sum() > 0:
        return dfv.sort_values("score", ascending=False).head(top_k)

    # Fallback ranking:
    # prefer high dominant ratio, high non_bg, high foreground, some purple/texture
    dfv["fallback_rank"] = (
        2.5 * dfv["dominant_ratio"]
        + 2.0 * dfv["non_bg_ratio"]
        + 1.5 * dfv["foreground_ratio"]
        + 1.0 * dfv["purple_ratio"]
        + 0.001 * dfv["laplacian_var"]
        - 1.5 * dfv["white_ratio"]
        - 0.15 * dfv["num_unique_clusters"]
    )
    return dfv.sort_values("fallback_rank", ascending=False).head(top_k)


def select_diverse_samples(
    df_cluster: pd.DataFrame,
    n: int = 36,
    random_state: int = 42,
) -> pd.DataFrame:
    dfv = df_cluster[df_cluster["exists"] == 1].copy()
    if len(dfv) <= n:
        return dfv

    # stratify roughly by dominant_ratio
    if "dominant_ratio" in dfv.columns:
        dfv["dom_bin"] = pd.cut(dfv["dominant_ratio"], bins=6, labels=False, include_lowest=True)
        group_col = "dom_bin"
    else:
        dfv["fg_bin"] = pd.cut(dfv["foreground_ratio"], bins=6, labels=False, include_lowest=True)
        group_col = "fg_bin"

    samples = []
    per_bin = max(1, n // max(dfv[group_col].nunique(), 1))

    rng = np.random.default_rng(random_state)
    for _, g in dfv.groupby(group_col):
        take = min(per_bin, len(g))
        idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
        samples.append(dfv.loc[idx])

    out = pd.concat(samples, axis=0).drop_duplicates()

    if len(out) > n:
        out = out.sample(n=n, random_state=random_state)
    elif len(out) < n:
        remain = dfv.drop(index=out.index, errors="ignore")
        add_n = min(n - len(out), len(remain))
        if add_n > 0:
            out = pd.concat([out, remain.sample(n=add_n, random_state=random_state)], axis=0)

    return out.head(n)


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True,
                        help="CSV with at least columns: path, dominant_cluster. "
                             "Optional: dominant_ratio, non_bg_ratio, num_unique_clusters, score")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--path_col", type=str, default="path")
    parser.add_argument("--cluster_col", type=str, default="dominant_cluster")
    parser.add_argument("--score_col", type=str, default="score")

    parser.add_argument("--min_dominant_ratio", type=float, default=0.6)
    parser.add_argument("--min_non_bg_ratio", type=float, default=0.5)
    parser.add_argument("--max_num_unique_clusters", type=float, default=-1.0,
                        help="If > 0, only keep rows with num_unique_clusters <= this threshold")

    parser.add_argument("--exclude_clusters", type=str, default="",
                        help="Comma-separated cluster ids to exclude, e.g. '5' or '4,5'")

    parser.add_argument("--top_k", type=int, default=36)
    parser.add_argument("--sample_k", type=int, default=36)
    parser.add_argument("--thumb_size", type=int, default=128)
    parser.add_argument("--ncols", type=int, default=6)
    parser.add_argument("--save_patch_stats", action="store_true")

    args = parser.parse_args()

    safe_mkdir(args.output_dir)
    montage_dir = os.path.join(args.output_dir, "montages")
    safe_mkdir(montage_dir)

    df = pd.read_csv(args.input_csv)

    if args.path_col not in df.columns:
        raise ValueError(f"Missing path column: {args.path_col}")
    if args.cluster_col not in df.columns:
        raise ValueError(f"Missing cluster column: {args.cluster_col}")

    path_col = args.path_col
    cluster_col = args.cluster_col
    score_col = args.score_col if args.score_col in df.columns else None

    # optional filtering
    before_n = len(df)

    if "dominant_ratio" in df.columns:
        df = df[df["dominant_ratio"] >= args.min_dominant_ratio]

    if "non_bg_ratio" in df.columns:
        df = df[df["non_bg_ratio"] >= args.min_non_bg_ratio]

    if args.max_num_unique_clusters > 0 and "num_unique_clusters" in df.columns:
        df = df[df["num_unique_clusters"] <= args.max_num_unique_clusters]

    if args.exclude_clusters.strip():
        exclude_ids = [int(x) for x in args.exclude_clusters.split(",") if x.strip() != ""]
        df = df[~df[cluster_col].isin(exclude_ids)]

    df = df.reset_index(drop=True)

    after_n = len(df)
    print(f"Loaded {before_n} rows, kept {after_n} after filtering.")

    patch_stats_csv = os.path.join(args.output_dir, "patch_stats.csv")

    if args.save_patch_stats and os.path.exists(patch_stats_csv):
        print(f"Loading cached patch stats from {patch_stats_csv}")
        stats_df = pd.read_csv(patch_stats_csv)
    else:
        patch_stats: List[PatchStats] = []

        print("Computing patch-level statistics...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            path = row[path_col]
            cid = int(row[cluster_col])

            score = float(row[score_col]) if score_col is not None and pd.notna(row[score_col]) else None
            dominant_ratio = float(row["dominant_ratio"]) if "dominant_ratio" in row and pd.notna(row["dominant_ratio"]) else 1.0
            non_bg_ratio = float(row["non_bg_ratio"]) if "non_bg_ratio" in row and pd.notna(row["non_bg_ratio"]) else 1.0
            num_unique_clusters = float(row["num_unique_clusters"]) if "num_unique_clusters" in row and pd.notna(row["num_unique_clusters"]) else 1.0

            st = compute_patch_stats(
                path=path,
                cluster_id=cid,
                dominant_ratio=dominant_ratio,
                non_bg_ratio=non_bg_ratio,
                num_unique_clusters=num_unique_clusters,
                score=score,
            )
            patch_stats.append(st)

        stats_df = pd.DataFrame([asdict(x) for x in patch_stats])

        if args.save_patch_stats:
            stats_df.to_csv(patch_stats_csv, index=False)

    summary_df = summarize_clusters(stats_df)
    summary_csv = os.path.join(args.output_dir, "cluster_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    report = {
        "input_csv": args.input_csv,
        "num_rows_before_filter": int(before_n),
        "num_rows_after_filter": int(after_n),
        "num_clusters": int(summary_df["cluster_id"].nunique()) if len(summary_df) > 0 else 0,
        "filters": {
            "min_dominant_ratio": args.min_dominant_ratio,
            "min_non_bg_ratio": args.min_non_bg_ratio,
            "max_num_unique_clusters": args.max_num_unique_clusters,
            "exclude_clusters": args.exclude_clusters,
        },
        "clusters": summary_df.to_dict(orient="records"),
    }
    with open(os.path.join(args.output_dir, "cluster_quality_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if len(summary_df) > 0:
        print("\nCluster summary:")
        print(summary_df[[
            "cluster_id",
            "n_valid",
            "dominant_ratio_mean",
            "non_bg_ratio_mean",
            "foreground_ratio_mean",
            "white_ratio_mean",
            "purple_ratio_mean",
            "laplacian_var_mean",
            "prototype_consistency",
            "suggested_label",
            "reason",
        ]])
    else:
        print("\nNo clusters remaining after filtering.")

    merged = df.copy().reset_index(drop=True)
    merged["_row_id_"] = np.arange(len(merged))
    stats_df["_row_id_"] = np.arange(len(stats_df))

    keep_cols = [
        "_row_id_",
        "exists",
        "foreground_ratio",
        "white_ratio",
        "purple_ratio",
        "laplacian_var",
        "dominant_ratio",
        "non_bg_ratio",
        "num_unique_clusters",
        "path",
        "cluster_id",
        "score",
    ]
    keep_cols = [c for c in keep_cols if c in stats_df.columns]

    merged = merged.merge(
        stats_df[keep_cols],
        on="_row_id_",
        how="left",
        suffixes=("", "_stats"),
    )

    # 统一一个不会冲突的分析列名
    if path_col in merged.columns:
        merged["path_for_analysis"] = merged[path_col]
    else:
        merged["path_for_analysis"] = merged["path"]

    if cluster_col in merged.columns:
        merged["cluster_for_analysis"] = merged[cluster_col]
    else:
        merged["cluster_for_analysis"] = merged["cluster_id"]

    for cid, dfc in merged.groupby("cluster_for_analysis"):
        cid = int(cid)

        proto_df = select_prototypes(
            df_cluster=dfc,
            top_k=args.top_k,
            use_score=(score_col is not None),
        )
        proto_paths = proto_df["path_for_analysis"].tolist()
        proto_ann = []
        for _, r in proto_df.iterrows():
            txt = (
                f"dom={r.get('dominant_ratio', 0):.2f}, "
                f"nonbg={r.get('non_bg_ratio', 0):.2f}, "
                f"fg={r.get('foreground_ratio', 0):.2f}"
            )
            if pd.notna(r.get("score", np.nan)):
                txt += f", s={r['score']:.3f}"
            proto_ann.append(txt)

        make_montage(
            image_paths=proto_paths,
            save_path=os.path.join(montage_dir, f"cluster_{cid}_prototype_montage.png"),
            title=f"Cluster {cid} - Prototype Montage",
            thumb_size=args.thumb_size,
            ncols=args.ncols,
            max_items=args.top_k,
            annotate=proto_ann,
        )

        sample_df = select_diverse_samples(
            df_cluster=dfc,
            n=args.sample_k,
        )
        sample_paths = sample_df["path_for_analysis"].tolist()
        sample_ann = [
            (
                f"dom={r.get('dominant_ratio', 0):.2f}, "
                f"nonbg={r.get('non_bg_ratio', 0):.2f}, "
                f"uniq={r.get('num_unique_clusters', 0):.0f}"
            )
            for _, r in sample_df.iterrows()
        ]

        make_montage(
            image_paths=sample_paths,
            save_path=os.path.join(montage_dir, f"cluster_{cid}_sample_montage.png"),
            title=f"Cluster {cid} - Diverse Sample Montage",
            thumb_size=args.thumb_size,
            ncols=args.ncols,
            max_items=args.sample_k,
            annotate=sample_ann,
        )

    print(f"\nDone. Results saved to: {args.output_dir}")
    print(f"- Summary CSV: {summary_csv}")
    print(f"- JSON report: {os.path.join(args.output_dir, 'cluster_quality_report.json')}")
    print(f"- Montages dir: {montage_dir}")


if __name__ == "__main__":
    main()