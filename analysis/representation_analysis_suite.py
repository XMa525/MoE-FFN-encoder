#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# =========================================================
# utils
# =========================================================
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(seed: int, slide_id: str) -> int:
    h = hashlib.md5(str(slide_id).encode("utf-8")).hexdigest()
    return int(seed) + (int(h[:8], 16) % 100000)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / denom


# =========================================================
# io
# =========================================================
def read_split_csv(csv_path: str, split_names: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "slide_id" not in df.columns and "image_id" in df.columns:
        df["slide_id"] = df["image_id"]
    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("split csv missing label / slide_binary_label")
    required = {"slide_id", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split csv missing columns: {missing}")
    return df[df["split"].isin(split_names)].copy().reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, per_split_total: int, seed: int) -> pd.DataFrame:
    if per_split_total is None or per_split_total <= 0:
        return df.copy().reset_index(drop=True)

    parts = []
    for split, sub in df.groupby("split"):
        if len(sub) <= per_split_total:
            parts.append(sub.copy())
            continue

        labels = sorted(sub["label"].unique().tolist())
        total = len(sub)

        alloc: Dict[int, int] = {}
        for y in labels:
            alloc[y] = max(1, round(per_split_total * int((sub["label"] == y).sum()) / total))

        cur = sum(alloc.values())
        while cur > per_split_total:
            y = max(alloc, key=alloc.get)
            if alloc[y] > 1:
                alloc[y] -= 1
                cur -= 1
            else:
                break
        while cur < per_split_total:
            y = max(labels, key=lambda yy: int((sub["label"] == yy).sum()) - alloc.get(yy, 0))
            alloc[y] += 1
            cur += 1

        split_parts = []
        for y in labels:
            ss = sub[sub["label"] == y]
            k = min(len(ss), alloc[y])
            split_parts.append(ss.sample(n=k, random_state=seed))
        parts.append(pd.concat(split_parts, axis=0))

    out = pd.concat(parts, axis=0).drop_duplicates("slide_id").reset_index(drop=True)
    return out


def load_bag_feature(feature_dir: str, slide_id: str) -> Dict[str, Any]:
    path = Path(feature_dir) / f"{slide_id}.pt"
    if not path.exists():
        raise FileNotFoundError(path)

    obj = torch.load(path, map_location="cpu", weights_only=False)
    feats = obj["features"]
    coords = obj.get("coords", None)

    if torch.is_tensor(feats):
        feats = feats.cpu().numpy()
    if coords is not None and torch.is_tensor(coords):
        coords = coords.cpu().numpy()

    return {
        "features": np.asarray(feats, dtype=np.float32),
        "coords": np.asarray(coords, dtype=np.int64) if coords is not None else None,
        "slide_id": str(obj.get("slide_id", slide_id)),
        "label": int(obj.get("label", -1)),
    }


def sample_matched_patches(
    frozen_feats: np.ndarray,
    frozen_coords: Optional[np.ndarray],
    adapted_feats: np.ndarray,
    adapted_coords: Optional[np.ndarray],
    slide_id: str,
    seed: int,
    max_patches: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    rng = np.random.default_rng(stable_seed(seed, slide_id))

    if frozen_coords is not None and adapted_coords is not None:
        f_map = {tuple(map(int, c)): i for i, c in enumerate(frozen_coords)}
        a_map = {tuple(map(int, c)): i for i, c in enumerate(adapted_coords)}
        common = list(set(f_map.keys()) & set(a_map.keys()))
        if len(common) > 0:
            if len(common) > max_patches:
                sel = rng.choice(len(common), size=max_patches, replace=False)
                common = [common[i] for i in sel]
            f_idx = np.array([f_map[c] for c in common], dtype=np.int64)
            a_idx = np.array([a_map[c] for c in common], dtype=np.int64)
            return frozen_feats[f_idx], adapted_feats[a_idx], np.asarray(common, dtype=np.int64)

    n = min(len(frozen_feats), len(adapted_feats), max_patches)
    f_idx = rng.choice(len(frozen_feats), size=n, replace=False) if len(frozen_feats) > n else np.arange(n)
    a_idx = rng.choice(len(adapted_feats), size=n, replace=False) if len(adapted_feats) > n else np.arange(n)
    coords = frozen_coords[f_idx] if frozen_coords is not None else None
    return frozen_feats[f_idx], adapted_feats[a_idx], coords


# =========================================================
# score csv helpers
# =========================================================
def load_score_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or str(path).strip() == "":
        return None
    df = pd.read_csv(path)
    required = {"slide_id", "coord_x", "coord_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"score csv missing required columns: {missing}")
    return df.copy()


def merge_patch_scores(
    patch_df: pd.DataFrame,
    frozen_score_df: Optional[pd.DataFrame],
    adapted_score_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    out = patch_df.copy()

    key_cols = ["slide_id", "coord_x", "coord_y"]

    if frozen_score_df is not None:
        f_cols = [c for c in frozen_score_df.columns if c not in key_cols]
        frozen_score_df = frozen_score_df[key_cols + f_cols].copy()
        rename_map = {c: f"frozen__{c}" for c in f_cols}
        frozen_score_df = frozen_score_df.rename(columns=rename_map)
        out = out.merge(frozen_score_df, on=key_cols, how="left")

    if adapted_score_df is not None:
        a_cols = [c for c in adapted_score_df.columns if c not in key_cols]
        adapted_score_df = adapted_score_df[key_cols + a_cols].copy()
        rename_map = {c: f"adapted__{c}" for c in a_cols}
        adapted_score_df = adapted_score_df.rename(columns=rename_map)
        out = out.merge(adapted_score_df, on=key_cols, how="left")

    return out


# =========================================================
# reducers / clustering / plotting
# =========================================================
def fit_reducer(X: np.ndarray, seed: int, n_neighbors: int = 15, min_dist: float = 0.1):
    if HAS_UMAP:
        return umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=seed,
        ).fit(X)
    return PCA(n_components=2, random_state=seed).fit(X)


def transform_reducer(reducer, X: np.ndarray) -> np.ndarray:
    return reducer.transform(X)


def plot_umap_by_cluster(ax, emb: np.ndarray, cluster_ids: np.ndarray, title: str):
    uniq = sorted(np.unique(cluster_ids).tolist())
    cmap = plt.get_cmap("tab20")
    for i, cid in enumerate(uniq):
        m = cluster_ids == cid
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.70, color=cmap(i % 20), label=f"c={cid}")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_umap_by_label(ax, emb: np.ndarray, labels: np.ndarray, title: str):
    uniq = sorted(np.unique(labels).tolist())
    for y in uniq:
        m = labels == y
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.70, label=f"label={y}")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8)


# =========================================================
# probe / purity
# =========================================================
def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def run_patch_probe(
    X: np.ndarray,
    patch_df: pd.DataFrame,
    label_col: str,
    slide_train_ids: Sequence[str],
    slide_test_ids: Sequence[str],
) -> Dict[str, float]:
    train_mask = patch_df["slide_id"].isin(slide_train_ids).values
    test_mask = patch_df["slide_id"].isin(slide_test_ids).values

    y = patch_df[label_col].values.astype(int)
    valid = ~pd.isna(y)

    train_mask = train_mask & valid
    test_mask = test_mask & valid

    if train_mask.sum() < 10 or test_mask.sum() < 10:
        return {
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "auc": float("nan"),
            "acc": float("nan"),
            "f1": float("nan"),
        }

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
    )
    clf.fit(X[train_mask], y[train_mask])

    prob = clf.predict_proba(X[test_mask])[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "auc": safe_auc(y[test_mask], prob),
        "acc": float(accuracy_score(y[test_mask], pred)),
        "f1": float(f1_score(y[test_mask], pred, zero_division=0)),
    }


def knn_purity(X: np.ndarray, y: np.ndarray, k: int) -> float:
    y = np.asarray(y)
    valid = ~pd.isna(y)
    X = X[valid]
    y = y[valid]

    if len(X) <= k + 1 or len(np.unique(y)) < 2:
        return float("nan")

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nbrs.fit(X)
    idx = nbrs.kneighbors(X, return_distance=False)[:, 1:]
    same = (y[idx] == y[:, None]).mean(axis=1)
    return float(np.mean(same))


# =========================================================
# pseudo labels from score csv
# =========================================================
def build_pseudo_patch_labels(
    patch_df: pd.DataFrame,
    version_prefix: str,
    pos_quantile: float = 0.80,
    neg_quantile: float = 0.20,
) -> pd.Series:
    gap_col = f"{version_prefix}__tumor_gap"
    if gap_col not in patch_df.columns:
        return pd.Series([np.nan] * len(patch_df), index=patch_df.index)

    out = pd.Series([np.nan] * len(patch_df), index=patch_df.index, dtype=float)

    pos_mask = patch_df["label"] == 1
    neg_mask = patch_df["label"] == 0

    if pos_mask.sum() > 0:
        pos_thr = patch_df.loc[pos_mask, gap_col].quantile(pos_quantile)
        out[pos_mask & (patch_df[gap_col] >= pos_thr)] = 1.0

    if neg_mask.sum() > 0:
        neg_thr = patch_df.loc[neg_mask, gap_col].quantile(neg_quantile)
        out[neg_mask & (patch_df[gap_col] <= neg_thr)] = 0.0

    return out


# =========================================================
# cluster semantics / enrichment
# =========================================================
def summarize_cluster_semantics(
    patch_df: pd.DataFrame,
    cluster_col: str,
    prefix: str,
) -> pd.DataFrame:
    rows = []
    score_cols = [c for c in patch_df.columns if c.startswith(f"{prefix}__")]

    for cid, sub in patch_df.groupby(cluster_col):
        row = {
            "cluster_id": int(cid),
            "n": int(len(sub)),
            "label1_ratio": float((sub["label"] == 1).mean()),
        }

        for c in score_cols:
            if pd.api.types.is_numeric_dtype(sub[c]):
                row[f"mean::{c}"] = float(sub[c].mean())

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_top_score_enrichment(
    patch_df: pd.DataFrame,
    cluster_col: str,
    prefix: str,
    top_frac: float,
) -> pd.DataFrame:
    gap_col = f"{prefix}__tumor_gap"
    prob_col = f"{prefix}__tumor_prob"
    if gap_col not in patch_df.columns:
        return pd.DataFrame()

    rows = []
    for slide_id, sub in patch_df.groupby("slide_id"):
        n = len(sub)
        k = max(1, int(math.ceil(n * top_frac)))
        top_sub = sub.nlargest(k, gap_col)

        row = {
            "slide_id": slide_id,
            "label": int(sub["label"].iloc[0]),
            "n_total": int(n),
            "n_top": int(k),
            "mean_top_gap": float(top_sub[gap_col].mean()),
        }
        if prob_col in top_sub.columns:
            row["mean_top_prob"] = float(top_sub[prob_col].mean())

        top_cluster_vc = top_sub[cluster_col].value_counts(normalize=True)
        for cid, ratio in top_cluster_vc.items():
            row[f"top_cluster_ratio__{int(cid)}"] = float(ratio)

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Representation analysis suite")
    parser.add_argument("--frozen-dir", type=str, required=True)
    parser.add_argument("--adapted-dir", type=str, required=True)
    parser.add_argument("--slides-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--sample-slides-per-split", type=int, default=20)
    parser.add_argument("--max-patches-per-slide", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patch-clusters", type=int, default=10)
    parser.add_argument("--slide-clusters", type=int, default=6)
    parser.add_argument("--pca-dim-before-cluster", type=int, default=32)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--probe-test-slide-frac", type=float, default=0.30)

    parser.add_argument("--frozen-score-csv", type=str, default="")
    parser.add_argument("--adapted-score-csv", type=str, default="")
    parser.add_argument("--top-score-frac", type=float, default=0.05)

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    set_seed(args.seed)

    # -----------------------------
    # 1) sample slides
    # -----------------------------
    df = read_split_csv(args.slides_csv, args.splits)
    df = stratified_sample(df, args.sample_slides_per_split, args.seed)
    df.to_csv(Path(args.out_dir) / "sampled_slides.csv", index=False)

    # -----------------------------
    # 2) load matched patch features
    # -----------------------------
    frozen_X_list, adapted_X_list = [], []
    patch_rows = []

    slide_frozen_feats, slide_adapted_feats = [], []
    slide_meta = []

    for _, row in df.iterrows():
        slide_id = str(row["slide_id"])
        label = int(row["label"])
        split = str(row["split"])

        f_obj = load_bag_feature(args.frozen_dir, slide_id)
        a_obj = load_bag_feature(args.adapted_dir, slide_id)

        f_feats, a_feats, coords = sample_matched_patches(
            f_obj["features"], f_obj["coords"],
            a_obj["features"], a_obj["coords"],
            slide_id=slide_id,
            seed=args.seed,
            max_patches=args.max_patches_per_slide,
        )

        frozen_X_list.append(f_feats)
        adapted_X_list.append(a_feats)

        for i in range(len(f_feats)):
            patch_rows.append({
                "slide_id": slide_id,
                "split": split,
                "label": label,
                "coord_x": int(coords[i, 0]) if coords is not None else -1,
                "coord_y": int(coords[i, 1]) if coords is not None else -1,
            })

        slide_frozen_feats.append(f_feats.mean(axis=0))
        slide_adapted_feats.append(a_feats.mean(axis=0))
        slide_meta.append({
            "slide_id": slide_id,
            "split": split,
            "label": label,
            "n_instances": int(len(f_feats)),
        })

    frozen_X = np.concatenate(frozen_X_list, axis=0).astype(np.float32)
    adapted_X = np.concatenate(adapted_X_list, axis=0).astype(np.float32)
    patch_df = pd.DataFrame(patch_rows)

    slide_frozen = np.stack(slide_frozen_feats, axis=0).astype(np.float32)
    slide_adapted = np.stack(slide_adapted_feats, axis=0).astype(np.float32)
    slide_df = pd.DataFrame(slide_meta)

    # -----------------------------
    # 3) merge optional score csv
    # -----------------------------
    frozen_score_df = load_score_csv(args.frozen_score_csv)
    adapted_score_df = load_score_csv(args.adapted_score_csv)
    patch_df = merge_patch_scores(patch_df, frozen_score_df, adapted_score_df)

    # -----------------------------
    # 4) UMAP + unified cluster
    # -----------------------------
    patch_all = np.concatenate([frozen_X, adapted_X], axis=0)
    patch_all_norm = l2_normalize_np(patch_all)

    patch_pca_dim = min(args.pca_dim_before_cluster, patch_all_norm.shape[1], max(2, patch_all_norm.shape[0] - 1))
    patch_pca = PCA(n_components=patch_pca_dim, random_state=args.seed)
    patch_all_pca = patch_pca.fit_transform(patch_all_norm)

    patch_kmeans = KMeans(n_clusters=args.patch_clusters, random_state=args.seed, n_init=10)
    patch_cluster_all = patch_kmeans.fit_predict(patch_all_pca)
    patch_cluster_frozen = patch_cluster_all[:len(frozen_X)]
    patch_cluster_adapted = patch_cluster_all[len(frozen_X):]

    patch_reducer = fit_reducer(patch_all_norm, args.seed, 15, 0.1)
    frozen_patch_emb = transform_reducer(patch_reducer, l2_normalize_np(frozen_X))
    adapted_patch_emb = transform_reducer(patch_reducer, l2_normalize_np(adapted_X))

    patch_df["frozen_cluster"] = patch_cluster_frozen
    patch_df["adapted_cluster"] = patch_cluster_adapted
    patch_df["frozen_umap_x"] = frozen_patch_emb[:, 0]
    patch_df["frozen_umap_y"] = frozen_patch_emb[:, 1]
    patch_df["adapted_umap_x"] = adapted_patch_emb[:, 0]
    patch_df["adapted_umap_y"] = adapted_patch_emb[:, 1]
    patch_df.to_csv(Path(args.out_dir) / "patch_points.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_umap_by_cluster(axes[0], frozen_patch_emb, patch_cluster_frozen, "Frozen patch UMAP (cluster)")
    plot_umap_by_cluster(axes[1], adapted_patch_emb, patch_cluster_adapted, "Adapted patch UMAP (cluster)")
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "patch_umap_by_cluster.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_umap_by_label(axes[0], frozen_patch_emb, patch_df["label"].values, "Frozen patch UMAP")
    plot_umap_by_label(axes[1], adapted_patch_emb, patch_df["label"].values, "Adapted patch UMAP")
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "patch_umap_by_label.png", dpi=220)
    plt.close(fig)

    slide_all = np.concatenate([slide_frozen, slide_adapted], axis=0)
    slide_all_norm = l2_normalize_np(slide_all)

    slide_pca_dim = min(args.pca_dim_before_cluster, slide_all_norm.shape[1], max(2, slide_all_norm.shape[0] - 1))
    slide_pca = PCA(n_components=slide_pca_dim, random_state=args.seed)
    slide_all_pca = slide_pca.fit_transform(slide_all_norm)

    slide_kmeans = KMeans(n_clusters=args.slide_clusters, random_state=args.seed, n_init=10)
    slide_cluster_all = slide_kmeans.fit_predict(slide_all_pca)
    slide_cluster_frozen = slide_cluster_all[:len(slide_frozen)]
    slide_cluster_adapted = slide_cluster_all[len(slide_frozen):]

    slide_reducer = fit_reducer(slide_all_norm, args.seed, 10, 0.15)
    frozen_slide_emb = transform_reducer(slide_reducer, l2_normalize_np(slide_frozen))
    adapted_slide_emb = transform_reducer(slide_reducer, l2_normalize_np(slide_adapted))

    slide_df["frozen_cluster"] = slide_cluster_frozen
    slide_df["adapted_cluster"] = slide_cluster_adapted
    slide_df["frozen_umap_x"] = frozen_slide_emb[:, 0]
    slide_df["frozen_umap_y"] = frozen_slide_emb[:, 1]
    slide_df["adapted_umap_x"] = adapted_slide_emb[:, 0]
    slide_df["adapted_umap_y"] = adapted_slide_emb[:, 1]
    slide_df.to_csv(Path(args.out_dir) / "slide_points.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_umap_by_cluster(axes[0], frozen_slide_emb, slide_cluster_frozen, "Frozen slide UMAP (cluster)")
    plot_umap_by_cluster(axes[1], adapted_slide_emb, slide_cluster_adapted, "Adapted slide UMAP (cluster)")
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "slide_umap_by_cluster.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_umap_by_label(axes[0], frozen_slide_emb, slide_df["label"].values, "Frozen slide UMAP")
    plot_umap_by_label(axes[1], adapted_slide_emb, slide_df["label"].values, "Adapted slide UMAP")
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "slide_umap_by_label.png", dpi=220)
    plt.close(fig)

    # -----------------------------
    # 5) occupancy shift
    # -----------------------------
    occ_rows = []
    for level, version, cluster_ids in [
        ("patch", "frozen", patch_cluster_frozen),
        ("patch", "adapted", patch_cluster_adapted),
        ("slide", "frozen", slide_cluster_frozen),
        ("slide", "adapted", slide_cluster_adapted),
    ]:
        vc = pd.Series(cluster_ids).value_counts(normalize=False).sort_index()
        vc_ratio = pd.Series(cluster_ids).value_counts(normalize=True).sort_index()
        for cid in sorted(vc.index.tolist()):
            occ_rows.append({
                "level": level,
                "version": version,
                "cluster_id": int(cid),
                "count": int(vc[cid]),
                "ratio": float(vc_ratio[cid]),
            })
    pd.DataFrame(occ_rows).to_csv(Path(args.out_dir) / "cluster_occupancy.csv", index=False)

    # -----------------------------
    # 6) weak patch probe
    # -----------------------------
    slide_train_df, slide_test_df = train_test_split(
        slide_df[["slide_id", "label"]].drop_duplicates(),
        test_size=args.probe_test_slide_frac,
        stratify=slide_df["label"].values,
        random_state=args.seed,
    )
    train_slide_ids = slide_train_df["slide_id"].tolist()
    test_slide_ids = slide_test_df["slide_id"].tolist()

    probe_rows = []
    weak_frozen = run_patch_probe(frozen_X, patch_df, "label", train_slide_ids, test_slide_ids)
    weak_adapted = run_patch_probe(adapted_X, patch_df, "label", train_slide_ids, test_slide_ids)
    weak_frozen.update({"probe_type": "weak_slide_label", "version": "frozen"})
    weak_adapted.update({"probe_type": "weak_slide_label", "version": "adapted"})
    probe_rows.extend([weak_frozen, weak_adapted])

    # pseudo label probe if score csv exists
    if "frozen__tumor_gap" in patch_df.columns and "adapted__tumor_gap" in patch_df.columns:
        patch_df["pseudo_patch_label_frozen"] = build_pseudo_patch_labels(patch_df, "frozen")
        patch_df["pseudo_patch_label_adapted"] = build_pseudo_patch_labels(patch_df, "adapted")

        pseudo_frozen = run_patch_probe(
            frozen_X, patch_df, "pseudo_patch_label_frozen",
            train_slide_ids, test_slide_ids
        )
        pseudo_adapted = run_patch_probe(
            adapted_X, patch_df, "pseudo_patch_label_adapted",
            train_slide_ids, test_slide_ids
        )
        pseudo_frozen.update({"probe_type": "pseudo_patch_label", "version": "frozen"})
        pseudo_adapted.update({"probe_type": "pseudo_patch_label", "version": "adapted"})
        probe_rows.extend([pseudo_frozen, pseudo_adapted])

    pd.DataFrame(probe_rows).to_csv(Path(args.out_dir) / "probe_results.csv", index=False)

    # -----------------------------
    # 7) kNN purity
    # -----------------------------
    purity_rows = []
    purity_rows.append({
        "version": "frozen",
        "label_type": "weak_slide_label",
        "knn_purity": knn_purity(l2_normalize_np(frozen_X), patch_df["label"].values, args.knn_k),
    })
    purity_rows.append({
        "version": "adapted",
        "label_type": "weak_slide_label",
        "knn_purity": knn_purity(l2_normalize_np(adapted_X), patch_df["label"].values, args.knn_k),
    })

    if "pseudo_patch_label_frozen" in patch_df.columns:
        purity_rows.append({
            "version": "frozen",
            "label_type": "pseudo_patch_label",
            "knn_purity": knn_purity(l2_normalize_np(frozen_X), patch_df["pseudo_patch_label_frozen"].values, args.knn_k),
        })
        purity_rows.append({
            "version": "adapted",
            "label_type": "pseudo_patch_label",
            "knn_purity": knn_purity(l2_normalize_np(adapted_X), patch_df["pseudo_patch_label_adapted"].values, args.knn_k),
        })

    pd.DataFrame(purity_rows).to_csv(Path(args.out_dir) / "knn_purity.csv", index=False)

    # -----------------------------
    # 8) cluster semantics if score csv provided
    # -----------------------------
    if frozen_score_df is not None:
        frozen_sem = summarize_cluster_semantics(patch_df, "frozen_cluster", "frozen")
        frozen_sem["version"] = "frozen"
        frozen_sem.to_csv(Path(args.out_dir) / "frozen_cluster_semantics.csv", index=False)

        frozen_top = summarize_top_score_enrichment(
            patch_df, "frozen_cluster", "frozen", args.top_score_frac
        )
        frozen_top["version"] = "frozen"
        frozen_top.to_csv(Path(args.out_dir) / "frozen_top_score_enrichment.csv", index=False)

    if adapted_score_df is not None:
        adapted_sem = summarize_cluster_semantics(patch_df, "adapted_cluster", "adapted")
        adapted_sem["version"] = "adapted"
        adapted_sem.to_csv(Path(args.out_dir) / "adapted_cluster_semantics.csv", index=False)

        adapted_top = summarize_top_score_enrichment(
            patch_df, "adapted_cluster", "adapted", args.top_score_frac
        )
        adapted_top["version"] = "adapted"
        adapted_top.to_csv(Path(args.out_dir) / "adapted_top_score_enrichment.csv", index=False)

    # -----------------------------
    # 9) summary
    # -----------------------------
    summary = {
        "sampled_n_slides": int(len(slide_df)),
        "patch_n_points": int(len(patch_df)),
        "slide_n_points": int(len(slide_df)),
        "has_umap": bool(HAS_UMAP),
        "has_frozen_score_csv": frozen_score_df is not None,
        "has_adapted_score_csv": adapted_score_df is not None,
        "note": (
            "Mainline outputs: cluster_occupancy.csv, probe_results.csv, knn_purity.csv, "
            "and optional cluster semantics / top-score enrichment if score csvs are provided."
        ),
    }
    with open(Path(args.out_dir) / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved analysis to", args.out_dir)


if __name__ == "__main__":
    main()