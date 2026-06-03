#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import json
import random
import argparse
import atexit
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageFile, ImageDraw
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# global caches
# =========================================================
_SLIDE_CACHE: Dict[str, openslide.OpenSlide] = {}
_PATCH_CACHE: Dict[str, Image.Image] = {}


def close_all_slides():
    for _, slide in list(_SLIDE_CACHE.items()):
        try:
            slide.close()
        except Exception:
            pass
    _SLIDE_CACHE.clear()


atexit.register(close_all_slides)


# =========================================================
# utils
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def make_patch_uid(row: pd.Series) -> str:
    return f"{canonicalize_path(row['svs_path'])}__{int(row['coord_x'])}__{int(row['coord_y'])}"


def attach_patch_uid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["patch_uid"] = df.apply(make_patch_uid, axis=1)
    return df


def get_slide(svs_path: str) -> openslide.OpenSlide:
    svs_path = canonicalize_path(svs_path)
    if svs_path not in _SLIDE_CACHE:
        _SLIDE_CACHE[svs_path] = openslide.OpenSlide(svs_path)
    return _SLIDE_CACHE[svs_path]


def open_patch_from_svs(
    svs_path: str,
    coord_x: int,
    coord_y: int,
    patch_level: int,
    patch_size: int,
    use_patch_cache: bool = True,
) -> Image.Image:
    svs_path = canonicalize_path(svs_path)
    cache_key = f"{svs_path}__{int(coord_x)}__{int(coord_y)}__{int(patch_level)}__{int(patch_size)}"

    if use_patch_cache and cache_key in _PATCH_CACHE:
        return _PATCH_CACHE[cache_key].copy()

    slide = get_slide(svs_path)
    img = slide.read_region(
        (int(coord_x), int(coord_y)),
        int(patch_level),
        (int(patch_size), int(patch_size)),
    ).convert("RGB")

    if use_patch_cache:
        _PATCH_CACHE[cache_key] = img.copy()

    return img


def infer_token_grid(num_tokens: int) -> int:
    g = int(round(math.sqrt(num_tokens)))
    if g * g != num_tokens:
        raise ValueError(f"num_tokens={num_tokens} is not a perfect square.")
    return g


def draw_token_box_on_patch(
    patch_img: Image.Image,
    token_idx: int,
    num_tokens_per_patch: int,
    color=(255, 0, 0),
    width: int = 3,
) -> Image.Image:
    img = patch_img.copy()
    W, H = img.size
    grid = infer_token_grid(num_tokens_per_patch)

    cell_w = W / grid
    cell_h = H / grid

    r = int(token_idx // grid)
    c = int(token_idx % grid)

    x0 = int(math.floor(c * cell_w))
    y0 = int(math.floor(r * cell_h))
    x1 = int(math.ceil((c + 1) * cell_w))
    y1 = int(math.ceil((r + 1) * cell_h))

    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W))
    y1 = max(y0 + 1, min(y1, H))

    draw = ImageDraw.Draw(img)
    max_k = min(width, max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2))
    for k in range(max_k):
        draw.rectangle([x0 + k, y0 + k, x1 - 1 - k, y1 - 1 - k], outline=color)
    return img


def dedup_indices_by_patch(
    df: pd.DataFrame,
    indices: List[int],
    max_per_slide: int = 2,
) -> List[int]:
    out = []
    seen_patch = set()
    slide_counter: Dict[str, int] = {}

    for idx in indices:
        row = df.iloc[idx]
        patch_uid = str(row["patch_uid"])
        slide_id = str(row["slide_id"])

        if patch_uid in seen_patch:
            continue
        if slide_counter.get(slide_id, 0) >= max_per_slide:
            continue

        seen_patch.add(patch_uid)
        slide_counter[slide_id] = slide_counter.get(slide_id, 0) + 1
        out.append(int(idx))

    return out


# =========================================================
# neighbor / selection
# =========================================================
def build_neighbor_graph(embedding: np.ndarray, k: int = 30):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(embedding)
    dist, ind = nn.kneighbors(embedding)
    return dist[:, 1:], ind[:, 1:]


def find_mixed_points(
    df: pd.DataFrame,
    embedding: np.ndarray,
    k: int = 30,
    min_neighbor_disagree_ratio: float = 0.45,
    topn: int = 48,
    dedup_by_patch: bool = True,
    gallery_max_per_slide: int = 2,
    show_progress: bool = True,
):
    _, nbr_idx = build_neighbor_graph(embedding, k=k)

    expert_ids = df["expert_id"].to_numpy()
    nearest_role = df["nearest_role"].astype(str).to_numpy()

    scores = []
    iterator = range(len(df))
    if show_progress:
        iterator = tqdm(iterator, desc="Scoring mixed points", leave=False)

    for i in iterator:
        nb = nbr_idx[i]
        expert_disagree = np.mean(expert_ids[nb] != expert_ids[i])
        role_disagree = np.mean(nearest_role[nb] != nearest_role[i])
        mixed_score = 0.5 * expert_disagree + 0.5 * role_disagree
        scores.append(mixed_score)

    scores = np.asarray(scores, dtype=np.float32)

    candidate = np.where(scores >= min_neighbor_disagree_ratio)[0]
    if len(candidate) == 0:
        ranked = np.argsort(-scores).tolist()
    else:
        ranked = candidate[np.argsort(-scores[candidate])].tolist()

    if dedup_by_patch:
        ranked = dedup_indices_by_patch(
            df=df,
            indices=ranked,
            max_per_slide=gallery_max_per_slide,
        )

    ranked = ranked[:topn]
    return np.asarray(ranked, dtype=np.int64), scores


def find_dense_cluster_centers(
    df: pd.DataFrame,
    embedding: np.ndarray,
    topn: int = 12,
    k: int = 30,
    center_min_dist_quantile: float = 0.08,
    neighborhood_size: int = 3072,
    overlap_max: float = 0.35,
    show_progress: bool = True,
):
    dist, _ = build_neighbor_graph(embedding, k=k)
    density_score = -dist.mean(axis=1)

    ranked = np.argsort(-density_score).tolist()

    N = len(df)
    if N <= 1:
        min_center_dist = 0.0
    else:
        sample_n = min(2000, N)
        sample_idx = np.random.choice(N, size=sample_n, replace=False)
        sample_emb = embedding[sample_idx]   # [S, 2]

        tri = np.triu_indices(sample_n, k=1)
        pair_d = np.linalg.norm(
            sample_emb[:, None, :] - sample_emb[None, :, :],
            axis=-1
        )[tri]

        min_center_dist = np.quantile(pair_d, center_min_dist_quantile) if len(pair_d) > 0 else 0.0

    selected = []
    selected_nbr_sets = []
    selected_patch_uids = set()

    iterator = ranked
    if show_progress:
        iterator = tqdm(iterator, desc="Selecting dense cluster centers", leave=False)

    for idx in iterator:
        row = df.iloc[idx]
        patch_uid = row["patch_uid"]

        if patch_uid in selected_patch_uids:
            continue

        too_close = False
        for sidx in selected:
            d = np.linalg.norm(embedding[idx] - embedding[sidx])
            if d < min_center_dist:
                too_close = True
                break
        if too_close:
            continue

        nbr = np.argsort(np.linalg.norm(embedding - embedding[idx], axis=1))[:neighborhood_size]
        nbr_set = set(map(int, nbr.tolist()))

        overlap_bad = False
        for prev_set in selected_nbr_sets:
            inter = len(nbr_set & prev_set)
            union = max(1, min(len(nbr_set), len(prev_set)))
            overlap = inter / union
            if overlap > overlap_max:
                overlap_bad = True
                break
        if overlap_bad:
            continue

        selected.append(int(idx))
        selected_nbr_sets.append(nbr_set)
        selected_patch_uids.add(patch_uid)

        if len(selected) >= topn:
            break

    return np.asarray(selected, dtype=np.int64), density_score


def get_local_group_indices(
    center_idx: int,
    embedding: np.ndarray,
    neighborhood_size: int = 3072,
):
    d = np.linalg.norm(embedding - embedding[center_idx], axis=1)
    local_idx = np.argsort(d)[:neighborhood_size]
    return local_idx, d


# =========================================================
# local subclustering
# =========================================================
def cluster_local_patches_by_feature(
    local_indices: np.ndarray,
    features: np.ndarray,
    min_clusters: int = 2,
    max_clusters: int = 6,
    random_state: int = 42,
):
    if len(local_indices) == 0:
        return np.asarray([], dtype=np.int64), 0

    X = features[local_indices]
    X = normalize(X, norm="l2")

    n = len(X)
    if n < 8:
        return np.zeros(n, dtype=np.int64), 1

    k = int(round(np.sqrt(n / 12)))
    k = max(min_clusters, min(max_clusters, k))
    k = min(k, n)

    if k <= 1:
        return np.zeros(n, dtype=np.int64), 1

    km = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
    )
    labels = km.fit_predict(X)
    return labels.astype(np.int64), k


def select_representatives_per_subcluster(
    df_local: pd.DataFrame,
    features_local: np.ndarray,
    subcluster_labels: np.ndarray,
    max_per_subcluster: int = 8,
    gallery_max_per_slide: int = 2,
):
    df_local = df_local.copy()
    X = normalize(features_local, norm="l2")

    chosen_by_cluster = {}
    summary_by_cluster = {}

    for cid in sorted(pd.unique(subcluster_labels).tolist()):
        mask = (subcluster_labels == cid)
        sub = df_local.loc[mask].copy()
        Xc = X[mask]

        if len(sub) == 0:
            continue

        center = Xc.mean(axis=0, keepdims=True)
        center = normalize(center, norm="l2")
        _, dist = pairwise_distances_argmin_min(Xc, center, metric="euclidean")
        sub["dist_to_subcluster_center"] = dist

        sub = sub.sort_values("dist_to_subcluster_center", ascending=True)
        sub = sub.drop_duplicates(subset=["patch_uid"], keep="first").copy()

        keep_rows = []
        slide_counter: Dict[str, int] = {}
        for _, row in sub.iterrows():
            sid = str(row["slide_id"])
            if slide_counter.get(sid, 0) >= gallery_max_per_slide:
                continue
            keep_rows.append(row)
            slide_counter[sid] = slide_counter.get(sid, 0) + 1

        if len(keep_rows) == 0:
            continue

        sub2 = pd.DataFrame(keep_rows)
        sub2 = sub2.nsmallest(max_per_subcluster, "dist_to_subcluster_center").copy()

        chosen_by_cluster[int(cid)] = sub2.index.to_list()
        summary_by_cluster[int(cid)] = {
            "num_points": int(mask.sum()),
            "num_unique_patches_after_dedup": int(len(sub2)),
            "expert_ratio": sub["expert_id"].value_counts(normalize=True).to_dict(),
            "role_ratio": sub["nearest_role"].value_counts(normalize=True).to_dict(),
        }

    return chosen_by_cluster, summary_by_cluster


# =========================================================
# render
# =========================================================
def render_umap_with_points(
    embedding: np.ndarray,
    df: pd.DataFrame,
    selected_idx: List[int],
    save_path: str,
    color_by: str = "expert_id",
):
    plt.figure(figsize=(9, 9))

    if color_by == "expert_id":
        labels = df["expert_id"].to_numpy()
        uniq = sorted(pd.unique(labels).tolist())
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 1)))
        for i, u in enumerate(uniq):
            idx = labels == u
            plt.scatter(embedding[idx, 0], embedding[idx, 1], s=5, alpha=0.25, color=cmap[i], label=f"E{u}")
    else:
        labels = df["nearest_role"].astype(str).to_numpy()
        uniq = sorted(pd.unique(labels).tolist())
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 1)))
        for i, u in enumerate(uniq):
            idx = labels == u
            plt.scatter(embedding[idx, 0], embedding[idx, 1], s=5, alpha=0.25, color=cmap[i], label=u)

    sel = np.asarray(selected_idx, dtype=np.int64)
    if len(sel) > 0:
        plt.scatter(
            embedding[sel, 0],
            embedding[sel, 1],
            s=40,
            c="black",
            marker="x",
            linewidths=1.2,
            label="selected",
        )

    plt.legend(markerscale=2)
    plt.title(f"UMAP with selected points ({color_by})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def render_full_patch_gallery(
    df: pd.DataFrame,
    indices: List[int],
    save_path: str,
    title: str,
    ncols: int = 4,
    patch_size: int = 180,
    max_items: int = 24,
    show_progress: bool = True,
):
    indices = list(indices)[:max_items]
    if len(indices) == 0:
        return

    n = len(indices)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(4.4 * ncols, 4.3 * nrows))
    fig.suptitle(title, fontsize=16)

    iterator = enumerate(indices)
    if show_progress:
        iterator = tqdm(list(iterator), desc=f"Rendering {os.path.basename(save_path)}", leave=False)

    for plot_i, idx in iterator:
        row = df.loc[idx]

        patch = open_patch_from_svs(
            svs_path=row["svs_path"],
            coord_x=row["coord_x"],
            coord_y=row["coord_y"],
            patch_level=row["patch_level"],
            patch_size=row["patch_size"],
            use_patch_cache=True,
        ).resize((patch_size, patch_size))

        ax = plt.subplot(nrows, ncols, plot_i + 1)
        ax.imshow(patch)
        ax.axis("off")

        txt = (
            f"E{int(row['expert_id'])} | {row['nearest_role']}\n"
            f"slide={row['slide_id']}\n"
            f"coord=({int(row['coord_x'])},{int(row['coord_y'])})"
        )

        if "role_affinity_tumor" in df.columns:
            role_cols = [c for c in df.columns if c.startswith("role_affinity_")]
            tumor = float(row["role_affinity_tumor"])
            other_max = max([float(row[c]) for c in role_cols if c != "role_affinity_tumor"], default=0.0)
            gap = tumor - other_max
            txt += f"\n tumor={tumor:.3f} other={other_max:.3f} gap={gap:.3f}"

        if "dist_to_subcluster_center" in row and pd.notna(row["dist_to_subcluster_center"]):
            txt += f"\n subdist={float(row['dist_to_subcluster_center']):.3f}"

        ax.set_title(txt, fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(save_path, dpi=220)
    plt.close()


def render_patch_with_tokenbox_gallery(
    df: pd.DataFrame,
    indices: List[int],
    save_path: str,
    title: str,
    num_tokens_per_patch: int,
    ncols: int = 4,
    patch_size: int = 180,
    max_items: int = 24,
    show_progress: bool = True,
):
    indices = list(indices)[:max_items]
    if len(indices) == 0:
        return

    n = len(indices)
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(4.4 * ncols, 4.3 * nrows))
    fig.suptitle(title, fontsize=16)

    iterator = enumerate(indices)
    if show_progress:
        iterator = tqdm(list(iterator), desc=f"Rendering {os.path.basename(save_path)}", leave=False)

    for plot_i, idx in iterator:
        row = df.loc[idx]

        patch = open_patch_from_svs(
            svs_path=row["svs_path"],
            coord_x=row["coord_x"],
            coord_y=row["coord_y"],
            patch_level=row["patch_level"],
            patch_size=row["patch_size"],
            use_patch_cache=True,
        )
        patch = draw_token_box_on_patch(
            patch_img=patch,
            token_idx=int(row["token_idx"]),
            num_tokens_per_patch=num_tokens_per_patch,
        ).resize((patch_size, patch_size))

        ax = plt.subplot(nrows, ncols, plot_i + 1)
        ax.imshow(patch)
        ax.axis("off")

        txt = (
            f"E{int(row['expert_id'])} | {row['nearest_role']}\n"
            f"slide={row['slide_id']}\n"
            f"tok={int(row['token_idx'])}"
        )
        ax.set_title(txt, fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(save_path, dpi=220)
    plt.close()


def render_subcluster_galleries_for_local_cluster(
    df_local: pd.DataFrame,
    chosen_by_cluster: Dict[int, List[int]],
    summary_by_cluster: Dict[int, Dict],
    save_dir: str,
    prefix: str,
    num_tokens_per_patch: int,
):
    for cid, idxs in tqdm(
        sorted(chosen_by_cluster.items(), key=lambda x: x[0]),
        desc=f"{prefix}: rendering subcluster galleries",
        leave=False,
    ):
        title_prefix = (
            f"{prefix} | subcluster={cid} | "
            f"n={summary_by_cluster[cid]['num_points']}"
        )

        render_full_patch_gallery(
            df=df_local,
            indices=idxs,
            save_path=os.path.join(save_dir, f"{prefix}_subcluster_{cid:02d}_full_patch_gallery.png"),
            title=title_prefix + " | full patch",
            ncols=4,
            max_items=len(idxs),
            show_progress=False,
        )

        render_patch_with_tokenbox_gallery(
            df=df_local,
            indices=idxs,
            save_path=os.path.join(save_dir, f"{prefix}_subcluster_{cid:02d}_patch_with_tokenbox_gallery.png"),
            title=title_prefix + " | patch + token box",
            num_tokens_per_patch=num_tokens_per_patch,
            ncols=4,
            max_items=len(idxs),
            show_progress=False,
        )


# =========================================================
# main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=str, required=True,
                        help="目录里应包含 token_level_analysis.csv / umap_embedding.npy / features_moe.npy")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--num-tokens-per-patch", type=int, default=196)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mixed-topn", type=int, default=32)
    parser.add_argument("--cluster-topn", type=int, default=12)
    parser.add_argument("--neighbor-k", type=int, default=30)

    parser.add_argument("--gallery-max-per-slide", type=int, default=2)
    parser.add_argument("--mixed-min-neighbor-disagree-ratio", type=float, default=0.45)

    parser.add_argument("--cluster-center-min-dist-quantile", type=float, default=0.08)
    parser.add_argument("--cluster-neighborhood-size", type=int, default=3072)
    parser.add_argument("--cluster-neighborhood-overlap-max", type=float, default=0.35)

    parser.add_argument("--subcluster-min-k", type=int, default=2)
    parser.add_argument("--subcluster-max-k", type=int, default=6)
    parser.add_argument("--max-per-subcluster", type=int, default=8)
    parser.add_argument("--disable-patch-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    if args.disable_patch_cache:
        global _PATCH_CACHE
        _PATCH_CACHE = {}

    token_csv = os.path.join(args.analysis_dir, "token_level_analysis.csv")
    umap_npy = os.path.join(args.analysis_dir, "umap_embedding.npy")
    feat_npy = os.path.join(args.analysis_dir, "features_moe.npy")

    if not os.path.exists(token_csv):
        raise FileNotFoundError(token_csv)
    if not os.path.exists(umap_npy):
        raise FileNotFoundError(umap_npy)
    if not os.path.exists(feat_npy):
        raise FileNotFoundError(feat_npy)

    print("[1/6] Loading analysis files...")
    df = pd.read_csv(token_csv)
    df["svs_path"] = df["svs_path"].map(canonicalize_path)
    df = attach_patch_uid(df)

    embedding = np.load(umap_npy)
    features = np.load(feat_npy)

    if len(df) != len(embedding):
        raise ValueError(f"csv rows ({len(df)}) != embedding rows ({len(embedding)})")
    if len(df) != len(features):
        raise ValueError(f"csv rows ({len(df)}) != features rows ({len(features)})")

    print(f"Loaded {len(df)} tokens")
    print("[2/6] Finding mixed points...")

    mixed_idx, mixed_scores = find_mixed_points(
        df=df,
        embedding=embedding,
        k=args.neighbor_k,
        min_neighbor_disagree_ratio=args.mixed_min_neighbor_disagree_ratio,
        topn=args.mixed_topn,
        dedup_by_patch=True,
        gallery_max_per_slide=args.gallery_max_per_slide,
        show_progress=True,
    )
    df["mixed_score"] = mixed_scores

    print("[3/6] Rendering mixed-point overview...")
    render_umap_with_points(
        embedding=embedding,
        df=df,
        selected_idx=mixed_idx.tolist(),
        save_path=os.path.join(args.output_dir, "umap_selected_mixed_points_expert.png"),
        color_by="expert_id",
    )
    render_umap_with_points(
        embedding=embedding,
        df=df,
        selected_idx=mixed_idx.tolist(),
        save_path=os.path.join(args.output_dir, "umap_selected_mixed_points_role.png"),
        color_by="nearest_role",
    )

    render_full_patch_gallery(
        df=df,
        indices=mixed_idx.tolist(),
        save_path=os.path.join(args.output_dir, "mixed_points_full_patch_gallery.png"),
        title="Suspicious mixed points on UMAP | full patch",
        ncols=4,
        max_items=args.mixed_topn,
        show_progress=True,
    )

    print("[4/6] Selecting representative dense cluster centers...")
    center_idx, density_score = find_dense_cluster_centers(
        df=df,
        embedding=embedding,
        topn=args.cluster_topn,
        k=args.neighbor_k,
        center_min_dist_quantile=args.cluster_center_min_dist_quantile,
        neighborhood_size=args.cluster_neighborhood_size,
        overlap_max=args.cluster_neighborhood_overlap_max,
        show_progress=True,
    )
    df["density_score"] = density_score

    render_umap_with_points(
        embedding=embedding,
        df=df,
        selected_idx=center_idx.tolist(),
        save_path=os.path.join(args.output_dir, "umap_cluster_centers_expert.png"),
        color_by="expert_id",
    )
    render_umap_with_points(
        embedding=embedding,
        df=df,
        selected_idx=center_idx.tolist(),
        save_path=os.path.join(args.output_dir, "umap_cluster_centers_role.png"),
        color_by="nearest_role",
    )

    print("[5/6] Analyzing local clusters and rendering subcluster galleries...")
    cluster_meta = []

    for rank, cidx in tqdm(
        list(enumerate(center_idx)),
        desc="Processing local clusters",
        leave=True,
    ):
        local_idx, d_all = get_local_group_indices(
            center_idx=int(cidx),
            embedding=embedding,
            neighborhood_size=args.cluster_neighborhood_size,
        )

        df_local = df.iloc[local_idx].copy()
        df_local["dist_to_cluster_center"] = d_all[local_idx]
        feat_local = features[local_idx]

        sub_labels, k_used = cluster_local_patches_by_feature(
            local_indices=np.arange(len(local_idx)),
            features=feat_local,
            min_clusters=args.subcluster_min_k,
            max_clusters=args.subcluster_max_k,
            random_state=args.seed,
        )
        df_local["local_subcluster_id"] = sub_labels

        chosen_by_cluster, summary_by_cluster = select_representatives_per_subcluster(
            df_local=df_local,
            features_local=feat_local,
            subcluster_labels=sub_labels,
            max_per_subcluster=args.max_per_subcluster,
            gallery_max_per_slide=args.gallery_max_per_slide,
        )

        prefix = f"cluster_{rank:02d}"
        render_subcluster_galleries_for_local_cluster(
            df_local=df_local,
            chosen_by_cluster=chosen_by_cluster,
            summary_by_cluster=summary_by_cluster,
            save_dir=args.output_dir,
            prefix=prefix,
            num_tokens_per_patch=args.num_tokens_per_patch,
        )

        cluster_meta.append({
            "cluster_rank": int(rank),
            "center_global_idx": int(cidx),
            "num_local_points": int(len(local_idx)),
            "subcluster_k": int(k_used),
            "subclusters": summary_by_cluster,
        })

        with open(os.path.join(args.output_dir, "cluster_subcluster_summary.json"), "w", encoding="utf-8") as f:
            json.dump(cluster_meta, f, ensure_ascii=False, indent=2)

    print("[6/6] Saving outputs...")
    with open(os.path.join(args.output_dir, "cluster_subcluster_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cluster_meta, f, ensure_ascii=False, indent=2)

    df.to_csv(os.path.join(args.output_dir, "token_level_analysis_with_scores.csv"), index=False)

    print(f"[Done] saved to: {args.output_dir}")
    print(f"[Cache] opened slides: {len(_SLIDE_CACHE)}")
    print(f"[Cache] cached patches: {len(_PATCH_CACHE)}")


if __name__ == "__main__":
    main()