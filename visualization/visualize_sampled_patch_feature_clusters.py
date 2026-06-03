#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import random
from collections import Counter
from typing import Dict, List, Optional

from tqdm import tqdm

import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.v2 as T
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import yaml

from models.encoders.moe_encoder import MoEEncoder

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# utils
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def l2_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def build_transform(image_size: int = 224):
    return T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


def try_get_font(size: int = 18):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def ratio_dict(values: List[str]) -> Dict[str, float]:
    if len(values) == 0:
        return {}
    c = Counter(values)
    total = sum(c.values())
    return {k: float(v / total) for k, v in sorted(c.items(), key=lambda x: (-x[1], x[0]))}


def ratio_dict_int(values: List[int]) -> Dict[str, float]:
    if len(values) == 0:
        return {}
    c = Counter(values)
    total = sum(c.values())
    return {str(k): float(v / total) for k, v in sorted(c.items(), key=lambda x: x[0])}


# =========================================================
# model loading
# =========================================================
def load_stage2_bundle(config_path: str, full_ckpt_path: str, device: str = "cuda"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not os.path.exists(full_ckpt_path):
        raise FileNotFoundError(f"Full checkpoint not found: {full_ckpt_path}")

    ckpt = torch.load(full_ckpt_path, map_location="cpu")

    if "student_state_dict" not in ckpt:
        raise KeyError("student_state_dict not found in full checkpoint")
    if "distiller_state_dict" not in ckpt:
        raise KeyError("distiller_state_dict not found in full checkpoint")

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    model.load_state_dict(ckpt["student_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    distiller_sd = ckpt["distiller_state_dict"]
    if "proj_l12.weight" not in distiller_sd or "proj_l12.bias" not in distiller_sd:
        raise KeyError("proj_l12 not found in distiller_state_dict")

    proj_out_dim, proj_in_dim = distiller_sd["proj_l12.weight"].shape
    role_proj_head = nn.Linear(proj_in_dim, proj_out_dim)
    role_proj_head.load_state_dict({
        "weight": distiller_sd["proj_l12.weight"],
        "bias": distiller_sd["proj_l12.bias"],
    })
    role_proj_head = role_proj_head.to(device)
    role_proj_head.eval()

    print("Loaded matched student + proj_l12 from best_full checkpoint")
    print(f"Current moe_layers_idx = {model.moe_layers_idx}")
    print(f"proj_l12 shape: {proj_in_dim} -> {proj_out_dim}")

    return model, role_proj_head, cfg


def load_role_prototypes(role_proto_dir: str):
    proto_path = os.path.join(role_proto_dir, "role_prototypes_init.npy")
    names_path = os.path.join(role_proto_dir, "role_names.json")

    if not os.path.exists(proto_path):
        raise FileNotFoundError(f"Missing prototype file: {proto_path}")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Missing role names file: {names_path}")

    protos = np.load(proto_path).astype(np.float32)
    with open(names_path, "r", encoding="utf-8") as f:
        role_names = json.load(f)

    protos = normalize(protos, norm="l2")
    print(f"[RoleProto] role names = {role_names}")
    print(f"[RoleProto] proto shape = {protos.shape}")
    return protos, role_names


# =========================================================
# dataset
# =========================================================
class TCGAPoolDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        transform,
        max_rows: Optional[int] = None,
        sample_per_project: Optional[int] = None,
        sample_per_label: Optional[int] = None,
        seed: int = 42,
        keep_projects: Optional[List[str]] = None,
        keep_labels: Optional[List[str]] = None,
        csv_chunksize: int = 50000,
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"csv not found: {csv_path}")

        self.transform = transform
        rng = np.random.default_rng(seed)

        required = ["svs_path", "coord_x", "coord_y", "patch_level", "patch_size"]
        optional = ["project", "slide_id", "pred_label", "coord_idx"]
        usecols = required + optional

        print(f"[Dataset] stream reading csv: {csv_path}")
        chunk_list = []
        total_rows_seen = 0
        total_rows_kept = 0

        reader = pd.read_csv(
            csv_path,
            usecols=lambda c: c in usecols,
            chunksize=csv_chunksize,
            low_memory=False,
        )

        for chunk_id, chunk in enumerate(reader):
            total_rows_seen += len(chunk)

            missing = [c for c in required if c not in chunk.columns]
            if missing:
                raise ValueError(f"csv missing required columns: {missing}")

            chunk = chunk.copy()
            chunk["svs_path"] = chunk["svs_path"].map(canonicalize_path)

            if keep_projects is not None and "project" in chunk.columns:
                chunk = chunk[chunk["project"].isin(keep_projects)]

            if keep_labels is not None and "pred_label" in chunk.columns:
                chunk = chunk[chunk["pred_label"].isin(keep_labels)]

            if len(chunk) == 0:
                continue

            chunk_list.append(chunk)
            total_rows_kept += len(chunk)

            if (chunk_id + 1) % 10 == 0:
                print(f"[Dataset] chunk {chunk_id+1}: seen={total_rows_seen}, kept={total_rows_kept}")

            if (
                max_rows is not None
                and sample_per_project is None
                and sample_per_label is None
                and total_rows_kept >= max_rows * 5
            ):
                print(f"[Dataset] early stop stream read at kept={total_rows_kept}")
                break

        if len(chunk_list) == 0:
            raise ValueError("No rows kept after streaming/filtering.")

        df = pd.concat(chunk_list, axis=0, ignore_index=True)
        print(f"[Dataset] streamed df rows = {len(df)}")

        if sample_per_project is not None and "project" in df.columns:
            print(f"[Dataset] applying sample_per_project={sample_per_project}")
            parts = []
            for _, sub in df.groupby("project", sort=False):
                if len(sub) > sample_per_project:
                    choose_idx = rng.choice(len(sub), size=sample_per_project, replace=False)
                    sub = sub.iloc[choose_idx].copy()
                parts.append(sub)
            df = pd.concat(parts, axis=0, ignore_index=True)
            print(f"[Dataset] after sample_per_project: {len(df)}")

        if sample_per_label is not None and "pred_label" in df.columns:
            print(f"[Dataset] applying sample_per_label={sample_per_label}")
            parts = []
            for _, sub in df.groupby("pred_label", sort=False):
                if len(sub) > sample_per_label:
                    choose_idx = rng.choice(len(sub), size=sample_per_label, replace=False)
                    sub = sub.iloc[choose_idx].copy()
                parts.append(sub)
            df = pd.concat(parts, axis=0, ignore_index=True)
            print(f"[Dataset] after sample_per_label: {len(df)}")

        if max_rows is not None and len(df) > max_rows:
            print(f"[Dataset] applying final max_rows={max_rows}")
            choose_idx = rng.choice(len(df), size=max_rows, replace=False)
            df = df.iloc[choose_idx].copy().reset_index(drop=True)

        self.df = df.reset_index(drop=True)

        print(f"[Dataset] final rows = {len(self.df)}")
        if "project" in self.df.columns:
            print("[Dataset] project counts:")
            print(self.df["project"].value_counts())
        if "pred_label" in self.df.columns:
            print("[Dataset] label counts:")
            print(self.df["pred_label"].value_counts())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        svs_path = str(row["svs_path"])
        if not os.path.exists(svs_path):
            raise FileNotFoundError(f"Missing svs_path: {svs_path}")

        x = int(row["coord_x"])
        y = int(row["coord_y"])
        patch_level = int(row["patch_level"])
        patch_size = int(row["patch_size"])

        slide = openslide.OpenSlide(svs_path)
        try:
            img = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
        finally:
            slide.close()

        img_tensor = self.transform(img)

        meta = {
            "row_idx": int(idx),
            "project": str(row["project"]) if "project" in row and pd.notna(row["project"]) else "",
            "slide_id": str(row["slide_id"]) if "slide_id" in row and pd.notna(row["slide_id"]) else "",
            "pred_label": str(row["pred_label"]) if "pred_label" in row and pd.notna(row["pred_label"]) else "",
            "svs_path": svs_path,
            "coord_x": x,
            "coord_y": y,
            "coord_idx": int(row["coord_idx"]) if "coord_idx" in row and pd.notna(row["coord_idx"]) else -1,
            "patch_level": patch_level,
            "patch_size": patch_size,
        }
        return img_tensor, meta


def collate_with_meta(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    metas = [x[1] for x in batch]
    return images, metas


# =========================================================
# forward helpers
# =========================================================
@torch.no_grad()
def run_model_and_collect(model, img_tensor):
    final_feats, gate_info_list, _, moe_feature_list = model(
        img_tensor,
        return_gates=True,
        return_features=True,
        is_eval=True,
    )

    if len(gate_info_list) == 0:
        raise RuntimeError("gate_info_list is empty")
    if len(moe_feature_list) == 0:
        raise RuntimeError("moe_feature_list is empty")

    return final_feats, gate_info_list, moe_feature_list


def get_expert_assignment_from_gate_info(gate_info, seq_len):
    dispatch = gate_info["dispatch_weight"]  # [B*seq_len, E]
    total_tokens, num_experts = dispatch.shape
    B = total_tokens // seq_len
    dispatch = dispatch.view(B, seq_len, num_experts)[:, 1:, :]
    expert_id = dispatch.argmax(dim=-1)  # [B, N]
    return expert_id


@torch.no_grad()
def project_features_to_role_space(features: np.ndarray, proj_head, device="cpu", batch_size=4096):
    outs = []
    n = len(features)
    for start in tqdm(
        range(0, n, batch_size),
        total=(n + batch_size - 1) // batch_size,
        desc="Project->role-space",
        leave=True,
    ):
        x = torch.from_numpy(features[start:start+batch_size]).float().to(device)
        y = proj_head(x)
        y = F.normalize(y, dim=-1)
        outs.append(y.cpu().numpy())
    return np.concatenate(outs, axis=0)


def compute_role_affinity(features_role_space: np.ndarray, role_prototypes: np.ndarray):
    feats = normalize(features_role_space, norm="l2")
    protos = normalize(role_prototypes, norm="l2")
    return feats @ protos.T


def nearest_role_assignment(role_affinity: np.ndarray, role_names: List[str]):
    idx = role_affinity.argmax(axis=1)
    labels = np.array([role_names[i] for i in idx], dtype=object)
    return idx, labels


# =========================================================
# clustering helpers
# =========================================================
def cluster_features_kmeans(
    feats: np.ndarray,
    n_clusters: int,
    seed: int = 42,
    batch_size: int = 2048,
):
    n_clusters = min(n_clusters, len(feats))
    if n_clusters <= 1:
        labels = np.zeros(len(feats), dtype=np.int64)
        centers = feats[:1].copy()
        return labels, centers

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=batch_size,
        n_init="auto",
    )
    labels = km.fit_predict(feats)
    centers = km.cluster_centers_
    return labels, centers


def get_nearest_indices_to_center(feats: np.ndarray, center: np.ndarray, topk: int):
    d = np.sum((feats - center[None, :]) ** 2, axis=1)
    return np.argsort(d)[:topk]


def get_random_indices_in_cluster(indices: np.ndarray, topk: int, seed: int):
    rng = np.random.default_rng(seed)
    if len(indices) <= topk:
        return np.array(indices, dtype=np.int64)
    return np.array(rng.choice(indices, size=topk, replace=False), dtype=np.int64)


# =========================================================
# patch visualization
# =========================================================
def read_patch_from_meta(meta: dict, out_size: int = 160) -> Image.Image:
    slide = openslide.OpenSlide(meta["svs_path"])
    try:
        img = slide.read_region(
            (int(meta["coord_x"]), int(meta["coord_y"])),
            int(meta["patch_level"]),
            (int(meta["patch_size"]), int(meta["patch_size"])),
        ).convert("RGB")
    finally:
        slide.close()

    if out_size is not None:
        img = img.resize((out_size, out_size), resample=Image.BICUBIC)
    return img


def draw_patch_tile(img: Image.Image, text_lines: List[str], tile_w: int = 180, img_h: int = 160, text_h: int = 56):
    font = try_get_font(14)
    tile = Image.new("RGB", (tile_w, img_h + text_h), (255, 255, 255))
    img = ImageOps.fit(img, (tile_w, img_h), method=Image.BICUBIC)
    tile.paste(img, (0, 0))

    draw = ImageDraw.Draw(tile)
    y = img_h + 3
    for line in text_lines[:3]:
        draw.text((4, y), line, fill=(0, 0, 0), font=font)
        y += 16
    return tile


def paste_grid(
    rows_tiles: List[List[Image.Image]],
    row_titles: List[str],
    save_path: str,
    title: str = "",
    left_title_w: int = 300,
    pad: int = 10,
    bg=(245, 245, 245),
):
    font_title = try_get_font(26)
    font_row = try_get_font(18)

    n_rows = len(rows_tiles)
    if n_rows == 0:
        return

    tile_h = max(tile.height for row in rows_tiles for tile in row)
    tile_w = max(tile.width for row in rows_tiles for tile in row)
    max_cols = max(len(row) for row in rows_tiles)

    top_title_h = 50 if title else 10
    canvas_w = left_title_w + max_cols * tile_w + (max_cols + 1) * pad
    canvas_h = top_title_h + n_rows * tile_h + (n_rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((pad, 8), title, fill=(0, 0, 0), font=font_title)

    y0 = top_title_h + pad
    for r, row in enumerate(rows_tiles):
        row_y = y0 + r * tile_h
        draw.text((10, row_y + 6), row_titles[r], fill=(0, 0, 0), font=font_row)

        x0 = left_title_w + pad
        for c, tile in enumerate(row):
            x = x0 + c * tile_w
            canvas.paste(tile, (x, row_y))

    canvas.save(save_path)
    print(f"[Saved] {save_path}")


# =========================================================
# feature viz
# =========================================================
def _prepare_feature_subset_for_vis(features: np.ndarray, max_points: int, seed: int):
    if len(features) <= max_points:
        idx = np.arange(len(features), dtype=np.int64)
        return features, idx
    rng = np.random.default_rng(seed)
    idx = np.array(rng.choice(len(features), size=max_points, replace=False), dtype=np.int64)
    return features[idx], idx


def _safe_import_umap():
    try:
        import umap  # type: ignore
        return umap
    except Exception:
        return None


def _scatter_plot(
    emb: np.ndarray,
    labels: np.ndarray,
    centers_2d: Optional[np.ndarray],
    save_path: str,
    title: str,
):
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=labels,
        s=12,
        alpha=0.8,
        cmap="viridis",
    )
    if centers_2d is not None:
        plt.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            marker="X",
            s=450,
            c="#1f77b4",
            edgecolors="black",
            linewidths=1.8,
        )
        for i, (x, y) in enumerate(centers_2d):
            plt.text(x + 0.06, y + 0.06, str(i), fontsize=14, weight="bold")

    plt.xlabel(title.split(" ")[0] + "1")
    plt.ylabel(title.split(" ")[0] + "2")
    plt.title(title, fontsize=22)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"[Saved] {save_path}")


def build_feature_visualizations(
    features_for_cluster: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    out_dir: str,
    seed: int = 42,
    max_points: int = 5000,
    make_pca: bool = True,
    make_umap: bool = True,
    make_tsne: bool = False,
    umap_n_neighbors: int = 12,
    umap_min_dist: float = 0.05,
):
    ensure_dir(out_dir)

    feats_sub, idx_sub = _prepare_feature_subset_for_vis(
        features_for_cluster,
        max_points=max_points,
        seed=seed,
    )
    labels_sub = cluster_labels[idx_sub]

    print(f"[FeatureViz] using {len(feats_sub)} / {len(features_for_cluster)} points")

    # PCA
    if make_pca:
        pca = PCA(n_components=2, random_state=seed)
        emb_pca = pca.fit_transform(feats_sub)
        centers_pca = pca.transform(cluster_centers)
        _scatter_plot(
            emb=emb_pca,
            labels=labels_sub,
            centers_2d=centers_pca,
            save_path=os.path.join(out_dir, "pca_by_auto_clusters.png"),
            title="PCA by auto clusters",
        )

    # UMAP
    if make_umap:
        umap_mod = _safe_import_umap()
        if umap_mod is None:
            print("[Warn] umap-learn not installed, skip UMAP")
        else:
            reducer = umap_mod.UMAP(
                n_components=2,
                n_neighbors=umap_n_neighbors,
                min_dist=umap_min_dist,
                metric="euclidean",
                random_state=seed,
            )
            emb_umap = reducer.fit_transform(feats_sub)
            centers_umap = reducer.transform(cluster_centers)
            _scatter_plot(
                emb=emb_umap,
                labels=labels_sub,
                centers_2d=centers_umap,
                save_path=os.path.join(out_dir, "umap_by_auto_clusters.png"),
                title=f"UMAP by auto clusters (k={len(cluster_centers)})",
            )

    # t-SNE
    if make_tsne:
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, len(feats_sub) // 100)),
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        emb_tsne = tsne.fit_transform(feats_sub)
        _scatter_plot(
            emb=emb_tsne,
            labels=labels_sub,
            centers_2d=None,
            save_path=os.path.join(out_dir, "tsne_by_auto_clusters.png"),
            title="t-SNE by auto clusters",
        )


# =========================================================
# board builders
# =========================================================
def build_cluster_board(
    features: np.ndarray,
    metas: List[dict],
    expert_ids: np.ndarray,
    nearest_roles: np.ndarray,
    out_dir: str,
    board_name: str,
    n_clusters: int,
    patches_per_cluster: int,
    selection_mode: str,
    seed: int,
    min_cluster_size: int = 8,
):
    ensure_dir(out_dir)

    feats_norm = l2_normalize_np(features)
    labels, centers = cluster_features_kmeans(feats_norm, n_clusters=n_clusters, seed=seed)

    summary_rows = []
    rows_tiles = []
    row_titles = []

    uniq = sorted(np.unique(labels).tolist())
    for cid in uniq:
        idx = np.where(labels == cid)[0]
        if len(idx) < min_cluster_size:
            continue

        feats_c = feats_norm[idx]
        center_c = centers[cid]

        if selection_mode == "center":
            local_pick = get_nearest_indices_to_center(feats_c, center_c, topk=patches_per_cluster)
            pick_idx = idx[local_pick]
        elif selection_mode == "random":
            pick_idx = get_random_indices_in_cluster(idx, topk=patches_per_cluster, seed=seed + cid)
        else:
            k1 = max(1, patches_per_cluster // 2)
            k2 = patches_per_cluster - k1
            local_center = get_nearest_indices_to_center(feats_c, center_c, topk=min(k1, len(idx)))
            center_idx = idx[local_center]
            remain = np.array([x for x in idx if x not in set(center_idx.tolist())], dtype=np.int64)
            if len(remain) > 0 and k2 > 0:
                rand_idx = get_random_indices_in_cluster(remain, topk=min(k2, len(remain)), seed=seed + cid)
                pick_idx = np.concatenate([center_idx, rand_idx], axis=0)
            else:
                pick_idx = center_idx

        expert_ratio = ratio_dict_int(expert_ids[idx].tolist())
        role_ratio = ratio_dict(nearest_roles[idx].tolist())

        summary_rows.append({
            "cluster_id": int(cid),
            "cluster_size": int(len(idx)),
            "expert_ratio_json": json.dumps(expert_ratio, ensure_ascii=False),
            "role_ratio_json": json.dumps(role_ratio, ensure_ascii=False),
        })

        tiles = []
        for j in pick_idx:
            m = metas[j]
            img = read_patch_from_meta(m, out_size=160)
            txt = [
                f"slide={m['slide_id']}",
                f"E{int(expert_ids[j])} | role={nearest_roles[j]}",
                f"({m['coord_x']},{m['coord_y']})",
            ]
            tiles.append(draw_patch_tile(img, txt, tile_w=180, img_h=160, text_h=56))

        row_title = (
            f"C{cid} | n={len(idx)} | "
            f"expert={dict(list(expert_ratio.items())[:3])} | "
            f"role={dict(list(role_ratio.items())[:3])}"
        )
        rows_tiles.append(tiles)
        row_titles.append(row_title)

    summary_csv = os.path.join(out_dir, f"{board_name}_cluster_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")

    board_path = os.path.join(out_dir, f"{board_name}_cluster_board.jpg")
    paste_grid(
        rows_tiles=rows_tiles,
        row_titles=row_titles,
        save_path=board_path,
        title=f"{board_name} clusters",
        left_title_w=520,
        pad=10,
    )

    return labels, centers


def build_per_expert_boards(
    features: np.ndarray,
    metas: List[dict],
    expert_ids: np.ndarray,
    nearest_roles: np.ndarray,
    out_dir: str,
    base_k: int,
    patches_per_cluster: int,
    selection_mode: str,
    seed: int,
    min_tokens_per_expert: int = 50,
    min_cluster_size: int = 6,
):
    ensure_dir(out_dir)

    feats_norm = l2_normalize_np(features)
    uniq_experts = sorted(np.unique(expert_ids).tolist())
    summary_rows = []

    for e in uniq_experts:
        idx_e = np.where(expert_ids == e)[0]
        n_e = len(idx_e)
        if n_e < min_tokens_per_expert:
            print(f"[Skip] E{e}: too few samples ({n_e})")
            continue

        feats_e = feats_norm[idx_e]
        k_e = min(base_k, max(2, n_e // max(20, patches_per_cluster)))
        labels_e, centers_e = cluster_features_kmeans(feats_e, n_clusters=k_e, seed=seed + e)

        rows_tiles = []
        row_titles = []

        uniq_c = sorted(np.unique(labels_e).tolist())
        for cid in uniq_c:
            local_idx = np.where(labels_e == cid)[0]
            if len(local_idx) < min_cluster_size:
                continue

            global_idx = idx_e[local_idx]
            feats_c = feats_e[local_idx]
            center_c = centers_e[cid]

            if selection_mode == "center":
                pick_local = get_nearest_indices_to_center(feats_c, center_c, topk=patches_per_cluster)
                pick_idx = global_idx[pick_local]
            elif selection_mode == "random":
                pick_idx = get_random_indices_in_cluster(global_idx, topk=patches_per_cluster, seed=seed + e * 100 + cid)
            else:
                k1 = max(1, patches_per_cluster // 2)
                k2 = patches_per_cluster - k1
                pick_center_local = get_nearest_indices_to_center(feats_c, center_c, topk=min(k1, len(global_idx)))
                pick_center_idx = global_idx[pick_center_local]
                remain = np.array([x for x in global_idx if x not in set(pick_center_idx.tolist())], dtype=np.int64)
                if len(remain) > 0 and k2 > 0:
                    pick_rand_idx = get_random_indices_in_cluster(remain, topk=min(k2, len(remain)), seed=seed + e * 100 + cid)
                    pick_idx = np.concatenate([pick_center_idx, pick_rand_idx], axis=0)
                else:
                    pick_idx = pick_center_idx

            role_ratio = ratio_dict(nearest_roles[global_idx].tolist())

            summary_rows.append({
                "expert_id": int(e),
                "cluster_id": int(cid),
                "cluster_size": int(len(global_idx)),
                "role_ratio_json": json.dumps(role_ratio, ensure_ascii=False),
            })

            tiles = []
            for j in pick_idx:
                m = metas[j]
                img = read_patch_from_meta(m, out_size=160)
                txt = [
                    f"slide={m['slide_id']}",
                    f"E{int(expert_ids[j])} | role={nearest_roles[j]}",
                    f"({m['coord_x']},{m['coord_y']})",
                ]
                tiles.append(draw_patch_tile(img, txt, tile_w=180, img_h=160, text_h=56))

            row_title = f"E{e}-C{cid} | n={len(global_idx)} | role={dict(list(role_ratio.items())[:3])}"
            rows_tiles.append(tiles)
            row_titles.append(row_title)

        board_path = os.path.join(out_dir, f"expert_E{e}_cluster_board.jpg")
        paste_grid(
            rows_tiles=rows_tiles,
            row_titles=row_titles,
            save_path=board_path,
            title=f"Per-expert clusters: E{e}",
            left_title_w=420,
            pad=10,
        )

    summary_csv = os.path.join(out_dir, "per_expert_cluster_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")


# =========================================================
# main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Sample patches online -> extract feature -> cluster boards")

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--full-ckpt", type=str, required=True)
    parser.add_argument("--role-proto-dir", type=str, required=True)
    parser.add_argument("--pool-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--sample-per-project", type=int, default=None)
    parser.add_argument("--sample-per-label", type=int, default=None)

    parser.add_argument("--keep-projects", nargs="+", default=None)
    parser.add_argument("--keep-labels", nargs="+", default=None)

    parser.add_argument("--global-k", type=int, default=16)
    parser.add_argument("--per-expert-k", type=int, default=8)
    parser.add_argument("--patches-per-cluster", type=int, default=6)
    parser.add_argument("--selection-mode", type=str, default="center_random_mix",
                        choices=["center", "random", "center_random_mix"])

    parser.add_argument("--make-global-board", action="store_true")
    parser.add_argument("--make-per-expert-board", action="store_true")

    parser.add_argument("--make-feature-viz", action="store_true")
    parser.add_argument("--make-pca", action="store_true")
    parser.add_argument("--make-umap", action="store_true")
    parser.add_argument("--make-tsne", action="store_true")
    parser.add_argument("--viz-max-points", type=int, default=5000)

    parser.add_argument("--feature-level", type=str, default="token", choices=["token", "patch"])
    parser.add_argument("--cluster-feature-space", type=str, default="role", choices=["raw", "role"])

    parser.add_argument("--umap-n-neighbors", type=int, default=12)
    parser.add_argument("--umap-min-dist", type=float, default=0.05)

    parser.add_argument("--csv-chunksize", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    if not args.make_global_board and not args.make_per_expert_board and not args.make_feature_viz:
        args.make_global_board = True
        args.make_per_expert_board = True
        args.make_feature_viz = True
        args.make_pca = True
        args.make_umap = True

    if args.make_feature_viz and (not args.make_pca and not args.make_umap and not args.make_tsne):
        args.make_pca = True
        args.make_umap = True

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    model, role_proj_head, _ = load_stage2_bundle(
        args.config,
        args.full_ckpt,
        device=device,
    )
    role_prototypes, role_names = load_role_prototypes(args.role_proto_dir)

    transform = build_transform(image_size=args.image_size)
    dataset = TCGAPoolDataset(
        csv_path=args.pool_csv,
        transform=transform,
        max_rows=args.max_rows,
        sample_per_project=args.sample_per_project,
        sample_per_label=args.sample_per_label,
        seed=args.seed,
        keep_projects=args.keep_projects,
        keep_labels=args.keep_labels,
        csv_chunksize=args.csv_chunksize,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_with_meta,
    )

    print(f"[Debug] len(dataset) = {len(dataset)}")
    print(f"[Debug] len(loader) = {len(loader)}")

    all_features_raw = []
    all_expert_ids = []
    all_metas = []

    for batch_idx, (images, metas) in enumerate(tqdm(loader, total=len(loader), desc="Extract")):
        if batch_idx == 0:
            print(f"[Debug] first batch arrived: images.shape={tuple(images.shape)}")

        images = images.to(device, non_blocking=True)

        final_feats, gate_info_list, moe_feature_list = run_model_and_collect(model, images)
        seq_len = final_feats.shape[1]

        last_gate_info = gate_info_list[-1]
        last_moe_feats = moe_feature_list[-1]      # [B, seq_len, D]
        expert_ids_tok = get_expert_assignment_from_gate_info(last_gate_info, seq_len=seq_len)  # [B, N]
        moe_feats = last_moe_feats[:, 1:, :]       # [B, N, D]

        B, N, D = moe_feats.shape

        if args.feature_level == "token":
            feats_raw = moe_feats.detach().cpu().reshape(B * N, D).numpy()
            exp_np = expert_ids_tok.detach().cpu().reshape(B * N).numpy()

            all_features_raw.append(feats_raw)
            all_expert_ids.append(exp_np)

            for b in range(B):
                meta_b = metas[b]
                for _ in range(N):
                    all_metas.append(meta_b.copy())

        else:  # patch
            patch_feats = moe_feats.mean(dim=1)  # [B, D]

            expert_ids_patch = []
            expert_ids_np = expert_ids_tok.detach().cpu().numpy()
            for b in range(B):
                vals, counts = np.unique(expert_ids_np[b], return_counts=True)
                expert_ids_patch.append(int(vals[np.argmax(counts)]))
            expert_ids_patch = np.array(expert_ids_patch, dtype=np.int64)

            all_features_raw.append(patch_feats.detach().cpu().numpy())
            all_expert_ids.append(expert_ids_patch)

            for b in range(B):
                all_metas.append(metas[b].copy())

    features_raw = np.concatenate(all_features_raw, axis=0).astype(np.float32)
    expert_ids = np.concatenate(all_expert_ids, axis=0).astype(np.int64)

    print("\n===== Summary =====")
    print("feature_level:", args.feature_level)
    print("raw features shape:", features_raw.shape)
    uniq_e, cnt_e = np.unique(expert_ids, return_counts=True)
    for e, c in zip(uniq_e, cnt_e):
        print(f"  E{int(e)}: {int(c)}")

    features_role = project_features_to_role_space(
        features_raw,
        role_proj_head,
        device=device,
    )
    role_affinity = compute_role_affinity(features_role, role_prototypes)
    nearest_role_ids, nearest_role_labels = nearest_role_assignment(role_affinity, role_names)

    # choose feature space for clustering / viz
    if args.cluster_feature_space == "role":
        features_for_cluster = features_role.astype(np.float32)
    else:
        features_for_cluster = features_raw.astype(np.float32)

    analysis_df = pd.DataFrame(all_metas)
    analysis_df["expert_id"] = expert_ids
    analysis_df["nearest_role"] = nearest_role_labels
    analysis_df["nearest_role_id"] = nearest_role_ids
    analysis_df.to_csv(os.path.join(args.output_dir, "sampled_token_analysis.csv"), index=False)
    print(f"[Saved] {os.path.join(args.output_dir, 'sampled_token_analysis.csv')}")

    # global cluster board + cluster labels
    cluster_labels_global = None
    cluster_centers_global = None
    if args.make_global_board or args.make_feature_viz:
        cluster_labels_global, cluster_centers_global = build_cluster_board(
            features=features_for_cluster,
            metas=all_metas,
            expert_ids=expert_ids,
            nearest_roles=nearest_role_labels,
            out_dir=os.path.join(args.output_dir, "global_clusters"),
            board_name="global",
            n_clusters=args.global_k,
            patches_per_cluster=args.patches_per_cluster,
            selection_mode=args.selection_mode,
            seed=args.seed,
        )

    if args.make-per-expert-board if False else False:
        pass

    if args.make_per_expert_board:
        build_per_expert_boards(
            features=features_for_cluster,
            metas=all_metas,
            expert_ids=expert_ids,
            nearest_roles=nearest_role_labels,
            out_dir=os.path.join(args.output_dir, "per_expert_clusters"),
            base_k=args.per_expert_k,
            patches_per_cluster=args.patches_per_cluster,
            selection_mode=args.selection_mode,
            seed=args.seed,
        )

    if args.make_feature_viz:
        if cluster_labels_global is None or cluster_centers_global is None:
            cluster_labels_global, cluster_centers_global = cluster_features_kmeans(
                l2_normalize_np(features_for_cluster),
                n_clusters=args.global_k,
                seed=args.seed,
            )

        build_feature_visualizations(
            features_for_cluster=l2_normalize_np(features_for_cluster),
            cluster_labels=cluster_labels_global,
            cluster_centers=l2_normalize_np(cluster_centers_global),
            out_dir=os.path.join(args.output_dir, "feature_viz"),
            seed=args.seed,
            max_points=args.viz_max_points,
            make_pca=args.make_pca,
            make_umap=args.make_umap,
            make_tsne=args.make_tsne,
            umap_n_neighbors=args.umap_n_neighbors,
            umap_min_dist=args.umap_min_dist,
        )

    print(f"\n[Done] saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()