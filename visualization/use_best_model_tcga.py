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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.v2 as T

import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import umap
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
    ):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"csv not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required = ["svs_path", "coord_x", "coord_y", "patch_level", "patch_size"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"csv missing required columns: {missing}")

        df = df.copy()
        df["svs_path"] = df["svs_path"].map(canonicalize_path)

        if keep_projects is not None and "project" in df.columns:
            df = df[df["project"].isin(keep_projects)].copy()

        if keep_labels is not None and "pred_label" in df.columns:
            df = df[df["pred_label"].isin(keep_labels)].copy()

        rng = np.random.default_rng(seed)

        if sample_per_project is not None and "project" in df.columns:
            parts = []
            for _, sub in df.groupby("project"):
                if len(sub) > sample_per_project:
                    choose_idx = rng.choice(len(sub), size=sample_per_project, replace=False)
                    sub = sub.iloc[choose_idx].copy()
                parts.append(sub)
            df = pd.concat(parts, axis=0).reset_index(drop=True)

        if sample_per_label is not None and "pred_label" in df.columns:
            parts = []
            for _, sub in df.groupby("pred_label"):
                if len(sub) > sample_per_label:
                    choose_idx = rng.choice(len(sub), size=sample_per_label, replace=False)
                    sub = sub.iloc[choose_idx].copy()
                parts.append(sub)
            df = pd.concat(parts, axis=0).reset_index(drop=True)

        if max_rows is not None and len(df) > max_rows:
            choose_idx = rng.choice(len(df), size=max_rows, replace=False)
            df = df.iloc[choose_idx].copy().reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        self.transform = transform

        print(f"[Dataset] rows = {len(self.df)}")
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
        x = int(row["coord_x"])
        y = int(row["coord_y"])
        patch_level = int(row["patch_level"])
        patch_size = int(row["patch_size"])

        slide = openslide.OpenSlide(svs_path)
        try:
            img = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
        finally:
            slide.close()

        img = self.transform(img)

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
        return img, meta


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
    dispatch = dispatch.view(B, seq_len, num_experts)[:, 1:, :]  # remove CLS
    expert_id = dispatch.argmax(dim=-1)  # [B, num_patches]
    return expert_id


@torch.no_grad()
def project_features_to_role_space(features: np.ndarray, proj_head, device="cpu", batch_size=4096):
    outs = []
    for start in range(0, len(features), batch_size):
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
# tables / plots
# =========================================================
def build_expert_role_affinity_table(role_affinity, expert_ids, num_experts=None):
    if num_experts is None:
        num_experts = int(np.max(expert_ids)) + 1
    R = role_affinity.shape[1]
    table = np.zeros((num_experts, R), dtype=np.float32)
    for e in range(num_experts):
        idx = expert_ids == e
        if idx.sum() == 0:
            continue
        table[e] = role_affinity[idx].mean(axis=0)
    return table


def build_expert_role_ratio_table(nearest_role_ids, expert_ids, num_roles, num_experts=None):
    if num_experts is None:
        num_experts = int(np.max(expert_ids)) + 1
    table = np.zeros((num_experts, num_roles), dtype=np.float32)
    for e in range(num_experts):
        idx = expert_ids == e
        if idx.sum() == 0:
            continue
        total = idx.sum()
        for r in range(num_roles):
            table[e, r] = np.sum(nearest_role_ids[idx] == r) / max(total, 1)
    return table


def build_expert_category_ratio_table(category_values, expert_ids, category_names, num_experts=None):
    if num_experts is None:
        num_experts = int(np.max(expert_ids)) + 1
    table = np.zeros((num_experts, len(category_names)), dtype=np.float32)
    cat_to_idx = {c: i for i, c in enumerate(category_names)}
    for e in range(num_experts):
        idx = expert_ids == e
        if idx.sum() == 0:
            continue
        vals = category_values[idx]
        total = len(vals)
        for v in vals:
            if v in cat_to_idx:
                table[e, cat_to_idx[v]] += 1
        table[e] /= max(total, 1)
    return table


def plot_heatmap(table, row_labels, col_labels, title, save_path, cmap="YlGnBu", vmin=None, vmax=None, cbar_label=None):
    plt.figure(figsize=(1.2 * len(col_labels) + 3, 0.9 * len(row_labels) + 2))
    im = plt.imshow(table, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    plt.xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    cb = plt.colorbar(im)
    if cbar_label is not None:
        cb.set_label(cbar_label)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def plot_umap(embedding, labels, label_names, title, save_path):
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(label_names), 1)))
    label_to_color = {name: colors[i] for i, name in enumerate(label_names)}

    for name in label_names:
        idx = labels == name
        if idx.sum() == 0:
            continue
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c=[label_to_color[name]],
            s=6,
            alpha=0.75,
            label=f"{name} (n={int(idx.sum())})"
        )

    plt.legend(markerscale=2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


def analyze_free_expert_residue(role_affinity, expert_ids, role_names, free_expert_id=3):
    idx = expert_ids == free_expert_id
    if idx.sum() == 0:
        return None

    aff = role_affinity[idx]
    mean_aff = aff.mean(axis=0)
    std_aff = aff.std(axis=0)

    out = {
        "count": int(idx.sum()),
        "mean_affinity": {role_names[i]: float(mean_aff[i]) for i in range(len(role_names))},
        "std_affinity": {role_names[i]: float(std_aff[i]) for i in range(len(role_names))},
    }
    return out


# =========================================================
# main analysis
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze TCGA stage2 role-proto model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--full-ckpt", type=str, required=True)
    parser.add_argument("--role-proto-dir", type=str, required=True)
    parser.add_argument("--pool-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--max-rows", type=int, default=400)
    parser.add_argument("--sample-per-project", type=int, default=None)
    parser.add_argument("--sample-per-label", type=int, default=None)

    parser.add_argument("--keep-projects", nargs="+", default=None)
    parser.add_argument("--keep-labels", nargs="+", default=None)

    parser.add_argument("--free-expert-id", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", type=str, default="cosine")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    model, role_proj_head, cfg = load_stage2_bundle(
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

    all_features_moe = []
    all_expert_ids = []
    all_projects = []
    all_pred_labels = []
    all_slide_ids = []
    all_token_metas = []

    for images, metas in loader:
        images = images.to(device, non_blocking=True)

        final_feats, gate_info_list, moe_feature_list = run_model_and_collect(model, images)
        seq_len = final_feats.shape[1]

        last_gate_info = gate_info_list[-1]
        last_moe_feats = moe_feature_list[-1]     # [B, seq_len, D]
        expert_ids = get_expert_assignment_from_gate_info(last_gate_info, seq_len=seq_len)  # [B, N]

        moe_feats = last_moe_feats[:, 1:, :]      # [B, N, D]
        B, N, D = moe_feats.shape

        all_features_moe.append(moe_feats.detach().cpu().reshape(B * N, D).numpy())
        all_expert_ids.append(expert_ids.detach().cpu().reshape(B * N).numpy())

        for b in range(B):
            meta = metas[b]
            for t in range(N):
                all_projects.append(meta["project"])
                all_pred_labels.append(meta["pred_label"])
                all_slide_ids.append(meta["slide_id"])
                all_token_metas.append({
                    "project": meta["project"],
                    "slide_id": meta["slide_id"],
                    "pred_label": meta["pred_label"],
                    "svs_path": meta["svs_path"],
                    "coord_x": meta["coord_x"],
                    "coord_y": meta["coord_y"],
                    "coord_idx": meta["coord_idx"],
                    "patch_level": meta["patch_level"],
                    "patch_size": meta["patch_size"],
                    "token_idx": int(t),
                    "expert_id": int(expert_ids[b, t].item()),
                })

    features_moe = np.concatenate(all_features_moe, axis=0).astype(np.float32)
    expert_ids = np.concatenate(all_expert_ids, axis=0).astype(np.int64)
    project_arr = np.array(all_projects, dtype=object)
    pred_label_arr = np.array(all_pred_labels, dtype=object)
    slide_id_arr = np.array(all_slide_ids, dtype=object)

    print("\n===== Summary =====")
    print("features_moe shape:", features_moe.shape)
    print("expert_ids shape :", expert_ids.shape)
    print("expert counts:")
    uniq_e, cnt_e = np.unique(expert_ids, return_counts=True)
    for e, c in zip(uniq_e, cnt_e):
        print(f"  E{int(e)}: {int(c)}")

    # role-space projection
    features_role_space = project_features_to_role_space(
        features_moe,
        role_proj_head,
        device=device,
    )
    role_affinity = compute_role_affinity(features_role_space, role_prototypes)
    nearest_role_ids, nearest_role_labels = nearest_role_assignment(role_affinity, role_names)

    # save token-level csv
    token_df = pd.DataFrame(all_token_metas)
    token_df["nearest_role"] = nearest_role_labels
    token_df["nearest_role_id"] = nearest_role_ids
    for i, rname in enumerate(role_names):
        token_df[f"role_affinity_{rname}"] = role_affinity[:, i]
    token_csv_path = os.path.join(args.output_dir, "token_level_analysis.csv")
    token_df.to_csv(token_csv_path, index=False)
    print(f"[Saved] {token_csv_path}")

    # tables
    num_experts = int(np.max(expert_ids)) + 1
    expert_row_names = [f"E{i}" for i in range(num_experts)]

    aff_table = build_expert_role_affinity_table(role_affinity, expert_ids, num_experts=num_experts)
    ratio_table = build_expert_role_ratio_table(nearest_role_ids, expert_ids, num_roles=len(role_names), num_experts=num_experts)

    project_names = sorted(pd.unique(project_arr).tolist())
    pred_label_names = sorted(pd.unique(pred_label_arr).tolist())

    project_ratio_table = build_expert_category_ratio_table(project_arr, expert_ids, project_names, num_experts=num_experts)
    label_ratio_table = build_expert_category_ratio_table(pred_label_arr, expert_ids, pred_label_names, num_experts=num_experts)

    np.save(os.path.join(args.output_dir, "expert_role_affinity_table.npy"), aff_table)
    np.save(os.path.join(args.output_dir, "expert_role_ratio_table.npy"), ratio_table)
    np.save(os.path.join(args.output_dir, "expert_project_ratio_table.npy"), project_ratio_table)
    np.save(os.path.join(args.output_dir, "expert_pred_label_ratio_table.npy"), label_ratio_table)

    plot_heatmap(
        aff_table,
        expert_row_names,
        role_names,
        title="Expert x Role Affinity (last MoE block)",
        save_path=os.path.join(args.output_dir, "expert_role_affinity_heatmap.png"),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        cbar_label="Mean cosine to role prototype",
    )

    plot_heatmap(
        ratio_table,
        expert_row_names,
        role_names,
        title="Expert x Nearest-Role Ratio (last MoE block)",
        save_path=os.path.join(args.output_dir, "expert_nearest_role_ratio_heatmap.png"),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Ratio",
    )

    plot_heatmap(
        project_ratio_table,
        expert_row_names,
        project_names,
        title="Expert x Project Ratio (last MoE block)",
        save_path=os.path.join(args.output_dir, "expert_project_ratio_heatmap.png"),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Ratio",
    )

    plot_heatmap(
        label_ratio_table,
        expert_row_names,
        pred_label_names,
        title="Expert x Pred-label Ratio (last MoE block)",
        save_path=os.path.join(args.output_dir, "expert_pred_label_ratio_heatmap.png"),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Ratio",
    )

    # UMAP on moe features
    reducer = umap.UMAP(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    )
    features_norm = l2_normalize_np(features_moe)
    embedding = reducer.fit_transform(features_norm)

    expert_label_names = [f"E{i}" for i in range(num_experts)]
    expert_name_arr = np.array([f"E{i}" for i in expert_ids], dtype=object)

    plot_umap(
        embedding,
        expert_name_arr,
        expert_label_names,
        title="UMAP colored by expert (last MoE block)",
        save_path=os.path.join(args.output_dir, "umap_by_expert.png"),
    )

    plot_umap(
        embedding,
        nearest_role_labels,
        role_names,
        title="UMAP colored by nearest role (last MoE block)",
        save_path=os.path.join(args.output_dir, "umap_by_nearest_role.png"),
    )

    plot_umap(
        embedding,
        pred_label_arr,
        pred_label_names,
        title="UMAP colored by pred_label (last MoE block)",
        save_path=os.path.join(args.output_dir, "umap_by_pred_label.png"),
    )

    free_stats = analyze_free_expert_residue(
        role_affinity=role_affinity,
        expert_ids=expert_ids,
        role_names=role_names,
        free_expert_id=args.free_expert_id,
    )

    np.save(os.path.join(args.output_dir, "features_moe.npy"), features_moe)
    np.save(os.path.join(args.output_dir, "features_role_space.npy"), features_role_space)
    np.save(os.path.join(args.output_dir, "role_affinity.npy"), role_affinity)
    np.save(os.path.join(args.output_dir, "umap_embedding.npy"), embedding)

    summary = {
        "role_names": role_names,
        "num_tokens": int(len(expert_ids)),
        "num_experts": int(num_experts),
        "expert_counts": {f"E{int(e)}": int(c) for e, c in zip(uniq_e, cnt_e)},
        "free_expert_id": int(args.free_expert_id),
        "free_expert_stats": free_stats,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== Free Expert Residue =====")
    if free_stats is None:
        print(f"No tokens assigned to free expert E{args.free_expert_id}")
    else:
        print(json.dumps(free_stats, indent=2))

    print(f"\n[Done] saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()