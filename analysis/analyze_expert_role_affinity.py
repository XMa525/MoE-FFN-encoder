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
from typing import List, Optional

import numpy as np
import pandas as pd
import openslide
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

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
# loading
# =========================================================
def load_stage2_bundle(config_path: str, full_ckpt_path: str, device: str = "cuda"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(full_ckpt_path, map_location="cpu")

    if "student_state_dict" not in ckpt:
        raise KeyError("student_state_dict not found in checkpoint")
    if "distiller_state_dict" not in ckpt:
        raise KeyError("distiller_state_dict not found in checkpoint")

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    model.load_state_dict(ckpt["student_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    distiller_sd = ckpt["distiller_state_dict"]
    if "proj_l12.weight" not in distiller_sd or "proj_l12.bias" not in distiller_sd:
        raise KeyError("proj_l12 not found in distiller_state_dict")

    proj_out_dim, proj_in_dim = distiller_sd["proj_l12.weight"].shape
    proj_l12 = nn.Linear(proj_in_dim, proj_out_dim)
    proj_l12.load_state_dict({
        "weight": distiller_sd["proj_l12.weight"],
        "bias": distiller_sd["proj_l12.bias"],
    })
    proj_l12 = proj_l12.to(device)
    proj_l12.eval()

    print("Loaded stage2 student + proj_l12")
    print(f"Current moe_layers_idx = {model.moe_layers_idx}")
    print(f"proj_l12 shape: {proj_in_dim} -> {proj_out_dim}")
    return model, proj_l12, cfg


def load_role_proto_from_dir(role_proto_dir: str):
    proto_path = os.path.join(role_proto_dir, "role_prototypes_init.npy")
    names_path = os.path.join(role_proto_dir, "role_names.json")

    if not os.path.exists(proto_path):
        raise FileNotFoundError(f"Missing prototype file: {proto_path}")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Missing role names file: {names_path}")

    protos = np.load(proto_path).astype(np.float32)
    with open(names_path, "r", encoding="utf-8") as f:
        role_names = json.load(f)

    protos = l2_normalize_np(protos)
    print(f"[RoleProto] loaded from dir: {role_proto_dir}")
    print(f"[RoleProto] role names = {role_names}")
    print(f"[RoleProto] shape = {protos.shape}")
    return protos, role_names


def load_role_proto_from_stage3_ckpt(stage3_ckpt_path: str, role_proto_dir: str):
    if not os.path.exists(stage3_ckpt_path):
        raise FileNotFoundError(f"stage3 ckpt not found: {stage3_ckpt_path}")

    names_path = os.path.join(role_proto_dir, "role_names.json")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Missing role names file: {names_path}")

    with open(names_path, "r", encoding="utf-8") as f:
        role_names = json.load(f)

    ckpt = torch.load(stage3_ckpt_path, map_location="cpu")
    if "stage3_state_dict" in ckpt:
        sd = ckpt["stage3_state_dict"]
    else:
        sd = ckpt

    key = "shared_role_proto.prototypes"
    if key not in sd:
        raise KeyError(f"{key} not found in stage3 checkpoint")

    protos = sd[key].detach().cpu().numpy().astype(np.float32)
    protos = l2_normalize_np(protos)

    print(f"[RoleProto] loaded refined proto from stage3 ckpt: {stage3_ckpt_path}")
    print(f"[RoleProto] role names = {role_names}")
    print(f"[RoleProto] shape = {protos.shape}")
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
        split_csv: Optional[str] = None,
        split_keep: Optional[List[str]] = None,
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

        if "prefilter_white" in df.columns:
            df = df[df["prefilter_white"].fillna(0).astype(int) == 0].copy()

        if keep_projects is not None and "project" in df.columns:
            df = df[df["project"].isin(keep_projects)].copy()

        if keep_labels is not None and "pred_label" in df.columns:
            df = df[df["pred_label"].isin(keep_labels)].copy()

        if split_csv is not None and split_keep is not None:
            sdf = pd.read_csv(split_csv).copy()
            if "source_path" in sdf.columns:
                sdf["match_key"] = sdf["source_path"].astype(str).map(
                    lambda x: os.path.basename(canonicalize_path(x)).lower().strip()
                )
                df["match_key"] = df["svs_path"].astype(str).map(
                    lambda x: os.path.basename(canonicalize_path(x)).lower().strip()
                )
            elif "slide_id" in sdf.columns and "slide_id" in df.columns:
                sdf["match_key"] = sdf["slide_id"].astype(str).str.strip()
                df["match_key"] = df["slide_id"].astype(str).str.strip()
            else:
                raise ValueError("split matching requires source_path or slide_id")

            keep_keys = set(
                sdf.loc[sdf["split"].isin(split_keep), "match_key"].tolist()
            )
            df = df[df["match_key"].isin(keep_keys)].copy()

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
            print("[Dataset] pred_label counts:")
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
    final_feats, gate_info_list, feature_dict, moe_feature_list = model(
        img_tensor,
        return_gates=True,
        return_features=True,
        is_eval=True,
    )

    if len(gate_info_list) == 0:
        raise RuntimeError("gate_info_list is empty")
    if len(moe_feature_list) == 0:
        raise RuntimeError("moe_feature_list is empty")

    return final_feats, gate_info_list, feature_dict, moe_feature_list


def get_last_dispatch_weight(gate_info, seq_len):
    dispatch = gate_info["dispatch_weight"]  # [B*seq_len, E]
    total_tokens, num_experts = dispatch.shape
    B = total_tokens // seq_len
    dispatch = dispatch.view(B, seq_len, num_experts)[:, 1:, :]
    return dispatch  # [B, N, E]


@torch.no_grad()
def project_features_to_role_space(
    features: np.ndarray,
    proj_head,
    device="cpu",
    batch_size=4096,
    show_pbar: bool = True,
):
    outs = []

    iterator = range(0, len(features), batch_size)
    if show_pbar:
        iterator = tqdm(
            iterator,
            total=(len(features) + batch_size - 1) // batch_size,
            desc="Projecting to role space",
            leave=False,
        )

    for start in iterator:
        x = torch.from_numpy(features[start:start + batch_size]).float().to(device)
        y = proj_head(x)
        y = F.normalize(y, dim=-1)
        outs.append(y.cpu().numpy())

    return np.concatenate(outs, axis=0)


def compute_role_affinity(features_role_space: np.ndarray, role_prototypes: np.ndarray):
    feats = l2_normalize_np(features_role_space)
    protos = l2_normalize_np(role_prototypes)
    return feats @ protos.T


def maybe_subsample_tokens(
    expert_ids: np.ndarray,
    dispatch_conf: np.ndarray,
    role_affinity: np.ndarray,
    token_meta: List[dict],
    max_token_points: int,
    seed: int = 42,
):
    num_tokens = len(expert_ids)
    if max_token_points is None or max_token_points <= 0 or num_tokens <= max_token_points:
        print(f"[Stage] token subsample skipped: num_tokens={num_tokens}")
        return expert_ids, dispatch_conf, role_affinity, token_meta

    print(f"[Stage] subsampling tokens: {num_tokens} -> {max_token_points}")
    rng = np.random.default_rng(seed)
    keep_idx = rng.choice(num_tokens, size=max_token_points, replace=False)
    keep_idx = np.sort(keep_idx)

    token_meta_sub = [token_meta[i] for i in keep_idx.tolist()]

    return (
        expert_ids[keep_idx],
        dispatch_conf[keep_idx],
        role_affinity[keep_idx],
        token_meta_sub,
    )


# =========================================================
# analysis
# =========================================================
def build_mean_affinity_table(role_affinity, expert_ids, num_experts):
    R = role_affinity.shape[1]
    table = np.zeros((num_experts, R), dtype=np.float32)
    counts = np.zeros((num_experts,), dtype=np.int64)

    for e in range(num_experts):
        idx = (expert_ids == e)
        counts[e] = int(idx.sum())
        if idx.sum() > 0:
            table[e] = role_affinity[idx].mean(axis=0)
    return table, counts


def build_nearest_role_ratio_table(nearest_role_ids, expert_ids, num_roles, num_experts):
    table = np.zeros((num_experts, num_roles), dtype=np.float32)
    for e in range(num_experts):
        idx = (expert_ids == e)
        total = int(idx.sum())
        if total == 0:
            continue
        for r in range(num_roles):
            table[e, r] = float((nearest_role_ids[idx] == r).sum()) / total
    return table


def summarize_global_role_ratio(nearest_role_ids, role_names):
    out = {}
    total = len(nearest_role_ids)
    for r, name in enumerate(role_names):
        out[name] = float((nearest_role_ids == r).sum()) / max(total, 1)
    return out


def summarize_semantic_expert_role_ratio(
    nearest_role_ids,
    expert_ids,
    semantic_expert_ids,
    role_names,
):
    mask = np.isin(expert_ids, np.array(semantic_expert_ids))
    out = {}
    total = int(mask.sum())
    for r, name in enumerate(role_names):
        if total == 0:
            out[name] = 0.0
        else:
            out[name] = float((nearest_role_ids[mask] == r).sum()) / total
    return out


def plot_heatmap(table, row_labels, col_labels, title, save_path, cmap="YlGnBu", vmin=None, vmax=None):
    plt.figure(figsize=(1.2 * len(col_labels) + 3, 0.9 * len(row_labels) + 2))
    im = plt.imshow(table, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    plt.xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)
    plt.colorbar(im)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            plt.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()


# =========================================================
# main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze expert-role affinity")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage2-full-ckpt", type=str, required=True)
    parser.add_argument("--role-proto-dir", type=str, required=True)
    parser.add_argument("--pool-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--stage3-ckpt", type=str, default=None,
                        help="If provided, use refined shared_role_proto from stage3 ckpt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--max-rows", type=int, default=400)
    parser.add_argument("--max-token-points", type=int, default=0,
                        help="If > 0, randomly subsample tokens after role affinity")
    parser.add_argument("--sample-per-project", type=int, default=None)
    parser.add_argument("--sample-per-label", type=int, default=None)
    parser.add_argument("--keep-projects", nargs="+", default=None)
    parser.add_argument("--keep-labels", nargs="+", default=None)

    parser.add_argument("--split-csv", type=str, default=None)
    parser.add_argument("--split-keep", nargs="+", default=None)

    parser.add_argument("--semantic-expert-ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--free-expert-id", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    model, proj_l12, cfg = load_stage2_bundle(
        config_path=args.config,
        full_ckpt_path=args.stage2_full_ckpt,
        device=device,
    )

    if args.stage3_ckpt is not None:
        role_prototypes, role_names = load_role_proto_from_stage3_ckpt(
            stage3_ckpt_path=args.stage3_ckpt,
            role_proto_dir=args.role_proto_dir,
        )
        proto_source = "stage3_refined"
    else:
        role_prototypes, role_names = load_role_proto_from_dir(args.role_proto_dir)
        proto_source = "stage2_init"

    transform = build_transform(args.image_size)
    dataset = TCGAPoolDataset(
        csv_path=args.pool_csv,
        transform=transform,
        max_rows=args.max_rows,
        sample_per_project=args.sample_per_project,
        sample_per_label=args.sample_per_label,
        seed=args.seed,
        keep_projects=args.keep_projects,
        keep_labels=args.keep_labels,
        split_csv=args.split_csv,
        split_keep=args.split_keep,
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

    all_role_affinity = []
    all_expert_ids = []
    all_dispatch_conf = []
    all_token_meta = []

    print("[Stage] start model forward + token collection")

    for batch_idx, (images, metas) in enumerate(tqdm(loader, desc="Running model on patches")):
        images = images.to(device, non_blocking=True)

        final_feats, gate_info_list, feature_dict, moe_feature_list = run_model_and_collect(model, images)

        seq_len = final_feats.shape[1]
        last_gate = gate_info_list[-1]
        last_dispatch = get_last_dispatch_weight(last_gate, seq_len=seq_len)  # [B, N, E]

        last_moe_feat = moe_feature_list[-1][:, 1:, :]  # [B, N, D_student]
        B, N, D = last_moe_feat.shape

        expert_ids = last_dispatch.argmax(dim=-1)       # [B, N]
        dispatch_conf = last_dispatch.max(dim=-1).values

        feat_student_np = last_moe_feat.detach().cpu().reshape(B * N, D).numpy().astype(np.float32)

        feat_role_np = project_features_to_role_space(
            feat_student_np,
            proj_l12,
            device=device,
            show_pbar=False,
        )

        role_affinity_np = compute_role_affinity(feat_role_np, role_prototypes).astype(np.float32)

        all_role_affinity.append(role_affinity_np)
        all_expert_ids.append(expert_ids.detach().cpu().reshape(B * N).numpy())
        all_dispatch_conf.append(dispatch_conf.detach().cpu().reshape(B * N).numpy())

        for b in range(B):
            meta = metas[b]
            for t in range(N):
                all_token_meta.append({
                    "project": meta["project"],
                    "slide_id": meta["slide_id"],
                    "pred_label": meta["pred_label"],
                    "svs_path": meta["svs_path"],
                    "coord_x": meta["coord_x"],
                    "coord_y": meta["coord_y"],
                    "coord_idx": meta["coord_idx"],
                    "token_idx": int(t),
                })

        if (batch_idx + 1) % 5 == 0:
            cur_tokens = sum(len(x) for x in all_expert_ids)
            print(f"[Stage] collected batches={batch_idx+1}, tokens_so_far={cur_tokens}")

    print(f"[Debug] num batch chunks role_affinity = {len(all_role_affinity)}")
    print(f"[Debug] num batch chunks expert_ids    = {len(all_expert_ids)}")
    print(f"[Debug] num batch chunks dispatch_conf = {len(all_dispatch_conf)}")
    print(f"[Debug] total token_meta length        = {len(all_token_meta)}")
    if len(all_role_affinity) > 0:
        print(f"[Debug] first role_affinity chunk shape = {all_role_affinity[0].shape}")

    print("[Stage] concatenating role_affinity ...")
    role_affinity = np.concatenate(all_role_affinity, axis=0).astype(np.float32)
    print(f"[Stage] role_affinity done: shape={role_affinity.shape}, dtype={role_affinity.dtype}")

    print("[Stage] concatenating expert_ids ...")
    expert_ids = np.concatenate(all_expert_ids, axis=0).astype(np.int64)
    print(f"[Stage] expert_ids done: shape={expert_ids.shape}")

    print("[Stage] concatenating dispatch_conf ...")
    dispatch_conf = np.concatenate(all_dispatch_conf, axis=0).astype(np.float32)
    print(f"[Stage] dispatch_conf done: shape={dispatch_conf.shape}")

    print(f"[Stage] concatenation done: num_tokens={len(expert_ids)}")

    print("[Stage] applying optional token subsampling ...")
    (
        expert_ids,
        dispatch_conf,
        role_affinity,
        all_token_meta,
    ) = maybe_subsample_tokens(
        expert_ids=expert_ids,
        dispatch_conf=dispatch_conf,
        role_affinity=role_affinity,
        token_meta=all_token_meta,
        max_token_points=args.max_token_points,
        seed=args.seed,
    )

    print(f"[Stage] after subsample: num_tokens={len(expert_ids)}, role_affinity={role_affinity.shape}")

    print("[Stage] computing nearest roles ...")
    nearest_role_ids = role_affinity.argmax(axis=1)
    nearest_role_labels = np.array([role_names[i] for i in nearest_role_ids], dtype=object)
    print("[Stage] nearest roles done")

    num_experts = int(expert_ids.max()) + 1
    expert_names = [f"E{i}" for i in range(num_experts)]

    print("[Stage] building summary tables ...")
    mean_aff_table, expert_counts = build_mean_affinity_table(
        role_affinity, expert_ids, num_experts=num_experts
    )
    nearest_role_ratio_table = build_nearest_role_ratio_table(
        nearest_role_ids, expert_ids, num_roles=len(role_names), num_experts=num_experts
    )

    global_role_ratio = summarize_global_role_ratio(nearest_role_ids, role_names)
    semantic_role_ratio = summarize_semantic_expert_role_ratio(
        nearest_role_ids=nearest_role_ids,
        expert_ids=expert_ids,
        semantic_expert_ids=args.semantic_expert_ids,
        role_names=role_names,
    )
    print("[Stage] summary tables done")

    print("[Stage] building token dataframe ...")
    token_df = pd.DataFrame(all_token_meta)
    token_df["expert_id"] = expert_ids
    token_df["dispatch_conf"] = dispatch_conf
    token_df["nearest_role_id"] = nearest_role_ids
    token_df["nearest_role"] = nearest_role_labels
    for i, name in enumerate(role_names):
        token_df[f"affinity_{name}"] = role_affinity[:, i]
    print(f"[Stage] token dataframe done: shape={token_df.shape}")

    print("[Stage] saving token csv ...")
    token_csv = os.path.join(args.output_dir, "token_role_affinity.csv")
    token_df.to_csv(token_csv, index=False)
    print(f"[Stage] token csv saved: {token_csv}")

    print("[Stage] preparing summary json ...")
    summary = {
        "proto_source": proto_source,
        "role_names": role_names,
        "num_tokens": int(len(expert_ids)),
        "num_experts": int(num_experts),
        "expert_counts": {f"E{i}": int(expert_counts[i]) for i in range(num_experts)},
        "global_nearest_role_ratio": global_role_ratio,
        "semantic_expert_ids": args.semantic_expert_ids,
        "semantic_expert_nearest_role_ratio": semantic_role_ratio,
        "free_expert_id": args.free_expert_id,
    }

    expert_details = {}
    for e in range(num_experts):
        idx = (expert_ids == e)
        detail = {
            "count": int(idx.sum()),
            "mean_dispatch_conf": float(dispatch_conf[idx].mean()) if idx.sum() > 0 else 0.0,
            "mean_affinity": {
                role_names[r]: float(mean_aff_table[e, r]) for r in range(len(role_names))
            },
            "nearest_role_ratio": {
                role_names[r]: float(nearest_role_ratio_table[e, r]) for r in range(len(role_names))
            },
        }
        expert_details[f"E{e}"] = detail
    summary["expert_details"] = expert_details

    summary_json_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Stage] summary json saved: {summary_json_path}")

    print("[Stage] plotting heatmap: mean affinity ...")
    plot_heatmap(
        mean_aff_table,
        expert_names,
        role_names,
        title=f"Expert x Mean Role Affinity ({proto_source})",
        save_path=os.path.join(args.output_dir, "expert_mean_role_affinity.png"),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    print("[Stage] plotting heatmap: nearest role ratio ...")
    plot_heatmap(
        nearest_role_ratio_table,
        expert_names,
        role_names,
        title=f"Expert x Nearest Role Ratio ({proto_source})",
        save_path=os.path.join(args.output_dir, "expert_nearest_role_ratio.png"),
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
    )

    print("\n===== Global nearest-role ratio =====")
    print(json.dumps(global_role_ratio, indent=2, ensure_ascii=False))

    print("\n===== Semantic experts nearest-role ratio =====")
    print(json.dumps(semantic_role_ratio, indent=2, ensure_ascii=False))

    print(f"\n[Saved] {token_csv}")
    print(f"[Saved] {summary_json_path}")
    print(f"[Done] outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()