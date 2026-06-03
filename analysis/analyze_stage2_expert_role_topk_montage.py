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
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageDraw, ImageFile

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
def _load_cfg_from_ckpt_or_file(
    ckpt: dict,
    config_path: Optional[str] = None,
):
    """
    优先使用 checkpoint 内保存的 cfg。
    只有 ckpt 里没有 cfg 时，才退回外部 yaml。
    """
    if "cfg" in ckpt and ckpt["cfg"] is not None:
        cfg = ckpt["cfg"]
        print("[load_stage2_bundle] Using cfg stored inside checkpoint.")
        return cfg

    if config_path is None:
        raise KeyError(
            "Checkpoint does not contain cfg, and no --config was provided."
        )

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("[load_stage2_bundle] Checkpoint has no cfg, fallback to external yaml.")
    return cfg


def _print_patch_guided_cfg(cfg: dict):
    moe_cfg = cfg.get("moe_encoder", {})
    print("[MoE cfg] use_patch_guided_routing =", moe_cfg.get("use_patch_guided_routing", None))
    print("[MoE cfg] patch_guided_mode       =", moe_cfg.get("patch_guided_mode", None))
    print("[MoE cfg] patch_context_alpha     =", moe_cfg.get("patch_context_alpha", None))
    print("[MoE cfg] use_routing_proj        =", moe_cfg.get("use_routing_proj", None))


def _inspect_patch_context_proj(model: nn.Module):
    print("[Model inspect] checking patch_context_proj on moe blocks...")
    depth = len(model.blocks)
    for idx in getattr(model, "moe_layers_idx", []):
        real_idx = idx if idx >= 0 else depth + idx
        blk = model.blocks[real_idx]
        has_attr = hasattr(blk.mlp, "patch_context_proj")
        val = getattr(blk.mlp, "patch_context_proj", None)
        print(f"  block {real_idx}: has patch_context_proj = {has_attr}, value = {type(val).__name__ if val is not None else None}")


def load_stage2_bundle(
    full_ckpt_path: str,
    config_path: Optional[str] = None,
    device: str = "cuda",
):
    ckpt = torch.load(full_ckpt_path, map_location="cpu")

    if "student_state_dict" not in ckpt:
        raise KeyError("student_state_dict not found in checkpoint")
    if "distiller_state_dict" not in ckpt:
        raise KeyError("distiller_state_dict not found in checkpoint")

    cfg = _load_cfg_from_ckpt_or_file(ckpt, config_path=config_path)
    _print_patch_guided_cfg(cfg)

    if "base_encoder" not in cfg or "moe_encoder" not in cfg:
        raise KeyError("cfg must contain both 'base_encoder' and 'moe_encoder'")

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    _inspect_patch_context_proj(model)

    load_ret = model.load_state_dict(ckpt["student_state_dict"], strict=True)
    if load_ret is not None:
        # torch 一般 strict=True 成功时返回 <All keys matched successfully>
        print(load_ret)

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

            keep_keys = set(sdf.loc[sdf["split"].isin(split_keep), "match_key"].tolist())
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
    if total_tokens % seq_len != 0:
        raise ValueError(f"dispatch total_tokens={total_tokens} cannot be divided by seq_len={seq_len}")
    B = total_tokens // seq_len
    dispatch = dispatch.view(B, seq_len, num_experts)[:, 1:, :]
    return dispatch  # [B, N, E]


@torch.no_grad()
def project_features_to_role_space(features: np.ndarray, proj_head, device="cpu", batch_size=4096):
    outs = []
    iterator = tqdm(
        range(0, len(features), batch_size),
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


# =========================================================
# visualization
# =========================================================
def read_patch_image(meta: dict, out_size: int = 224) -> Image.Image:
    slide = openslide.OpenSlide(meta["svs_path"])
    try:
        img = slide.read_region(
            (int(meta["coord_x"]), int(meta["coord_y"])),
            int(meta["patch_level"]),
            (int(meta["patch_size"]), int(meta["patch_size"]))
        ).convert("RGB")
    finally:
        slide.close()

    if img.size != (out_size, out_size):
        img = img.resize((out_size, out_size))
    return img


def draw_token_box(img: Image.Image, token_idx: int, tokens_per_side: int = 37, color=(255, 0, 0), width: int = 2):
    img = img.copy()
    draw = ImageDraw.Draw(img)

    patch_w, patch_h = img.size
    cell_w = patch_w / tokens_per_side
    cell_h = patch_h / tokens_per_side

    row = token_idx // tokens_per_side
    col = token_idx % tokens_per_side

    x0 = int(round(col * cell_w))
    y0 = int(round(row * cell_h))
    x1 = int(round((col + 1) * cell_w))
    y1 = int(round((row + 1) * cell_h))

    for k in range(width):
        draw.rectangle([x0 + k, y0 + k, x1 - k, y1 - k], outline=color)
    return img


def make_montage(
    records: List[dict],
    save_path: str,
    title: str,
    role_names: List[str],
    patch_size_vis: int = 224,
    ncols: int = 6,
):
    if len(records) == 0:
        print(f"[Warn] no records for montage: {title}")
        return

    n = len(records)
    nrows = math.ceil(n / ncols)

    fig_w = ncols * 2.4
    fig_h = nrows * 3.2 + 0.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    fig.suptitle(title, fontsize=16)

    for ax in axes.flat:
        ax.axis("off")

    for i, rec in enumerate(records):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        img = read_patch_image(rec, out_size=patch_size_vis)
        img = draw_token_box(
            img,
            token_idx=int(rec["token_idx"]),
            tokens_per_side=int(rec["tokens_per_side"]),
            color=(255, 0, 0),
            width=2,
        )

        ax.imshow(img)
        ax.axis("off")

        aff_tumor = rec.get("aff_tumor", float("nan"))
        aff_stroma = rec.get("aff_stroma", float("nan"))
        aff_necrosis = rec.get("aff_necrosis", float("nan"))

        pred_role = rec.get("pred_role", "")
        pred_conf = rec.get("pred_conf", 0.0)
        pred_margin = rec.get("pred_margin", 0.0)

        text = (
            f"slide={rec['slide_id']}\n"
            f"proj={rec['project']} pred={rec['pred_label']}\n"
            f"E={rec['expert_id']} route={rec['route_conf']:.3f}\n"
            f"pred_role={pred_role} conf={pred_conf:.3f} margin={pred_margin:.3f}\n"
            f"tumor={rec['aff_tumor']:.3f} str={rec['aff_stroma']:.3f} nec={rec['aff_necrosis']:.3f}"
        )
        ax.set_title(text, fontsize=7)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(save_path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {save_path}")


# =========================================================
# selection helpers
# =========================================================
def topk_unique_by_slide(
    df: pd.DataFrame,
    score_col: str,
    topk: int,
    max_per_slide: int = 2,
):
    if len(df) == 0:
        return df.copy()

    df = df.sort_values(score_col, ascending=False).copy()
    keep_rows = []
    slide_counts = {}

    for _, row in df.iterrows():
        sid = row["slide_id"]
        cur = slide_counts.get(sid, 0)
        if cur >= max_per_slide:
            continue
        keep_rows.append(row)
        slide_counts[sid] = cur + 1
        if len(keep_rows) >= topk:
            break

    if len(keep_rows) == 0:
        return df.head(0).copy()
    return pd.DataFrame(keep_rows).reset_index(drop=True)


# =========================================================
# main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 expert/role topk montage analysis")
    parser.add_argument("--config", type=str, default=None, help="Optional fallback yaml. Normally not needed if ckpt contains cfg.")
    parser.add_argument("--stage2-full-ckpt", type=str, required=True)
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

    parser.add_argument("--split-csv", type=str, default=None)
    parser.add_argument("--split-keep", nargs="+", default=None)

    parser.add_argument("--expert-ids", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--topk-per-expert", type=int, default=30)
    parser.add_argument("--topk-per-role", type=int, default=30)
    parser.add_argument("--max-per-slide", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    model, proj_l12, cfg = load_stage2_bundle(
        full_ckpt_path=args.stage2_full_ckpt,
        config_path=args.config,
        device=device,
    )
    role_prototypes, role_names = load_role_proto_from_dir(args.role_proto_dir)

    if len(role_names) != 3:
        print(f"[Warn] role_names != 3 : {role_names}")

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

    all_feat_student = []
    all_expert_ids = []
    all_route_conf = []
    all_meta = []

    print("[Stage] running model and collecting token-level data")

    for images, metas in tqdm(loader, desc="Running model on patches"):
        images = images.to(device, non_blocking=True)

        final_feats, gate_info_list, feature_dict, moe_feature_list = run_model_and_collect(model, images)

        seq_len = final_feats.shape[1]
        tokens_per_side = int(round(math.sqrt(seq_len - 1)))

        last_gate = gate_info_list[-1]
        last_dispatch = get_last_dispatch_weight(last_gate, seq_len=seq_len)  # [B, N, E]
        last_moe_feat = moe_feature_list[-1][:, 1:, :]  # [B, N, D]

        B, N, D = last_moe_feat.shape
        expert_ids = last_dispatch.argmax(dim=-1)       # [B, N]
        route_conf = last_dispatch.max(dim=-1).values   # [B, N]

        feat_student_np = last_moe_feat.detach().cpu().reshape(B * N, D).numpy().astype(np.float32)

        all_feat_student.append(feat_student_np)
        all_expert_ids.append(expert_ids.detach().cpu().reshape(B * N).numpy().astype(np.int64))
        all_route_conf.append(route_conf.detach().cpu().reshape(B * N).numpy().astype(np.float32))

        for b in range(B):
            meta = metas[b]
            for t in range(N):
                all_meta.append({
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
                    "tokens_per_side": int(tokens_per_side),
                })

    print("[Stage] concatenating arrays")
    feat_student = np.concatenate(all_feat_student, axis=0)
    expert_ids = np.concatenate(all_expert_ids, axis=0)
    route_conf = np.concatenate(all_route_conf, axis=0)

    print(f"[Stage] feat_student shape = {feat_student.shape}")
    print(f"[Stage] num token metas = {len(all_meta)}")

    print("[Stage] projecting to role space")
    feat_role = project_features_to_role_space(
        feat_student,
        proj_l12,
        device=device,
        batch_size=4096,
    )

    print("[Stage] computing role affinity")
    role_affinity = compute_role_affinity(feat_role, role_prototypes)
    nearest_role_ids = role_affinity.argmax(axis=1)
    nearest_role_labels = np.array([role_names[i] for i in nearest_role_ids], dtype=object)

    # -----------------------------------------------------
    # build dataframe first
    # -----------------------------------------------------
    token_df = pd.DataFrame(all_meta)
    token_df["expert_id"] = expert_ids
    token_df["route_conf"] = route_conf
    token_df["nearest_role_id"] = nearest_role_ids
    token_df["nearest_role"] = nearest_role_labels

    for i, role_name in enumerate(role_names):
        token_df[f"aff_{role_name}"] = role_affinity[:, i]

    # -----------------------------------------------------
    # predicted role stats
    # -----------------------------------------------------
    aff_t = torch.from_numpy(role_affinity).float()
    pred_prob_t = torch.softmax(aff_t, dim=1)

    top2_vals, top2_idx = torch.topk(aff_t, k=min(2, aff_t.shape[1]), dim=1)
    top1_idx = top2_idx[:, 0].cpu().numpy()

    if aff_t.shape[1] >= 2:
        top2_only_idx = top2_idx[:, 1]
        pred_margin = (top2_vals[:, 0] - top2_vals[:, 1]).cpu().numpy()
    else:
        top2_only_idx = top2_idx[:, 0]
        pred_margin = np.zeros(len(top1_idx), dtype=np.float32)

    pred_conf = pred_prob_t[torch.arange(aff_t.shape[0]), top2_idx[:, 0]].cpu().numpy()
    pred_role_labels = np.array([role_names[i] for i in top1_idx], dtype=object)

    token_df["pred_role_id"] = top1_idx
    token_df["pred_role"] = pred_role_labels
    token_df["pred_conf"] = pred_conf
    token_df["pred_margin"] = pred_margin

    for i, role_name in enumerate(role_names):
        token_df[f"aff_{role_name}"] = role_affinity[:, i]

    csv_path = os.path.join(args.output_dir, "stage2_token_analysis.csv")
    token_df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

    # -----------------------------------------------------
    # 1) by expert: top routed tokens
    # -----------------------------------------------------
    print("[Stage] building expert montages")
    for expert_id in args.expert_ids:
        sub = token_df[token_df["expert_id"] == int(expert_id)].copy()
        if len(sub) == 0:
            print(f"[Warn] no token for expert E{expert_id}")
            continue

        sub = sub.sort_values("route_conf", ascending=False).copy()
        sub = topk_unique_by_slide(
            sub,
            score_col="route_conf",
            topk=args.topk_per_expert,
            max_per_slide=args.max_per_slide,
        )

        records = sub.to_dict("records")
        save_path = os.path.join(args.output_dir, f"expert_E{expert_id}_top_route_conf.png")
        make_montage(
            records=records,
            save_path=save_path,
            title=f"Stage2 Expert E{expert_id} top routed tokens",
            role_names=role_names,
            patch_size_vis=args.image_size,
            ncols=6,
        )

    # -----------------------------------------------------
    # 2) by role: top affinity tokens
    # -----------------------------------------------------
    print("[Stage] building role montages")
    for role_name in role_names:
        score_col = f"aff_{role_name}"
        sub = token_df.sort_values(score_col, ascending=False).copy()
        sub = topk_unique_by_slide(
            sub,
            score_col=score_col,
            topk=args.topk_per_role,
            max_per_slide=args.max_per_slide,
        )

        records = sub.to_dict("records")
        save_path = os.path.join(args.output_dir, f"role_{role_name}_top_affinity.png")
        make_montage(
            records=records,
            save_path=save_path,
            title=f"Stage2 Role {role_name} top affinity tokens",
            role_names=role_names,
            patch_size_vis=args.image_size,
            ncols=6,
        )

    # -----------------------------------------------------
    # 3) by predicted role: top confident predictions
    # -----------------------------------------------------
    print("[Stage] building predicted-role montages")
    for role_name in role_names:
        sub = token_df[token_df["pred_role"] == role_name].copy()
        if len(sub) == 0:
            print(f"[Warn] no predicted token for role={role_name}")
            continue

        sub = topk_unique_by_slide(
            sub.sort_values("pred_margin", ascending=False).copy(),
            score_col="pred_margin",
            topk=args.topk_per_role,
            max_per_slide=args.max_per_slide,
        )

        records = sub.to_dict("records")
        save_path = os.path.join(args.output_dir, f"pred_role_{role_name}_top_margin.png")
        make_montage(
            records=records,
            save_path=save_path,
            title=f"Stage2 Predicted Role {role_name} top-margin tokens",
            role_names=role_names,
            patch_size_vis=args.image_size,
            ncols=6,
        )

    summary = {
        "num_tokens": int(len(token_df)),
        "role_names": role_names,
        "expert_counts": {
            f"E{int(k)}": int(v)
            for k, v in token_df["expert_id"].value_counts().sort_index().to_dict().items()
        },
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {summary_path}")

    print(f"[Done] outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()