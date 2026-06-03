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
from pathlib import Path
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
import yaml
from tqdm import tqdm

from models.encoders.moe_encoder import MoEEncoder
from stage3.stage3_role_refinement import Stage3RoleRefiner

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


def make_patch_uid(row: pd.Series) -> str:
    coord_idx = int(row["coord_idx"]) if "coord_idx" in row and pd.notna(row["coord_idx"]) else -1
    return (
        f"{canonicalize_path(row['svs_path'])}"
        f"__x{int(row['coord_x'])}"
        f"__y{int(row['coord_y'])}"
        f"__lvl{int(row['patch_level'])}"
        f"__psz{int(row['patch_size'])}"
        f"__cidx{coord_idx}"
    )


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

    print("[Load] stage2 student + proj_l12 loaded")
    print(f"[Load] moe_layers_idx = {model.moe_layers_idx}")
    print(f"[Load] proj_l12 shape: {proj_in_dim} -> {proj_out_dim}")
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
    return protos, role_names


def load_stage3_bundle(
    config_path: str,
    stage2_full_ckpt: str,
    stage3_ckpt: str,
    role_proto_dir: str,
    device: str = "cuda",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    stage2_model, proj_l12, _ = load_stage2_bundle(
        config_path=config_path,
        full_ckpt_path=stage2_full_ckpt,
        device=device,
    )

    stage3_cfg = cfg["stage3_train"]
    refiner = Stage3RoleRefiner(
        student_model=stage2_model,
        proj_l12=proj_l12,
        role_proto_dir=role_proto_dir,
        stage3_cfg=stage3_cfg,
    ).to(device)

    ckpt = torch.load(stage3_ckpt, map_location="cpu")
    if "stage3_state_dict" in ckpt:
        refiner.load_state_dict(ckpt["stage3_state_dict"], strict=True)
    else:
        refiner.load_state_dict(ckpt, strict=True)

    refiner.eval()

    role_names = list(refiner.shared_role_proto.role_names)
    role_protos = refiner.shared_role_proto.get_prototypes().detach().cpu().numpy().astype(np.float32)
    role_protos = l2_normalize_np(role_protos)

    print("[Load] stage3 refiner loaded")
    return refiner.student, refiner.proj_l12, role_protos, role_names, cfg


# =========================================================
# dataset
# =========================================================
class PatchRowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.transform = build_transform(image_size=image_size)

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
            img_pil = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
        finally:
            slide.close()

        img = self.transform(img_pil)

        meta = row.to_dict()
        meta["patch_uid"] = make_patch_uid(row)
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
    dispatch = dispatch.view(B, seq_len, num_experts)[:, 1:, :]  # remove CLS
    return dispatch  # [B, N, E]


@torch.no_grad()
def project_features_to_role_space(
    features: np.ndarray,
    proj_head,
    device="cpu",
    batch_size=4096,
):
    outs = []
    for start in range(0, len(features), batch_size):
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
# slide sampling / selection
# =========================================================
def sample_slides_for_replay(
    df: pd.DataFrame,
    num_pos_slides: int,
    num_neg_slides: int,
    seed: int = 42,
    split_csv: Optional[str] = None,
    split_keep: Optional[List[str]] = None,
):
    df = df.copy()

    if "slide_label" not in df.columns:
        raise ValueError("pool csv must contain slide_label for pos/neg slide sampling")

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

    slide_df = df.groupby("slide_id", as_index=False).agg(
        slide_label=("slide_label", "first"),
        n_patches=("svs_path", "size"),
        svs_path=("svs_path", "first"),
        project=("project", "first"),
    )

    pos_df = slide_df[slide_df["slide_label"].astype(int) == 1].copy()
    neg_df = slide_df[slide_df["slide_label"].astype(int) == 0].copy()

    rng = np.random.default_rng(seed)

    if len(pos_df) < num_pos_slides:
        raise ValueError(f"Not enough positive slides: need {num_pos_slides}, got {len(pos_df)}")
    if len(neg_df) < num_neg_slides:
        raise ValueError(f"Not enough negative slides: need {num_neg_slides}, got {len(neg_df)}")

    pos_choose = pos_df.iloc[rng.choice(len(pos_df), size=num_pos_slides, replace=False)].copy()
    neg_choose = neg_df.iloc[rng.choice(len(neg_df), size=num_neg_slides, replace=False)].copy()

    chosen = pd.concat([pos_choose, neg_choose], axis=0).reset_index(drop=True)
    print("[Sample] chosen slides:")
    print(chosen[["slide_id", "slide_label", "project", "n_patches"]])

    return chosen["slide_id"].astype(str).tolist()


# =========================================================
# core extraction
# =========================================================
def extract_token_table_for_rows(
    rows_df: pd.DataFrame,
    model,
    proj_l12,
    role_prototypes: np.ndarray,
    role_names: List[str],
    device: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    desc: str,
):
    dataset = PatchRowDataset(rows_df, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_with_meta,
    )

    all_records = []

    for images, metas in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)

        final_feats, gate_info_list, feature_dict, moe_feature_list = run_model_and_collect(model, images)

        seq_len = final_feats.shape[1]
        last_gate = gate_info_list[-1]
        last_dispatch = get_last_dispatch_weight(last_gate, seq_len=seq_len)  # [B, N, E]
        last_moe_feat = moe_feature_list[-1][:, 1:, :]                        # [B, N, D_student]

        B, N, D = last_moe_feat.shape
        expert_ids = last_dispatch.argmax(dim=-1)      # [B, N]
        dispatch_conf = last_dispatch.max(dim=-1).values

        feat_student_np = last_moe_feat.detach().cpu().reshape(B * N, D).numpy().astype(np.float32)
        feat_role_np = project_features_to_role_space(
            feat_student_np,
            proj_l12,
            device=device,
        )
        role_aff = compute_role_affinity(feat_role_np, role_prototypes)  # [B*N, R]

        role_aff = role_aff.reshape(B, N, -1)

        for b in range(B):
            meta = metas[b]
            for t in range(N):
                rec = {
                    "patch_uid": meta["patch_uid"],
                    "token_uid": f"{meta['patch_uid']}__tok{t}",
                    "slide_id": str(meta.get("slide_id", "")),
                    "slide_label": int(meta.get("slide_label", -1)) if pd.notna(meta.get("slide_label", None)) else -1,
                    "project": str(meta.get("project", "")),
                    "svs_path": str(meta["svs_path"]),
                    "coord_x": int(meta["coord_x"]),
                    "coord_y": int(meta["coord_y"]),
                    "coord_idx": int(meta["coord_idx"]) if pd.notna(meta.get("coord_idx", None)) else -1,
                    "patch_level": int(meta["patch_level"]),
                    "patch_size": int(meta["patch_size"]),
                    "token_idx": int(t),
                    "expert_id": int(expert_ids[b, t].item()),
                    "dispatch_conf": float(dispatch_conf[b, t].item()),
                }

                cur_aff = role_aff[b, t]
                rec["nearest_role_id"] = int(cur_aff.argmax())
                rec["nearest_role"] = role_names[int(cur_aff.argmax())]

                for rid, rname in enumerate(role_names):
                    rec[f"sim_{rname}"] = float(cur_aff[rid])

                other_scores = [cur_aff[r] for r in range(len(role_names))]
                tumor_id = role_names.index("tumor") if "tumor" in role_names else 0
                tumor_score = float(cur_aff[tumor_id])

                other_wo_tumor = [float(cur_aff[r]) for r in range(len(role_names)) if r != tumor_id]
                rec["tumor_minus_other"] = tumor_score - max(other_wo_tumor) if len(other_wo_tumor) > 0 else tumor_score

                if "necrosis" in role_names:
                    nec_id = role_names.index("necrosis")
                    rec["tumor_minus_necrosis"] = tumor_score - float(cur_aff[nec_id])
                else:
                    rec["tumor_minus_necrosis"] = np.nan

                all_records.append(rec)

    token_df = pd.DataFrame(all_records)
    return token_df


def select_reference_tokens(
    token_df: pd.DataFrame,
    per_slide_topk: int,
    per_slide_bottomk: int,
):
    selected_parts = []

    for slide_id, sdf in token_df.groupby("slide_id"):
        slide_label = int(sdf["slide_label"].iloc[0])

        sdf = sdf.copy()
        sdf = sdf.sort_values("sim_tumor", ascending=False)

        if slide_label == 0:
            risky = sdf.head(per_slide_topk).copy()
            risky["selection_group"] = "neg_topk_tumor"

            suppressed = sdf.tail(per_slide_bottomk).copy()
            suppressed["selection_group"] = "neg_bottomk_tumor"

            selected_parts.extend([risky, suppressed])

        elif slide_label == 1:
            evidence = sdf.head(per_slide_topk).copy()
            evidence["selection_group"] = "pos_topk_tumor"

            low = sdf.tail(per_slide_bottomk).copy()
            low["selection_group"] = "pos_bottomk_tumor"

            selected_parts.extend([evidence, low])

    out = pd.concat(selected_parts, axis=0).reset_index(drop=True)

    # 防止重复：同一 token 可能被多次选中
    out = out.drop_duplicates(subset=["token_uid"]).reset_index(drop=True)
    return out


# =========================================================
# replay helpers
# =========================================================
def build_patch_subset_from_selected(selected_token_df: pd.DataFrame, full_df: pd.DataFrame):
    patch_keys = selected_token_df["patch_uid"].unique().tolist()

    full_df = full_df.copy()
    full_df["patch_uid"] = full_df.apply(make_patch_uid, axis=1)

    sub = full_df[full_df["patch_uid"].isin(patch_keys)].copy().reset_index(drop=True)
    return sub


def replay_stage3_on_selected_tokens(
    selected_token_df: pd.DataFrame,
    replay_token_df: pd.DataFrame,
):
    keep_cols = [
        "token_uid",
        "expert_id",
        "dispatch_conf",
        "nearest_role_id",
        "nearest_role",
        "sim_tumor",
        "sim_stroma",
        "sim_necrosis",
        "tumor_minus_other",
        "tumor_minus_necrosis",
    ]

    stage2_df = selected_token_df.copy()
    stage3_df = replay_token_df[keep_cols].copy()

    stage3_df = stage3_df.rename(columns={
        "expert_id": "stage3_expert_id",
        "dispatch_conf": "stage3_dispatch_conf",
        "nearest_role_id": "stage3_nearest_role_id",
        "nearest_role": "stage3_nearest_role",
        "sim_tumor": "stage3_sim_tumor",
        "sim_stroma": "stage3_sim_stroma",
        "sim_necrosis": "stage3_sim_necrosis",
        "tumor_minus_other": "stage3_tumor_minus_other",
        "tumor_minus_necrosis": "stage3_tumor_minus_necrosis",
    })

    merged = stage2_df.merge(stage3_df, on="token_uid", how="left")

    merged["delta_sim_tumor"] = merged["stage3_sim_tumor"] - merged["sim_tumor"]
    merged["delta_sim_stroma"] = merged["stage3_sim_stroma"] - merged["sim_stroma"]
    merged["delta_sim_necrosis"] = merged["stage3_sim_necrosis"] - merged["sim_necrosis"]
    merged["delta_tumor_minus_other"] = merged["stage3_tumor_minus_other"] - merged["tumor_minus_other"]
    merged["delta_tumor_minus_necrosis"] = merged["stage3_tumor_minus_necrosis"] - merged["tumor_minus_necrosis"]

    merged["stage2_route_e0"] = (merged["expert_id"] == 0).astype(int)
    merged["stage3_route_e0"] = (merged["stage3_expert_id"] == 0).astype(int)

    return merged


def summarize_delta(merged_df: pd.DataFrame):
    summary = {}

    for group, sdf in merged_df.groupby("selection_group"):
        summary[group] = {
            "num_tokens": int(len(sdf)),
            "stage2_mean_tumor": float(sdf["sim_tumor"].mean()),
            "stage3_mean_tumor": float(sdf["stage3_sim_tumor"].mean()),
            "delta_mean_tumor": float(sdf["delta_sim_tumor"].mean()),

            "stage2_mean_tumor_minus_other": float(sdf["tumor_minus_other"].mean()),
            "stage3_mean_tumor_minus_other": float(sdf["stage3_tumor_minus_other"].mean()),
            "delta_mean_tumor_minus_other": float(sdf["delta_tumor_minus_other"].mean()),

            "stage2_mean_tumor_minus_necrosis": float(sdf["tumor_minus_necrosis"].mean()),
            "stage3_mean_tumor_minus_necrosis": float(sdf["stage3_tumor_minus_necrosis"].mean()),
            "delta_mean_tumor_minus_necrosis": float(sdf["delta_tumor_minus_necrosis"].mean()),

            "stage2_route_e0_ratio": float(sdf["stage2_route_e0"].mean()),
            "stage3_route_e0_ratio": float(sdf["stage3_route_e0"].mean()),
            "delta_route_e0_ratio": float(sdf["stage3_route_e0"].mean() - sdf["stage2_route_e0"].mean()),
        }

    return summary


# =========================================================
# visualization
# =========================================================
def draw_token_box_on_patch(
    svs_path: str,
    coord_x: int,
    coord_y: int,
    patch_level: int,
    patch_size: int,
    token_idx: int,
    out_path: str,
    image_size: int = 224,
    token_grid_size: int = 16,
):
    slide = openslide.OpenSlide(svs_path)
    try:
        img = slide.read_region(
            (coord_x, coord_y),
            patch_level,
            (patch_size, patch_size)
        ).convert("RGB")
    finally:
        slide.close()

    img = img.resize((image_size, image_size))
    draw = ImageDraw.Draw(img)

    cell = image_size // token_grid_size
    row = token_idx // token_grid_size
    col = token_idx % token_grid_size

    x0 = col * cell
    y0 = row * cell
    x1 = (col + 1) * cell - 1
    y1 = (row + 1) * cell - 1

    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    img.save(out_path)


def save_visual_examples(
    merged_df: pd.DataFrame,
    out_dir: str,
    image_size: int = 224,
    token_grid_size: int = 16,
    per_group: int = 12,
):
    ensure_dir(out_dir)

    for group, sdf in merged_df.groupby("selection_group"):
        # 对每组挑变化最明显的
        sdf = sdf.copy()
        if "delta_sim_tumor" in sdf.columns:
            sdf = sdf.sort_values("delta_sim_tumor", ascending=False)
        sdf = sdf.head(per_group)

        group_dir = os.path.join(out_dir, group)
        ensure_dir(group_dir)

        for i, (_, row) in enumerate(sdf.iterrows()):
            fname = (
                f"{i:03d}"
                f"__slide_{row['slide_id']}"
                f"__tok{int(row['token_idx'])}"
                f"__dTumor_{row['delta_sim_tumor']:+.3f}.png"
            )
            save_path = os.path.join(group_dir, fname)

            draw_token_box_on_patch(
                svs_path=row["svs_path"],
                coord_x=int(row["coord_x"]),
                coord_y=int(row["coord_y"]),
                patch_level=int(row["patch_level"]),
                patch_size=int(row["patch_size"]),
                token_idx=int(row["token_idx"]),
                out_path=save_path,
                image_size=image_size,
                token_grid_size=token_grid_size,
            )


# =========================================================
# main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 token mining + Stage3 replay analysis")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage2-full-ckpt", type=str, required=True)
    parser.add_argument("--stage3-ckpt", type=str, required=True)
    parser.add_argument("--role-proto-dir", type=str, required=True)
    parser.add_argument("--pool-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--split-csv", type=str, default=None)
    parser.add_argument("--split-keep", nargs="+", default=None)

    parser.add_argument("--num-pos-slides", type=int, default=20)
    parser.add_argument("--num-neg-slides", type=int, default=20)
    parser.add_argument("--max-patches-per-slide", type=int, default=256)

    parser.add_argument("--per-slide-topk", type=int, default=32)
    parser.add_argument("--per-slide-bottomk", type=int, default=32)

    parser.add_argument("--token-grid-size", type=int, default=16)
    parser.add_argument("--save-patch-vis", action="store_true")
    parser.add_argument("--vis-per-group", type=int, default=12)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # ---------- load data ----------
    full_df = pd.read_csv(args.pool_csv)
    full_df["svs_path"] = full_df["svs_path"].map(canonicalize_path)

    if "prefilter_white" in full_df.columns:
        full_df = full_df[full_df["prefilter_white"].fillna(0).astype(int) == 0].copy()

    if "slide_id" not in full_df.columns:
        full_df["slide_id"] = full_df["svs_path"].astype(str)

    if "slide_label" not in full_df.columns:
        raise ValueError("pool csv must contain slide_label")

    full_df = full_df.reset_index(drop=True)

    # ---------- sample slides ----------
    chosen_slide_ids = sample_slides_for_replay(
        df=full_df,
        num_pos_slides=args.num_pos_slides,
        num_neg_slides=args.num_neg_slides,
        seed=args.seed,
        split_csv=args.split_csv,
        split_keep=args.split_keep,
    )

    work_df = full_df[full_df["slide_id"].astype(str).isin(chosen_slide_ids)].copy()

    # 可选：每张 slide 限制 patch 数，避免太慢
    if args.max_patches_per_slide > 0:
        rng = np.random.default_rng(args.seed)
        parts = []
        for sid, sdf in work_df.groupby("slide_id"):
            if len(sdf) > args.max_patches_per_slide:
                keep_idx = rng.choice(len(sdf), size=args.max_patches_per_slide, replace=False)
                sdf = sdf.iloc[keep_idx].copy()
            parts.append(sdf)
        work_df = pd.concat(parts, axis=0).reset_index(drop=True)

    print(f"[Data] sampled rows for stage2 mining = {len(work_df)}")

    # ---------- load stage2 ----------
    stage2_model, stage2_proj, cfg = load_stage2_bundle(
        config_path=args.config,
        full_ckpt_path=args.stage2_full_ckpt,
        device=device,
    )
    stage2_proto, role_names = load_role_proto_from_dir(args.role_proto_dir)

    # ---------- stage2 mining ----------
    stage2_token_df = extract_token_table_for_rows(
        rows_df=work_df,
        model=stage2_model,
        proj_l12=stage2_proj,
        role_prototypes=stage2_proto,
        role_names=role_names,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        desc="Stage2 mining",
    )

    stage2_token_path = os.path.join(args.output_dir, "stage2_all_tokens.csv")
    stage2_token_df.to_csv(stage2_token_path, index=False)
    print(f"[Save] {stage2_token_path}")

    selected_df = select_reference_tokens(
        token_df=stage2_token_df,
        per_slide_topk=args.per_slide_topk,
        per_slide_bottomk=args.per_slide_bottomk,
    )
    selected_path = os.path.join(args.output_dir, "stage2_selected_tokens.csv")
    selected_df.to_csv(selected_path, index=False)
    print(f"[Save] {selected_path}")

    # ---------- build replay patch subset ----------
    replay_patch_df = build_patch_subset_from_selected(selected_df, work_df)
    print(f"[Replay] unique replay patches = {len(replay_patch_df)}")

    # ---------- load stage3 ----------
    stage3_model, stage3_proj, stage3_proto, stage3_role_names, _ = load_stage3_bundle(
        config_path=args.config,
        stage2_full_ckpt=args.stage2_full_ckpt,
        stage3_ckpt=args.stage3_ckpt,
        role_proto_dir=args.role_proto_dir,
        device=device,
    )

    if stage3_role_names != role_names:
        raise ValueError(f"role name mismatch: stage2={role_names}, stage3={stage3_role_names}")

    # ---------- stage3 replay on same patches ----------
    stage3_token_df = extract_token_table_for_rows(
        rows_df=replay_patch_df,
        model=stage3_model,
        proj_l12=stage3_proj,
        role_prototypes=stage3_proto,
        role_names=role_names,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        desc="Stage3 replay",
    )

    stage3_token_path = os.path.join(args.output_dir, "stage3_replayed_patch_tokens.csv")
    stage3_token_df.to_csv(stage3_token_path, index=False)
    print(f"[Save] {stage3_token_path}")

    # ---------- compare only selected tokens ----------
    merged_df = replay_stage3_on_selected_tokens(
        selected_token_df=selected_df,
        replay_token_df=stage3_token_df,
    )
    merged_path = os.path.join(args.output_dir, "stage2_stage3_token_delta.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"[Save] {merged_path}")

    summary = summarize_delta(merged_df)
    summary["meta"] = {
        "num_pos_slides": args.num_pos_slides,
        "num_neg_slides": args.num_neg_slides,
        "max_patches_per_slide": args.max_patches_per_slide,
        "per_slide_topk": args.per_slide_topk,
        "per_slide_bottomk": args.per_slide_bottomk,
        "role_names": role_names,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Save] {summary_path}")

    print("\n===== Summary =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # ---------- visualize ----------
    if args.save_patch_vis:
        vis_dir = os.path.join(args.output_dir, "patch_visuals")
        save_visual_examples(
            merged_df=merged_df,
            out_dir=vis_dir,
            image_size=args.image_size,
            token_grid_size=args.token_grid_size,
            per_group=args.vis_per_group,
        )
        print(f"[Save] patch visuals -> {vis_dir}")


if __name__ == "__main__":
    main()