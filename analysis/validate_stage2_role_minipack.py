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
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import yaml
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    # 允许 patch-guided routing 等新字段兼容
    missing, unexpected = model.load_state_dict(ckpt["student_state_dict"], strict=False)
    print("[load student] missing keys:", len(missing))
    print("[load student] unexpected keys:", len(unexpected))
    if len(missing) > 0:
        print("  first missing:", missing[:10])
    if len(unexpected) > 0:
        print("  first unexpected:", unexpected[:10])

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

        if "slide_id" not in df.columns:
            df["slide_id"] = df["svs_path"].astype(str)

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
            "slide_label": int(row["slide_label"]) if "slide_label" in row and pd.notna(row["slide_label"]) else -1,
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
# image stats
# =========================================================
def compute_patch_image_stats(meta: dict) -> Dict[str, float]:
    slide = openslide.OpenSlide(meta["svs_path"])
    try:
        img = slide.read_region(
            (int(meta["coord_x"]), int(meta["coord_y"])),
            int(meta["patch_level"]),
            (int(meta["patch_size"]), int(meta["patch_size"])),
        ).convert("RGB")
    finally:
        slide.close()

    arr = np.asarray(img).astype(np.float32) / 255.0   # [H, W, 3]

    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mean_intensity = float(gray.mean())
    std_intensity = float(gray.std())

    # tissue occupancy: very crude, enough for quick sanity check
    tissue_mask = gray < 0.90
    tissue_occupancy = float(tissue_mask.mean())

    # local variance / texture proxy
    local_var = float(gray.var())

    rgb_max = arr.max(axis=-1)
    rgb_min = arr.min(axis=-1)
    saturation = np.where(rgb_max > 1e-6, (rgb_max - rgb_min) / np.maximum(rgb_max, 1e-6), 0.0)
    mean_saturation = float(saturation.mean())

    colorfulness = float(arr.std(axis=(0, 1)).mean())

    return {
        "img_mean_intensity": mean_intensity,
        "img_std_intensity": std_intensity,
        "img_tissue_occupancy": tissue_occupancy,
        "img_local_variance": local_var,
        "img_mean_saturation": mean_saturation,
        "img_colorfulness": colorfulness,
    }


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
def project_features_to_role_space(features: np.ndarray, proj_head, device="cpu", batch_size=4096):
    outs = []
    for start in tqdm(
        range(0, len(features), batch_size),
        total=(len(features) + batch_size - 1) // batch_size,
        desc="Projecting to role space",
        leave=False,
    ):
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
# summaries
# =========================================================
def summarize_role_purity(token_df: pd.DataFrame, role_names: List[str]) -> pd.DataFrame:
    rows = []

    for role_id, role_name in enumerate(role_names):
        sub = token_df[token_df["pred_role_id"] == role_id].copy()
        if len(sub) == 0:
            rows.append({
                "role_id": role_id,
                "role_name": role_name,
                "num_tokens": 0,
            })
            continue

        row = {
            "role_id": role_id,
            "role_name": role_name,
            "num_tokens": int(len(sub)),
            "mean_top1_affinity": float(sub["top1_affinity"].mean()),
            "mean_top1_margin": float(sub["top1_margin"].mean()),
            "mean_route_conf": float(sub["route_conf"].mean()),
            "mean_patch_role_entropy": float(sub["role_entropy"].mean()),
        }

        # expert ratio
        expert_ratio = sub["expert_id"].value_counts(normalize=True).sort_index()
        for e, v in expert_ratio.items():
            row[f"expert_ratio_e{int(e)}"] = float(v)

        # pred_label ratio
        if "pred_label" in sub.columns:
            pred_ratio = sub["pred_label"].value_counts(normalize=True)
            for k, v in pred_ratio.items():
                row[f"pred_label_ratio_{str(k)}"] = float(v)

        # slide label ratio
        if "slide_label" in sub.columns and (sub["slide_label"] >= 0).any():
            slide_ratio = sub["slide_label"].value_counts(normalize=True).sort_index()
            for k, v in slide_ratio.items():
                row[f"slide_label_ratio_{int(k)}"] = float(v)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_role_failure_modes(
    token_df: pd.DataFrame,
    role_names: List[str],
    topk_per_role: int = 300,
    bottomk_per_role: int = 300,
) -> pd.DataFrame:
    rows = []

    for role_id, role_name in enumerate(role_names):
        sub = token_df[token_df["pred_role_id"] == role_id].copy()
        if len(sub) == 0:
            continue

        sub_top = sub.sort_values("top1_margin", ascending=False).head(topk_per_role).copy()
        sub_bottom = sub.sort_values("top1_margin", ascending=True).head(bottomk_per_role).copy()

        for tag, cur in [("top_margin", sub_top), ("bottom_margin", sub_bottom)]:
            if len(cur) == 0:
                continue

            rows.append({
                "role_id": role_id,
                "role_name": role_name,
                "subset": tag,
                "num_tokens": int(len(cur)),
                "mean_top1_margin": float(cur["top1_margin"].mean()),
                "mean_top1_affinity": float(cur["top1_affinity"].mean()),
                "mean_route_conf": float(cur["route_conf"].mean()),
                "mean_img_mean_intensity": float(cur["img_mean_intensity"].mean()),
                "mean_img_std_intensity": float(cur["img_std_intensity"].mean()),
                "mean_img_tissue_occupancy": float(cur["img_tissue_occupancy"].mean()),
                "mean_img_local_variance": float(cur["img_local_variance"].mean()),
                "mean_img_mean_saturation": float(cur["img_mean_saturation"].mean()),
                "mean_img_colorfulness": float(cur["img_colorfulness"].mean()),
            })

    return pd.DataFrame(rows)


def build_slide_features(token_df: pd.DataFrame, role_names: List[str]) -> pd.DataFrame:
    rows = []

    for slide_id, sub in token_df.groupby("slide_id"):
        row = {
            "slide_id": slide_id,
            "project": str(sub["project"].iloc[0]) if "project" in sub.columns else "",
            "slide_label": int(sub["slide_label"].iloc[0]) if "slide_label" in sub.columns else -1,
            "num_tokens": int(len(sub)),
        }

        # baseline feature summaries
        row["mean_route_conf"] = float(sub["route_conf"].mean())
        row["mean_top1_margin"] = float(sub["top1_margin"].mean())

        # role histogram
        role_hist = sub["pred_role_id"].value_counts(normalize=True).sort_index()
        for rid in range(len(role_names)):
            row[f"role_hist_r{rid}"] = float(role_hist.get(rid, 0.0))

        # role-wise affinity / margin
        for rid, rname in enumerate(role_names):
            sub_r = sub[sub["pred_role_id"] == rid]
            row[f"role_{rname}_count"] = int(len(sub_r))
            row[f"role_{rname}_top1_margin_mean"] = float(sub_r["top1_margin"].mean()) if len(sub_r) > 0 else 0.0
            row[f"role_{rname}_top1_affinity_mean"] = float(sub_r["top1_affinity"].mean()) if len(sub_r) > 0 else 0.0

            aff_col = f"aff_{rname}"
            if aff_col in sub.columns:
                row[f"role_{rname}_affinity_mean_all"] = float(sub[aff_col].mean())
                row[f"role_{rname}_affinity_top10p_mean"] = float(sub[aff_col].quantile(0.90))

        rows.append(row)

    return pd.DataFrame(rows)


def run_slide_probe(slide_df: pd.DataFrame) -> Dict[str, dict]:
    if "slide_label" not in slide_df.columns:
        return {"error": "slide_label not found"}

    df = slide_df[slide_df["slide_label"].isin([0, 1])].copy()
    if len(df) < 20:
        return {"error": f"too few valid slides: {len(df)}"}

    y = df["slide_label"].values.astype(np.int64)

    feature_sets = {
        "baseline_small": [
            "mean_route_conf",
            "mean_top1_margin",
        ],
        "role_hist_only": [c for c in df.columns if c.startswith("role_hist_")],
        "role_stats_only": [
            c for c in df.columns
            if (
                c.endswith("_top1_margin_mean")
                or c.endswith("_top1_affinity_mean")
                or c.endswith("_affinity_mean_all")
                or c.endswith("_affinity_top10p_mean")
            )
        ],
        "role_all": [
            c for c in df.columns
            if c not in ["slide_id", "project", "slide_label"]
        ],
    }

    results = {}

    for feat_name, cols in feature_sets.items():
        cols = [c for c in cols if c in df.columns]
        if len(cols) == 0:
            results[feat_name] = {"error": "no valid features"}
            continue

        X = df[cols].fillna(0.0).values.astype(np.float32)

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except Exception as e:
            results[feat_name] = {"error": f"split failed: {str(e)}"}
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        clf = LogisticRegression(
            random_state=42,
            max_iter=2000,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)

        prob = clf.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(np.int64)

        out = {
            "num_features": len(cols),
            "auc": float(roc_auc_score(y_val, prob)),
            "acc": float(accuracy_score(y_val, pred)),
            "f1": float(f1_score(y_val, pred)),
            "feature_names": cols,
        }

        # feature importance
        coef = np.abs(clf.coef_[0])
        order = np.argsort(-coef)[:10]
        out["top_features"] = [
            {"name": cols[i], "abs_coef": float(coef[i])}
            for i in order
        ]

        results[feat_name] = out

    return results


# =========================================================
# main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Stage2 role validation mini-pack")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage2-full-ckpt", type=str, required=True)
    parser.add_argument("--role-proto-dir", type=str, required=True)
    parser.add_argument("--pool-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--sample-per-project", type=int, default=None)
    parser.add_argument("--sample-per-label", type=int, default=None)
    parser.add_argument("--keep-projects", nargs="+", default=None)
    parser.add_argument("--keep-labels", nargs="+", default=None)

    parser.add_argument("--split-csv", type=str, default=None)
    parser.add_argument("--split-keep", nargs="+", default=None)

    parser.add_argument("--failure-topk", type=int, default=300)
    parser.add_argument("--failure-bottomk", type=int, default=300)

    parser.add_argument("--compute-image-stats", action="store_true")
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
    role_prototypes, role_names = load_role_proto_from_dir(args.role_proto_dir)

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
                    "slide_label": meta["slide_label"],
                    "svs_path": meta["svs_path"],
                    "coord_x": meta["coord_x"],
                    "coord_y": meta["coord_y"],
                    "coord_idx": meta["coord_idx"],
                    "patch_level": meta["patch_level"],
                    "patch_size": meta["patch_size"],
                    "token_idx": int(t),
                    "tokens_per_side": int(tokens_per_side),
                })

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

    top1_idx = role_affinity.argmax(axis=1)
    top1_aff = role_affinity[np.arange(len(role_affinity)), top1_idx]

    sorted_aff = np.sort(role_affinity, axis=1)[:, ::-1]
    top2_aff = sorted_aff[:, 1] if role_affinity.shape[1] >= 2 else np.zeros_like(top1_aff)
    top1_margin = top1_aff - top2_aff

    # entropy on role probs
    role_probs = np.exp(role_affinity / 0.07)
    role_probs = role_probs / np.clip(role_probs.sum(axis=1, keepdims=True), 1e-8, None)
    role_entropy = -np.sum(role_probs * np.log(np.clip(role_probs, 1e-8, None)), axis=1)

    token_df = pd.DataFrame(all_meta)
    token_df["expert_id"] = expert_ids
    token_df["route_conf"] = route_conf
    token_df["pred_role_id"] = top1_idx
    token_df["pred_role"] = [role_names[i] for i in top1_idx]
    token_df["top1_affinity"] = top1_aff
    token_df["top2_affinity"] = top2_aff
    token_df["top1_margin"] = top1_margin
    token_df["role_entropy"] = role_entropy

    for i, role_name in enumerate(role_names):
        token_df[f"aff_{role_name}"] = role_affinity[:, i]

    # optional image stats
    if args.compute_image_stats:
        print("[Stage] computing image stats")
        stats_rows = []
        for _, row in tqdm(token_df.iterrows(), total=len(token_df), desc="Image stats"):
            stats_rows.append(compute_patch_image_stats(row.to_dict()))
        stats_df = pd.DataFrame(stats_rows)
        token_df = pd.concat([token_df.reset_index(drop=True), stats_df.reset_index(drop=True)], axis=1)
    else:
        # keep columns for failure-mode summary compatibility
        for col in [
            "img_mean_intensity",
            "img_std_intensity",
            "img_tissue_occupancy",
            "img_local_variance",
            "img_mean_saturation",
            "img_colorfulness",
        ]:
            token_df[col] = 0.0

    # save token table
    token_csv = os.path.join(args.output_dir, "token_role_analysis.csv")
    token_df.to_csv(token_csv, index=False)
    print(f"[Saved] {token_csv}")

    # role purity
    purity_df = summarize_role_purity(token_df, role_names)
    purity_csv = os.path.join(args.output_dir, "role_purity_summary.csv")
    purity_df.to_csv(purity_csv, index=False)
    print(f"[Saved] {purity_csv}")

    # failure mode summary
    failure_df = summarize_role_failure_modes(
        token_df,
        role_names,
        topk_per_role=args.failure_topk,
        bottomk_per_role=args.failure_bottomk,
    )
    failure_csv = os.path.join(args.output_dir, "role_failure_mode_summary.csv")
    failure_df.to_csv(failure_csv, index=False)
    print(f"[Saved] {failure_csv}")

    # slide-level features
    slide_df = build_slide_features(token_df, role_names)
    slide_csv = os.path.join(args.output_dir, "slide_role_features.csv")
    slide_df.to_csv(slide_csv, index=False)
    print(f"[Saved] {slide_csv}")

    # probe
    probe_res = run_slide_probe(slide_df)
    probe_json = os.path.join(args.output_dir, "slide_probe_results.json")
    with open(probe_json, "w", encoding="utf-8") as f:
        json.dump(probe_res, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {probe_json}")

    # overall meta
    meta = {
        "num_tokens": int(len(token_df)),
        "num_slides": int(slide_df.shape[0]),
        "role_names": role_names,
        "expert_counts": {
            f"E{int(k)}": int(v)
            for k, v in token_df["expert_id"].value_counts().sort_index().to_dict().items()
        },
        "pred_role_counts": {
            str(k): int(v)
            for k, v in token_df["pred_role"].value_counts().to_dict().items()
        },
    }
    meta_json = os.path.join(args.output_dir, "meta_summary.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {meta_json}")

    print(f"[Done] outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()