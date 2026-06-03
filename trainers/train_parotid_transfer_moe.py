#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import random
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr

import h5py
import numpy as np
import pandas as pd
import openslide
from PIL import ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml
import torchvision.transforms.v2 as T
from torchmil.models import ABMIL, DSMIL

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.encoders.moe_encoder import MoEEncoder
from models.plugins.shared_role_prototype import SharedRolePrototype, PatchRoleSummaryFromSharedProto

try:
    from models.plugins.role_aware_tail_plugin import RoleAwareTailWithSharedSummary as PluginClass
except Exception:
    try:
        from models.plugins.role_aware_tail_plugin import RoleAwareTailPlugin as PluginClass
    except Exception:
        PluginClass = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEBUG_ENCODER_OUTPUT = True
DEBUG_ENCODER_OUTPUT_ONCE = True


# =========================================================
# logging / utils
# =========================================================
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def try_compute_auc(labels: List[int], probs: List[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, probs))
    except Exception:
        return float("nan")


def normalize_binary_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0]
    if logits.ndim == 2 and logits.shape[1] == 2:
        return logits[:, 1]
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


@torch.no_grad()
def compute_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    logits = normalize_binary_logits(logits)
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).long()

    acc = (preds == labels).float().mean()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    spec = tn / max(tn + fp, 1)

    auc = try_compute_auc(labels.detach().cpu().tolist(), probs.detach().cpu().tolist())
    return {
        "acc": safe_float(acc),
        "f1": float(f1),
        "auc": float(auc),
        "sens": float(recall),
        "spec": float(spec),
    }


def build_transform(img_size: int = 224, is_train: bool = False):
    if is_train:
        return T.Compose([
            T.ToImage(),
            T.Resize((img_size, img_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ])
    return T.Compose([
        T.ToImage(),
        T.Resize((img_size, img_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def print_trainable_params(module: nn.Module, prefix: str):
    total = 0
    trainable = 0
    for _, p in module.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"[{prefix}] total={total:,} trainable={trainable:,}")


# =========================================================
# dataset
# =========================================================
def subsample_slide_dataframe(
    df: pd.DataFrame,
    max_slides: Optional[int],
    seed: int = 42,
    balance_by_label: bool = True,
) -> pd.DataFrame:
    if max_slides is None or max_slides >= len(df):
        return df.reset_index(drop=True).copy()

    rng = np.random.default_rng(seed)

    if (not balance_by_label) or ("label" not in df.columns):
        idx = rng.choice(len(df), size=max_slides, replace=False)
        return df.iloc[idx].reset_index(drop=True).copy()

    parts = []
    remain_budget = max_slides
    grouped = list(df.groupby("label"))
    n_groups = len(grouped)
    per_group = max_slides // max(n_groups, 1)
    rem = max_slides % max(n_groups, 1)

    for gi, (_, sub) in enumerate(grouped):
        take = per_group + (1 if gi < rem else 0)
        take = min(take, len(sub))
        if take > 0:
            idx = rng.choice(len(sub), size=take, replace=False)
            parts.append(sub.iloc[idx])
            remain_budget -= take

    if remain_budget > 0:
        used_slide_ids = set(pd.concat(parts)["slide_id"].astype(str).tolist()) if len(parts) > 0 else set()
        leftover = df[~df["slide_id"].astype(str).isin(used_slide_ids)].copy()
        if len(leftover) > 0:
            take = min(remain_budget, len(leftover))
            idx = rng.choice(len(leftover), size=take, replace=False)
            parts.append(leftover.iloc[idx])

    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


class ParotidSlideBagDataset(Dataset):
    def __init__(
        self,
        split_csv: str,
        split: str,
        max_slides: Optional[int] = None,
        seed: int = 42,
        balance_by_label: bool = True,
    ):
        if not os.path.exists(split_csv):
            raise FileNotFoundError(split_csv)

        df = pd.read_csv(split_csv)
        need = ["slide_id", "label", "split", "svs_path", "h5_path"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise ValueError(f"split csv missing columns: {miss}")

        split_df = df[df["split"] == split].reset_index(drop=True).copy()
        if len(split_df) == 0:
            raise ValueError(f"No rows found for split={split}")

        self.df = subsample_slide_dataframe(
            df=split_df,
            max_slides=max_slides,
            seed=seed,
            balance_by_label=balance_by_label,
        )

        print(
            f"[{split}] num_slides = {len(self.df)} "
            f"(original={len(split_df)}, max_slides={max_slides})"
        )
        print(self.df["label"].value_counts().sort_index())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return {
            "slide_id": str(row["slide_id"]),
            "label": int(row["label"]),
            "svs_path": str(row["svs_path"]),
            "h5_path": str(row["h5_path"]),
            "project": str(row["project"]) if "project" in row and pd.notna(row["project"]) else "",
        }


def slide_bag_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "slide_id": [x["slide_id"] for x in batch],
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "svs_path": [x["svs_path"] for x in batch],
        "h5_path": [x["h5_path"] for x in batch],
        "project": [x["project"] for x in batch],
    }


# =========================================================
# MIL heads
# =========================================================
class ABMILWrapper(nn.Module):
    def __init__(self, in_dim: int, device: str):
        super().__init__()
        self.model = ABMIL(
            in_shape=(in_dim,),
            att_dim=256,
            att_act="tanh",
            gated=False,
        )
        self.to(device)

    def forward(self, bag_feats: torch.Tensor):
        out = self.model(bag_feats)
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict):
            for key in ["logits", "pred", "scores", "output"]:
                if key in out:
                    return out[key]
        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    return item
        raise TypeError(f"Unsupported ABMIL output type: {type(out)}")


class DSMILWrapper(nn.Module):
    def __init__(self, in_dim: int, device: str):
        super().__init__()
        self.model = DSMIL(
            in_shape=(in_dim,),
            att_dim=128,
            nonlinear_q=False,
            nonlinear_v=False,
            dropout=0.0,
        )
        self.to(device)

    def forward(self, bag_feats: torch.Tensor):
        if bag_feats.ndim == 2:
            bag_feats = bag_feats.unsqueeze(0)
        out = self.model(bag_feats)
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, dict):
            for key in ["logits", "pred", "scores", "output", "Y_pred"]:
                if key in out:
                    return out[key]
        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    return item
        raise TypeError(f"Unsupported DSMIL output type: {type(out)}")


def build_aggregator(mil_model: str, in_dim: int, device: str):
    if mil_model == "abmil":
        return ABMILWrapper(in_dim=in_dim, device=device)
    if mil_model == "dsmil":
        return DSMILWrapper(in_dim=in_dim, device=device)
    raise ValueError(f"Unsupported mil_model: {mil_model}")


def load_aggregator_init_if_available(
    aggregator: nn.Module,
    init_path: Optional[str],
    device: str,
):
    if init_path is None or str(init_path).strip() == "":
        print("[MIL Init] no init checkpoint provided, use random init.")
        return

    if not os.path.exists(init_path):
        raise FileNotFoundError(f"mil init ckpt not found: {init_path}")

    ckpt = torch.load(init_path, map_location=device)
    loaded = False

    if isinstance(ckpt, dict):
        for key in ["aggregator_state_dict", "model_state_dict", "state_dict"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                aggregator.load_state_dict(ckpt[key], strict=True)
                loaded = True
                print(f"[MIL Init] loaded from key '{key}' in {init_path}")
                break

    if (not loaded) and isinstance(ckpt, dict):
        try:
            aggregator.load_state_dict(ckpt, strict=True)
            loaded = True
            print(f"[MIL Init] loaded raw state_dict from {init_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MIL init from {init_path}. "
                f"Available top-level keys={list(ckpt.keys())}; error={e}"
            )

    if not loaded:
        raise RuntimeError(f"Could not load MIL init checkpoint: {init_path}")


# =========================================================
# stage2 loader
# =========================================================
def build_encoder_and_proj_from_stage2(
    base_encoder_cfg,
    moe_encoder_cfg,
    stage2_full_ckpt: str,
    device: str,
):
    ckpt = torch.load(stage2_full_ckpt, map_location="cpu")
    if "student_state_dict" not in ckpt:
        raise KeyError("student_state_dict not found in checkpoint")
    if "distiller_state_dict" not in ckpt:
        raise KeyError("distiller_state_dict not found in checkpoint")

    encoder = MoEEncoder(base_encoder_cfg, moe_encoder_cfg)
    encoder.load_state_dict(ckpt["student_state_dict"], strict=True)
    encoder = encoder.to(device)

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
    freeze_module(proj_l12)

    print("[Encoder] loaded student_state_dict + proj_l12 from stage2 full ckpt")
    if hasattr(encoder, "moe_layers_idx"):
        print(f"[Encoder] moe_layers_idx = {encoder.moe_layers_idx}")
    print(f"[Proj] proj_l12: {proj_in_dim} -> {proj_out_dim}")

    return encoder, proj_l12


def debug_print_moe_param_names(encoder: nn.Module, max_lines: int = 300):
    print("\n[Debug] MoE-related parameter names:")
    cnt = 0
    for name, _ in encoder.named_parameters():
        if any(k in name.lower() for k in ["expert", "experts", "gate", "router", "routing"]):
            print("  ", name)
            cnt += 1
            if cnt >= max_lines:
                print("  ... truncated ...")
                break
    print(f"[Debug] printed {cnt} MoE-related parameter names\n")


def resolve_last_moe_layer_idx(moe_layers, num_layers: int) -> int:
    if len(moe_layers) == 0:
        raise ValueError("moe_layers is empty")
    last_idx = moe_layers[-1]
    if last_idx < 0:
        last_idx = num_layers + last_idx
    return int(last_idx)


def unfreeze_last_moe_params_by_name(
    encoder: nn.Module,
    moe_layers: List[int],
    num_layers: int = 12,
    train_gate: bool = True,
    train_experts: bool = True,
):
    last_moe_layer_idx = resolve_last_moe_layer_idx(moe_layers, num_layers)
    print(f"[Encoder] resolved last MoE layer idx = {last_moe_layer_idx}")

    gate_keys = ["router", "gate", "gating", "routing"]
    expert_keys = ["expert", "experts"]
    layer_patterns = [
        f".layer.{last_moe_layer_idx}.",
        f".blocks.{last_moe_layer_idx}.",
        f"blocks.{last_moe_layer_idx}.",
        f"layer.{last_moe_layer_idx}.",
    ]

    trainable_names = []
    for name, p in encoder.named_parameters():
        lname = name.lower()
        if not any(pat in name for pat in layer_patterns):
            continue

        enable = False
        if train_gate and any(k in lname for k in gate_keys):
            enable = True
        if train_experts and any(k in lname for k in expert_keys):
            enable = True

        if enable:
            p.requires_grad = True
            trainable_names.append(name)

    print("[Encoder] trainable params in last MoE layer:")
    for n in trainable_names:
        print("  ", n)

    if len(trainable_names) == 0:
        print("[Warn] no params were unfrozen. Please inspect encoder.named_parameters().")


# =========================================================
# patch IO
# =========================================================
def sample_coords_from_h5(
    h5_path: str,
    max_patches_per_slide: int,
    random_sample: bool,
    seed: int,
):
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs.items())

    patch_size = int(attrs.get("patch_size", 256))
    patch_level = int(attrs.get("patch_level", 0))

    n = len(coords)
    if n > max_patches_per_slide:
        if random_sample:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=max_patches_per_slide, replace=False)
            coords = coords[idx]
        else:
            coords = coords[:max_patches_per_slide]

    return coords, patch_size, patch_level


def read_patch_batch(
    slide,
    coords: np.ndarray,
    patch_size: int,
    patch_level: int,
    transform,
):
    imgs = []
    for xy in coords:
        x, y = int(xy[0]), int(xy[1])
        img = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
        imgs.append(transform(img))
    return torch.stack(imgs, dim=0)


# =========================================================
# stage2-consistent feature extraction
# =========================================================
def forward_encoder_for_stage2_features(
    encoder: nn.Module,
    patch_imgs: torch.Tensor,
):
    out = encoder(
        patch_imgs,
        return_gates=True,
        mask=None,
        is_eval=True,
        return_features=True,
        offline_cluster_ids=None,
    )

    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise RuntimeError(
            f"Unexpected encoder output format. "
            f"Expected 4-tuple, got type={type(out)}"
        )

    student_out, gate_info_list, feature_dict, moe_feature_list = out

    global DEBUG_ENCODER_OUTPUT_ONCE
    if DEBUG_ENCODER_OUTPUT and DEBUG_ENCODER_OUTPUT_ONCE:
        print("[DEBUG] forward_encoder_for_stage2_features output summary:")
        print(f"  student_out: type={type(student_out)} shape={tuple(student_out.shape) if torch.is_tensor(student_out) else 'N/A'}")
        print(f"  feature_dict keys: {list(feature_dict.keys()) if isinstance(feature_dict, dict) else 'N/A'}")
        if isinstance(moe_feature_list, (tuple, list)):
            print(f"  moe_feature_list len: {len(moe_feature_list)}")
            for i, x in enumerate(moe_feature_list):
                if torch.is_tensor(x):
                    print(f"    moe_feature_list[{i}] shape={tuple(x.shape)}")
                else:
                    print(f"    moe_feature_list[{i}] type={type(x)}")
    return student_out, gate_info_list, feature_dict, moe_feature_list


def encode_patch_batch(
    encoder: nn.Module,
    patch_imgs: torch.Tensor,
    use_last_moe_output: bool = True,
) -> torch.Tensor:
    _, _, feature_dict, moe_feature_list = forward_encoder_for_stage2_features(
        encoder=encoder,
        patch_imgs=patch_imgs,
    )

    if use_last_moe_output and isinstance(moe_feature_list, (tuple, list)) and len(moe_feature_list) > 0:
        feat_tokens = moe_feature_list[-1]
        source_name = "moe_feature_list[-1]"
    else:
        if (not isinstance(feature_dict, dict)) or ("layer_12" not in feature_dict):
            raise KeyError(
                f"'layer_12' not found in feature_dict. "
                f"Available keys={list(feature_dict.keys()) if isinstance(feature_dict, dict) else 'N/A'}"
            )
        feat_tokens = feature_dict["layer_12"]
        source_name = "feature_dict['layer_12']"

    if not torch.is_tensor(feat_tokens) or feat_tokens.ndim != 3:
        raise ValueError(
            f"Expected token tensor [B, T+1, D], got "
            f"type={type(feat_tokens)} "
            f"shape={tuple(feat_tokens.shape) if torch.is_tensor(feat_tokens) else 'N/A'}"
        )

    patch_tokens = feat_tokens[:, 1:, :]
    if patch_tokens.shape[1] == 0:
        raise RuntimeError(f"No patch tokens found, got shape={tuple(patch_tokens.shape)}")

    feat = patch_tokens.mean(dim=1)

    global DEBUG_ENCODER_OUTPUT_ONCE
    if DEBUG_ENCODER_OUTPUT and DEBUG_ENCODER_OUTPUT_ONCE:
        print(
            f"[DEBUG] use {source_name}: "
            f"raw_tokens={tuple(feat_tokens.shape)} "
            f"patch_tokens={tuple(patch_tokens.shape)} "
            f"pooled_feat={tuple(feat.shape)}"
        )
        DEBUG_ENCODER_OUTPUT_ONCE = False

    return feat


@torch.no_grad()
def infer_feat_dim(
    encoder: nn.Module,
    device: str,
    img_size: int,
    use_last_moe_output: bool = True,
):
    dummy = torch.randn(2, 3, img_size, img_size, device=device)
    feat = encode_patch_batch(
        encoder=encoder,
        patch_imgs=dummy,
        use_last_moe_output=use_last_moe_output,
    )
    return int(feat.shape[-1])


def encode_one_slide_bag(
    encoder: nn.Module,
    svs_path: str,
    h5_path: str,
    transform,
    device: str,
    max_patches_per_slide: int,
    patch_batch_size: int,
    random_sample_patches: bool,
    seed: int,
    use_last_moe_output: bool = True,
) -> torch.Tensor:
    coords, patch_size, patch_level = sample_coords_from_h5(
        h5_path=h5_path,
        max_patches_per_slide=max_patches_per_slide,
        random_sample=random_sample_patches,
        seed=seed,
    )

    slide = openslide.OpenSlide(svs_path)
    feats = []
    try:
        for i in range(0, len(coords), patch_batch_size):
            coord_chunk = coords[i:i + patch_batch_size]
            imgs = read_patch_batch(
                slide=slide,
                coords=coord_chunk,
                patch_size=patch_size,
                patch_level=patch_level,
                transform=transform,
            ).to(device, non_blocking=True)

            feat = encode_patch_batch(
                encoder=encoder,
                patch_imgs=imgs,
                use_last_moe_output=use_last_moe_output,
            )
            feats.append(feat)
    finally:
        slide.close()

    return torch.cat(feats, dim=0)


# =========================================================
# role proto helpers
# =========================================================
def load_role_proto_bundle(role_proto_dir: str, device: str):
    proto_path = os.path.join(role_proto_dir, "role_prototypes_init.npy")
    names_path = os.path.join(role_proto_dir, "role_names.json")
    if not os.path.exists(proto_path):
        raise FileNotFoundError(f"Missing prototype file: {proto_path}")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Missing role names file: {names_path}")

    protos = torch.from_numpy(np.load(proto_path).astype("float32")).to(device)
    protos = F.normalize(protos, dim=-1)

    with open(names_path, "r", encoding="utf-8") as f:
        role_names = json.load(f)

    role_to_idx = {str(n): i for i, n in enumerate(role_names)}
    return protos, role_names, role_to_idx


def build_role_summary_from_teacher_feats(
    teacher_feats: torch.Tensor,       # [N, 1280]
    role_protos: torch.Tensor,         # [R, 1280]
    role_tau: float = 1.0,
):
    teacher_feats = F.normalize(teacher_feats, dim=-1)
    role_protos = F.normalize(role_protos, dim=-1)

    logits = teacher_feats @ role_protos.t()   # [N, R]
    probs = torch.softmax(logits / role_tau, dim=-1)

    gaps = []
    R = logits.shape[-1]
    for r in range(R):
        cur = logits[:, r]
        other_ids = [i for i in range(R) if i != r]
        if len(other_ids) == 0:
            other_max = torch.zeros_like(cur)
        else:
            other_max = logits[:, other_ids].max(dim=-1).values
        gaps.append(cur - other_max)
    role_gaps = torch.stack(gaps, dim=-1)

    top2 = torch.topk(logits, k=min(2, R), dim=-1).values
    if R >= 2:
        role_top1_gap = top2[:, 0] - top2[:, 1]
    else:
        role_top1_gap = torch.ones_like(top2[:, 0])

    return {
        "role_logits": logits,
        "role_probs": probs,
        "role_gaps": role_gaps,
        "role_top1_gap": role_top1_gap,
    }


def compute_slide_role_proto_loss(
    patch_feat_raw: torch.Tensor,          # [N, 384]
    slide_label: int,
    proj_l12: nn.Module,
    role_protos: torch.Tensor,             # [R, 1280]
    role_to_idx: Dict[str, int],
    role_tau: float,
    proto_tumor_name: str,
    proto_negative_role_names: List[str],
    loss_weight: float,
    margin: float,
    conf_thresh: float,
    top1_gap_thresh: float,
    min_kept: int,
):
    """
    新目标域迁移里最稳妥的版本：
    - 先用 proj_l12 投到 teacher space
    - 再和新的 role proto 做匹配
    - 正片：鼓励 tumor > max(neg roles) + margin
    - 负片：鼓励 max(neg roles) > tumor + margin
    - 只在高置信 token 上做；若太少，退化为 top-k
    """
    if loss_weight <= 0:
        zero = patch_feat_raw.new_tensor(0.0)
        return zero, {
            "proto_kept": 0,
            "proto_loss": 0.0,
        }

    if proto_tumor_name not in role_to_idx:
        raise KeyError(f"proto_tumor_name='{proto_tumor_name}' not found in role names")

    neg_ids = []
    for name in proto_negative_role_names:
        if name not in role_to_idx:
            raise KeyError(f"negative role '{name}' not found in role names")
        neg_ids.append(role_to_idx[name])

    tumor_id = role_to_idx[proto_tumor_name]

    teacher_feats = proj_l12(patch_feat_raw)       # [N, 1280]
    teacher_feats = F.normalize(teacher_feats, dim=-1)

    role_summary = build_role_summary_from_teacher_feats(
        teacher_feats=teacher_feats,
        role_protos=role_protos,
        role_tau=role_tau,
    )
    logits = role_summary["role_logits"]           # [N, R]
    probs = role_summary["role_probs"]
    role_top1_gap = role_summary["role_top1_gap"]

    pred = probs.argmax(dim=-1)
    conf = probs.max(dim=-1).values

    tumor_logit = logits[:, tumor_id]
    neg_logit = logits[:, neg_ids].max(dim=-1).values if len(neg_ids) > 0 else torch.zeros_like(tumor_logit)

    if int(slide_label) == 1:
        keep = (pred == tumor_id) & (conf >= conf_thresh) & (role_top1_gap >= top1_gap_thresh)
        score = tumor_logit - neg_logit
        per_token_loss = F.relu(margin - score)
    else:
        keep_neg_pred = torch.zeros_like(pred, dtype=torch.bool)
        for nid in neg_ids:
            keep_neg_pred |= (pred == nid)
        keep = keep_neg_pred & (conf >= conf_thresh) & (role_top1_gap >= top1_gap_thresh)
        score = neg_logit - tumor_logit
        per_token_loss = F.relu(margin - score)

    kept_idx = torch.where(keep)[0]

    if kept_idx.numel() < min_kept:
        topk = min(max(min_kept, 1), patch_feat_raw.shape[0])
        if int(slide_label) == 1:
            rank_score = tumor_logit - neg_logit
        else:
            rank_score = neg_logit - tumor_logit
        kept_idx = torch.topk(rank_score, k=topk, largest=True).indices

    loss = per_token_loss[kept_idx].mean() * loss_weight

    stats = {
        "proto_kept": int(kept_idx.numel()),
        "proto_loss": float(loss.detach().cpu().item()),
        "proto_mean_conf": float(conf[kept_idx].mean().detach().cpu().item()) if kept_idx.numel() > 0 else 0.0,
    }
    return loss, stats


# =========================================================
# plugin helpers
# =========================================================
def build_role_summary_from_feats(
    patch_feat: torch.Tensor,
    summary_builder: PatchRoleSummaryFromSharedProto,
    use_role_logits: bool,
):
    role_dict = summary_builder(patch_feat.unsqueeze(0))
    return {
        "patch_role_probs": role_dict["patch_role_probs"],
        "patch_role_gaps": role_dict["patch_role_gaps"],
        "patch_top1_gap": role_dict["patch_top1_gap"],
        "patch_role_logits": role_dict["patch_role_logits"] if use_role_logits else None,
    }


def feature_preserve_loss(
    feat_raw: torch.Tensor,
    feat_plugin: torch.Tensor,
    patch_mask: torch.Tensor,
    mode: str = "l2",
):
    mask = patch_mask.unsqueeze(-1).float()
    if mode == "l2":
        diff = ((feat_plugin - feat_raw) ** 2) * mask
        denom = mask.sum().clamp_min(1.0) * feat_raw.shape[-1]
        return diff.sum() / denom
    if mode == "cosine":
        raw = F.normalize(feat_raw, dim=-1)
        plugin = F.normalize(feat_plugin, dim=-1)
        cos = (raw * plugin).sum(dim=-1)
        loss = (1.0 - cos) * patch_mask.float()
        return loss.sum() / patch_mask.float().sum().clamp_min(1.0)
    raise ValueError(f"Unsupported preserve mode: {mode}")


# =========================================================
# per-slide forward
# =========================================================
def forward_one_slide(
    encoder: nn.Module,
    aggregator: nn.Module,
    plugin: Optional[nn.Module],
    summary_builder: PatchRoleSummaryFromSharedProto,
    proj_l12: nn.Module,
    role_protos: torch.Tensor,
    role_to_idx: Dict[str, int],
    svs_path: str,
    h5_path: str,
    transform,
    device: str,
    max_patches_per_slide: int,
    patch_batch_size: int,
    random_sample_patches: bool,
    seed: int,
    use_plugin: bool,
    use_role_logits: bool,
    use_last_moe_output: bool,
    slide_label: int,
    enable_phase1_role_proto_loss: bool,
    role_proto_loss_weight: float,
    role_proto_margin: float,
    role_proto_conf_thresh: float,
    role_proto_top1_gap_thresh: float,
    role_proto_min_kept: int,
    role_tau: float,
    proto_tumor_name: str,
    proto_negative_role_names: List[str],
):
    patch_feat_raw = encode_one_slide_bag(
        encoder=encoder,
        svs_path=svs_path,
        h5_path=h5_path,
        transform=transform,
        device=device,
        max_patches_per_slide=max_patches_per_slide,
        patch_batch_size=patch_batch_size,
        random_sample_patches=random_sample_patches,
        seed=seed,
        use_last_moe_output=use_last_moe_output,
    )

    patch_feat = patch_feat_raw.unsqueeze(0)
    patch_mask = torch.ones(1, patch_feat.shape[1], dtype=torch.bool, device=device)

    loss_preserve = None
    if use_plugin:
        role_summary = build_role_summary_from_feats(
            patch_feat=patch_feat_raw,
            summary_builder=summary_builder,
            use_role_logits=use_role_logits,
        )
        plugin_out = plugin(
            patch_feat=patch_feat,
            patch_role_probs=role_summary["patch_role_probs"],
            patch_role_gaps=role_summary["patch_role_gaps"],
            patch_role_logits=role_summary["patch_role_logits"],
            patch_top1_gap=role_summary["patch_top1_gap"],
            return_aux=True,
        )
        if isinstance(plugin_out, tuple):
            patch_feat_plugin, _ = plugin_out
        else:
            patch_feat_plugin = plugin_out

        loss_preserve = feature_preserve_loss(
            feat_raw=patch_feat,
            feat_plugin=patch_feat_plugin,
            patch_mask=patch_mask,
            mode="l2",
        )
        patch_feat = patch_feat_plugin

    logits = aggregator(patch_feat)
    logits = normalize_binary_logits(logits)

    proto_loss = patch_feat_raw.new_tensor(0.0)
    proto_stats = {
        "proto_kept": 0,
        "proto_loss": 0.0,
        "proto_mean_conf": 0.0,
    }
    if enable_phase1_role_proto_loss and role_proto_loss_weight > 0:
        proto_loss, proto_stats = compute_slide_role_proto_loss(
            patch_feat_raw=patch_feat_raw,
            slide_label=int(slide_label),
            proj_l12=proj_l12,
            role_protos=role_protos,
            role_to_idx=role_to_idx,
            role_tau=role_tau,
            proto_tumor_name=proto_tumor_name,
            proto_negative_role_names=proto_negative_role_names,
            loss_weight=role_proto_loss_weight,
            margin=role_proto_margin,
            conf_thresh=role_proto_conf_thresh,
            top1_gap_thresh=role_proto_top1_gap_thresh,
            min_kept=role_proto_min_kept,
        )

    return logits.view(1), loss_preserve, proto_loss, proto_stats


# =========================================================
# save helpers
# =========================================================
def save_checkpoint(
    path: str,
    phase: str,
    epoch: int,
    encoder: nn.Module,
    aggregator: nn.Module,
    plugin: Optional[nn.Module],
    cfg: dict,
    val_auc: float,
):
    torch.save(
        {
            "phase": phase,
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "aggregator_state_dict": aggregator.state_dict(),
            "plugin_state_dict": plugin.state_dict() if plugin is not None else None,
            "args": cfg,
            "val_auc": float(val_auc),
        },
        path,
    )


def save_full_bundle(
    path: str,
    phase: str,
    epoch: int,
    encoder: nn.Module,
    aggregator: nn.Module,
    plugin: Optional[nn.Module],
    proj_l12: nn.Module,
    cfg: dict,
    val_auc: float,
):
    torch.save(
        {
            "phase": phase,
            "epoch": epoch,
            "student_state_dict": encoder.state_dict(),
            "aggregator_state_dict": aggregator.state_dict(),
            "plugin_state_dict": plugin.state_dict() if plugin is not None else None,
            "proj_l12_state_dict": proj_l12.state_dict(),
            "cfg": cfg,
            "val_auc": float(val_auc),
        },
        path,
    )


# =========================================================
# train / eval
# =========================================================
def train_one_epoch(
    encoder: nn.Module,
    aggregator: nn.Module,
    plugin: Optional[nn.Module],
    summary_builder: PatchRoleSummaryFromSharedProto,
    proj_l12: nn.Module,
    role_protos: torch.Tensor,
    role_to_idx: Dict[str, int],
    loader: DataLoader,
    transform,
    device: str,
    args,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    desc_prefix: str = "Train",
    enable_role_proto_loss: bool = False,
):
    encoder.train()
    aggregator.train()
    if plugin is not None:
        plugin.train()

    train_losses = []
    logits_all = []
    labels_all = []

    proto_loss_vals = []
    proto_kept_vals = []

    pbar = tqdm(loader, desc=f"{desc_prefix} {epoch}/{total_epochs}", leave=False)
    for step, batch in enumerate(pbar):
        labels = batch["labels"].to(device).float()
        batch_logits = []
        batch_loss = 0.0

        optimizer.zero_grad()

        for i in range(len(batch["slide_id"])):
            logits_i, loss_preserve_i, proto_loss_i, proto_stats_i = forward_one_slide(
                encoder=encoder,
                aggregator=aggregator,
                plugin=plugin,
                summary_builder=summary_builder,
                proj_l12=proj_l12,
                role_protos=role_protos,
                role_to_idx=role_to_idx,
                svs_path=batch["svs_path"][i],
                h5_path=batch["h5_path"][i],
                transform=transform,
                device=device,
                max_patches_per_slide=args.max_patches_per_slide,
                patch_batch_size=args.patch_batch_size,
                random_sample_patches=args.random_sample_patches,
                seed=args.seed + epoch * 10000 + step * 100 + i,
                use_plugin=args.use_plugin,
                use_role_logits=args.use_role_logits,
                use_last_moe_output=args.use_last_moe_output,
                slide_label=int(batch["labels"][i].item()),
                enable_phase1_role_proto_loss=enable_role_proto_loss,
                role_proto_loss_weight=float(getattr(args, "role_proto_loss_weight", 0.0)),
                role_proto_margin=float(getattr(args, "role_proto_margin", 0.15)),
                role_proto_conf_thresh=float(getattr(args, "role_proto_conf_thresh", 0.55)),
                role_proto_top1_gap_thresh=float(getattr(args, "role_proto_top1_gap_thresh", 0.05)),
                role_proto_min_kept=int(getattr(args, "role_proto_min_kept", 1)),
                role_tau=float(getattr(args, "role_tau", 1.0)),
                proto_tumor_name=str(getattr(args, "proto_tumor_name", "tumor")),
                proto_negative_role_names=list(getattr(args, "proto_negative_role_names", ["stroma", "normal_epithelium"])),
            )

            loss_i = F.binary_cross_entropy_with_logits(logits_i, labels[i:i+1])

            if args.use_plugin and loss_preserve_i is not None:
                loss_i = loss_i + args.feat_preserve_weight * loss_preserve_i

            if enable_role_proto_loss:
                loss_i = loss_i + proto_loss_i
                proto_loss_vals.append(float(proto_stats_i["proto_loss"]))
                proto_kept_vals.append(float(proto_stats_i["proto_kept"]))

            batch_logits.append(logits_i.detach().cpu())
            batch_loss = batch_loss + loss_i

        batch_loss = batch_loss / len(batch["slide_id"])
        batch_loss.backward()
        optimizer.step()

        train_losses.append(batch_loss.item())
        logits_all.append(torch.cat(batch_logits, dim=0))
        labels_all.append(labels.detach().cpu())

        postfix = {"loss": f"{np.mean(train_losses):.4f}"}
        if len(proto_loss_vals) > 0:
            postfix["proto"] = f"{np.mean(proto_loss_vals):.4f}"
            postfix["kept"] = f"{np.mean(proto_kept_vals):.2f}"
        pbar.set_postfix(**postfix)

    logits_all = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
    labels_all = torch.cat(labels_all, dim=0).long() if labels_all else torch.empty(0, dtype=torch.long)
    train_metric = compute_metrics_from_logits(logits_all, labels_all)

    out = {
        "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
        "train_acc": train_metric["acc"],
        "train_f1": train_metric["f1"],
        "train_auc": train_metric["auc"],
        "train_sens": train_metric["sens"],
        "train_spec": train_metric["spec"],
    }
    if len(proto_loss_vals) > 0:
        out["train_proto_loss"] = float(np.mean(proto_loss_vals))
        out["train_proto_kept"] = float(np.mean(proto_kept_vals))
    else:
        out["train_proto_loss"] = 0.0
        out["train_proto_kept"] = 0.0
    return out

def evaluate_checkpoint_and_save(
    ckpt_path: str,
    phase_name: str,
    encoder: nn.Module,
    aggregator: nn.Module,
    plugin: Optional[nn.Module],
    summary_builder: PatchRoleSummaryFromSharedProto,
    test_loader: DataLoader,
    eval_transform,
    device: str,
    args,
    out_dir: str,
):
    if not os.path.exists(ckpt_path):
        print(f"[Skip Test] {phase_name}: ckpt not found -> {ckpt_path}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    aggregator.load_state_dict(ckpt["aggregator_state_dict"])
    if plugin is not None and ckpt.get("plugin_state_dict", None) is not None:
        plugin.load_state_dict(ckpt["plugin_state_dict"])

    test_stats, test_pred_df = evaluate(
        encoder=encoder,
        aggregator=aggregator,
        plugin=plugin,
        summary_builder=summary_builder,
        loader=test_loader,
        transform=eval_transform,
        device=device,
        args=args,
        desc=f"Test-{phase_name}",
    )

    pred_path = os.path.join(out_dir, f"test_predictions_{phase_name}.csv")
    metrics_path = os.path.join(out_dir, f"final_test_metrics_{phase_name}.json")

    test_pred_df.to_csv(pred_path, index=False)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "phase": phase_name,
                "ckpt_path": ckpt_path,
                "test_stats": test_stats,
            },
            f,
            indent=2,
        )

    print(
        f"[Test-{phase_name}] "
        f"test_loss={test_stats['loss']:.4f} "
        f"test_auc={test_stats['auc']:.4f} "
        f"test_acc={test_stats['acc']:.4f} "
        f"test_f1={test_stats['f1']:.4f} "
        f"test_sens={test_stats['sens']:.4f} "
        f"test_spec={test_stats['spec']:.4f}"
    )

    return {
        "phase": phase_name,
        "ckpt_path": ckpt_path,
        "test_stats": test_stats,
        "pred_path": pred_path,
        "metrics_path": metrics_path,
    }

@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    aggregator: nn.Module,
    plugin: Optional[nn.Module],
    summary_builder: PatchRoleSummaryFromSharedProto,
    loader: DataLoader,
    transform,
    device: str,
    args,
    desc: str = "Eval",
):
    encoder.eval()
    aggregator.eval()
    if plugin is not None:
        plugin.eval()

    losses = []
    logits_all = []
    labels_all = []
    slide_ids_all = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for step, batch in enumerate(pbar):
        labels = batch["labels"].to(device).float()
        batch_logits = []
        batch_loss = 0.0

        for i in range(len(batch["slide_id"])):
            patch_feat_raw = encode_one_slide_bag(
                encoder=encoder,
                svs_path=batch["svs_path"][i],
                h5_path=batch["h5_path"][i],
                transform=transform,
                device=device,
                max_patches_per_slide=args.max_patches_per_slide,
                patch_batch_size=args.patch_batch_size,
                random_sample_patches=False,
                seed=args.seed + step * 1000 + i,
                use_last_moe_output=args.use_last_moe_output,
            )

            patch_feat = patch_feat_raw.unsqueeze(0)

            if args.use_plugin and plugin is not None:
                role_summary = build_role_summary_from_feats(
                    patch_feat=patch_feat_raw,
                    summary_builder=summary_builder,
                    use_role_logits=args.use_role_logits,
                )
                plugin_out = plugin(
                    patch_feat=patch_feat,
                    patch_role_probs=role_summary["patch_role_probs"],
                    patch_role_gaps=role_summary["patch_role_gaps"],
                    patch_role_logits=role_summary["patch_role_logits"],
                    patch_top1_gap=role_summary["patch_top1_gap"],
                    return_aux=True,
                )
                if isinstance(plugin_out, tuple):
                    patch_feat, _ = plugin_out
                else:
                    patch_feat = plugin_out

            logits_i = aggregator(patch_feat)
            logits_i = normalize_binary_logits(logits_i).view(1)

            loss_i = F.binary_cross_entropy_with_logits(logits_i, labels[i:i+1])
            batch_logits.append(logits_i.detach().cpu())
            batch_loss += loss_i.item()
            slide_ids_all.append(batch["slide_id"][i])

        batch_loss = batch_loss / len(batch["slide_id"])
        losses.append(batch_loss)

        logits_all.append(torch.cat(batch_logits, dim=0))
        labels_all.append(labels.detach().cpu())

    logits_all = torch.cat(logits_all, dim=0) if logits_all else torch.empty(0)
    labels_all = torch.cat(labels_all, dim=0).long() if labels_all else torch.empty(0, dtype=torch.long)

    metric = compute_metrics_from_logits(logits_all, labels_all) if len(labels_all) > 0 else {
        "acc": 0.0, "f1": 0.0, "auc": float("nan"), "sens": 0.0, "spec": 0.0
    }

    probs = torch.sigmoid(normalize_binary_logits(logits_all)).numpy() if len(logits_all) > 0 else np.array([])
    pred_df = pd.DataFrame({
        "slide_id": slide_ids_all,
        "y_true": labels_all.numpy() if len(labels_all) > 0 else [],
        "y_prob": probs,
    })

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        **metric,
    }, pred_df


# =========================================================
# phase runner
# =========================================================
def run_phase(
    phase_name: str,
    epochs: int,
    encoder: nn.Module,
    aggregator: nn.Module,
    plugin: Optional[nn.Module],
    summary_builder: PatchRoleSummaryFromSharedProto,
    proj_l12: nn.Module,
    role_protos: torch.Tensor,
    role_to_idx: Dict[str, int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_transform,
    eval_transform,
    device: str,
    args,
    optimizer: torch.optim.Optimizer,
    history: List[dict],
    out_dir: str,
    cfg: dict,
    enable_role_proto_loss: bool = False,
    initial_best_metric: float = -1e18,
):
    best_metric = initial_best_metric

    best_ckpt = os.path.join(out_dir, f"best_transfer_{phase_name}.pt")
    last_ckpt = os.path.join(out_dir, f"last_transfer_{phase_name}.pt")
    best_full_ckpt = os.path.join(out_dir, f"best_full_transfer_{phase_name}.pth")
    last_full_ckpt = os.path.join(out_dir, f"last_full_transfer_{phase_name}.pth")

    for epoch in range(1, epochs + 1):
        train_row = train_one_epoch(
            encoder=encoder,
            aggregator=aggregator,
            plugin=plugin,
            summary_builder=summary_builder,
            proj_l12=proj_l12,
            role_protos=role_protos,
            role_to_idx=role_to_idx,
            loader=train_loader,
            transform=train_transform,
            device=device,
            args=args,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=epochs,
            desc_prefix=f"Train-{phase_name}",
            enable_role_proto_loss=enable_role_proto_loss,
        )

        val_stats, val_pred_df = evaluate(
            encoder=encoder,
            aggregator=aggregator,
            plugin=plugin,
            summary_builder=summary_builder,
            loader=val_loader,
            transform=eval_transform,
            device=device,
            args=args,
            desc=f"Val-{phase_name}",
        )

        row = {
            "phase": phase_name,
            "epoch": epoch,
            **train_row,
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_f1": val_stats["f1"],
            "val_auc": val_stats["auc"],
            "val_sens": val_stats["sens"],
            "val_spec": val_stats["spec"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "train_history.csv"), index=False)

        print(
            f"[{phase_name} Epoch {epoch:03d}] "
            f"train_loss={row['train_loss']:.4f} "
            f"train_auc={row['train_auc']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_auc={row['val_auc']:.4f} "
            f"val_acc={row['val_acc']:.4f} "
            f"val_f1={row['val_f1']:.4f}"
        )

        # always save last
        save_checkpoint(
            path=last_ckpt,
            phase=phase_name,
            epoch=epoch,
            encoder=encoder,
            aggregator=aggregator,
            plugin=plugin,
            cfg=cfg,
            val_auc=row["val_auc"],
        )
        save_full_bundle(
            path=last_full_ckpt,
            phase=phase_name,
            epoch=epoch,
            encoder=encoder,
            aggregator=aggregator,
            plugin=plugin,
            proj_l12=proj_l12,
            cfg=cfg,
            val_auc=row["val_auc"],
        )

        cur_metric = row["val_auc"]
        if (math.isnan(cur_metric) and best_metric == -1e18) or (not math.isnan(cur_metric) and cur_metric > best_metric):
            best_metric = cur_metric
            save_checkpoint(
                path=best_ckpt,
                phase=phase_name,
                epoch=epoch,
                encoder=encoder,
                aggregator=aggregator,
                plugin=plugin,
                cfg=cfg,
                val_auc=row["val_auc"],
            )
            save_full_bundle(
                path=best_full_ckpt,
                phase=phase_name,
                epoch=epoch,
                encoder=encoder,
                aggregator=aggregator,
                plugin=plugin,
                proj_l12=proj_l12,
                cfg=cfg,
                val_auc=row["val_auc"],
            )
            val_pred_df.to_csv(os.path.join(out_dir, f"best_val_predictions_{phase_name}.csv"), index=False)
            print(f"[{phase_name} Best] epoch={epoch}, val_auc={best_metric:.4f}")

    return {
        "best_metric": best_metric,
        "best_ckpt": best_ckpt,
        "last_ckpt": last_ckpt,
        "best_full_ckpt": best_full_ckpt,
        "last_full_ckpt": last_full_ckpt,
    }


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Parotid transfer finetuning with MIL warmup + last-MoE adaptation")
    parser.add_argument("--config", type=str, required=True)
    args_cmd = parser.parse_args()

    with open(args_cmd.config, "r") as f:
        cfg = yaml.safe_load(f)

    class Args:
        pass

    args = Args()
    args.config = args_cmd.config
    for k, v in cfg.items():
        setattr(args, k, v)

    # defaults
    if not hasattr(args, "train_batch_size"):
        args.train_batch_size = getattr(args, "batch_size", 1)
    if not hasattr(args, "val_batch_size"):
        args.val_batch_size = getattr(args, "batch_size", 1)
    if not hasattr(args, "test_batch_size"):
        args.test_batch_size = getattr(args, "batch_size", 1)
    if not hasattr(args, "use_last_moe_output"):
        args.use_last_moe_output = True
    if not hasattr(args, "phase0_epochs"):
        args.phase0_epochs = 3
    if not hasattr(args, "phase1_epochs"):
        args.phase1_epochs = 5
    if not hasattr(args, "phase2_epochs"):
        args.phase2_epochs = 10
    if not hasattr(args, "run_phase2_only_mil"):
        args.run_phase2_only_mil = True
    if not hasattr(args, "phase2_train_plugin"):
        args.phase2_train_plugin = False
    if not hasattr(args, "phase0_train_plugin"):
        args.phase0_train_plugin = False
    if not hasattr(args, "phase0_lr"):
        args.phase0_lr = getattr(args, "lr", 1e-4)
    if not hasattr(args, "phase2_lr"):
        args.phase2_lr = getattr(args, "lr", 1e-4)
    if not hasattr(args, "mil_init_ckpt"):
        args.mil_init_ckpt = None
    if not hasattr(args, "enable_phase1_role_proto_loss"):
        args.enable_phase1_role_proto_loss = False
    if not hasattr(args, "role_proto_loss_weight"):
        args.role_proto_loss_weight = 0.0
    if not hasattr(args, "role_proto_margin"):
        args.role_proto_margin = 0.15
    if not hasattr(args, "role_proto_conf_thresh"):
        args.role_proto_conf_thresh = 0.55
    if not hasattr(args, "role_proto_top1_gap_thresh"):
        args.role_proto_top1_gap_thresh = 0.05
    if not hasattr(args, "role_proto_min_kept"):
        args.role_proto_min_kept = 1
    if not hasattr(args, "proto_tumor_name"):
        args.proto_tumor_name = "tumor"
    if not hasattr(args, "proto_negative_role_names"):
        args.proto_negative_role_names = ["stroma", "normal_epithelium"]

    ensure_dir(args.out_dir)
    log_path = os.path.join(args.out_dir, "train.log")

    # overwrite log instead of append
    with open(log_path, "w", encoding="utf-8") as log_f:
        tee = Tee(sys.stdout, log_f)
        with redirect_stdout(tee), redirect_stderr(tee):
            set_seed(args.seed)

            print("=" * 80)
            print(json.dumps(cfg, indent=2, ensure_ascii=False))
            print("=" * 80)

            device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

            train_transform = build_transform(args.img_size, is_train=True)
            eval_transform = build_transform(args.img_size, is_train=False)

            train_set = ParotidSlideBagDataset(
                args.split_csv,
                "train",
                max_slides=getattr(args, "max_train_slides", None),
                seed=args.seed,
                balance_by_label=getattr(args, "balance_train_slides_by_label", True),
            )
            val_set = ParotidSlideBagDataset(
                args.split_csv,
                "val",
                max_slides=getattr(args, "max_val_slides", None),
                seed=args.seed + 1,
                balance_by_label=getattr(args, "balance_val_slides_by_label", False),
            )
            test_set = ParotidSlideBagDataset(
                args.split_csv,
                "test",
                max_slides=getattr(args, "max_test_slides", None),
                seed=args.seed + 2,
                balance_by_label=getattr(args, "balance_test_slides_by_label", False),
            )

            train_loader = DataLoader(
                train_set,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=slide_bag_collate_fn,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=args.val_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=slide_bag_collate_fn,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=args.test_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=slide_bag_collate_fn,
            )

            encoder, proj_l12 = build_encoder_and_proj_from_stage2(
                base_encoder_cfg=cfg["base_encoder"],
                moe_encoder_cfg=cfg["moe_encoder"],
                stage2_full_ckpt=args.stage2_full_ckpt,
                device=device,
            )

            shared_role_proto = SharedRolePrototype.from_files(
                role_proto_dir=args.role_proto_dir,
                normalize=True,
                learnable=False,
                device=device,
            )

            summary_builder = PatchRoleSummaryFromSharedProto(
                shared_role_proto=shared_role_proto,
                tau=args.role_tau,
                use_softmax=True,
            ).to(device)

            role_protos, role_names, role_to_idx = load_role_proto_bundle(
                role_proto_dir=args.role_proto_dir,
                device=device,
            )
            print(f"[RoleProto] role_names = {role_names}")

            debug_print_moe_param_names(encoder)

            feat_dim = infer_feat_dim(
                encoder=encoder,
                device=device,
                img_size=args.img_size,
                use_last_moe_output=args.use_last_moe_output,
            )
            print(f"[Info] inferred feat_dim = {feat_dim}")

            aggregator = build_aggregator(
                mil_model=args.mil_model,
                in_dim=feat_dim,
                device=device,
            )
            load_aggregator_init_if_available(
                aggregator=aggregator,
                init_path=args.mil_init_ckpt,
                device=device,
            )

            plugin = None
            if getattr(args, "use_plugin", False):
                if PluginClass is None:
                    raise ImportError("PluginClass import failed, but use_plugin=true")
                plugin = PluginClass(
                    feat_dim=feat_dim,
                    num_roles=shared_role_proto.num_roles,
                    hidden_dim=args.plugin_hidden_dim,
                    dropout=args.plugin_dropout,
                    use_role_logits=args.use_role_logits,
                    use_top1_gap=args.use_top1_gap,
                    use_beta=args.use_beta,
                    init_scale=args.plugin_init_scale,
                ).to(device)

            history = []

            # =====================================================
            # Phase0
            # =====================================================
            print("\n" + "=" * 80)
            print("[Phase0] Freeze encoder, warm up MIL head")
            print("=" * 80)

            freeze_module(encoder)
            encoder.eval()

            unfreeze_module(aggregator)
            if plugin is not None:
                if bool(getattr(args, "phase0_train_plugin", False)):
                    unfreeze_module(plugin)
                else:
                    freeze_module(plugin)

            print_trainable_params(encoder, "Encoder-Phase0")
            print_trainable_params(aggregator, "Aggregator-Phase0")
            if plugin is not None:
                print_trainable_params(plugin, "Plugin-Phase0")

            phase0_params = list(aggregator.parameters())
            if plugin is not None and bool(getattr(args, "phase0_train_plugin", False)):
                phase0_params += list(plugin.parameters())

            optimizer_phase0 = torch.optim.AdamW(
                phase0_params,
                lr=float(getattr(args, "phase0_lr", args.lr)),
                weight_decay=args.weight_decay,
            )

            phase0_result = run_phase(
                phase_name="phase0",
                epochs=int(args.phase0_epochs),
                encoder=encoder,
                aggregator=aggregator,
                plugin=plugin,
                summary_builder=summary_builder,
                proj_l12=proj_l12,
                role_protos=role_protos,
                role_to_idx=role_to_idx,
                train_loader=train_loader,
                val_loader=val_loader,
                train_transform=train_transform,
                eval_transform=eval_transform,
                device=device,
                args=args,
                optimizer=optimizer_phase0,
                history=history,
                out_dir=args.out_dir,
                cfg=cfg,
                enable_role_proto_loss=False,
                initial_best_metric=-1e18,
            )

            best_metric_global = phase0_result["best_metric"]
            best_ckpt_global = phase0_result["best_ckpt"]

            # =====================================================
            # Phase1
            # =====================================================
            print("\n" + "=" * 80)
            print("[Phase1] Load best Phase0, unfreeze last MoE + continue MIL training")
            print("=" * 80)

            ckpt = torch.load(phase0_result["best_ckpt"], map_location=device)
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            aggregator.load_state_dict(ckpt["aggregator_state_dict"])
            if plugin is not None and ckpt["plugin_state_dict"] is not None:
                plugin.load_state_dict(ckpt["plugin_state_dict"])

            freeze_module(encoder)
            unfreeze_last_moe_params_by_name(
                encoder=encoder,
                moe_layers=cfg["moe_encoder"]["moe_layers"],
                num_layers=12,
                train_gate=args.train_last_gate,
                train_experts=args.train_last_experts,
            )
            unfreeze_module(aggregator)
            if plugin is not None:
                unfreeze_module(plugin) if args.use_plugin else freeze_module(plugin)

            print_trainable_params(encoder, "Encoder-Phase1")
            print_trainable_params(aggregator, "Aggregator-Phase1")
            if plugin is not None:
                print_trainable_params(plugin, "Plugin-Phase1")

            phase1_params = [p for p in encoder.parameters() if p.requires_grad]
            phase1_params += list(aggregator.parameters())
            if plugin is not None and args.use_plugin:
                phase1_params += list(plugin.parameters())

            optimizer_phase1 = torch.optim.AdamW(
                phase1_params,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            phase1_result = run_phase(
                phase_name="phase1",
                epochs=int(args.phase1_epochs),
                encoder=encoder,
                aggregator=aggregator,
                plugin=plugin,
                summary_builder=summary_builder,
                proj_l12=proj_l12,
                role_protos=role_protos,
                role_to_idx=role_to_idx,
                train_loader=train_loader,
                val_loader=val_loader,
                train_transform=train_transform,
                eval_transform=eval_transform,
                device=device,
                args=args,
                optimizer=optimizer_phase1,
                history=history,
                out_dir=args.out_dir,
                cfg=cfg,
                enable_role_proto_loss=bool(getattr(args, "enable_phase1_role_proto_loss", False)),
                initial_best_metric=phase0_result["best_metric"],
            )

            # global best can stay phase0 if phase1没超
            if (not math.isnan(phase1_result["best_metric"])) and (
                math.isnan(best_metric_global) or phase1_result["best_metric"] > best_metric_global
            ):
                best_metric_global = phase1_result["best_metric"]
                best_ckpt_global = phase1_result["best_ckpt"]

            # =====================================================
            # Phase2
            # =====================================================
            if bool(getattr(args, "run_phase2_only_mil", False)) and int(getattr(args, "phase2_epochs", 0)) > 0:
                print("\n" + "=" * 80)
                print("[Phase2] Freeze encoder, train MIL only")
                print("=" * 80)

                # use phase1 last, not necessarily best
                ckpt = torch.load(phase1_result["last_ckpt"], map_location=device)
                encoder.load_state_dict(ckpt["encoder_state_dict"])
                aggregator.load_state_dict(ckpt["aggregator_state_dict"])
                if plugin is not None and ckpt["plugin_state_dict"] is not None:
                    plugin.load_state_dict(ckpt["plugin_state_dict"])

                freeze_module(encoder)
                encoder.eval()

                if plugin is not None and bool(getattr(args, "phase2_train_plugin", False)):
                    unfreeze_module(plugin)
                elif plugin is not None:
                    freeze_module(plugin)

                unfreeze_module(aggregator)

                print_trainable_params(encoder, "Encoder-Phase2")
                print_trainable_params(aggregator, "Aggregator-Phase2")
                if plugin is not None:
                    print_trainable_params(plugin, "Plugin-Phase2")

                phase2_params = list(aggregator.parameters())
                if plugin is not None and bool(getattr(args, "phase2_train_plugin", False)):
                    phase2_params += list(plugin.parameters())

                optimizer_phase2 = torch.optim.AdamW(
                    phase2_params,
                    lr=float(getattr(args, "phase2_lr", args.lr)),
                    weight_decay=args.weight_decay,
                )

                phase2_result = run_phase(
                    phase_name="phase2",
                    epochs=int(args.phase2_epochs),
                    encoder=encoder,
                    aggregator=aggregator,
                    plugin=plugin,
                    summary_builder=summary_builder,
                    proj_l12=proj_l12,
                    role_protos=role_protos,
                    role_to_idx=role_to_idx,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    train_transform=train_transform,
                    eval_transform=eval_transform,
                    device=device,
                    args=args,
                    optimizer=optimizer_phase2,
                    history=history,
                    out_dir=args.out_dir,
                    cfg=cfg,
                    enable_role_proto_loss=False,
                    initial_best_metric=max(phase0_result["best_metric"], phase1_result["best_metric"]),
                )

                if (not math.isnan(phase2_result["best_metric"])) and (
                    math.isnan(best_metric_global) or phase2_result["best_metric"] > best_metric_global
                ):
                    best_metric_global = phase2_result["best_metric"]
                    best_ckpt_global = phase2_result["best_ckpt"]

            # =====================================================
            # final test
            # =====================================================
                        # =====================================================
            # final test: evaluate phase0 / phase1 / phase2 separately
            # =====================================================
            print("\n" + "=" * 80)
            print("[Final Test] Evaluate all phase checkpoints separately")
            print("=" * 80)

            best_ckpt_phase0 = os.path.join(args.out_dir, "best_transfer_phase0.pt")
            best_ckpt_phase1 = os.path.join(args.out_dir, "best_transfer_phase1.pt")
            best_ckpt_phase2 = os.path.join(args.out_dir, "best_transfer_phase2.pt")

            final_results = {}

            res0 = evaluate_checkpoint_and_save(
                ckpt_path=best_ckpt_phase0,
                phase_name="phase0",
                encoder=encoder,
                aggregator=aggregator,
                plugin=plugin,
                summary_builder=summary_builder,
                test_loader=test_loader,
                eval_transform=eval_transform,
                device=device,
                args=args,
                out_dir=args.out_dir,
            )
            if res0 is not None:
                final_results["phase0"] = res0

            res1 = evaluate_checkpoint_and_save(
                ckpt_path=best_ckpt_phase1,
                phase_name="phase1",
                encoder=encoder,
                aggregator=aggregator,
                plugin=plugin,
                summary_builder=summary_builder,
                test_loader=test_loader,
                eval_transform=eval_transform,
                device=device,
                args=args,
                out_dir=args.out_dir,
            )
            if res1 is not None:
                final_results["phase1"] = res1

            res2 = evaluate_checkpoint_and_save(
                ckpt_path=best_ckpt_phase2,
                phase_name="phase2",
                encoder=encoder,
                aggregator=aggregator,
                plugin=plugin,
                summary_builder=summary_builder,
                test_loader=test_loader,
                eval_transform=eval_transform,
                device=device,
                args=args,
                out_dir=args.out_dir,
            )
            if res2 is not None:
                final_results["phase2"] = res2

            # summary compare
            compare_dict = {}
            for k, v in final_results.items():
                compare_dict[k] = v["test_stats"]

            with open(os.path.join(args.out_dir, "final_comparison.json"), "w") as f:
                json.dump(compare_dict, f, indent=2)

            print("[Done] final comparison saved to:",
                  os.path.join(args.out_dir, "final_comparison.json"))
            print(f"[Done] log saved to: {log_path}")



if __name__ == "__main__":
    main()