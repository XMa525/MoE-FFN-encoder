#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torchmil.models import ABMIL, DSMIL


# =========================================================
# Utils
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


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 0:
        logits = logits.view(1)

    if logits.ndim == 1:
        return torch.sigmoid(logits)

    if logits.ndim == 2 and logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])

    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def normalize_attention(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = np.clip(a, eps, None)
    s = a.sum()
    if s <= 0:
        raise ValueError("Attention sum <= 0")
    return (a / s).astype(np.float32)


def attention_entropy(a: np.ndarray, eps: float = 1e-12) -> float:
    a = np.clip(a, eps, 1.0)
    return float(-(a * np.log(a)).sum())


def topk_mass(a: np.ndarray, frac: float) -> float:
    k = max(1, int(math.ceil(len(a) * frac)))
    idx = np.argsort(a)[-k:]
    return float(a[idx].sum())


def gini_coefficient(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, eps, None)
    x = np.sort(x)
    n = len(x)
    if n == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * np.sum(x)))


def effective_num_instances(att: np.ndarray, eps: float = 1e-12) -> float:
    att = np.asarray(att, dtype=np.float64)
    att = np.clip(att, eps, None)
    att = att / att.sum()
    return float(1.0 / np.sum(att ** 2))


def build_grid_attention_map(
    coords: np.ndarray,
    att: np.ndarray,
    cell_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = coords[:, 0]
    y = coords[:, 1]

    gx = (x // cell_size).astype(np.int64)
    gy = (y // cell_size).astype(np.int64)

    gx_unique = np.unique(gx)
    gy_unique = np.unique(gy)
    gx_to_id = {v: i for i, v in enumerate(gx_unique)}
    gy_to_id = {v: i for i, v in enumerate(gy_unique)}

    grid = np.zeros((len(gy_unique), len(gx_unique)), dtype=np.float32)

    for i in range(len(coords)):
        yy = gy_to_id[gy[i]]
        xx = gx_to_id[gx[i]]
        grid[yy, xx] += float(att[i])

    return grid, gx_unique, gy_unique


def grid_iou_topmass(
    coords_a: np.ndarray,
    att_a: np.ndarray,
    coords_b: np.ndarray,
    att_b: np.ndarray,
    cell_size: int = 512,
    frac: float = 0.1,
) -> float:
    def _top_cells(coords: np.ndarray, att: np.ndarray):
        x = coords[:, 0]
        y = coords[:, 1]
        gx = (x // cell_size).astype(np.int64)
        gy = (y // cell_size).astype(np.int64)

        cell_map: Dict[Tuple[int, int], float] = {}
        for i in range(len(coords)):
            k = (int(gx[i]), int(gy[i]))
            cell_map[k] = cell_map.get(k, 0.0) + float(att[i])

        items = sorted(cell_map.items(), key=lambda z: z[1], reverse=True)
        k = max(1, int(math.ceil(len(items) * frac)))
        return set([it[0] for it in items[:k]])

    sa = _top_cells(coords_a, att_a)
    sb = _top_cells(coords_b, att_b)
    return float(len(sa & sb) / max(1, len(sa | sb)))


# =========================================================
# WSI helpers
# =========================================================
def find_wsi_path(raw_dir: str, slide_id: str) -> str:
    raw_dir = Path(raw_dir)
    exts = [".tif", ".tiff", ".svs", ".ndpi", ".mrxs"]

    exact_matches = []
    for ext in exts:
        exact_matches.extend(raw_dir.rglob(f"{slide_id}{ext}"))
    if len(exact_matches) == 1:
        return str(exact_matches[0])
    if len(exact_matches) > 1:
        raise RuntimeError(
            f"Found multiple exact WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in exact_matches[:10])
        )

    fuzzy_matches = []
    for ext in exts:
        fuzzy_matches.extend(raw_dir.rglob(f"{slide_id}*{ext}"))
    if len(fuzzy_matches) == 1:
        return str(fuzzy_matches[0])
    if len(fuzzy_matches) > 1:
        exact_name = [p for p in fuzzy_matches if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])
        raise RuntimeError(
            f"Found multiple WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in fuzzy_matches[:10])
        )

    raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {raw_dir}")


def read_patch_from_wsi(
    slide: openslide.OpenSlide,
    coord_xy: Tuple[int, int],
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    return slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")


def make_montage(
    pil_images: List[Image.Image],
    tile_size: int = 224,
    n_cols: int = 4,
) -> Image.Image:
    n = len(pil_images)
    n_rows = math.ceil(n / n_cols)
    canvas = Image.new("RGB", (n_cols * tile_size, n_rows * tile_size), color=(255, 255, 255))

    for i, img in enumerate(pil_images):
        r = i // n_cols
        c = i % n_cols
        x0 = c * tile_size
        y0 = r * tile_size
        img = ImageOps.fit(img, (tile_size, tile_size), method=Image.BICUBIC)
        canvas.paste(img, (x0, y0))

    return canvas


# =========================================================
# Feature loading
# =========================================================
def load_feature_pt(path: str | Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)

    feats = obj["features"]
    coords = obj.get("coords", None)

    if not torch.is_tensor(feats):
        raise ValueError(f"'features' should be tensor in {path}")
    feats = feats.float().cpu()

    if coords is None:
        raise KeyError(f"'coords' missing in {path}")
    if not torch.is_tensor(coords):
        coords = torch.tensor(coords, dtype=torch.long)
    coords = coords.long().cpu()

    return {
        "features": feats,
        "coords": coords,
        "slide_id": str(obj.get("slide_id", Path(path).stem)),
        "label": int(obj.get("label", -1)),
        "num_instances": int(obj.get("num_instances", feats.shape[0])),
    }


# =========================================================
# Model defs compatible with user's training script
# =========================================================
class ABMILWrapper(nn.Module):
    def __init__(self, in_dim: int, device: str, att_dim: int = 128, gated: bool = False):
        super().__init__()
        self.model = ABMIL(
            in_shape=(in_dim,),
            att_dim=att_dim,
            att_act="tanh",
            gated=gated,
        )
        self.device = device
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
    def __init__(
        self,
        in_dim: int,
        device: str,
        att_dim: int = 128,
        nonlinear_q: bool = False,
        nonlinear_v: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model = DSMIL(
            in_shape=(in_dim,),
            att_dim=att_dim,
            nonlinear_q=nonlinear_q,
            nonlinear_v=nonlinear_v,
            dropout=dropout,
        )
        self.device = device
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


class MeanPoolMIL(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1, device: str = "cuda"):
        super().__init__()
        self.classifier = nn.Linear(in_dim, out_dim)
        self.device = device
        self.to(device)

    def forward(self, bag_feats: torch.Tensor):
        if bag_feats.ndim == 3:
            bag_repr = bag_feats.mean(dim=1)
        elif bag_feats.ndim == 2:
            bag_repr = bag_feats.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported bag_feats shape: {tuple(bag_feats.shape)}")

        return self.classifier(bag_repr)


class MILWithRoleAux(nn.Module):
    def __init__(self, base_model: nn.Module, aux_dim: int = 3, hidden_dim: int = 16, device: str = "cuda"):
        super().__init__()
        self.base_model = base_model
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.device = device
        self.to(device)

    def forward(self, bag_feats: torch.Tensor, role_aux: Optional[torch.Tensor] = None):
        logits = self.base_model(bag_feats)

        if role_aux is None:
            return logits

        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        elif logits.ndim == 2 and logits.shape[1] == 2:
            raise ValueError("MILWithRoleAux expects one-logit binary output, not [B,2].")

        aux_bias = self.aux_mlp(role_aux)
        return logits + aux_bias


# =========================================================
# Checkpoint loading
# =========================================================
def load_checkpoint(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(obj)}")
    return obj


def build_model_from_ckpt(
    ckpt: Dict[str, Any],
    override_feat_dim: Optional[int],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    args = ckpt.get("args", {}) or {}
    feat_dim = int(override_feat_dim if override_feat_dim is not None else ckpt.get("feat_dim", args.get("feat_dim", 0)))
    if feat_dim <= 0:
        raise ValueError("Could not infer feat_dim from checkpoint; pass --feat-dim")

    mil_model = args.get("mil_model", "abmil")
    att_dim = int(args.get("att_dim", 128))
    abmil_gated = bool(args.get("abmil_gated", False))
    use_role_aux = bool(args.get("use_role_aux", False))
    role_aux_hidden_dim = int(args.get("role_aux_hidden_dim", 16))

    if mil_model == "abmil":
        model: nn.Module = ABMILWrapper(
            in_dim=feat_dim,
            device=str(device),
            att_dim=att_dim,
            gated=abmil_gated,
        )
    elif mil_model == "dsmil":
        model = DSMILWrapper(
            in_dim=feat_dim,
            device=str(device),
            att_dim=att_dim,
            nonlinear_q=bool(args.get("dsmil_nonlinear_q", False)),
            nonlinear_v=bool(args.get("dsmil_nonlinear_v", False)),
            dropout=float(args.get("dsmil_dropout", 0.0)),
        )
    elif mil_model == "meanpool":
        model = MeanPoolMIL(
            in_dim=feat_dim,
            out_dim=1,
            device=str(device),
        )
    else:
        raise ValueError(f"Unsupported mil_model from ckpt: {mil_model}")

    if use_role_aux:
        model = MILWithRoleAux(
            base_model=model,
            aux_dim=3,
            hidden_dim=role_aux_hidden_dim,
            device=str(device),
        )

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    info = {
        "feat_dim": feat_dim,
        "mil_model": mil_model,
        "att_dim": att_dim,
        "abmil_gated": abmil_gated,
        "use_role_aux": use_role_aux,
        "role_aux_hidden_dim": role_aux_hidden_dim,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "raw_args": args,
    }
    return model, info


# =========================================================
# Strict attention extraction
# =========================================================
def _unwrap_base_abmil(model: nn.Module) -> nn.Module:
    if isinstance(model, MILWithRoleAux):
        model = model.base_model
    if isinstance(model, ABMILWrapper):
        return model.model
    return model


@torch.no_grad()
def extract_abmil_attention_strict(
    model: nn.Module,
    bag_feats: torch.Tensor,
    role_aux: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    For current torchmil ABMIL:
    attention logits come from base.pool.fc2 output with shape [1, N, 1].
    """
    if bag_feats.ndim != 3 or bag_feats.shape[0] != 1:
        raise ValueError(f"bag_feats should be [1, N, D], got {tuple(bag_feats.shape)}")

    n_instances = int(bag_feats.shape[1])

    base = _unwrap_base_abmil(model)
    if not isinstance(base, ABMIL):
        raise TypeError("This script is intended for ABMIL checkpoints. Got non-ABMIL base model.")

    if not hasattr(base, "pool"):
        raise RuntimeError("ABMIL base model has no attribute 'pool'")
    if not hasattr(base.pool, "fc2"):
        raise RuntimeError("ABMIL pool has no attribute 'fc2'")

    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inputs, output):
        captured["fc2_out"] = output.detach().cpu()

    handle = base.pool.fc2.register_forward_hook(hook_fn)
    try:
        _ = base(bag_feats)
    finally:
        handle.remove()

    if "fc2_out" not in captured:
        raise RuntimeError("Failed to capture attention logits from base.pool.fc2")

    fc2_out = captured["fc2_out"]

    if not torch.is_tensor(fc2_out):
        raise RuntimeError(f"Captured fc2_out is not tensor: {type(fc2_out)}")

    if fc2_out.ndim != 3 or fc2_out.shape[0] != 1 or fc2_out.shape[1] != n_instances:
        raise RuntimeError(
            f"Unexpected fc2_out shape: got {tuple(fc2_out.shape)}, expected [1, {n_instances}, 1]"
        )

    att_logits = fc2_out.squeeze(0).squeeze(-1).float()
    if att_logits.ndim != 1 or att_logits.shape[0] != n_instances:
        raise RuntimeError(f"Unexpected att_logits shape after squeeze: {tuple(att_logits.shape)}")

    att_weights = torch.softmax(att_logits, dim=0)

    if isinstance(model, MILWithRoleAux):
        logits = model(bag_feats, role_aux=role_aux)
    else:
        logits = model(bag_feats)

    prob = logits_to_prob(logits).reshape(-1)[0].detach().cpu()

    att_weights_np = normalize_attention(att_weights.numpy())

    return {
        "att_logits": att_logits.numpy().astype(np.float32),
        "att_weights": att_weights_np,
        "bag_prob": float(prob.item()),
        "bag_logits": logits.detach().cpu(),
        "extract_meta": {
            "source": "hook",
            "key": "base.pool.fc2",
        },
    }


# =========================================================
# Plot helpers
# =========================================================
def plot_attention_map(ax, coords: np.ndarray, att: np.ndarray, title: str):
    x = coords[:, 0]
    y = coords[:, 1]
    sc = ax.scatter(x, -y, c=att, s=10, cmap="inferno")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def plot_grid_heatmap(ax, grid: np.ndarray, title: str):
    im = ax.imshow(grid, cmap="inferno")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Full-bag ABMIL attention comparison")

    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--frozen_feature_dir", type=str, required=True)
    parser.add_argument("--adapted_feature_dir", type=str, required=True)
    parser.add_argument("--frozen_ckpt", type=str, required=True)
    parser.add_argument("--adapted_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--slide_ids", nargs="*", default=[])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_slides", type=int, default=None)

    parser.add_argument("--feat_dim", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--raw_dir", type=str, default="")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--gallery_topk", type=int, default=16)
    parser.add_argument("--grid_cell_size", type=int, default=512)

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(Path(args.out_dir) / "per_slide")
    ensure_dir(Path(args.out_dir) / "galleries")
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    slides_df = pd.read_csv(args.slides_csv)
    if "slide_id" not in slides_df.columns and "image_id" in slides_df.columns:
        slides_df["slide_id"] = slides_df["image_id"]
    if "label" not in slides_df.columns:
        if "slide_binary_label" in slides_df.columns:
            slides_df["label"] = slides_df["slide_binary_label"]
        else:
            raise ValueError("slides_csv missing label / slide_binary_label")
    if "split" not in slides_df.columns:
        raise ValueError("slides_csv missing split column")

    if args.slide_ids:
        use_df = slides_df[slides_df["slide_id"].isin(args.slide_ids)].copy()
    else:
        use_df = slides_df[slides_df["split"] == args.split].copy()

    use_df = use_df.reset_index(drop=True)
    if args.max_slides is not None:
        use_df = use_df.iloc[:args.max_slides].copy()

    frozen_ckpt = load_checkpoint(args.frozen_ckpt, device)
    adapted_ckpt = load_checkpoint(args.adapted_ckpt, device)

    frozen_model, frozen_info = build_model_from_ckpt(frozen_ckpt, args.feat_dim, device)
    adapted_model, adapted_info = build_model_from_ckpt(adapted_ckpt, args.feat_dim, device)

    with open(Path(args.out_dir) / "load_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "frozen": frozen_info,
            "adapted": adapted_info,
        }, f, indent=2, ensure_ascii=False)

    if frozen_info["mil_model"] != "abmil" or adapted_info["mil_model"] != "abmil":
        raise ValueError(
            f"This script is for ABMIL attention comparison. "
            f"Got frozen={frozen_info['mil_model']}, adapted={adapted_info['mil_model']}"
        )

    summary_rows = []

    for _, row in use_df.iterrows():
        slide_id = str(row["slide_id"])
        label = int(row["label"])

        frozen_pt = Path(args.frozen_feature_dir) / f"{slide_id}.pt"
        adapted_pt = Path(args.adapted_feature_dir) / f"{slide_id}.pt"
        if not frozen_pt.exists():
            print(f"[WARN] missing frozen feature file: {frozen_pt}")
            continue
        if not adapted_pt.exists():
            print(f"[WARN] missing adapted feature file: {adapted_pt}")
            continue

        print(f"[Process] {slide_id}")

        frozen_obj = load_feature_pt(frozen_pt)
        adapted_obj = load_feature_pt(adapted_pt)

        frozen_coords = frozen_obj["coords"].numpy()
        adapted_coords = adapted_obj["coords"].numpy()

        frozen_feats = frozen_obj["features"]
        adapted_feats = adapted_obj["features"]

        frozen_role_aux = None
        adapted_role_aux = None
        if frozen_info["use_role_aux"]:
            raise RuntimeError("Frozen ckpt expects role_aux, but current script does not load role_probs from feature pt.")
        if adapted_info["use_role_aux"]:
            raise RuntimeError("Adapted ckpt expects role_aux, but current script does not load role_probs from feature pt.")

        frozen_feats_t = frozen_feats.to(device).unsqueeze(0)
        adapted_feats_t = adapted_feats.to(device).unsqueeze(0)

        frozen_out = extract_abmil_attention_strict(frozen_model, frozen_feats_t, frozen_role_aux)
        adapted_out = extract_abmil_attention_strict(adapted_model, adapted_feats_t, adapted_role_aux)

        frozen_att = frozen_out["att_weights"]
        adapted_att = adapted_out["att_weights"]

        frozen_entropy = attention_entropy(frozen_att)
        adapted_entropy = attention_entropy(adapted_att)

        frozen_top1 = float(np.max(frozen_att))
        adapted_top1 = float(np.max(adapted_att))

        frozen_top5 = topk_mass(frozen_att, 0.05)
        adapted_top5 = topk_mass(adapted_att, 0.05)

        frozen_gini = gini_coefficient(frozen_att)
        adapted_gini = gini_coefficient(adapted_att)

        frozen_eff_n = effective_num_instances(frozen_att)
        adapted_eff_n = effective_num_instances(adapted_att)

        spatial_top10_iou = grid_iou_topmass(
            frozen_coords, frozen_att,
            adapted_coords, adapted_att,
            cell_size=args.grid_cell_size,
            frac=0.10,
        )

        frozen_top_idx = np.argsort(frozen_att)[::-1][:max(1, int(math.ceil(len(frozen_att) * 0.05)))]
        adapted_top_idx = np.argsort(adapted_att)[::-1][:max(1, int(math.ceil(len(adapted_att) * 0.05)))]

        frozen_top_coords = set((int(frozen_coords[i, 0]), int(frozen_coords[i, 1])) for i in frozen_top_idx)
        adapted_top_coords = set((int(adapted_coords[i, 0]), int(adapted_coords[i, 1])) for i in adapted_top_idx)
        top5_patch_jaccard = float(len(frozen_top_coords & adapted_top_coords) / max(1, len(frozen_top_coords | adapted_top_coords)))

        frozen_df = pd.DataFrame({
            "slide_id": slide_id,
            "label": label,
            "version": "frozen",
            "coord_x": frozen_coords[:, 0],
            "coord_y": frozen_coords[:, 1],
            "attention": frozen_att,
        }).sort_values("attention", ascending=False).reset_index(drop=True)

        adapted_df = pd.DataFrame({
            "slide_id": slide_id,
            "label": label,
            "version": "adapted",
            "coord_x": adapted_coords[:, 0],
            "coord_y": adapted_coords[:, 1],
            "attention": adapted_att,
        }).sort_values("attention", ascending=False).reset_index(drop=True)

        frozen_df.to_csv(Path(args.out_dir) / "per_slide" / f"{slide_id}_frozen_attention_points.csv", index=False)
        adapted_df.to_csv(Path(args.out_dir) / "per_slide" / f"{slide_id}_adapted_attention_points.csv", index=False)

        summary_rows.append({
            "slide_id": slide_id,
            "label": label,
            "n_frozen_patches": int(len(frozen_coords)),
            "n_adapted_patches": int(len(adapted_coords)),
            "frozen_bag_prob_full": float(frozen_out["bag_prob"]),
            "adapted_bag_prob_full": float(adapted_out["bag_prob"]),
            "delta_bag_prob_full": float(adapted_out["bag_prob"] - frozen_out["bag_prob"]),
            "frozen_att_entropy": frozen_entropy,
            "adapted_att_entropy": adapted_entropy,
            "delta_att_entropy": adapted_entropy - frozen_entropy,
            "frozen_top1_mass": frozen_top1,
            "adapted_top1_mass": adapted_top1,
            "delta_top1_mass": adapted_top1 - frozen_top1,
            "frozen_top5pct_mass": frozen_top5,
            "adapted_top5pct_mass": adapted_top5,
            "delta_top5pct_mass": adapted_top5 - frozen_top5,
            "frozen_gini": frozen_gini,
            "adapted_gini": adapted_gini,
            "delta_gini": adapted_gini - frozen_gini,
            "frozen_effective_n": frozen_eff_n,
            "adapted_effective_n": adapted_eff_n,
            "delta_effective_n": adapted_eff_n - frozen_eff_n,
            "top5_patch_jaccard": top5_patch_jaccard,
            "top10_grid_iou": spatial_top10_iou,
            "frozen_extract_source": frozen_out["extract_meta"]["source"],
            "frozen_extract_key": frozen_out["extract_meta"]["key"],
            "adapted_extract_source": adapted_out["extract_meta"]["source"],
            "adapted_extract_key": adapted_out["extract_meta"]["key"],
        })

        frozen_grid, _, _ = build_grid_attention_map(
            frozen_coords, frozen_att, cell_size=args.grid_cell_size
        )
        adapted_grid, _, _ = build_grid_attention_map(
            adapted_coords, adapted_att, cell_size=args.grid_cell_size
        )

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        sc0 = plot_attention_map(
            axes[0, 0], frozen_coords, frozen_att,
            f"Frozen ABMIL attention\np={frozen_out['bag_prob']:.3f} | n={len(frozen_coords)}"
        )
        sc1 = plot_attention_map(
            axes[0, 1], adapted_coords, adapted_att,
            f"Adapted ABMIL attention\np={adapted_out['bag_prob']:.3f} | n={len(adapted_coords)}"
        )

        im0 = plot_grid_heatmap(
            axes[1, 0], frozen_grid,
            f"Frozen attention density grid\nentropy={frozen_entropy:.3f} | effN={frozen_eff_n:.1f}"
        )
        im1 = plot_grid_heatmap(
            axes[1, 1], adapted_grid,
            f"Adapted attention density grid\nentropy={adapted_entropy:.3f} | effN={adapted_eff_n:.1f}"
        )

        fig.suptitle(
            f"{slide_id} | y={label} | "
            f"Δp={adapted_out['bag_prob'] - frozen_out['bag_prob']:.3f} | "
            f"top5_patch_J={top5_patch_jaccard:.3f} | "
            f"grid_top10_IoU={spatial_top10_iou:.3f}",
            fontsize=11
        )

        fig.colorbar(sc0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        fig.colorbar(sc1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        fig.colorbar(im0, ax=axes[1, 0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(Path(args.out_dir) / "per_slide" / f"{slide_id}_attention_compare.png", dpi=220)
        plt.close(fig)

        if args.raw_dir.strip():
            try:
                slide_path = find_wsi_path(args.raw_dir, slide_id)
                slide = openslide.OpenSlide(slide_path)

                try:
                    top_frozen_idx = np.argsort(frozen_att)[::-1][:args.gallery_topk]
                    top_adapted_idx = np.argsort(adapted_att)[::-1][:args.gallery_topk]

                    frozen_imgs = []
                    adapted_imgs = []

                    for idx in top_frozen_idx:
                        frozen_imgs.append(read_patch_from_wsi(
                            slide, (int(frozen_coords[idx, 0]), int(frozen_coords[idx, 1])), args.patch_size, 0
                        ))

                    for idx in top_adapted_idx:
                        adapted_imgs.append(read_patch_from_wsi(
                            slide, (int(adapted_coords[idx, 0]), int(adapted_coords[idx, 1])), args.patch_size, 0
                        ))

                finally:
                    slide.close()

                if frozen_imgs:
                    make_montage(frozen_imgs, tile_size=224, n_cols=4).save(
                        Path(args.out_dir) / "galleries" / f"{slide_id}_top_frozen_attention.png"
                    )
                if adapted_imgs:
                    make_montage(adapted_imgs, tile_size=224, n_cols=4).save(
                        Path(args.out_dir) / "galleries" / f"{slide_id}_top_adapted_attention.png"
                    )

            except Exception as e:
                print(f"[WARN] failed to save patch galleries for {slide_id}: {e}")

    if len(summary_rows) == 0:
        raise RuntimeError("No slides processed successfully.")

    pd.DataFrame(summary_rows).to_csv(Path(args.out_dir) / "attention_summary.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(Path(args.out_dir) / "attention_summary.csv", index=False)

    # =========================
    # Dataset-level summary
    # =========================
    metrics = [
        "frozen_bag_prob_full",
        "adapted_bag_prob_full",
        "delta_bag_prob_full",
        "frozen_att_entropy",
        "adapted_att_entropy",
        "delta_att_entropy",
        "frozen_top1_mass",
        "adapted_top1_mass",
        "delta_top1_mass",
        "frozen_top5pct_mass",
        "adapted_top5pct_mass",
        "delta_top5pct_mass",
        "frozen_gini",
        "adapted_gini",
        "delta_gini",
        "frozen_effective_n",
        "adapted_effective_n",
        "delta_effective_n",
        "top5_patch_jaccard",
        "top10_grid_iou",
    ]

    dataset_rows = []

    # overall
    overall = {"group": "all", "n_slides": len(summary_df)}
    for m in metrics:
        overall[f"{m}_mean"] = float(summary_df[m].mean())
        overall[f"{m}_std"] = float(summary_df[m].std())
        overall[f"{m}_median"] = float(summary_df[m].median())
    dataset_rows.append(overall)

    # by label
    for label_value, sub_df in summary_df.groupby("label"):
        row = {
            "group": f"label_{label_value}",
            "label": int(label_value),
            "n_slides": len(sub_df),
        }
        for m in metrics:
            row[f"{m}_mean"] = float(sub_df[m].mean())
            row[f"{m}_std"] = float(sub_df[m].std())
            row[f"{m}_median"] = float(sub_df[m].median())
        dataset_rows.append(row)

    dataset_summary_df = pd.DataFrame(dataset_rows)
    dataset_summary_df.to_csv(
        Path(args.out_dir) / "attention_dataset_summary.csv",
        index=False,
    )

    # A cleaner long-format table for paper/statistics
    long_rows = []
    for group_name, sub_df in [("all", summary_df)] + [
        (f"label_{k}", v) for k, v in summary_df.groupby("label")
    ]:
        for m in metrics:
            long_rows.append({
                "group": group_name,
                "metric": m,
                "n_slides": len(sub_df),
                "mean": float(sub_df[m].mean()),
                "std": float(sub_df[m].std()),
                "median": float(sub_df[m].median()),
                "q25": float(sub_df[m].quantile(0.25)),
                "q75": float(sub_df[m].quantile(0.75)),
            })

    pd.DataFrame(long_rows).to_csv(
        Path(args.out_dir) / "attention_dataset_summary_long.csv",
        index=False,
    )

    with open(Path(args.out_dir) / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_slides": int(len(summary_rows)),
            "note": (
                "Full-bag ABMIL attention comparison. "
                "Frozen and adapted attentions are computed on their own complete bags; "
                "no coordinate intersection is used. "
                "Patch-level one-to-one delta is intentionally avoided."
            ),
        }, f, indent=2, ensure_ascii=False)

    print(f"[Done] saved to: {args.out_dir}")


if __name__ == "__main__":
    main()