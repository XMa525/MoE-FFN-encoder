#!/usr/bin/env python3
"""
CAMELYON WSI expert / role overlay visualization for a trained MoE pathology encoder.

This version supports:
1. expert overlay
2. role overlay
3. patch-level affinity export
4. slide-level top-k tumor evidence summary
5. top-k tumor evidence patch export
6. tumor-evidence overlay

Expected input layout
---------------------
raw_dir:
    patient_000_node_0.tif / .svs / .ndpi / .mrxs ...

h5_dir:
    patient_000_node_0.h5

The h5 is expected to contain:
    coords: [num_patches, 2]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml

try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

from models.encoders.moe_encoder import MoEEncoder


# =========================================================
# Utils
# =========================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def find_wsi_path(raw_dir: str, slide_id: str) -> str:
    candidates = [
        os.path.join(raw_dir, f"{slide_id}.tif"),
        os.path.join(raw_dir, f"{slide_id}.svs"),
        os.path.join(raw_dir, f"{slide_id}.ndpi"),
        os.path.join(raw_dir, f"{slide_id}.mrxs"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return canonicalize_path(p)
    raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {raw_dir}")


def find_h5_path(h5_dir: str, slide_id: str) -> str:
    p = os.path.join(h5_dir, f"{slide_id}.h5")
    if os.path.exists(p):
        return canonicalize_path(p)
    raise FileNotFoundError(f"H5 not found for slide_id={slide_id} in {h5_dir}")


def read_coords_from_h5(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
    return np.asarray(coords, dtype=np.int64)


def read_patch_from_wsi(
    slide: "openslide.OpenSlide",
    coord_xy: Tuple[int, int],
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    patch = slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")
    return patch


def _topk_mean(arr: np.ndarray, frac: float) -> float:
    if len(arr) == 0:
        return float("nan")
    k = max(1, int(round(len(arr) * frac)))
    idx = np.argpartition(arr, -k)[-k:]
    return float(arr[idx].mean())


# =========================================================
# Data structures
# =========================================================
@dataclass
class PatchPred:
    x: int
    y: int
    patch_size: int
    expert_id: int
    expert_prob: float
    role_id: int
    role_name: str
    role_prob: float

    sim_tumor: float
    sim_stroma: float
    sim_necrosis: float
    delta_tumor_minus_stroma: float
    delta_tumor_minus_max_other: float

    token_major_expert_hist: Dict[int, int]
    token_role_hist: Dict[str, int]


# =========================================================
# Model loading
# =========================================================
def load_stage2_bundle(config_path: str, full_ckpt_path: str, device: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(full_ckpt_path, map_location="cpu")

    if "student_state_dict" not in ckpt:
        raise KeyError(f"student_state_dict not found in {full_ckpt_path}")
    if "distiller_state_dict" not in ckpt:
        raise KeyError(f"distiller_state_dict not found in {full_ckpt_path}")

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    missing, unexpected = model.load_state_dict(ckpt["student_state_dict"], strict=False)
    print(f"[load_stage2_bundle:model] missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device)
    model.eval()

    distiller_sd = ckpt["distiller_state_dict"]
    if "proj_l12.weight" not in distiller_sd or "proj_l12.bias" not in distiller_sd:
        raise RuntimeError(f"proj_l12 not found in distiller_state_dict of {full_ckpt_path}")

    proj_out_dim, proj_in_dim = distiller_sd["proj_l12.weight"].shape
    proj_head = torch.nn.Linear(proj_in_dim, proj_out_dim)
    proj_head.load_state_dict({
        "weight": distiller_sd["proj_l12.weight"],
        "bias": distiller_sd["proj_l12.bias"],
    })
    proj_head = proj_head.to(device)
    proj_head.eval()

    print(f"[load_stage2_bundle] loaded matched model + proj_l12 from {full_ckpt_path}")
    print(f"[load_stage2_bundle] moe_layers_idx = {model.moe_layers_idx}")
    print(f"[load_stage2_bundle] proj_l12: {proj_in_dim} -> {proj_out_dim}")

    return model, proj_head, cfg


def build_transform() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])


# =========================================================
# Role prototype projection
# =========================================================
def load_role_prototypes(role_proto_dir: str) -> Tuple[np.ndarray, List[str]]:
    protos = np.load(os.path.join(role_proto_dir, "role_prototypes_init.npy")).astype(np.float32)
    with open(os.path.join(role_proto_dir, "role_names.json"), "r", encoding="utf-8") as f:
        role_names = json.load(f)
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    return protos, role_names


# =========================================================
# Forward helpers
# =========================================================
def get_last_dispatch_and_feature(model: MoEEncoder, img_tensor: torch.Tensor):
    with torch.inference_mode():
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(img_tensor.device.type == "cuda")):
            final_feats, gate_info_list, feature_dict, moe_feature_list = model(
                img_tensor,
                return_gates=True,
                return_features=True,
                is_eval=True,
            )

    last_gate = gate_info_list[-1]
    seq_len = final_feats.shape[1]

    dispatch_weight = last_gate["dispatch_weight"]
    num_experts = dispatch_weight.shape[-1]
    dispatch_weight = dispatch_weight.view(1, seq_len, num_experts)[:, 1:, :][0]  # [N, E]

    last_moe_feat = moe_feature_list[-1][0, 1:, :]  # [N, D]
    dispatch_weight = dispatch_weight.detach().cpu()
    last_moe_feat = last_moe_feat.detach().cpu()

    del final_feats, gate_info_list, feature_dict, moe_feature_list, last_gate
    return dispatch_weight, last_moe_feat


def infer_patch_prediction(
    model: MoEEncoder,
    proj_head,
    role_prototypes: np.ndarray,
    role_names: Sequence[str],
    patch_image: Image.Image,
    x: int,
    y: int,
    patch_size: int,
    transform: T.Compose,
    device: str,
) -> PatchPred:
    img_tensor = transform(patch_image).unsqueeze(0).to(device)

    dispatch_weight, last_moe_feat = get_last_dispatch_and_feature(model, img_tensor)  # [256,E], [256,D]

    patch_expert_score = dispatch_weight.mean(dim=0)                   # [E]
    expert_id = int(torch.argmax(patch_expert_score).item())
    expert_prob = float(patch_expert_score[expert_id].item())

    token_major_ids = dispatch_weight.argmax(dim=-1).numpy().tolist()
    token_major_hist = dict(Counter(token_major_ids))

    with torch.inference_mode():
        token_role_feat = proj_head(last_moe_feat.to(device).float())
        token_role_feat = F.normalize(token_role_feat, dim=-1)
        protos = torch.from_numpy(role_prototypes).to(device).float()  # [R,D]
        token_role_aff = token_role_feat @ protos.t()                  # [N,R]
        patch_role_aff = token_role_aff.mean(dim=0)                    # [R]

        role_id = int(torch.argmax(patch_role_aff).item())
        role_name = role_names[role_id]
        role_prob = float(torch.softmax(patch_role_aff, dim=0)[role_id].item())

        token_role_ids = torch.argmax(token_role_aff, dim=-1).cpu().numpy().tolist()
        role_to_idx = {name: i for i, name in enumerate(role_names)}

        sim_tumor = float(patch_role_aff[role_to_idx["tumor"]].item()) if "tumor" in role_to_idx else float("nan")
        sim_stroma = float(patch_role_aff[role_to_idx["stroma"]].item()) if "stroma" in role_to_idx else float("nan")
        sim_necrosis = float(patch_role_aff[role_to_idx["necrosis"]].item()) if "necrosis" in role_to_idx else float("nan")

        delta_tumor_minus_stroma = sim_tumor - sim_stroma
        delta_tumor_minus_max_other = sim_tumor - max(sim_stroma, sim_necrosis)

    token_role_hist = dict(Counter([role_names[r] for r in token_role_ids]))

    del token_role_feat, protos, token_role_aff, patch_role_aff
    return PatchPred(
        x=int(x),
        y=int(y),
        patch_size=int(patch_size),
        expert_id=expert_id,
        expert_prob=expert_prob,
        role_id=role_id,
        role_name=role_name,
        role_prob=role_prob,
        sim_tumor=sim_tumor,
        sim_stroma=sim_stroma,
        sim_necrosis=sim_necrosis,
        delta_tumor_minus_stroma=delta_tumor_minus_stroma,
        delta_tumor_minus_max_other=delta_tumor_minus_max_other,
        token_major_expert_hist=token_major_hist,
        token_role_hist=token_role_hist,
    )


# =========================================================
# WSI rendering
# =========================================================
EXPERT_COLORS = {
    0: (49, 130, 189),
    1: (222, 45, 38),
    2: (231, 138, 195),
    3: (65, 182, 196),
}

ROLE_COLORS = {
    "tumor": (215, 48, 39),
    "stroma": (49, 130, 189),
    "necrosis": (231, 138, 195),
    "free": (65, 182, 196),
}


def read_wsi_thumbnail(wsi_path: str, thumbnail_level: int = 6) -> Tuple[Image.Image, float]:
    if not HAS_OPENSLIDE:
        raise ImportError("openslide-python is required for WSI thumbnail visualization")

    slide = openslide.OpenSlide(wsi_path)
    level = min(thumbnail_level, slide.level_count - 1)
    dims = slide.level_dimensions[level]
    downsample = float(slide.level_downsamples[level])
    thumb = slide.read_region((0, 0), level, dims).convert("RGB")
    slide.close()
    return thumb, downsample


def draw_patch_overlay(
    thumbnail: Image.Image,
    patch_preds: Sequence[PatchPred],
    downsample: float,
    mode: str = "expert",
    alpha: int = 110,
) -> Image.Image:
    canvas = thumbnail.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for pred in patch_preds:
        x0 = int(pred.x / downsample)
        y0 = int(pred.y / downsample)
        ps = max(1, int(pred.patch_size / downsample))
        x1 = x0 + ps
        y1 = y0 + ps

        if mode == "expert":
            color = EXPERT_COLORS.get(pred.expert_id, (255, 255, 255))
        elif mode == "role":
            if pred.expert_id == 3:
                color = ROLE_COLORS["free"]
            else:
                color = ROLE_COLORS.get(pred.role_name, (255, 255, 255))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        draw.rectangle((x0, y0, x1, y1), fill=(*color, alpha))

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def draw_topk_score_overlay(
    thumbnail: Image.Image,
    patch_preds: Sequence[PatchPred],
    downsample: float,
    score_key: str,
    top_frac: float = 0.05,
    alpha: int = 150,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    if len(patch_preds) == 0:
        return thumbnail.copy()

    scores = np.array([getattr(p, score_key) for p in patch_preds], dtype=np.float32)
    k = max(1, int(round(len(scores) * top_frac)))
    idx = np.argpartition(scores, -k)[-k:]
    selected = [patch_preds[i] for i in idx]

    canvas = thumbnail.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for pred in selected:
        x0 = int(pred.x / downsample)
        y0 = int(pred.y / downsample)
        ps = max(1, int(pred.patch_size / downsample))
        x1 = x0 + ps
        y1 = y0 + ps
        draw.rectangle((x0, y0, x1, y1), fill=(*color, alpha))

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def save_legend(path: str, color_map: Dict[str, Tuple[int, int, int]], title: str) -> None:
    width = 320
    row_h = 38
    height = 40 + row_h * len(color_map)
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=(20, 20, 20))

    for i, (name, color) in enumerate(color_map.items()):
        y = 40 + i * row_h
        draw.rectangle((12, y, 42, y + 22), fill=color)
        draw.text((56, y + 2), str(name), fill=(20, 20, 20))

    canvas.save(path)


# =========================================================
# Export + summary
# =========================================================
def export_patch_affinity_csv(path: str, patch_preds: Sequence[PatchPred]) -> None:
    fieldnames = [
        "x", "y", "patch_size",
        "expert_id", "expert_prob",
        "role_id", "role_name", "role_prob",
        "sim_tumor", "sim_stroma", "sim_necrosis",
        "delta_tumor_minus_stroma",
        "delta_tumor_minus_max_other",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in patch_preds:
            writer.writerow({
                "x": p.x,
                "y": p.y,
                "patch_size": p.patch_size,
                "expert_id": p.expert_id,
                "expert_prob": p.expert_prob,
                "role_id": p.role_id,
                "role_name": p.role_name,
                "role_prob": p.role_prob,
                "sim_tumor": p.sim_tumor,
                "sim_stroma": p.sim_stroma,
                "sim_necrosis": p.sim_necrosis,
                "delta_tumor_minus_stroma": p.delta_tumor_minus_stroma,
                "delta_tumor_minus_max_other": p.delta_tumor_minus_max_other,
            })


def summarize_role_affinity(patch_preds: Sequence[PatchPred]) -> Dict:
    if len(patch_preds) == 0:
        return {}

    sim_tumor = np.array([p.sim_tumor for p in patch_preds], dtype=np.float32)
    sim_stroma = np.array([p.sim_stroma for p in patch_preds], dtype=np.float32)
    sim_necrosis = np.array([p.sim_necrosis for p in patch_preds], dtype=np.float32)
    delta_ts = np.array([p.delta_tumor_minus_stroma for p in patch_preds], dtype=np.float32)
    delta_tmax = np.array([p.delta_tumor_minus_max_other for p in patch_preds], dtype=np.float32)

    nearest_role_counts = Counter([p.role_name for p in patch_preds])
    nearest_role_ratio = {
        str(k): float(v / len(patch_preds)) for k, v in nearest_role_counts.items()
    }

    summary = {
        "mean_sim_tumor": float(sim_tumor.mean()),
        "std_sim_tumor": float(sim_tumor.std()),
        "mean_sim_stroma": float(sim_stroma.mean()),
        "std_sim_stroma": float(sim_stroma.std()),
        "mean_sim_necrosis": float(sim_necrosis.mean()),
        "std_sim_necrosis": float(sim_necrosis.std()),

        "max_sim_tumor": float(sim_tumor.max()),
        "top1pct_mean_sim_tumor": _topk_mean(sim_tumor, 0.01),
        "top5pct_mean_sim_tumor": _topk_mean(sim_tumor, 0.05),
        "top10pct_mean_sim_tumor": _topk_mean(sim_tumor, 0.10),

        "mean_delta_tumor_minus_stroma": float(delta_ts.mean()),
        "std_delta_tumor_minus_stroma": float(delta_ts.std()),
        "top1pct_mean_delta_tumor_minus_stroma": _topk_mean(delta_ts, 0.01),
        "top5pct_mean_delta_tumor_minus_stroma": _topk_mean(delta_ts, 0.05),

        "mean_delta_tumor_minus_max_other": float(delta_tmax.mean()),
        "std_delta_tumor_minus_max_other": float(delta_tmax.std()),
        "top1pct_mean_delta_tumor_minus_max_other": _topk_mean(delta_tmax, 0.01),
        "top5pct_mean_delta_tumor_minus_max_other": _topk_mean(delta_tmax, 0.05),
        "top10pct_mean_delta_tumor_minus_max_other": _topk_mean(delta_tmax, 0.10),

        "frac_sim_tumor_gt_0.5": float((sim_tumor > 0.5).mean()),
        "frac_sim_tumor_gt_0.6": float((sim_tumor > 0.6).mean()),
        "frac_sim_tumor_gt_0.7": float((sim_tumor > 0.7).mean()),

        "frac_delta_tumor_minus_max_other_gt_0": float((delta_tmax > 0).mean()),
        "frac_delta_tumor_minus_max_other_gt_0.05": float((delta_tmax > 0.05).mean()),
        "frac_delta_tumor_minus_max_other_gt_0.10": float((delta_tmax > 0.10).mean()),

        "nearest_role_ratio": nearest_role_ratio,
    }
    return summary


def save_topk_patch_images(
    slide,
    patch_preds: Sequence[PatchPred],
    outdir: str,
    score_key: str = "sim_tumor",
    topk: int = 50,
    read_level: int = 0,
):
    ensure_dir(outdir)
    if len(patch_preds) == 0:
        return

    sorted_preds = sorted(
        patch_preds,
        key=lambda p: getattr(p, score_key),
        reverse=True,
    )[:topk]

    meta_rows = []
    for rank, pred in enumerate(sorted_preds, start=1):
        patch = read_patch_from_wsi(
            slide,
            coord_xy=(pred.x, pred.y),
            patch_size=pred.patch_size,
            read_level=read_level,
        )
        fname = f"{rank:03d}_{score_key}_{pred.x}_{pred.y}.png"
        patch.save(os.path.join(outdir, fname))

        meta_rows.append({
            "rank": rank,
            "x": pred.x,
            "y": pred.y,
            "expert_id": pred.expert_id,
            "role_name": pred.role_name,
            "sim_tumor": pred.sim_tumor,
            "sim_stroma": pred.sim_stroma,
            "sim_necrosis": pred.sim_necrosis,
            "delta_tumor_minus_stroma": pred.delta_tumor_minus_stroma,
            "delta_tumor_minus_max_other": pred.delta_tumor_minus_max_other,
            "filename": fname,
        })

    with open(os.path.join(outdir, "topk_meta.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        writer.writeheader()
        writer.writerows(meta_rows)


# =========================================================
# Main
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAMELYON WSI expert/role overlay visualization (raw_dir + h5_dir + slide_ids)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--full-ckpt", type=str, required=True)
    parser.add_argument("--role-proto-dir", type=str, required=True)
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--h5-dir", type=str, required=True)
    parser.add_argument("--slide-ids", nargs="+", required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--thumbnail-level", type=int, default=6)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--read-level", type=int, default=0)
    parser.add_argument("--max-patches-per-slide", type=int, default=0, help="0 means use all coords in h5")
    parser.add_argument("--topk-export", type=int, default=50)
    parser.add_argument("--tumor-overlay-frac", type=float, default=0.05, help="Top fraction for tumor evidence overlay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.outdir)

    model, proj_head, _ = load_stage2_bundle(
        args.config,
        args.full_ckpt,
        args.device,
    )
    transform = build_transform()
    role_prototypes, role_names = load_role_prototypes(args.role_proto_dir)

    save_legend(
        os.path.join(args.outdir, "legend_expert.png"),
        {f"E{k}": v for k, v in EXPERT_COLORS.items()},
        title="Expert legend",
    )
    save_legend(
        os.path.join(args.outdir, "legend_role.png"),
        {
            "tumor": ROLE_COLORS["tumor"],
            "stroma": ROLE_COLORS["stroma"],
            "necrosis": ROLE_COLORS["necrosis"],
            "free": ROLE_COLORS["free"],
        },
        title="Role legend",
    )

    for slide_id in args.slide_ids:
        print(f"\n[Slide] {slide_id}")
        wsi_path = find_wsi_path(args.raw_dir, slide_id)
        h5_path = find_h5_path(args.h5_dir, slide_id)
        coords = read_coords_from_h5(h5_path)

        if args.max_patches_per_slide > 0 and len(coords) > args.max_patches_per_slide:
            rng = np.random.default_rng(args.seed)
            keep_idx = rng.choice(len(coords), size=args.max_patches_per_slide, replace=False)
            coords = coords[keep_idx]

        print(f"  wsi_path: {wsi_path}")
        print(f"  h5_path : {h5_path}")
        print(f"  num coords: {len(coords)}")

        if not HAS_OPENSLIDE:
            raise ImportError("openslide-python is required for WSI visualization")

        slide = openslide.OpenSlide(wsi_path)
        thumb, downsample = read_wsi_thumbnail(wsi_path, thumbnail_level=args.thumbnail_level)

        patch_preds: List[PatchPred] = []
        for xy in coords:
            x, y = int(xy[0]), int(xy[1])
            patch = read_patch_from_wsi(
                slide,
                coord_xy=(x, y),
                patch_size=args.patch_size,
                read_level=args.read_level,
            )
            pred = infer_patch_prediction(
                model=model,
                proj_head=proj_head,
                role_prototypes=role_prototypes,
                role_names=role_names,
                patch_image=patch,
                x=x,
                y=y,
                patch_size=args.patch_size,
                transform=transform,
                device=args.device,
            )
            patch_preds.append(pred)

        slide_outdir = os.path.join(args.outdir, slide_id)
        ensure_dir(slide_outdir)

        thumb.save(os.path.join(slide_outdir, "thumbnail.png"))

        expert_overlay = draw_patch_overlay(thumb, patch_preds, downsample, mode="expert")
        role_overlay = draw_patch_overlay(thumb, patch_preds, downsample, mode="role")
        tumor_overlay = draw_topk_score_overlay(
            thumb,
            patch_preds,
            downsample=downsample,
            score_key="delta_tumor_minus_max_other",
            top_frac=args.tumor_overlay_frac,
            alpha=150,
            color=(255, 0, 0),
        )

        expert_overlay.save(os.path.join(slide_outdir, "expert_overlay.png"))
        role_overlay.save(os.path.join(slide_outdir, "role_overlay.png"))
        tumor_overlay.save(os.path.join(slide_outdir, "tumor_evidence_overlay.png"))

        expert_counts = Counter([p.expert_id for p in patch_preds])
        role_counts = Counter(["free" if p.expert_id == 3 else p.role_name for p in patch_preds])

        summary = {
            "slide_id": slide_id,
            "wsi_path": wsi_path,
            "h5_path": h5_path,
            "num_patches": len(patch_preds),
            "thumbnail_level": args.thumbnail_level,
            "patch_size": args.patch_size,
            "read_level": args.read_level,
            "downsample": downsample,
            "expert_counts": {str(k): int(v) for k, v in expert_counts.items()},
            "role_counts": {str(k): int(v) for k, v in role_counts.items()},
        }
        with open(os.path.join(slide_outdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        export_patch_affinity_csv(
            os.path.join(slide_outdir, "patch_affinity.csv"),
            patch_preds,
        )

        affinity_summary = summarize_role_affinity(patch_preds)
        with open(os.path.join(slide_outdir, "affinity_summary.json"), "w", encoding="utf-8") as f:
            json.dump(affinity_summary, f, ensure_ascii=False, indent=2)

        save_topk_patch_images(
            slide=slide,
            patch_preds=patch_preds,
            outdir=os.path.join(slide_outdir, "topk_sim_tumor"),
            score_key="sim_tumor",
            topk=args.topk_export,
            read_level=args.read_level,
        )
        save_topk_patch_images(
            slide=slide,
            patch_preds=patch_preds,
            outdir=os.path.join(slide_outdir, "topk_delta_tumor_minus_max_other"),
            score_key="delta_tumor_minus_max_other",
            topk=args.topk_export,
            read_level=args.read_level,
        )

        print("[Affinity Summary]")
        print(json.dumps(affinity_summary, ensure_ascii=False, indent=2))

        slide.close()

    print(f"\nDone. Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()