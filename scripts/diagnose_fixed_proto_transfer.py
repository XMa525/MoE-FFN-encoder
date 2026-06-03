#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import yaml
import h5py
import random
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import openslide

from PIL import Image, ImageDraw
from tqdm import tqdm
import torchvision.transforms.v2 as T

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoders.moe_encoder import MoEEncoder


# =========================================================
# utils
# =========================================================
VALID_WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_device(device_str: str) -> str:
    if device_str == "cpu":
        return "cpu"
    return device_str if torch.cuda.is_available() else "cpu"


def build_transform(image_size: int = 224):
    return T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


def make_label_column_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("Need 'label' or 'slide_binary_label' in slides_csv.")
    return df


def load_slides_csv(slides_csv: str, split: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(slides_csv)
    if "slide_id" not in df.columns:
        raise ValueError("slides_csv must contain 'slide_id'")
    df = make_label_column_if_needed(df)

    if split is not None:
        if "split" not in df.columns:
            raise ValueError("split specified but slides_csv has no 'split' column")
        df = df[df["split"] == split].copy()

    df = df.reset_index(drop=True)
    return df


def find_wsi_path(raw_dir: str, slide_id: str) -> str:
    raw_dir = Path(raw_dir)
    matches = []
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_WSI_EXTS:
            continue
        if slide_id in p.stem or slide_id in p.name:
            matches.append(str(p))
    if len(matches) == 0:
        raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {raw_dir}")
    matches = sorted(matches, key=lambda x: (len(x), x))
    return matches[0]


def find_h5_path(h5_dir: str, slide_id: str) -> str:
    h5_dir = Path(h5_dir)
    matches = []
    for p in h5_dir.rglob("*.h5"):
        if slide_id in p.stem or slide_id in p.name:
            matches.append(str(p))
    if len(matches) == 0:
        raise FileNotFoundError(f"H5 not found for slide_id={slide_id} in {h5_dir}")
    matches = sorted(matches, key=lambda x: (len(x), x))
    return matches[0]


def read_coords_from_h5(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
    return coords


def sample_coords(coords: np.ndarray, max_patches: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(coords)
    if n <= max_patches:
        idx = np.arange(n)
        return coords, idx
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_patches, replace=False)
    idx = np.sort(idx)
    return coords[idx], idx


def read_patch_from_wsi(
    slide: openslide.OpenSlide,
    coord_xy: Tuple[int, int],
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    patch = slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")
    return patch


def make_montage(images: List[Image.Image], captions: List[str], tile_size: int = 224, ncols: int = 4) -> Image.Image:
    assert len(images) == len(captions)
    if len(images) == 0:
        canvas = Image.new("RGB", (tile_size, tile_size), color=(255, 255, 255))
        return canvas

    n = len(images)
    nrows = math.ceil(n / ncols)
    cap_h = 28
    canvas = Image.new("RGB", (ncols * tile_size, nrows * (tile_size + cap_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, (img, cap) in enumerate(zip(images, captions)):
        r = i // ncols
        c = i % ncols
        x = c * tile_size
        y = r * (tile_size + cap_h)

        img = img.resize((tile_size, tile_size))
        canvas.paste(img, (x, y))
        draw.text((x + 4, y + tile_size + 4), cap, fill=(0, 0, 0))

    return canvas


# =========================================================
# load stage2 + proj_l12
# =========================================================
def load_stage2_bundle(
    config_path: str,
    full_ckpt_path: str,
    device: str = "cuda",
):
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

    for p in model.parameters():
        p.requires_grad = False
    for p in proj_l12.parameters():
        p.requires_grad = False

    print(f"[INFO] loaded stage2 model, proj_l12: {proj_in_dim} -> {proj_out_dim}")
    return model, proj_l12


def load_role_proto(role_proto_dir: str, device: str):
    proto_path = os.path.join(role_proto_dir, "role_prototypes_init.npy")
    names_path = os.path.join(role_proto_dir, "role_names.json")
    if not os.path.exists(proto_path):
        raise FileNotFoundError(proto_path)
    if not os.path.exists(names_path):
        raise FileNotFoundError(names_path)

    protos = torch.from_numpy(np.load(proto_path).astype("float32")).to(device)
    protos = F.normalize(protos, dim=-1)

    with open(names_path, "r", encoding="utf-8") as f:
        role_names = json.load(f)

    print(f"[INFO] loaded role proto: {protos.shape}, role_names={role_names}")
    return protos, role_names


# =========================================================
# feature extraction
# =========================================================
class Stage2PatchFeaturizer(nn.Module):
    def __init__(
        self,
        config_path: str,
        full_ckpt_path: str,
        device: str = "cuda",
        use_last_moe_output: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model, self.proj_l12 = load_stage2_bundle(config_path, full_ckpt_path, device)
        self.use_last_moe_output = bool(use_last_moe_output)
        self.transform = build_transform(image_size=224)

    @torch.no_grad()
    def extract_teacher_like_patch_feats(self, images: List[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.transform(img) for img in images]).to(self.device)  # [B,3,224,224]

        _, _, feature_dict, moe_feature_list = self.model(
            x,
            return_gates=True,
            mask=None,
            is_eval=True,
            return_features=True,
            offline_cluster_ids=None,
        )

        if self.use_last_moe_output and len(moe_feature_list) > 0:
            feat = moe_feature_list[-1]      # [B, T+1, 384]
        else:
            feat = feature_dict["layer_12"]  # [B, T+1, 384]

        patch_tokens = feat[:, 1:, :]        # [B, T, 384]
        patch_feat_raw = patch_tokens.mean(dim=1)  # [B, 384]

        patch_feat_teacher = self.proj_l12(patch_feat_raw)  # [B, 1280]
        patch_feat_teacher = F.normalize(patch_feat_teacher, dim=-1)
        return patch_feat_teacher


def compute_role_stats(
    patch_feat_teacher: torch.Tensor,   # [N, D]
    protos: torch.Tensor,               # [R, D]
    tau: float = 1.0,
):
    logits = patch_feat_teacher @ protos.t()      # [N, R]
    probs = torch.softmax(logits / tau, dim=-1)   # [N, R]

    R = logits.shape[-1]
    gaps = []
    for r in range(R):
        cur = logits[:, r]
        other_ids = [i for i in range(R) if i != r]
        other_max = logits[:, other_ids].max(dim=-1).values
        gaps.append(cur - other_max)
    role_gaps = torch.stack(gaps, dim=-1)         # [N, R]

    top2 = torch.topk(logits, k=min(2, R), dim=-1).values
    if R >= 2:
        top1_gap = top2[:, 0] - top2[:, 1]        # [N]
    else:
        top1_gap = torch.ones_like(top2[:, 0])

    top1_idx = logits.argmax(dim=-1)              # [N]

    return {
        "logits": logits,
        "probs": probs,
        "role_gaps": role_gaps,
        "top1_gap": top1_gap,
        "top1_idx": top1_idx,
    }


# =========================================================
# main diagnostic
# =========================================================
@torch.no_grad()
def diagnose_one_slide(
    featurizer: Stage2PatchFeaturizer,
    protos: torch.Tensor,
    role_names: List[str],
    slide_path: str,
    h5_path: str,
    slide_id: str,
    label: int,
    patch_size: int,
    max_patches: int,
    batch_size: int,
    role_tau: float,
    conf_thresh: float,
    vis_topk: int,
    vis_dir: str,
    seed: int,
):
    coords_all = read_coords_from_h5(h5_path)
    coords, sampled_idx = sample_coords(coords_all, max_patches=max_patches, seed=seed)

    slide = openslide.OpenSlide(slide_path)

    all_teacher = []
    all_coords = []
    all_imgs_small = []

    for start in range(0, len(coords), batch_size):
        end = min(start + batch_size, len(coords))
        batch_coords = coords[start:end]

        batch_imgs = []
        for xy in batch_coords.tolist():
            img = read_patch_from_wsi(slide, xy, patch_size=patch_size, read_level=0)
            batch_imgs.append(img)
            all_imgs_small.append(img.copy())

        teacher_feats = featurizer.extract_teacher_like_patch_feats(batch_imgs)  # [b, 1280]
        all_teacher.append(teacher_feats.cpu())
        all_coords.append(torch.from_numpy(batch_coords).long())

    slide.close()

    teacher_feats = torch.cat(all_teacher, dim=0)           # [N, 1280]
    coords_t = torch.cat(all_coords, dim=0)                 # [N, 2]

    role_out = compute_role_stats(
        patch_feat_teacher=teacher_feats.to(protos.device),
        protos=protos,
        tau=role_tau,
    )

    logits = role_out["logits"].cpu()
    probs = role_out["probs"].cpu()
    role_gaps = role_out["role_gaps"].cpu()
    top1_gap = role_out["top1_gap"].cpu()
    top1_idx = role_out["top1_idx"].cpu()

    # role index
    role_to_idx = {name: i for i, name in enumerate(role_names)}
    tumor_idx = role_to_idx.get("tumor", 0)

    slide_summary = {
        "slide_id": slide_id,
        "label": int(label),
        "num_sampled_patches": int(len(coords_t)),
        "mean_top1_gap": float(top1_gap.mean().item()),
        "median_top1_gap": float(top1_gap.median().item()),
        "high_conf_frac": float((top1_gap >= conf_thresh).float().mean().item()),
        "tumor_prob_mean": float(probs[:, tumor_idx].mean().item()),
        "tumor_logit_mean": float(logits[:, tumor_idx].mean().item()),
        "tumor_gap_mean": float(role_gaps[:, tumor_idx].mean().item()),
        "tumor_top1_frac": float((top1_idx == tumor_idx).float().mean().item()),
    }

    for r, name in enumerate(role_names):
        slide_summary[f"{name}_prob_mean"] = float(probs[:, r].mean().item())
        slide_summary[f"{name}_gap_mean"] = float(role_gaps[:, r].mean().item())
        slide_summary[f"{name}_top1_frac"] = float((top1_idx == r).float().mean().item())

    # patch-level sample table
    patch_rows = []
    for i in range(len(coords_t)):
        row = {
            "slide_id": slide_id,
            "label": int(label),
            "patch_idx": int(i),
            "coord_x": int(coords_t[i, 0].item()),
            "coord_y": int(coords_t[i, 1].item()),
            "top1_role": role_names[int(top1_idx[i].item())],
            "top1_gap": float(top1_gap[i].item()),
            "high_conf": int(top1_gap[i].item() >= conf_thresh),
            "tumor_prob": float(probs[i, tumor_idx].item()),
            "tumor_gap": float(role_gaps[i, tumor_idx].item()),
        }
        for r, name in enumerate(role_names):
            row[f"{name}_prob"] = float(probs[i, r].item())
            row[f"{name}_gap"] = float(role_gaps[i, r].item())
        patch_rows.append(row)

    # visualization: top confident patches per role
    ensure_dir(vis_dir)
    for r, role_name in enumerate(role_names):
        score = role_gaps[:, r]
        valid_mask = score >= conf_thresh
        valid_idx = torch.where(valid_mask)[0]

        if len(valid_idx) == 0:
            continue

        sort_idx = valid_idx[torch.argsort(score[valid_idx], descending=True)]
        sort_idx = sort_idx[:vis_topk]

        imgs = []
        caps = []
        for j in sort_idx.tolist():
            imgs.append(all_imgs_small[j])
            caps.append(
                f"{role_name} gap={score[j].item():.3f}\n"
                f"p={probs[j, r].item():.3f}"
            )

        montage = make_montage(imgs, caps, tile_size=224, ncols=min(4, vis_topk))
        save_path = os.path.join(vis_dir, f"{slide_id}__label{label}__top_{role_name}.png")
        montage.save(save_path)

    return slide_summary, patch_rows


def main():
    parser = argparse.ArgumentParser("Small-scale diagnostic for fixed role proto transferability")

    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--h5_dir", type=str, required=True)
    parser.add_argument("--stage2_config", type=str, required=True)
    parser.add_argument("--stage2_full_ckpt", type=str, required=True)
    parser.add_argument("--role_proto_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_pos_slides", type=int, default=5)
    parser.add_argument("--num_neg_slides", type=int, default=5)
    parser.add_argument("--max_patches_per_slide", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=256)

    parser.add_argument("--role_tau", type=float, default=1.0)
    parser.add_argument("--conf_thresh", type=float, default=0.20,
                        help="top1 gap threshold for high-confidence patches")
    parser.add_argument("--vis_topk", type=int, default=8)
    parser.add_argument("--use_last_moe_output", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    vis_dir = os.path.join(args.out_dir, "vis")
    ensure_dir(vis_dir)

    device = resolve_device(args.device)

    df = load_slides_csv(args.slides_csv, split=args.split)
    pos_df = df[df["label"] == 1].copy()
    neg_df = df[df["label"] == 0].copy()

    if len(pos_df) == 0 or len(neg_df) == 0:
        raise ValueError("Need both positive and negative slides in the selected set")

    pos_df = pos_df.sample(n=min(args.num_pos_slides, len(pos_df)), random_state=args.seed)
    neg_df = neg_df.sample(n=min(args.num_neg_slides, len(neg_df)), random_state=args.seed)
    use_df = pd.concat([pos_df, neg_df], axis=0).reset_index(drop=True)

    print(f"[INFO] selected slides = {len(use_df)} "
          f"(pos={len(pos_df)}, neg={len(neg_df)})")

    featurizer = Stage2PatchFeaturizer(
        config_path=args.stage2_config,
        full_ckpt_path=args.stage2_full_ckpt,
        device=device,
        use_last_moe_output=args.use_last_moe_output,
    )
    protos, role_names = load_role_proto(args.role_proto_dir, device=device)

    slide_rows = []
    patch_rows_all = []
    failed_rows = []

    for _, row in tqdm(use_df.iterrows(), total=len(use_df), desc="Diagnosing slides"):
        slide_id = str(row["slide_id"])
        label = int(row["label"])

        try:
            slide_path = find_wsi_path(args.raw_dir, slide_id)
            h5_path = find_h5_path(args.h5_dir, slide_id)

            slide_summary, patch_rows = diagnose_one_slide(
                featurizer=featurizer,
                protos=protos,
                role_names=role_names,
                slide_path=slide_path,
                h5_path=h5_path,
                slide_id=slide_id,
                label=label,
                patch_size=args.patch_size,
                max_patches=args.max_patches_per_slide,
                batch_size=args.batch_size,
                role_tau=args.role_tau,
                conf_thresh=args.conf_thresh,
                vis_topk=args.vis_topk,
                vis_dir=vis_dir,
                seed=args.seed,
            )

            slide_rows.append(slide_summary)
            patch_rows_all.extend(patch_rows)

        except Exception as e:
            print(f"[ERROR] {slide_id}: {e}")
            failed_rows.append({
                "slide_id": slide_id,
                "label": label,
                "error": str(e),
            })

    slide_csv = os.path.join(args.out_dir, "slide_level_summary.csv")
    patch_csv = os.path.join(args.out_dir, "patch_level_samples.csv")
    failed_csv = os.path.join(args.out_dir, "failed_slides.csv")

    pd.DataFrame(slide_rows).to_csv(slide_csv, index=False)
    pd.DataFrame(patch_rows_all).to_csv(patch_csv, index=False)
    if len(failed_rows) > 0:
        pd.DataFrame(failed_rows).to_csv(failed_csv, index=False)

    print(f"[Saved] {slide_csv}")
    print(f"[Saved] {patch_csv}")
    if len(failed_rows) > 0:
        print(f"[Saved] {failed_csv}")

    if len(slide_rows) > 0:
        sdf = pd.DataFrame(slide_rows)
        print("\n===== quick summary =====")
        group_cols = ["tumor_prob_mean", "tumor_gap_mean", "high_conf_frac", "mean_top1_gap"]
        print(sdf.groupby("label")[group_cols].mean())

    print(f"[Done] outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()