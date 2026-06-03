#!/usr/bin/env python3
"""
Layer-10 expert overlay and top-token gallery visualization for MoE pathology encoder.

What this script does
---------------------
1. Loads a trained MoEEncoder checkpoint.
2. Runs inference on sampled or user-specified patch images.
3. Extracts the LAST MoE block routing / dispatch information.
4. Saves:
   - hard expert assignment overlay on each image
   - per-expert binary overlays (E0 / E1 / E2 / E3)
   - top-k token galleries for each expert based on dispatch weight
   - optional top-k galleries restricted to a selected organ/category

Why this script is useful
-------------------------
It turns your statistical evidence (expert-role heatmaps / UMAP) into direct
image-level evidence of what each expert is actually handling.

Expected use
------------
- Best used on the stage2 roleproto static checkpoint.
- Focuses on the last MoE block, which is your main specialization layer.

Example
-------
python layer10_expert_overlay_and_gallery.py \
  --config configs/stage2.yaml \
  --ckpt results/stage2_best_model/moe_encoder_stage2_best.pth \
  --base-dir ../data/raw \
  --outdir analysis_outputs/layer10_overlay_gallery \
  --categories SPIDER-breast SPIDER-colorectal SPIDER-skin SPIDER-thorax \
  --n-samples-per-cat 20 \
  --topk-per-expert 96

If you already know the exact image paths to visualize, provide a txt file with one path per line:
python layer10_expert_overlay_and_gallery.py \
  --config configs/stage2.yaml \
  --ckpt results/stage2_best_model/moe_encoder_stage2_best.pth \
  --image-list selected_patches.txt \
  --outdir analysis_outputs/layer10_overlay_gallery
"""

from __future__ import annotations
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import math

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import yaml

from models.encoders.moe_encoder import MoEEncoder


# =========================
# Utilities
# =========================
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


def token_idx_to_row_col(token_idx: int, grid_size: int = 16) -> Tuple[int, int]:
    return token_idx // grid_size, token_idx % grid_size


def row_col_to_pixel_box(row: int, col: int, image_size: int = 224, grid_size: int = 16) -> Tuple[int, int, int, int]:
    patch_size = image_size // grid_size
    x0 = col * patch_size
    y0 = row * patch_size
    x1 = x0 + patch_size
    y1 = y0 + patch_size
    return x0, y0, x1, y1


# =========================
# Data structures
# =========================
@dataclass
class TokenRecord:
    image_path: str
    category: str
    token_idx: int
    row: int
    col: int
    expert_id: int
    expert_weight: float
    image_index: int


# =========================
# Model loading / inference
# =========================
def load_model(config_path: str, ckpt_path: str, device: str) -> Tuple[MoEEncoder, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # allow both full ckpt dict and raw state_dict
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "student_state_dict" in ckpt:
        state_dict = ckpt["student_state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load_model] missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device)
    model.eval()
    return model, cfg


def build_transform() -> T.Compose:
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])


def get_last_dispatch_info(gate_info_list: Sequence[dict], seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gate_info = gate_info_list[-1]
    dispatch_weight = gate_info["dispatch_weight"]  # [B*seq_len, E]
    dispatch_mask = gate_info["dispatch_mask"]      # [B*seq_len, E]

    total_tokens, num_experts = dispatch_weight.shape
    batch_size = total_tokens // seq_len

    dispatch_weight = dispatch_weight.view(batch_size, seq_len, num_experts)[:, 1:, :]  # remove CLS
    dispatch_mask = dispatch_mask.view(batch_size, seq_len, num_experts)[:, 1:, :]
    return dispatch_weight, dispatch_mask


def forward_one_image(model: MoEEncoder, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        final_feats, gate_info_list, _, moe_feature_list = model(
            img_tensor,
            return_gates=True,
            return_features=True,
            is_eval=True,
        )

    seq_len = final_feats.shape[1]
    dispatch_weight, dispatch_mask = get_last_dispatch_info(gate_info_list, seq_len)
    return dispatch_weight[0].detach().cpu(), dispatch_mask[0].detach().cpu()  # [N, E], [N, E]


# =========================
# Image selection
# =========================
def sample_images(base_dir: str, categories: Sequence[str], n_samples_per_cat: int) -> List[Tuple[str, str]]:
    selected: List[Tuple[str, str]] = []
    for cat in categories:
        img_dir = os.path.join(base_dir, cat, cat, "images")
        all_imgs = [os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.endswith(".png")]
        if len(all_imgs) < n_samples_per_cat:
            raise ValueError(f"{cat} has only {len(all_imgs)} images, fewer than requested {n_samples_per_cat}")
        sampled = random.sample(all_imgs, n_samples_per_cat)
        selected.extend((p, cat) for p in sampled)
    return selected


def load_image_list(image_list_path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(image_list_path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            rows.append((p, "unknown"))
    return rows


# =========================
# Overlay drawing
# =========================
EXPERT_COLORS = {
    0: (49, 130, 189),   # blue-ish
    1: (222, 45, 38),    # red-ish
    2: (231, 138, 195),  # pink-ish
    3: (65, 182, 196),   # cyan-ish
}


def draw_grid_lines(draw: ImageDraw.ImageDraw, image_size: int = 224, grid_size: int = 16) -> None:
    patch_size = image_size // grid_size
    for x in range(0, image_size + 1, patch_size):
        draw.line((x, 0, x, image_size), fill=(255, 255, 255, 80), width=1)
    for y in range(0, image_size + 1, patch_size):
        draw.line((0, y, image_size, y), fill=(255, 255, 255, 80), width=1)


def make_hard_assignment_overlay(
    image: Image.Image,
    expert_ids: np.ndarray,
    alpha: int = 90,
    grid_size: int = 16,
) -> Image.Image:
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for token_idx, expert_id in enumerate(expert_ids.tolist()):
        row, col = token_idx_to_row_col(token_idx, grid_size)
        x0, y0, x1, y1 = row_col_to_pixel_box(row, col, image.size[0], grid_size)
        color = EXPERT_COLORS.get(int(expert_id), (255, 255, 255))
        draw.rectangle((y0, x0, y1, x1), fill=(*color, alpha))

    draw_grid_lines(draw, image.size[0], grid_size)
    return Image.alpha_composite(image, overlay).convert("RGB")


def make_single_expert_overlay(
    image: Image.Image,
    expert_ids: np.ndarray,
    target_expert: int,
    alpha: int = 120,
    grid_size: int = 16,
) -> Image.Image:
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    color = EXPERT_COLORS.get(target_expert, (255, 255, 255))

    for token_idx, expert_id in enumerate(expert_ids.tolist()):
        row, col = token_idx_to_row_col(token_idx, grid_size)
        x0, y0, x1, y1 = row_col_to_pixel_box(row, col, image.size[0], grid_size)
        if int(expert_id) == target_expert:
            draw.rectangle((y0, x0, y1, x1), fill=(*color, alpha))
        else:
            draw.rectangle((y0, x0, y1, x1), fill=(0, 0, 0, 20))

    draw_grid_lines(draw, image.size[0], grid_size)
    return Image.alpha_composite(image, overlay).convert("RGB")


# =========================
# Top-k token gallery
# =========================
def collect_token_records(
    image_path: str,
    category: str,
    image_index: int,
    dispatch_weight: torch.Tensor,
) -> List[TokenRecord]:
    hard_ids = dispatch_weight.argmax(dim=-1).numpy()   # [N]
    out: List[TokenRecord] = []
    num_tokens, num_experts = dispatch_weight.shape

    for token_idx in range(num_tokens):
        row, col = token_idx_to_row_col(token_idx, 16)
        expert_id = int(hard_ids[token_idx])
        expert_weight = float(dispatch_weight[token_idx, expert_id].item())
        out.append(TokenRecord(
            image_path=image_path,
            category=category,
            token_idx=token_idx,
            row=row,
            col=col,
            expert_id=expert_id,
            expert_weight=expert_weight,
            image_index=image_index,
        ))
    return out


def crop_token_patch(image_path: str, token_idx: int, grid_size: int = 16, image_size: int = 224) -> Image.Image:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    row, col = token_idx_to_row_col(token_idx, grid_size)
    x0, y0, x1, y1 = row_col_to_pixel_box(row, col, image_size, grid_size)
    return image.crop((y0, x0, y1, x1))


def save_gallery(records: List[TokenRecord], out_path: str, title: str, ncols: int = 8, tile_size: int = 96) -> None:
    if len(records) == 0:
        return

    n = len(records)
    nrows = math.ceil(n / ncols)
    pad = 8
    header_h = 40
    canvas_w = ncols * (tile_size + pad) + pad
    canvas_h = header_h + nrows * (tile_size + 32 + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), title, fill=(20, 20, 20))

    for i, rec in enumerate(records):
        r = i // ncols
        c = i % ncols
        x = pad + c * (tile_size + pad)
        y = header_h + r * (tile_size + 32 + pad)

        patch = crop_token_patch(rec.image_path, rec.token_idx).resize((tile_size, tile_size))
        canvas.paste(patch, (x, y))
        info = f"E{rec.expert_id} w={rec.expert_weight:.3f}\n{os.path.basename(rec.image_path)}:{rec.token_idx}"
        draw.text((x, y + tile_size + 2), info, fill=(30, 30, 30))

    canvas.save(out_path)


# =========================
# Main
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer10 expert overlay and top-token gallery")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="../data/raw")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--categories", nargs="*", default=["SPIDER-breast", "SPIDER-colorectal", "SPIDER-skin", "SPIDER-thorax"])
    parser.add_argument("--n-samples-per-cat", type=int, default=20)
    parser.add_argument("--image-list", type=str, default="")
    parser.add_argument("--topk-per-expert", type=int, default=96)
    parser.add_argument("--save-per-image-overlays", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "hard_overlay"))
    ensure_dir(os.path.join(args.outdir, "single_expert_overlay"))
    ensure_dir(os.path.join(args.outdir, "galleries"))

    model, _ = load_model(args.config, args.ckpt, args.device)
    transform = build_transform()

    if args.image_list:
        selected = load_image_list(args.image_list)
    else:
        selected = sample_images(args.base_dir, args.categories, args.n_samples_per_cat)

    all_records: List[TokenRecord] = []

    for image_index, (image_path, category) in enumerate(selected):
        image = Image.open(image_path).convert("RGB").resize((224, 224))
        img_tensor = transform(image).unsqueeze(0).to(args.device)

        dispatch_weight, dispatch_mask = forward_one_image(model, img_tensor)
        hard_ids = dispatch_weight.argmax(dim=-1).numpy()  # [256]

        # collect token records for gallery ranking
        all_records.extend(collect_token_records(
            image_path=image_path,
            category=category,
            image_index=image_index,
            dispatch_weight=dispatch_weight,
        ))

        if args.save_per_image_overlays:
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            hard_overlay = make_hard_assignment_overlay(image, hard_ids)
            hard_overlay.save(os.path.join(args.outdir, "hard_overlay", f"{category}__{base_name}__hard.png"))

            for expert_id in range(dispatch_weight.shape[1]):
                single_overlay = make_single_expert_overlay(image, hard_ids, expert_id)
                single_overlay.save(
                    os.path.join(args.outdir, "single_expert_overlay", f"{category}__{base_name}__E{expert_id}.png")
                )

    # Save top-k token galleries per expert
    for expert_id in range(4):
        recs = [r for r in all_records if r.expert_id == expert_id]
        recs = sorted(recs, key=lambda x: x.expert_weight, reverse=True)[: args.topk_per_expert]
        save_gallery(
            recs,
            out_path=os.path.join(args.outdir, "galleries", f"topk_expert_{expert_id}.png"),
            title=f"Top-{len(recs)} token gallery for Expert {expert_id}",
        )

    # Also save per-category galleries to see organ-wise consistency
    for category in sorted(set(c for _, c in selected)):
        for expert_id in range(4):
            recs = [r for r in all_records if r.category == category and r.expert_id == expert_id]
            recs = sorted(recs, key=lambda x: x.expert_weight, reverse=True)[: min(48, len(recs))]
            if len(recs) == 0:
                continue
            save_gallery(
                recs,
                out_path=os.path.join(args.outdir, "galleries", f"topk_{category}_expert_{expert_id}.png"),
                title=f"Top-{len(recs)} token gallery for {category} / Expert {expert_id}",
                ncols=6,
            )

    print(f"Done. Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
