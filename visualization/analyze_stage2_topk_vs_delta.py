#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
from pathlib import Path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import math
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageDraw, ImageFont, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
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


# =========================================================
# model loading
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


def load_role_proto_from_stage3_ckpt(stage3_ckpt_path: str, role_proto_dir: str):
    names_path = os.path.join(role_proto_dir, "role_names.json")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Missing role names file: {names_path}")

    with open(names_path, "r", encoding="utf-8") as f:
        role_names = json.load(f)

    ckpt = torch.load(stage3_ckpt_path, map_location="cpu")
    sd = ckpt["stage3_state_dict"] if "stage3_state_dict" in ckpt else ckpt

    key = "shared_role_proto.prototypes"
    if key not in sd:
        raise KeyError(f"{key} not found in stage3 checkpoint")

    protos = sd[key].detach().cpu().numpy().astype(np.float32)
    protos = l2_normalize_np(protos)
    return protos, role_names


# =========================================================
# dataset
# =========================================================
def build_transform(image_size: int = 224):
    return T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


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
                    idx = rng.choice(len(sub), size=sample_per_project, replace=False)
                    sub = sub.iloc[idx].copy()
                parts.append(sub)
            df = pd.concat(parts, axis=0).reset_index(drop=True)

        if sample_per_label is not None and "pred_label" in df.columns:
            parts = []
            for _, sub in df.groupby("pred_label"):
                if len(sub) > sample_per_label:
                    idx = rng.choice(len(sub), size=sample_per_label, replace=False)
                    sub = sub.iloc[idx].copy()
                parts.append(sub)
            df = pd.concat(parts, axis=0).reset_index(drop=True)

        if max_rows is not None and len(df) > max_rows:
            idx = rng.choice(len(df), size=max_rows, replace=False)
            df = df.iloc[idx].copy().reset_index(drop=True)

        self.df = df.reset_index(drop=True)
        self.transform = transform

        print(f"[Dataset] rows = {len(self.df)}")
        if "slide_label" in self.df.columns:
            print("[Dataset] slide_label counts:")
            print(self.df["slide_label"].value_counts())
        if "project" in self.df.columns:
            print("[Dataset] project counts:")
            print(self.df["project"].value_counts())

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
    return final_feats, gate_info_list, feature_dict, moe_feature_list


def get_last_dispatch_weight(gate_info, seq_len):
    dispatch = gate_info["dispatch_weight"]  # [B*seq_len, E]
    total_tokens, num_experts = dispatch.shape
    B = total_tokens // seq_len
    dispatch = dispatch.view(B, seq_len, num_experts)[:, 1:, :]
    return dispatch


@torch.no_grad()
def project_features_to_role_space(features: np.ndarray, proj_head, device="cpu", batch_size=4096):
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
# image helpers
# =========================================================
def read_patch_for_vis(
    svs_path: str,
    x: int,
    y: int,
    patch_level: int,
    patch_size: int,
    out_size: int = 224,
):
    slide = openslide.OpenSlide(svs_path)
    try:
        img = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
    finally:
        slide.close()
    if out_size is not None and img.size != (out_size, out_size):
        img = img.resize((out_size, out_size))
    return img


def draw_token_box(img: Image.Image, token_idx: int, grid_size: int = 14, color=(255, 0, 0), width: int = 2):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    cell_w = W / grid_size
    cell_h = H / grid_size
    r = token_idx // grid_size
    c = token_idx % grid_size
    x0 = int(round(c * cell_w))
    y0 = int(round(r * cell_h))
    x1 = int(round((c + 1) * cell_w))
    y1 = int(round((r + 1) * cell_h))
    for k in range(width):
        draw.rectangle([x0 + k, y0 + k, x1 - k, y1 - k], outline=color)
    return img


def add_caption_below(img: Image.Image, lines: List[str], font_size: int = 12, pad: int = 4):
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    draw_tmp = ImageDraw.Draw(Image.new("RGB", (10, 10), "white"))
    line_heights = []
    max_w = img.size[0]
    for line in lines:
        bbox = draw_tmp.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_w = max(max_w, w + 2 * pad)
        line_heights.append(h)

    text_h = sum(line_heights) + pad * (len(lines) + 1)
    canvas = Image.new("RGB", (max_w, img.size[1] + text_h), "white")
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    y = img.size[1] + pad
    for line, h in zip(lines, line_heights):
        draw.text((pad, y), line, fill=(0, 0, 0), font=font)
        y += h + pad
    return canvas


def make_montage(df: pd.DataFrame, save_path: str, title: str, max_items: int = 36, cols: int = 6, out_size: int = 224):
    if len(df) == 0:
        return

    df = df.head(max_items).copy()
    tiles = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Montage {title}", leave=False):
        try:
            img = read_patch_for_vis(
                row["svs_path"],
                int(row["coord_x"]),
                int(row["coord_y"]),
                int(row["patch_level"]),
                int(row["patch_size"]),
                out_size=out_size,
            )
            img = draw_token_box(img, int(row["token_idx"]), grid_size=14, color=(255, 0, 0), width=2)
            lines = [
                f"slide={str(row['slide_id'])[:42]}",
                f"coord=({int(row['coord_x'])},{int(row['coord_y'])}) tok={int(row['token_idx'])}",
                f"base={float(row['stage2_tumor']):.3f} cur={float(row['stage3_tumor']):.3f}",
                f"delta={float(row['delta_tumor']):+.3f}",
            ]
            tile = add_caption_below(img, lines)
            tiles.append(tile)
        except Exception as e:
            print(f"[Warn] failed to make tile: {e}")

    if len(tiles) == 0:
        return

    cols = min(cols, len(tiles))
    rows = int(math.ceil(len(tiles) / cols))
    tile_w = max(t.size[0] for t in tiles)
    tile_h = max(t.size[1] for t in tiles)
    header_h = 60

    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 28)
    except Exception:
        font_title = ImageFont.load_default()
    draw.text((20, 15), title, fill=(0, 0, 0), font=font_title)

    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        canvas.paste(tile, (c * tile_w, header_h + r * tile_h))

    canvas.save(save_path)
    print(f"[Saved] montage: {save_path}")


# =========================================================
# analysis
# =========================================================
def export_groups(df: pd.DataFrame, out_dir: str, prefix: str, topk: int, make_vis: bool, max_montage_items: int):
    ensure_dir(out_dir)

    groups = {
        "abs_topk_stage2_tumor": df.sort_values("stage2_tumor", ascending=False).head(topk).copy(),
        "abs_bottomk_stage2_tumor": df.sort_values("stage2_tumor", ascending=True).head(topk).copy(),
        "delta_up_tumor": df.sort_values("delta_tumor", ascending=False).head(topk).copy(),
        "delta_down_tumor": df.sort_values("delta_tumor", ascending=True).head(topk).copy(),
    }

    summary = []
    for name, gdf in groups.items():
        csv_path = os.path.join(out_dir, f"{prefix}_{name}.csv")
        gdf.to_csv(csv_path, index=False)

        summary.append({
            "group": name,
            "num_tokens": len(gdf),
            "stage2_mean": float(gdf["stage2_tumor"].mean()) if len(gdf) > 0 else 0.0,
            "stage3_mean": float(gdf["stage3_tumor"].mean()) if len(gdf) > 0 else 0.0,
            "delta_mean": float(gdf["delta_tumor"].mean()) if len(gdf) > 0 else 0.0,
        })

        if make_vis:
            png_path = os.path.join(out_dir, f"{prefix}_{name}.png")
            make_montage(
                gdf,
                save_path=png_path,
                title=f"{prefix}: {name}",
                max_items=max_montage_items,
                cols=6,
                out_size=224,
            )

    pd.DataFrame(summary).to_csv(os.path.join(out_dir, f"{prefix}_summary.csv"), index=False)


# =========================================================
# main
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Analyze stage2 absolute-topk vs delta-up using stage2+stage3 forward")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage2-full-ckpt", type=str, required=True)
    parser.add_argument("--stage3-ckpt", type=str, required=True)
    parser.add_argument("--role-proto-dir", type=str, required=True)
    parser.add_argument("--pool-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--split-csv", type=str, default=None)
    parser.add_argument("--split-keep", nargs="+", default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)

    parser.add_argument("--max-rows", type=int, default=400)
    parser.add_argument("--sample-per-project", type=int, default=None)
    parser.add_argument("--sample-per-label", type=int, default=None)
    parser.add_argument("--keep-projects", nargs="+", default=None)
    parser.add_argument("--keep-labels", nargs="+", default=None)

    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--make-montage", action="store_true")
    parser.add_argument("--max-montage-items", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    ensure_dir(args.output_dir)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    print("[Load] stage2 bundle ...")
    stage2_model, proj_l12, cfg = load_stage2_bundle(
        config_path=args.config,
        full_ckpt_path=args.stage2_full_ckpt,
        device=device,
    )

    print("[Load] stage2 proto ...")
    stage2_protos, role_names_stage2 = load_role_proto_from_dir(args.role_proto_dir)

    print("[Load] stage3 proto ...")
    stage3_protos, role_names_stage3 = load_role_proto_from_stage3_ckpt(
        args.stage3_ckpt,
        args.role_proto_dir,
    )

    if role_names_stage2 != role_names_stage3:
        raise ValueError("role names mismatch between stage2 init and stage3 ckpt")

    role_names = role_names_stage2
    tumor_role_id = role_names.index("tumor")

    print("[Dataset] loading ...")
    dataset = TCGAPoolDataset(
        csv_path=args.pool_csv,
        transform=build_transform(args.image_size),
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

    rows = []

    print("[Stage] running stage2 forward + score extraction ...")
    for images, metas in tqdm(loader, desc="Forward"):
        images = images.to(device, non_blocking=True)

        with torch.no_grad():
            final_feats, gate_info_list, feature_dict, moe_feature_list = run_model_and_collect(stage2_model, images)

        last_moe_feat = moe_feature_list[-1][:, 1:, :]  # [B, N, D_student]
        B, N, D = last_moe_feat.shape

        feat_student_np = last_moe_feat.detach().cpu().reshape(B * N, D).numpy().astype(np.float32)
        feat_role_np = project_features_to_role_space(feat_student_np, proj_l12, device=device)

        aff_stage2 = compute_role_affinity(feat_role_np, stage2_protos)
        aff_stage3 = compute_role_affinity(feat_role_np, stage3_protos)

        stage2_tumor = aff_stage2[:, tumor_role_id]
        stage3_tumor = aff_stage3[:, tumor_role_id]

        idx = 0
        for b in range(B):
            meta = metas[b]
            for t in range(N):
                rows.append({
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
                    "stage2_tumor": float(stage2_tumor[idx]),
                    "stage3_tumor": float(stage3_tumor[idx]),
                    "delta_tumor": float(stage3_tumor[idx] - stage2_tumor[idx]),
                })
                idx += 1

    token_df = pd.DataFrame(rows)
    full_csv = os.path.join(args.output_dir, "token_level_stage2_vs_stage3.csv")
    token_df.to_csv(full_csv, index=False)
    print(f"[Saved] full token csv: {full_csv}")

    if "slide_label" not in token_df.columns:
        raise ValueError("token_df missing slide_label")

    pos_df = token_df[token_df["slide_label"].astype(int) == 1].copy()
    neg_df = token_df[token_df["slide_label"].astype(int) == 0].copy()

    print(f"[Info] total tokens = {len(token_df)}")
    print(f"[Info] positive tokens = {len(pos_df)}")
    print(f"[Info] negative tokens = {len(neg_df)}")

    export_groups(
        pos_df,
        out_dir=os.path.join(args.output_dir, "positive"),
        prefix="positive",
        topk=args.topk,
        make_vis=args.make_montage,
        max_montage_items=args.max_montage_items,
    )

    export_groups(
        neg_df,
        out_dir=os.path.join(args.output_dir, "negative"),
        prefix="negative",
        topk=args.topk,
        make_vis=args.make_montage,
        max_montage_items=args.max_montage_items,
    )

    print("[Done] analysis finished.")


if __name__ == "__main__":
    main()