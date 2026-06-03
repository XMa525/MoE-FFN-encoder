#!/usr/bin/env python3
import os
import csv
import math
import yaml
import h5py
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import openslide
import pandas as pd

from PIL import Image
from tqdm import tqdm

# ===== DINOv2 =====
from transformers import AutoModel, AutoImageProcessor

# ===== Virchow2 =====
import timm
from timm.layers import SwiGLUPacked
from timm.data.transforms_factory import create_transform

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def find_wsi_path(raw_dir: str, slide_id: str) -> str:
    candidates = [
        os.path.join(raw_dir, f"{slide_id}.tif"),
        os.path.join(raw_dir, f"{slide_id}.svs"),
        os.path.join(raw_dir, f"{slide_id}.ndpi"),
        os.path.join(raw_dir, f"{slide_id}.mrxs"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {raw_dir}")


def find_h5_path(h5_dir: str, slide_id: str) -> str:
    candidates = [os.path.join(h5_dir, f"{slide_id}.h5")]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"H5 not found for slide_id={slide_id} in {h5_dir}")


def read_coords_from_h5(h5_path: str) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
    return torch.from_numpy(coords).long()


def read_patch_from_wsi(
    slide: openslide.OpenSlide,
    coord_xy: Tuple[int, int],
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    patch = slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")
    return patch


def load_role_prototypes(role_proto_dir: str):
    proto_path = os.path.join(role_proto_dir, "role_prototypes_init.npy")
    names_path = os.path.join(role_proto_dir, "role_names.json")

    if not os.path.exists(proto_path):
        raise FileNotFoundError(f"Missing role prototype file: {proto_path}")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Missing role names file: {names_path}")

    protos = np.load(proto_path).astype(np.float32)
    protos = torch.from_numpy(protos)
    protos = F.normalize(protos, dim=-1)

    with open(names_path, "r", encoding="utf-8") as f:
        role_names = json.load(f)

    return protos, role_names


def load_proj_l12_from_distiller_ckpt(distiller_ckpt_path: str, in_dim: int, out_dim: int):
    ckpt = torch.load(distiller_ckpt_path, map_location="cpu")

    proj = nn.Linear(in_dim, out_dim)
    loaded = False

    if "distiller_state_dict" in ckpt:
        sd = ckpt["distiller_state_dict"]
        if "proj_l12.weight" in sd and "proj_l12.bias" in sd:
            proj.load_state_dict({
                "weight": sd["proj_l12.weight"],
                "bias": sd["proj_l12.bias"],
            })
            loaded = True

    if (not loaded) and ("proj_l12_state_dict" in ckpt):
        proj.load_state_dict(ckpt["proj_l12_state_dict"])
        loaded = True

    if not loaded:
        raise KeyError("proj_l12 not found in distiller checkpoint")

    proj.eval()
    for p in proj.parameters():
        p.requires_grad = False
    return proj


# =========================================================
# Encoders
# =========================================================
class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, local_model_path="./pretrained_models/dinov2-small", device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(local_model_path, local_files_only=True).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(local_model_path, local_files_only=True)

        self.embed_dim = self.model.config.hidden_size
        self.out_dim = self.embed_dim * 2

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        print(f"[DINOv2] embed_dim={self.embed_dim}, out_dim={self.out_dim}")

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image]) -> torch.Tensor:
        tokens = self.forward_tokens(images)
        cls = tokens[:, 0, :]
        patch_mean = tokens[:, 1:, :].mean(dim=1)
        return torch.cat([cls, patch_mean], dim=-1)


class Virchow2FeatureExtractor(nn.Module):
    def __init__(self, weight_path="models/distill_teacher/Virchow2/pytorch_model.bin", device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = timm.create_model(
            "vit_huge_patch14_224",
            pretrained=False,
            num_classes=0,
            reg_tokens=4,
            mlp_ratio=5.3375,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            init_values=1e-5,
        )

        state_dict = torch.load(weight_path, map_location="cpu")
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"[Virchow2] strict load failed, fallback strict=False: {e}")
            self.model.load_state_dict(state_dict, strict=False)

        if not hasattr(self.model, "pos_embed") or self.model.pos_embed is None:
            num_patches = self.model.patch_embed.num_patches + 1
            embed_dim = self.model.embed_dim
            self.model.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.transforms = create_transform(
            input_size=(3, 224, 224),
            interpolation="bicubic",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_pct=1.0,
        )

        self.embed_dim = self.model.embed_dim
        self.out_dim = self.embed_dim * 2
        print(f"[Virchow2] embed_dim={self.embed_dim}, out_dim={self.out_dim}")

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        x = torch.stack([self.transforms(img) for img in images]).to(self.device)
        tokens = self.model.forward_features(x)

        if isinstance(tokens, dict):
            if "x" in tokens:
                tokens = tokens["x"]
            elif "tokens" in tokens:
                tokens = tokens["tokens"]
            elif "features" in tokens:
                tokens = tokens["features"]
            else:
                raise TypeError(f"Unsupported Virchow2 forward_features dict keys: {tokens.keys()}")

        return tokens

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image]) -> torch.Tensor:
        tokens = self.forward_tokens(images)
        cls = tokens[:, 0, :]
        reg_tokens = getattr(self.model, "reg_tokens", 0)
        patch_start = 1 + reg_tokens
        patch_tokens = tokens[:, patch_start:, :]
        patch_mean = patch_tokens.mean(dim=1)
        return torch.cat([cls, patch_mean], dim=-1)


# =========================================================
# MoE Encoder
# =========================================================
def build_moe_encoder(config_path: str, ckpt_path: str, device: str):
    from models.encoders.moe_encoder import MoEEncoder

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, cfg


class MoEFeatureExtractor(nn.Module):
    """
    对 MoE encoder:
    - features: patch-level pooled feature (cls + patch_mean)
    - role_probs: patch-level role probabilities, aggregated from token-level role scores

    重要假设：
    1. MoEEncoder.forward(images, ...) 可以直接接收 Tensor[B,3,224,224]
    2. 当 return_features=True 时，返回 (..., feature_dict, moe_feature_list)
    3. moe_feature_list[-1] 是最后一个 MoE block 的 token 特征 [B, N, D]

    如果你的项目接口不同，只需要改 forward_tokens_and_roles() 里的 forward 部分。
    """
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        device="cuda",
        save_role_probs: bool = False,
        role_proto_dir: str = None,
        distiller_ckpt: str = None,
        role_temperature: float = 1.0,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, self.cfg = build_moe_encoder(config_path, ckpt_path, str(self.device))

        self.save_role_probs = save_role_probs
        self.role_temperature = role_temperature

        self.role_prototypes = None
        self.role_names = None
        self.proj_l12 = None

        self.student_dim = None
        self.out_dim = None

        # 统一给一个 224 输入的 transform
        self.transforms = create_transform(
            input_size=(3, 224, 224),
            interpolation="bicubic",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_pct=1.0,
        )

        if self.save_role_probs:
            if role_proto_dir is None or distiller_ckpt is None:
                raise ValueError("save_role_probs=True requires role_proto_dir and distiller_ckpt")

            role_prototypes, role_names = load_role_prototypes(role_proto_dir)
            self.role_prototypes = role_prototypes.to(self.device)
            self.role_names = role_names

            # 默认 student_dim=384；如果你的 MoE token dim 不是 384，请改这里
            student_dim = 384
            teacher_dim = self.role_prototypes.shape[1]

            self.proj_l12 = load_proj_l12_from_distiller_ckpt(
                distiller_ckpt_path=distiller_ckpt,
                in_dim=student_dim,
                out_dim=teacher_dim,
            ).to(self.device)
            self.student_dim = student_dim

        self.eval()
        print("[MoE] loaded.")

    @torch.no_grad()
    def forward_tokens_and_roles(self, images: List[Image.Image]):
        x = torch.stack([self.transforms(img) for img in images]).to(self.device)  # [B,3,224,224]

        # 这里按你项目当前分析脚本里用过的接口来写
        outputs = self.model(
            x,
            return_gates=True,
            return_features=True,
            is_eval=True,
        )

        if not isinstance(outputs, (tuple, list)) or len(outputs) < 4:
            raise RuntimeError(
                "Expected MoEEncoder forward to return (final_feats, gate_info_list, feature_dict, moe_feature_list). "
                "Please adapt this part to your real project forward interface."
            )

        final_feats, gate_info_list, feature_dict, moe_feature_list = outputs

        # 用最后一个 MoE block token 作为 feature/role 基础
        if moe_feature_list is None or len(moe_feature_list) == 0:
            raise RuntimeError("moe_feature_list is empty, cannot compute role_probs")

        tokens = moe_feature_list[-1]   # [B, N, D_student]
        patch_role_probs = None

        if self.save_role_probs:
            patch_tokens = tokens[:, 1:, :]   # [B, Nt, D_student]

            if patch_tokens.shape[-1] != self.student_dim:
                raise ValueError(
                    f"patch token dim {patch_tokens.shape[-1]} != proj_l12 input dim {self.student_dim}. "
                    "Please update student_dim in MoEFeatureExtractor.__init__."
                )

            proj_tokens = self.proj_l12(patch_tokens)          # [B, Nt, D_teacher]
            proj_tokens = F.normalize(proj_tokens, dim=-1)

            role_logits = torch.matmul(proj_tokens, self.role_prototypes.t())   # [B, Nt, R]
            role_probs_tok = torch.softmax(role_logits / self.role_temperature, dim=-1)
            patch_role_probs = role_probs_tok.mean(dim=1)   # [B, R]

        return tokens, patch_role_probs

    @torch.no_grad()
    def extract_features_and_roles(self, images: List[Image.Image]):
        tokens, patch_role_probs = self.forward_tokens_and_roles(images)

        cls = tokens[:, 0, :]
        patch_mean = tokens[:, 1:, :].mean(dim=1)
        feat = torch.cat([cls, patch_mean], dim=-1)

        if self.out_dim is None:
            self.out_dim = feat.shape[-1]

        return feat, patch_role_probs

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image]) -> torch.Tensor:
        feat, _ = self.extract_features_and_roles(images)
        return feat


# =========================================================
# Data reading
# =========================================================
def load_slides_csv(slides_csv: str, split: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(slides_csv)
    required_cols = {"slide_id", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"slides_csv missing columns: {missing}")

    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("slides_csv 需要包含 'label' 或 'slide_binary_label' 列")

    if split is not None:
        df = df[df["split"] == split].copy()

    return df.reset_index(drop=True)


def make_label_column_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("Need 'label' or 'slide_binary_label' in slides_csv.")
    return df


# =========================================================
# Main extraction
# =========================================================
def build_extractor(args):
    if args.encoder_name == "dinov2_small":
        extractor = DINOv2FeatureExtractor(
            local_model_path=args.dinov2_path,
            device=args.device,
        )
    elif args.encoder_name == "virchow2":
        extractor = Virchow2FeatureExtractor(
            weight_path=args.virchow2_weight,
            device=args.device,
        )
    elif args.encoder_name == "moe_stage1":
        extractor = MoEFeatureExtractor(
            config_path=args.moe_config,
            ckpt_path=args.moe_ckpt,
            device=args.device,
            save_role_probs=args.save_role_probs,
            role_proto_dir=args.role_proto_dir,
            distiller_ckpt=args.distiller_ckpt,
            role_temperature=args.role_temperature,
        )
    else:
        raise ValueError(f"Unknown encoder_name: {args.encoder_name}")
    return extractor


@torch.no_grad()
def extract_one_slide(
    extractor,
    slide_path: str,
    h5_path: str,
    label: int,
    slide_id: str,
    out_path: str,
    patch_size: int = 256,
    batch_size: int = 64,
    max_patches: Optional[int] = None,
):
    coords = read_coords_from_h5(h5_path)
    if max_patches is not None and coords.shape[0] > max_patches:
        perm = torch.randperm(coords.shape[0])[:max_patches]
        coords = coords[perm]

    slide = openslide.OpenSlide(slide_path)

    all_feats = []
    all_coords = []
    all_role_probs = []

    num_coords = coords.shape[0]
    for start in tqdm(range(0, num_coords, batch_size), desc=f"Extracting {slide_id}", leave=False):
        end = min(start + batch_size, num_coords)
        batch_coords = coords[start:end]

        batch_images = []
        for xy in batch_coords.tolist():
            img = read_patch_from_wsi(
                slide=slide,
                coord_xy=xy,
                patch_size=patch_size,
                read_level=0,
            )
            img = img.resize((224, 224), resample=Image.BICUBIC)
            batch_images.append(img)

        if hasattr(extractor, "extract_features_and_roles"):
            feats, role_probs = extractor.extract_features_and_roles(batch_images)
            feats = feats.cpu()
            role_probs = role_probs.cpu() if role_probs is not None else None
        else:
            feats = extractor.extract_features(batch_images).cpu()
            role_probs = None

        all_feats.append(feats)
        all_coords.append(batch_coords.cpu())

        if role_probs is not None:
            all_role_probs.append(role_probs)

    slide.close()

    features = torch.cat(all_feats, dim=0) if len(all_feats) > 0 else torch.empty(0)
    coords_out = torch.cat(all_coords, dim=0) if len(all_coords) > 0 else torch.empty(0, 2)
    role_probs_out = torch.cat(all_role_probs, dim=0) if len(all_role_probs) > 0 else None

    save_obj = {
        "features": features,
        "coords": coords_out,
        "label": int(label),
        "slide_id": slide_id,
        "num_instances": int(features.shape[0]),
        "feat_dim": int(features.shape[1]) if features.ndim == 2 and features.shape[0] > 0 else 0,
    }

    if role_probs_out is not None:
        save_obj["role_probs"] = role_probs_out   # [N, R]

    torch.save(save_obj, out_path)


def main():
    parser = argparse.ArgumentParser("Extract slide bag features from CLAM h5 + WSI")

    parser.add_argument("--slides_csv", type=str, required=True, help="CSV with slide_id / label / split")
    parser.add_argument("--raw_dir", type=str, required=True, help="Directory containing raw WSI files")
    parser.add_argument("--h5_dir", type=str, required=True, help="Directory containing CLAM patch h5 files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for bag feature .pt files")

    parser.add_argument("--encoder_name", type=str, required=True, choices=["dinov2_small", "virchow2", "moe_stage1"])
    parser.add_argument("--split", type=str, default=None, choices=[None, "train", "val", "test"], help="Optional split filter")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    # DINO
    parser.add_argument("--dinov2_path", type=str, default="./pretrained_models/dinov2-small")

    # Virchow2
    parser.add_argument("--virchow2_weight", type=str, default="models/distill_teacher/Virchow2/pytorch_model.bin")

    # MoE
    parser.add_argument("--moe_config", type=str, default="configs/phase2.yaml")
    parser.add_argument("--moe_ckpt", type=str, default="results/distilled_best_model/moe_encoder_best.pth")

    # Role probs saving
    parser.add_argument("--save_role_probs", action="store_true", help="Save patch-level role probabilities into each bag .pt")
    parser.add_argument("--role_proto_dir", type=str, default=None)
    parser.add_argument("--distiller_ckpt", type=str, default=None)
    parser.add_argument("--role_temperature", type=float, default=1.0)

    args = parser.parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    df = load_slides_csv(args.slides_csv, split=args.split)
    df = make_label_column_if_needed(df)

    extractor = build_extractor(args)

    meta_path = os.path.join(args.out_dir, "feature_meta.csv")
    meta_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extract {args.encoder_name}"):
        slide_id = row["slide_id"]
        label = int(row["label"])

        out_path = os.path.join(args.out_dir, f"{slide_id}.pt")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        slide_path = find_wsi_path(args.raw_dir, slide_id)
        h5_path = find_h5_path(args.h5_dir, slide_id)

        try:
            extract_one_slide(
                extractor=extractor,
                slide_path=slide_path,
                h5_path=h5_path,
                label=label,
                slide_id=slide_id,
                out_path=out_path,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                max_patches=args.max_patches,
            )

            obj = torch.load(out_path, map_location="cpu")
            meta_rows.append({
                "slide_id": slide_id,
                "label": label,
                "num_instances": obj["num_instances"],
                "feat_dim": obj["feat_dim"],
                "has_role_probs": int("role_probs" in obj),
                "out_path": out_path,
            })

        except Exception as e:
            print(f"[ERROR] slide_id={slide_id}: {e}")

    if len(meta_rows) > 0:
        pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
        print(f"[Saved meta] {meta_path}")

    print("Done.")


if __name__ == "__main__":
    main()
