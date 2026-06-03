import os
import csv
import math
import yaml
import h5py
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
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


from pathlib import Path

def find_wsi_path(raw_dir: str, slide_id: str) -> str:
    """
    在 raw_dir 下递归查找 slide_id 对应的 WSI。
    优先精确匹配 slide_id.ext，再做模糊匹配 slide_id*.ext
    """
    raw_dir = Path(raw_dir)
    exts = [".tif", ".tiff", ".svs", ".ndpi", ".mrxs"]

    exact_matches = []
    for ext in exts:
        exact_matches.extend(raw_dir.rglob(f"{slide_id}{ext}"))

    if len(exact_matches) == 1:
        return str(exact_matches[0])
    elif len(exact_matches) > 1:
        raise RuntimeError(
            f"Found multiple exact WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in exact_matches[:10])
        )

    fuzzy_matches = []
    for ext in exts:
        fuzzy_matches.extend(raw_dir.rglob(f"{slide_id}*{ext}"))

    if len(fuzzy_matches) == 1:
        return str(fuzzy_matches[0])
    elif len(fuzzy_matches) > 1:
        exact_name = [p for p in fuzzy_matches if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])

        raise RuntimeError(
            f"Found multiple WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in fuzzy_matches[:10])
        )

    raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {raw_dir}")


def resolve_wsi_path(row: pd.Series, raw_dir: Optional[str] = None) -> str:
    """
    优先使用 csv 中的 source_path；
    如果 source_path 不存在，再 fallback 到 raw_dir + slide_id 搜索。
    """
    if "source_path" in row and pd.notna(row["source_path"]):
        source_path = str(row["source_path"])
        if os.path.exists(source_path):
            return source_path
        else:
            print(f"[WARN] source_path not found, fallback search: {source_path}")

    if raw_dir is not None:
        return find_wsi_path(raw_dir, str(row["slide_id"]))

    raise FileNotFoundError(
        f"Cannot resolve WSI path for slide_id={row['slide_id']}. "
        f"source_path missing/invalid and raw_dir not provided."
    )


def find_h5_path(h5_dir: str, slide_id: str) -> str:
    h5_dir = Path(h5_dir)

    exact = list(h5_dir.rglob(f"{slide_id}.h5"))
    if len(exact) == 1:
        return str(exact[0])
    elif len(exact) > 1:
        raise RuntimeError(
            f"Found multiple exact h5 files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in exact[:10])
        )

    fuzzy = list(h5_dir.rglob(f"{slide_id}*.h5"))
    if len(fuzzy) == 1:
        return str(fuzzy[0])
    elif len(fuzzy) > 1:
        exact_name = [p for p in fuzzy if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])

        raise RuntimeError(
            f"Found multiple fuzzy h5 files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in fuzzy[:10])
        )

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



# =========================================================
# Encoders
# =========================================================
class DINOv2FeatureExtractor(nn.Module):
    """
    输出 tokens: [B, N, D]
    后续统一做 cls_mean pooling
    """
    def __init__(self, local_model_path="./pretrained_models/dinov2-small", device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(
            local_model_path,
            local_files_only=True
        ).to(self.device)
        self.processor = AutoImageProcessor.from_pretrained(
            local_model_path,
            local_files_only=True
        )

        self.embeddings = self.model.embeddings
        self.blocks = self.model.encoder.layer
        self.norm = self.model.layernorm

        self.embed_dim = self.model.config.hidden_size
        self.out_dim = self.embed_dim * 2  # cls_mean

        self.eval()
        for p in self.parameters():
            p.requires_grad = False

        print(f"[DINOv2] embed_dim={self.embed_dim}, out_dim={self.out_dim}")

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        """
        images: List[PIL]
        return: [B, N, D]
        """
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state   # [B, N, D]

        return tokens

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image]) -> torch.Tensor:
        tokens = self.forward_tokens(images)

        if tokens.ndim != 3:
            raise ValueError(f"[DINOv2] expected tokens [B, N, D], got shape={tokens.shape}")

        cls = tokens[:, 0, :]
        patch_mean = tokens[:, 1:, :].mean(dim=1)
        feat = torch.cat([cls, patch_mean], dim=-1)
        #print(f"[DINO debug] tokens.shape = {tokens.shape}")
        return feat


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base(x)
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return base_out + delta


def get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_keywords: Tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    block_ids: Optional[List[int]] = None,
):
    replace_names = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(k in name for k in target_keywords):
            continue

        if block_ids is not None:
            keep = False
            for bid in block_ids:
                if f"blocks.{bid}." in name:
                    keep = True
                    break
            if not keep:
                continue

        replace_names.append(name)

    for name in replace_names:
        parent, child_name = get_parent_module(model, name)
        old = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(old, rank=rank, alpha=alpha, dropout=dropout))

    return replace_names


class Virchow2FeatureExtractor(nn.Module):
    """
    支持两种权重：
    1) 原始 Virchow2 .bin
    2) LoRA 训练得到的 best_encoder_state_dict.pt
    """
    def __init__(
        self,
        weight_path="models/distill_teacher/Virchow2/pytorch_model.bin",
        device="cuda",
        use_lora: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_last_k_blocks: int = 2,
        lora_targets: Tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
    ):
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
            init_values=1e-5
        )

        # 先注入 LoRA 结构，再加载 LoRA pt
        if use_lora:
            num_blocks = len(self.model.blocks)
            block_ids = list(range(max(0, num_blocks - lora_last_k_blocks), num_blocks))
            replaced = inject_lora(
                self.model,
                target_keywords=lora_targets,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                block_ids=block_ids,
            )
            print(f"[Virchow2-LoRA] injected {len(replaced)} layers")

        state_dict = torch.load(weight_path, map_location="cpu")

        # 兼容几种常见保存格式
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "encoder" in state_dict:
                state_dict = state_dict["encoder"]

        # 去掉可能的前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("model.", "")
            if k.startswith("module."):
                k = k[len("module."):]
            new_state_dict[k] = v
        state_dict = new_state_dict

        try:
            self.model.load_state_dict(state_dict, strict=True)
            print(f"[Virchow2] strict load success: {weight_path}")
        except Exception as e:
            print(f"[Virchow2] strict load failed, fallback strict=False: {e}")
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"[Virchow2] missing={missing}")
            print(f"[Virchow2] unexpected={unexpected}")

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
            crop_pct=1.0
        )

        self.embed_dim = self.model.embed_dim
        self.out_dim = self.embed_dim * 2

        print(f"[Virchow2] embed_dim={self.embed_dim}, out_dim={self.out_dim}, use_lora={use_lora}")

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

        if tokens.ndim != 3:
            raise ValueError(f"Virchow2 forward_features should return [B, N, D], but got shape {tokens.shape}")

        cls = tokens[:, 0, :]
        reg_tokens = getattr(self.model, "reg_tokens", 0)
        patch_start = 1 + reg_tokens
        patch_tokens = tokens[:, patch_start:, :]
        patch_mean = patch_tokens.mean(dim=1)

        feat = torch.cat([cls, patch_mean], dim=-1)
        return feat

# =========================================================
# MoE Encoder
# 你项目里需要把这里的 import 改成你真实路径
# =========================================================
def build_moe_encoder(config_path: str, ckpt_path: str, device: str):
    from models.encoders.moe_encoder import MoEEncoder

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 兼容几种常见保存格式
    if isinstance(ckpt, dict):
        if "student_state_dict" in ckpt:
            state_dict = ckpt["student_state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"[MoE] missing keys ({len(missing)}): {missing[:20]}")
    if len(unexpected) > 0:
        print(f"[MoE] unexpected keys ({len(unexpected)}): {unexpected[:20]}")

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, cfg

class MoEFeatureExtractor(nn.Module):
    """
    用和 DINOv2 / Virchow2 尽量一致的方式导出 MoE 特征：

    1) 明确调用 MoEEncoder 的真实 forward 接口
    2) 明确取某一层 token:
       - 默认优先 last MoE output
       - 否则用 feature_dict["layer_12"]
    3) 明确做 cls + patch_mean pooling
    """

    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        device="cuda",
        use_last_moe_output: bool = True,
        moe_token_source: str = "auto",   # "auto" | "last_moe" | "layer_12"
        image_size: int = 224,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, self.cfg = build_moe_encoder(config_path, ckpt_path, str(self.device))

        self.use_last_moe_output = bool(use_last_moe_output)
        self.moe_token_source = str(moe_token_source)
        self.image_size = int(image_size)

        self.transform = T.Compose([
            T.ToImage(),
            T.Resize((self.image_size, self.image_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
        ])

        self.embed_dim = None
        self.out_dim = None

        self.eval()
        print(
            f"[MoE] loaded. "
            f"use_last_moe_output={self.use_last_moe_output}, "
            f"moe_token_source={self.moe_token_source}"
        )

    @torch.no_grad()
    def _forward_model(self, x: torch.Tensor):
        """
        与你项目里 stage2 / transfer 脚本保持一致的 MoEEncoder 调用方式
        """
        out = self.model(
            x,
            return_gates=True,
            mask=None,
            is_eval=True,
            return_features=True,
            offline_cluster_ids=None,
        )

        if not isinstance(out, (tuple, list)) or len(out) != 4:
            raise RuntimeError(
                f"[MoE] Unexpected encoder output: type={type(out)}, len={len(out) if isinstance(out, (tuple, list)) else 'NA'}"
            )

        student_out, gate_info_list, feature_dict, moe_feature_list = out
        return student_out, gate_info_list, feature_dict, moe_feature_list

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        """
        images: List[PIL]
        return: [B, N, D]
        """
        x = torch.stack([self.transform(img) for img in images]).to(self.device, non_blocking=True)

        _, _, feature_dict, moe_feature_list = self._forward_model(x)

        # 明确控制 token 来源
        if self.moe_token_source == "last_moe":
            if len(moe_feature_list) == 0:
                raise RuntimeError("[MoE] moe_feature_list is empty, cannot use last_moe")
            feat_tokens = moe_feature_list[-1]

        elif self.moe_token_source == "layer_12":
            if "layer_12" not in feature_dict:
                raise KeyError(f"[MoE] 'layer_12' not found in feature_dict keys={list(feature_dict.keys())}")
            feat_tokens = feature_dict["layer_12"]

        elif self.moe_token_source == "auto":
            if self.use_last_moe_output and len(moe_feature_list) > 0:
                feat_tokens = moe_feature_list[-1]
            else:
                if "layer_12" not in feature_dict:
                    raise KeyError(f"[MoE] 'layer_12' not found in feature_dict keys={list(feature_dict.keys())}")
                feat_tokens = feature_dict["layer_12"]
        else:
            raise ValueError(f"[MoE] Unsupported moe_token_source={self.moe_token_source}")

        if feat_tokens.ndim != 3:
            raise ValueError(f"[MoE] expected tokens [B, N, D], got shape={tuple(feat_tokens.shape)}")

        return feat_tokens

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image]) -> torch.Tensor:
        """
        与 DINOv2 / Virchow2 一致：
            feat = concat(cls, patch_mean)
        """
        tokens = self.forward_tokens(images)   # [B, N, D]

        if tokens.shape[1] < 2:
            raise RuntimeError(f"[MoE] token length too short: shape={tuple(tokens.shape)}")

        cls = tokens[:, 0, :]
        patch_tokens = tokens[:, 1:, :]
        patch_mean = patch_tokens.mean(dim=1)

        feat = torch.cat([cls, patch_mean], dim=-1)

        if self.embed_dim is None:
            self.embed_dim = int(tokens.shape[-1])
        if self.out_dim is None:
            self.out_dim = int(feat.shape[-1])

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

    df = df.reset_index(drop=True)
    return df


def make_label_column_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """
    兼容你现有 slides_split.csv 字段：
    如果没有 label，但有 slide_binary_label，就复制成 label。
    """
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
            device=args.device
        )
    elif args.encoder_name == "virchow2":
        extractor = Virchow2FeatureExtractor(
            weight_path=args.virchow2_weight,
            device=args.device,
            use_lora=args.virchow2_use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_last_k_blocks=args.lora_last_k_blocks,
            lora_targets=tuple(args.lora_targets),
        )
    elif args.encoder_name == "moe_encoder":
        extractor = MoEFeatureExtractor(
            config_path=args.moe_config,
            ckpt_path=args.moe_ckpt,
            device=args.device,
            use_last_moe_output=args.moe_use_last_moe_output,
            moe_token_source=args.moe_token_source,
            image_size=224,
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

    num_coords = coords.shape[0]
    for start in tqdm(
        range(0, num_coords, batch_size),
        desc=f"Extracting {slide_id}",
        leave=False
    ):
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
            # 统一先读 256，再 resize 到 224
            img = img.resize((224, 224), resample=Image.BICUBIC)
            batch_images.append(img)

        feats = extractor.extract_features(batch_images).cpu()  # [B, D]

        all_feats.append(feats)
        all_coords.append(batch_coords.cpu())

    slide.close()

    features = torch.cat(all_feats, dim=0) if len(all_feats) > 0 else torch.empty(0)
    coords_out = torch.cat(all_coords, dim=0) if len(all_coords) > 0 else torch.empty(0, 2)

    save_obj = {
        "features": features,              # [N, D]
        "coords": coords_out,              # [N, 2]
        "label": int(label),
        "slide_id": slide_id,
        "num_instances": int(features.shape[0]),
        "feat_dim": int(features.shape[1]) if features.ndim == 2 and features.shape[0] > 0 else 0,
    }

    torch.save(save_obj, out_path)


def main():
    parser = argparse.ArgumentParser("Extract slide bag features from CLAM h5 + WSI")

    parser.add_argument("--slides_csv", type=str, required=True,
                        help="CSV with slide_id / label / split")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Directory containing raw WSI files")
    parser.add_argument("--h5_dir", type=str, required=True,
                        help="Directory containing CLAM patch h5 files")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for bag feature .pt files")

    parser.add_argument("--encoder_name", type=str, required=True,
                        choices=["dinov2_small", "virchow2", "moe_encoder"])
    parser.add_argument("--virchow2_use_lora", action="store_true",
                        help="Whether to build Virchow2 with LoRA structure before loading weights")

    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_last_k_blocks", type=int, default=2)
    parser.add_argument("--lora_targets", nargs="+", default=["qkv", "proj", "fc1", "fc2"])
    parser.add_argument("--split", type=str, default=None,
                        choices=[None, "train", "val", "test"], help="Optional split filter")
    parser.add_argument("--moe_use_last_moe_output", action="store_true",
                    help="For MoE encoder, prefer last moe output when moe_token_source=auto")

    parser.add_argument("--moe_token_source", type=str, default="layer_12",
                        choices=["auto", "last_moe", "layer_12"],
                        help="Which token source to use for MoE feature extraction")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    # DINO
    parser.add_argument("--dinov2_path", type=str, default="./pretrained_models/dinov2-small")

    # Virchow2
    parser.add_argument("--virchow2_weight", type=str,
                        default="models/distill_teacher/Virchow2/pytorch_model.bin")

    # MoE
    parser.add_argument("--moe_config", type=str, default="configs/phase2.yaml")
    parser.add_argument("--moe_ckpt", type=str, default="results/distilled_best_model/moe_encoder_best.pth")

    args = parser.parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    df = load_slides_csv(args.slides_csv, split=args.split)
    df = make_label_column_if_needed(df)

    extractor = build_extractor(args)

    # 可选：记录元信息
    meta_path = os.path.join(args.out_dir, "feature_meta.csv")
    meta_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extract {args.encoder_name}"):
        slide_id = row["slide_id"]
        label = int(row["label"])

        out_path = os.path.join(args.out_dir, f"{slide_id}.pt")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        try:
            slide_path = resolve_wsi_path(row, raw_dir=args.raw_dir)
            h5_path = find_h5_path(args.h5_dir, slide_id)

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
                "out_path": out_path
            })

        except Exception as e:
            print(f"[ERROR] slide_id={slide_id}: {e}")

    if len(meta_rows) > 0:
        pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
        print(f"[Saved meta] {meta_path}")

    print("Done.")


if __name__ == "__main__":
    main()