#!/usr/bin/env python3
from __future__ import annotations

import os
import h5py
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import openslide
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Optional legacy DINOv2 fallback used by the original extraction script.
# This keeps old commands such as --encoder_name dinov2_small --dinov2_path ... working
# without requiring the project-level backbone factory to implement the plain DINO branch.
try:
    from transformers import AutoModel, AutoImageProcessor
except Exception:
    AutoModel = None
    AutoImageProcessor = None


import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Unified factory for current backbones / MoE adapters.
try:
    from models.encoders.backbone_moe_factory import build_feature_extractor as build_factory_feature_extractor
except Exception as e:
    build_factory_feature_extractor = None
    _FACTORY_IMPORT_ERROR = e


# =========================================================
# Utilities
# =========================================================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


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
# Legacy plain DINOv2 extractor
# =========================================================
class LegacyDINOv2FeatureExtractor(nn.Module):
    """
    Plain DINOv2 feature extractor from the old script.

    Output feature format:
        concat(CLS token, mean patch token) -> [B, 2 * D]

    This branch is useful for old local HuggingFace snapshots, e.g.
        --encoder_name dinov2_small --dinov2_path ./pretrained_models/dinov2-small
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/dinov2-small",
        weight_path: str = "",
        device: str = "cuda",
        local_files_only: bool = False,
    ):
        super().__init__()
        if AutoModel is None or AutoImageProcessor is None:
            raise ImportError(
                "transformers is required for the legacy DINOv2 extractor. "
                "Install it with `pip install transformers`."
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

        if weight_path:
            state = torch.load(weight_path, map_location="cpu")
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                elif "state_dict" in state:
                    state = state["state_dict"]
                elif "encoder" in state:
                    state = state["encoder"]
            cleaned = {}
            for k, v in state.items():
                k = k.replace("module.", "")
                k = k.replace("model.", "")
                cleaned[k] = v
            missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
            print(f"[Legacy DINOv2] loaded weight_path={weight_path}")
            print(f"[Legacy DINOv2] missing={len(missing)}, unexpected={len(unexpected)}")

        self.model.to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )

        self.embed_dim = int(self.model.config.hidden_size)
        self.out_dim = self.embed_dim * 2

        for p in self.parameters():
            p.requires_grad = False

        print(
            f"[Legacy DINOv2] model={model_name_or_path}, "
            f"embed_dim={self.embed_dim}, out_dim={self.out_dim}, "
            f"local_files_only={local_files_only}"
        )

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)
        outputs = self.model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        return tokens

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image]) -> torch.Tensor:
        tokens = self.forward_tokens(images)
        if tokens.ndim != 3:
            raise ValueError(
                f"[Legacy DINOv2] expected tokens [B, N, D], got shape={tuple(tokens.shape)}"
            )
        cls = tokens[:, 0, :]
        patch_mean = tokens[:, 1:, :].mean(dim=1)
        return torch.cat([cls, patch_mean], dim=-1)


def build_extractor(args):
    """
    Merge policy:
    1. Keep the current factory path for all new backbones and MoE adapters.
    2. Add the old plain-DINO path back as a legacy-compatible branch.
    """
    legacy_dino_names = {"dinov2_small", "dinov2_legacy", "dinov2_plain"}

    if args.encoder_name in legacy_dino_names or getattr(args, "use_legacy_dino", False):
        model_path = getattr(args, "dinov2_path", "") or getattr(args, "dinov2_model_name", "facebook/dinov2-small")
        if model_path in ("", None):
            model_path = "facebook/dinov2-small"

        return LegacyDINOv2FeatureExtractor(
            model_name_or_path=model_path,
            weight_path=getattr(args, "dinov2_weight", ""),
            device=args.device,
            local_files_only=getattr(args, "dinov2_local_files_only", False),
        )

    if build_factory_feature_extractor is None:
        raise ImportError(
            "Cannot import models.encoders.backbone_moe_factory.build_feature_extractor. "
            "Use --encoder_name dinov2_small for the legacy DINO branch, or fix the factory import. "
            f"Original import error: {_FACTORY_IMPORT_ERROR}"
        )

    return build_factory_feature_extractor(args)



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
        leave=False,
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

        
            img = img.resize((224, 224), resample=Image.BICUBIC)
            batch_images.append(img)

        feats = extractor.extract_features(batch_images).cpu()

        all_feats.append(feats)
        all_coords.append(batch_coords.cpu())

    slide.close()

    features = torch.cat(all_feats, dim=0) if len(all_feats) > 0 else torch.empty(0)
    coords_out = torch.cat(all_coords, dim=0) if len(all_coords) > 0 else torch.empty(0, 2)

    save_obj = {
        "features": features,
        "coords": coords_out,
        "label": int(label),
        "slide_id": slide_id,
        "num_instances": int(features.shape[0]),
        "feat_dim": int(features.shape[1]) if features.ndim == 2 and features.shape[0] > 0 else 0,
    }

    torch.save(save_obj, out_path)


def main():
    parser = argparse.ArgumentParser("Extract slide bag features from CLAM h5 + WSI")

    parser.add_argument(
        "--slides_csv",
        type=str,
        required=True,
        help="CSV with slide_id / label / split",
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Directory containing raw WSI files",
    )
    parser.add_argument(
        "--h5_dir",
        type=str,
        required=True,
        help="Directory containing CLAM patch h5 files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for bag feature .pt files",
    )

    parser.add_argument(
        "--encoder_name",
        type=str,
        required=True,
        choices=[
            # 原始 DINOv2 / MoE
            "dinov2",
            "dinov2_small",      # legacy name from old script
            "dinov2_legacy",     # explicit legacy plain-DINO branch
            "dinov2_plain",      # alias for legacy plain-DINO branch
            "dinov2_moe",
            "moe_encoder",

            # OpenCLIP
            "openclip",
            "open_clip",
            "openclip_b16",
            "openclip_b16_moe",
            "openclip_moe",
            "open_clip_moe",

            # Pathology foundation models
            "virchow2",
            "virchow2_moe",
            "uni",
            "uni_moe",
            "uni_moe_stage1",
            "uni_stage1_moe",
            "uni_moe_random",
            "uni_random_moe",
            "uni2_h",
            "uni2_h_moe",
            "hoptimus0",
            "hoptimus0_moe",
        ],
    )

    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=[None, "train", "val", "test"],
        help="Optional split filter",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    # =========================
    # OpenCLIP
    # =========================
    parser.add_argument(
        "--openclip_model_name",
        type=str,
        default="ViT-B-16",
        help="OpenCLIP model architecture name, e.g. ViT-B-16",
    )
    parser.add_argument(
        "--openclip_weight",
        type=str,
        default="",
        help="Local OpenCLIP weight file, e.g. open_clip_model.safetensors",
    )
    parser.add_argument(
        "--openclip_precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="AMP precision for OpenCLIP feature extraction",
    )
    parser.add_argument(
        "--no_openclip_normalize",
        action="store_true",
        help="Disable L2 normalization for OpenCLIP image features.",
    )

    # =========================
    # Virchow2
    # =========================
    parser.add_argument(
        "--virchow2_weight",
        type=str,
        default="models/distill_teacher/Virchow2/pytorch_model.bin",
    )

    # =========================
    # UNI / UNI2-h
    # =========================
    parser.add_argument("--uni_weight", type=str, default="")
    parser.add_argument("--uni2_weight", type=str, default="")

    # =========================
    # H-optimus-0
    # 优先推荐用本地 HF snapshot 方式
    # =========================
    parser.add_argument("--hopt_local_hf_hub_id", type=str, default="")
    parser.add_argument("--hopt_manual_arch_name", type=str, default="")
    parser.add_argument("--hopt_weight", type=str, default="")

    # =========================
    # DINOv2 / original MoE encoder
    # =========================
    parser.add_argument(
        "--dinov2_model_name",
        type=str,
        default="facebook/dinov2-small",
        help="DINOv2 model name or local path",
    )
    parser.add_argument(
        "--dinov2_weight",
        type=str,
        default="",
        help="Optional local DINOv2 weight path",
    )
    parser.add_argument(
        "--dinov2_path",
        type=str,
        default="",
        help=(
            "Legacy alias for a local DINOv2 HuggingFace snapshot. "
            "Used with --encoder_name dinov2_small / dinov2_legacy."
        ),
    )
    parser.add_argument(
        "--use_legacy_dino",
        action="store_true",
        help=(
            "Force the old plain-DINO extraction branch instead of the project factory. "
            "Useful when the factory does not implement plain DINOv2."
        ),
    )
    parser.add_argument(
        "--dinov2_local_files_only",
        action="store_true",
        help="Load DINOv2 only from local files in the legacy branch.",
    )
    parser.add_argument(
        "--moe_ckpt",
        type=str,
        default="",
        help="Original MoEEncoder checkpoint path",
    )
    parser.add_argument(
        "--moe_config",
        type=str,
        default="",
        help="Optional config yaml for original MoEEncoder",
    )
    parser.add_argument(
        "--dino_feature_layer",
        type=int,
        default=-1,
        help="DINOv2 hidden layer index if supported",
    )
    parser.add_argument(
        "--moe_use_last_moe_output",
        action="store_true",
        help="Legacy compatibility flag for the original MoE encoder.",
    )
    parser.add_argument(
        "--moe_token_source",
        type=str,
        default="layer_12",
        choices=["auto", "last_moe", "layer_12"],
        help="Legacy compatibility option for original MoE feature extraction.",
    )

    # =========================
    # Shared direct-bridge MoE args
    # =========================
    parser.add_argument(
        "--stage2_ckpt",
        type=str,
        default="",
        help="Checkpoint containing trained DINO-stage2 MoE weights",
    )
    parser.add_argument(
        "--shared_alpha_override",
        type=float,
        default=None,
        help=(
            "Override shared_alpha after loading MoE checkpoint. "
            "Useful for inference-time shared expert sensitivity, e.g. 0.0, 0.05, 0.1."
        ),
    )
    parser.add_argument(
        "--target_block_1",
        type=int,
        default=-3,
        help="Target backbone block index for first injected MoE",
    )
    parser.add_argument(
        "--target_block_2",
        type=int,
        default=-2,
        help="Target backbone block index for second injected MoE",
    )

    parser.add_argument(
        "--source_stage2_layer_1",
        type=int,
        default=9,
        help="Source DINO-stage2 MoE layer index for target_block_1",
    )
    parser.add_argument(
        "--source_stage2_layer_2",
        type=int,
        default=10,
        help="Source DINO-stage2 MoE layer index for target_block_2",
    )
    parser.add_argument(
        "--source_stage1_block_1",
        type=int,
        default=None,
        help="Source block index in stage1 checkpoint for target_block_1. Default: target_block_1.",
    )

    parser.add_argument(
        "--source_stage1_block_2",
        type=int,
        default=None,
        help="Source block index in stage1 checkpoint for target_block_2. Default: target_block_2.",
    )
    parser.add_argument(
        "--adapter_dim",
        type=int,
        default=384,
        help="Bridge adapter latent dim",
    )
    parser.add_argument(
        "--adapter_hidden_dim",
        type=int,
        default=1536,
        help="Bridge MoE hidden dim",
    )

    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--shared_expert", action="store_true")
    parser.add_argument("--routing_strategy", type=str, default="proto_topany")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--init_threshold", type=float, default=0.0)
    parser.add_argument("--min_experts", type=int, default=1)
    parser.add_argument("--max_experts", type=int, default=2)
    parser.add_argument("--gate_init_scale", type=float, default=2.0)
    parser.add_argument("--gate_noise_std", type=float, default=0.02)
    parser.add_argument("--shared_alpha", type=float, default=0.05)
    parser.add_argument("--use_routing_proj", action="store_true")
    parser.add_argument("--routing_metric", type=str, default="cosine")
    parser.add_argument("--freeze_backbone_except_moe", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    df = load_slides_csv(args.slides_csv, split=args.split)
    df = make_label_column_if_needed(df)

    extractor = build_extractor(args)
    

    if args.shared_alpha_override is not None:
        target = extractor

        # 有些 factory 可能返回 wrapper，真正 encoder 在 .encoder / .model / .base_encoder 里
        if hasattr(target, "set_shared_alpha"):
            target.set_shared_alpha(args.shared_alpha_override)
        elif hasattr(target, "encoder") and hasattr(target.encoder, "set_shared_alpha"):
            target.encoder.set_shared_alpha(args.shared_alpha_override)
        elif hasattr(target, "model") and hasattr(target.model, "set_shared_alpha"):
            target.model.set_shared_alpha(args.shared_alpha_override)
        else:
            raise AttributeError(
                "Current extractor does not support set_shared_alpha(). "
                f"extractor type={type(extractor)}. "
                "Please check build_feature_extractor(args) return object."
            )

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