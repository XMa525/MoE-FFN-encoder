#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Dict

import torch
import torch.nn as nn
from PIL import Image

import timm
from timm.layers import SwiGLUPacked
from timm.data.transforms_factory import create_transform

from models.encoders.dinov2_encoder import DINOv2Encoder

from models.encoders.virchow2_moe_encoder import Virchow2MoEEncoder
from models.encoders.timm_pathology_moe_encoder import (
    UNIEncoder,
    UNI2HEncoder,
    HOptimus0Encoder,
    UNIMoEEncoder,
    UNI2HMoEEncoder,
    HOptimus0MoEEncoder,
)
from models.encoders.openclip_moe_encoder import (
    OpenCLIPEncoder,
    OpenCLIPMoEEncoder,
)


# =========================================================
# Common state-dict helpers
# =========================================================
def _unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "encoder" in ckpt:
            return ckpt["encoder"]
        elif "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            return ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        else:
            return ckpt

    raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")


def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}

    for k, v in state_dict.items():
        nk = k

        # 常见包装前缀
        for prefix in [
            "module.",
            "model.",
            "student.",
            "student_model.",
            "encoder.",
            "backbone.",
            "base_encoder.",
            "base_encoder.model.",
        ]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]

        cleaned[nk] = v

    return cleaned


def _load_dinov2_finetuned_weight(enc: DINOv2Encoder, weight_path: str):
    if weight_path is None or weight_path == "":
        print("[DINOv2] no finetuned weight provided, using default pretrained DINOv2.")
        return

    ckpt = torch.load(weight_path, map_location="cpu")
    state_dict = _unwrap_state_dict(ckpt)

    cleaned = {}

    for k, v in state_dict.items():
        nk = k

        # remove only outer wrappers
        for prefix in [
            "module.",
            "model.",
            "student.",
            "student_model.",
            "encoder.",
            "backbone.",
            "base_encoder.",
        ]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]

        # Important:
        # train_finetune_baseline may export DINO blocks as:
        #   layer.0.xxx
        # but HuggingFace Dinov2Model expects:
        #   encoder.layer.0.xxx
        if nk.startswith("layer."):
            nk = "encoder." + nk

        # If exported as model.encoder.layer.xxx and the first "model." was removed,
        # it is already encoder.layer.xxx. Keep it unchanged.
        cleaned[nk] = v

    try:
        msg = enc.model.load_state_dict(cleaned, strict=True)
        print(f"[DINOv2] strict load success: {weight_path}")
        print(msg)
    except Exception as e:
        print(f"[DINOv2] strict load failed, fallback strict=False: {e}")
        msg = enc.model.load_state_dict(cleaned, strict=False)
        print(f"[DINOv2] loaded weight: {weight_path}")
        print(msg)

    # Useful sanity check
    matched = 0
    model_keys = set(enc.model.state_dict().keys())
    for k in cleaned.keys():
        if k in model_keys:
            matched += 1
    print(f"[DINOv2] matched keys: {matched} / {len(cleaned)}")

# =========================================================
# Frozen DINOv2
# =========================================================
class FrozenDINOv2FeatureExtractor(nn.Module):
    """
    DINOv2-small feature extractor.

    Output:
        CLS + mean patch token
        For DINOv2-small: 384 * 2 = 768
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        weight_path: str = "",
        device: str = "cuda",
        cache_dir: str = "./pretrained_models",
    ):
        super().__init__()

        self.encoder = DINOv2Encoder(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
        )

        self.device = self.encoder.device

        if weight_path is not None and weight_path != "":
            _load_dinov2_finetuned_weight(self.encoder, weight_path)

        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.embed_dim = self.encoder.embed_dim
        self.reg_tokens = int(getattr(self.encoder, "reg_tokens", 0))
        self.out_dim = self.embed_dim * 2

        print(
            f"[FrozenDINOv2FeatureExtractor] loaded: "
            f"embed_dim={self.embed_dim}, reg_tokens={self.reg_tokens}, "
            f"out_dim={self.out_dim}"
        )

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image], feature_mode: str = "final") -> torch.Tensor:
        if feature_mode not in ["final", "layer_last", "layer_m4"]:
            raise ValueError(
                f"DINOv2 only supports feature_mode='final', 'layer_last', or 'layer_m4', "
                f"got {feature_mode}"
            )

        if isinstance(images, Image.Image):
            images = [images]

        # 这里不用 self.encoder(images, return_tokens=True)，
        # 因为你当前 DINOv2Encoder.forward 可能会丢 batch 维。
        inputs = self.encoder.processor(
            images=[img.convert("RGB") for img in images],
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].to(self.device, non_blocking=True)

        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)

        x = self.encoder.embeddings(pixel_values)

        if x.dim() != 3:
            raise RuntimeError(
                f"[DINOv2] embeddings should be [B, N, D], got {tuple(x.shape)}"
            )

        feature_dict = {}

        for i, blk in enumerate(self.encoder.blocks):
            out = blk(x)
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out

            if i == len(self.encoder.blocks) - 4:
                feature_dict["layer_m4"] = x
            if i == len(self.encoder.blocks) - 1:
                feature_dict["layer_last"] = x

        x = self.encoder.norm(x)

        if x.dim() != 3:
            raise RuntimeError(
                f"[DINOv2] final tokens should be [B, N, D], got {tuple(x.shape)}"
            )

        if feature_mode == "final":
            return x
        elif feature_mode == "layer_last":
            if "layer_last" not in feature_dict:
                raise RuntimeError("DINOv2 feature_dict['layer_last'] not found.")
            return feature_dict["layer_last"]
        elif feature_mode == "layer_m4":
            if "layer_m4" not in feature_dict:
                raise RuntimeError("DINOv2 feature_dict['layer_m4'] not found.")
            return feature_dict["layer_m4"]

        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image], feature_mode: str = "final") -> torch.Tensor:
        tokens = self.forward_tokens(images, feature_mode=feature_mode)

        cls = tokens[:, 0, :]
        patch_start = 1 + self.reg_tokens

        if patch_start >= tokens.shape[1]:
            patch_start = 1

        patch_mean = tokens[:, patch_start:, :].mean(dim=1)
        feat = torch.cat([cls, patch_mean], dim=-1)

        return feat


# =========================================================
# Frozen Virchow2
# =========================================================
class FrozenVirchow2FeatureExtractor(nn.Module):
    def __init__(self, weight_path: str, device: str = "cuda"):
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
        state_dict = _unwrap_state_dict(state_dict)
        new_state_dict = _clean_state_dict_keys(state_dict)

        try:
            self.model.load_state_dict(new_state_dict, strict=True)
            print(f"[FrozenVirchow2] strict load success: {weight_path}")
        except Exception as e:
            print(f"[FrozenVirchow2] strict load failed, fallback strict=False: {e}")
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"[FrozenVirchow2] missing={missing[:20]}")
            print(f"[FrozenVirchow2] unexpected={unexpected[:20]}")

        if not hasattr(self.model, "pos_embed") or self.model.pos_embed is None:
            num_patches = self.model.patch_embed.num_patches + 1
            embed_dim = self.model.embed_dim
            self.model.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            print("[FrozenVirchow2] pos_embed was missing, created fallback pos_embed.")

        self.model = self.model.eval().to(self.device)
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
        self.reg_tokens = getattr(self.model, "reg_tokens", 0)
        self.out_dim = self.embed_dim * 2

        print(
            f"[FrozenVirchow2] loaded: embed_dim={self.embed_dim}, "
            f"reg_tokens={self.reg_tokens}, out_dim={self.out_dim}"
        )

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image], feature_mode: str = "final") -> torch.Tensor:
        if feature_mode not in ["final", "moe_last", "layer_last", "layer_m4"]:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        if isinstance(images, Image.Image):
            images = [images]

        x = torch.stack(
            [self.transforms(img.convert("RGB")) for img in images],
            dim=0,
        ).to(self.device, non_blocking=True)

        tokens = self.model.forward_features(x)

        if isinstance(tokens, dict):
            if "x" in tokens:
                tokens = tokens["x"]
            elif "tokens" in tokens:
                tokens = tokens["tokens"]
            elif "features" in tokens:
                tokens = tokens["features"]
            else:
                raise TypeError(
                    f"Unsupported Virchow2 forward_features dict keys: {tokens.keys()}"
                )

        return tokens

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image], feature_mode: str = "final") -> torch.Tensor:
        tokens = self.forward_tokens(images, feature_mode=feature_mode)

        cls = tokens[:, 0, :]
        patch_start = 1 + self.reg_tokens
        patch_mean = tokens[:, patch_start:, :].mean(dim=1)

        feat = torch.cat([cls, patch_mean], dim=-1)
        return feat


# =========================================================
# Virchow2 + MoE Adapter
# =========================================================
class Virchow2MoEFeatureExtractor(nn.Module):
    def __init__(
        self,
        virchow2_weight: str,
        stage2_ckpt: str,
        device: str = "cuda",
        target_block_1: int = 29,
        target_block_2: int = 30,
        source_stage2_layer_1: int = 9,
        source_stage2_layer_2: int = 10,
        adapter_dim: int = 384,
        adapter_hidden_dim: int = 1536,
        num_experts: int = 4,
        shared_expert: bool = True,
        routing_strategy: str = "proto_topany",
        top_k: int = 2,
        init_threshold: float = 0.0,
        min_experts: int = 1,
        max_experts: int = 2,
        gate_init_scale: float = 2.0,
        gate_noise_std: float = 0.02,
        shared_alpha: float = 0.05,
        use_routing_proj: bool = False,
        routing_metric: str = "cosine",
        freeze_backbone_except_moe: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_block_2 = target_block_2

        virchow2_cfg = {
            "weight_path": virchow2_weight,
            "device": str(self.device),
        }

        moe_cfg = {
            "moe_layers": [target_block_1, target_block_2],
            "adapter_dim": adapter_dim,
            "adapter_hidden_dim": adapter_hidden_dim,
            "num_experts": num_experts,
            "shared_expert": shared_expert,
            "routing_strategy": routing_strategy,
            "top_k": top_k,
            "init_threshold": init_threshold,
            "min_experts": min_experts,
            "max_experts": max_experts,
            "gate_init_scale": gate_init_scale,
            "gate_noise_std": gate_noise_std,
            "shared_alpha": shared_alpha,
            "use_routing_proj": use_routing_proj,
            "routing_metric": routing_metric,
            "bridge_mode": "direct",
        }

        self.model = Virchow2MoEEncoder(virchow2_cfg, moe_cfg).to(self.device)
        self.model.load_stage2_moe_from_ckpt(
            stage2_ckpt_path=stage2_ckpt,
            target_to_source_layer_map={
                target_block_1: source_stage2_layer_1,
                target_block_2: source_stage2_layer_2,
            },
            strict=False,
        )

        if freeze_backbone_except_moe:
            for p in self.model.parameters():
                p.requires_grad = False
            for idx in self.model.moe_layer_map.keys():
                for p in self.model.blocks[idx].mlp.parameters():
                    p.requires_grad = True

        self.model.eval()

        self.embed_dim = self.model.embed_dim
        self.reg_tokens = getattr(self.model, "reg_tokens", 0)
        self.out_dim = self.embed_dim * 2

        print(
            f"[Virchow2MoEFeatureExtractor] loaded: embed_dim={self.embed_dim}, "
            f"reg_tokens={self.reg_tokens}, out_dim={self.out_dim}"
        )

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image], feature_mode: str = "final") -> torch.Tensor:
        if feature_mode == "final":
            return self.model(images, return_gates=False, is_eval=True)

        elif feature_mode in ["moe_last", "layer_last", "layer_m4"]:
            try:
                x_out, feature_dict, moe_feature_list = self.model(
                    images,
                    return_gates=False,
                    is_eval=True,
                    return_features=True,
                )
            except TypeError:
                raise RuntimeError(
                    "Virchow2MoEEncoder does not currently support return_features=True. "
                    "Please add the same intermediate-output logic as in TimmPathologyMoEEncoder."
                )

            if feature_mode == "moe_last":
                if len(moe_feature_list) == 0:
                    raise RuntimeError("Requested feature_mode='moe_last' but moe_feature_list is empty.")
                return moe_feature_list[-1]

            elif feature_mode == "layer_last":
                if "layer_last" not in feature_dict:
                    raise RuntimeError("Requested feature_mode='layer_last' but feature_dict['layer_last'] not found.")
                return feature_dict["layer_last"]

            elif feature_mode == "layer_m4":
                if "layer_m4" not in feature_dict:
                    raise RuntimeError("Requested feature_mode='layer_m4' but feature_dict['layer_m4'] not found.")
                return feature_dict["layer_m4"]

        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image], feature_mode: str = "final") -> torch.Tensor:
        tokens = self.forward_tokens(images, feature_mode=feature_mode)

        cls = tokens[:, 0, :]
        patch_start = 1 + self.reg_tokens
        patch_mean = tokens[:, patch_start:, :].mean(dim=1)

        feat = torch.cat([cls, patch_mean], dim=-1)
        return feat


# =========================================================
# Generic frozen extractor for timm pathology encoders
# =========================================================
class FrozenTimmPathologyFeatureExtractor(nn.Module):
    def __init__(self, encoder: nn.Module, target_block_2: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.device = encoder.device
        self.embed_dim = encoder.embed_dim
        self.reg_tokens = encoder.reg_tokens
        self.out_dim = self.embed_dim * 2
        self.target_block_2 = target_block_2

        print(
            f"[FrozenTimmPathologyFeatureExtractor] loaded: "
            f"embed_dim={self.embed_dim}, reg_tokens={self.reg_tokens}, "
            f"out_dim={self.out_dim}, target_block_2={self.target_block_2}"
        )

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image], feature_mode: str = "final") -> torch.Tensor:
        if feature_mode == "final":
            return self.encoder(images, return_tokens=True)

        elif feature_mode in ["moe_last", "layer_last", "layer_m4"]:
            tokens, feature_dict = self.encoder(images, return_tokens=True, return_features=True)

            if feature_mode == "moe_last":
                if self.target_block_2 is None:
                    raise RuntimeError("feature_mode='moe_last' requires target_block_2 for frozen extractor.")
                if "block_outputs" not in feature_dict or self.target_block_2 not in feature_dict["block_outputs"]:
                    raise RuntimeError(
                        f"Frozen encoder missing block output for target_block_2={self.target_block_2}."
                    )
                return feature_dict["block_outputs"][self.target_block_2]

            elif feature_mode == "layer_last":
                if "layer_last" not in feature_dict:
                    raise RuntimeError("Frozen encoder missing feature_dict['layer_last'].")
                return feature_dict["layer_last"]

            elif feature_mode == "layer_m4":
                if "layer_m4" not in feature_dict:
                    raise RuntimeError("Frozen encoder missing feature_dict['layer_m4'].")
                return feature_dict["layer_m4"]

        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image], feature_mode: str = "final") -> torch.Tensor:
        tokens = self.forward_tokens(images, feature_mode=feature_mode)

        cls = tokens[:, 0, :]
        patch_start = 1 + self.reg_tokens
        patch_mean = tokens[:, patch_start:, :].mean(dim=1)

        feat = torch.cat([cls, patch_mean], dim=-1)
        return feat


# =========================================================
# Generic direct-bridge MoE extractor
# =========================================================
class DirectBridgeMoEFeatureExtractor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        freeze_backbone_except_moe: bool = True,
    ):
        super().__init__()
        self.model = model
        self.device = model.device
        self.embed_dim = model.embed_dim
        self.reg_tokens = model.reg_tokens
        self.out_dim = self.embed_dim * 2

        if freeze_backbone_except_moe:
            for p in self.model.parameters():
                p.requires_grad = False
            for idx in self.model.moe_layer_map.keys():
                for p in self.model.blocks[idx].mlp.parameters():
                    p.requires_grad = True

        self.model.eval()

        print(
            f"[DirectBridgeMoEFeatureExtractor] loaded: embed_dim={self.embed_dim}, "
            f"reg_tokens={self.reg_tokens}, out_dim={self.out_dim}, "
            f"freeze_backbone_except_moe={freeze_backbone_except_moe}, "
            f"moe_blocks={sorted(list(self.model.moe_layer_map.keys()))}"
        )

    @torch.no_grad()
    def forward_tokens(self, images: List[Image.Image], feature_mode: str = "final") -> torch.Tensor:
        if feature_mode == "final":
            x_out = self.model(images, return_gates=False, is_eval=True)
            return x_out

        elif feature_mode == "moe_last":
            x_out, feature_dict, moe_feature_list = self.model(
                images,
                return_gates=False,
                is_eval=True,
                return_features=True,
            )
            if len(moe_feature_list) == 0:
                raise RuntimeError("Requested feature_mode='moe_last' but moe_feature_list is empty.")
            return moe_feature_list[-1]

        elif feature_mode == "layer_last":
            x_out, feature_dict, moe_feature_list = self.model(
                images,
                return_gates=False,
                is_eval=True,
                return_features=True,
            )
            if "layer_last" not in feature_dict:
                raise RuntimeError("Requested feature_mode='layer_last' but feature_dict['layer_last'] not found.")
            return feature_dict["layer_last"]

        elif feature_mode == "layer_m4":
            x_out, feature_dict, moe_feature_list = self.model(
                images,
                return_gates=False,
                is_eval=True,
                return_features=True,
            )
            if "layer_m4" not in feature_dict:
                raise RuntimeError("Requested feature_mode='layer_m4' but feature_dict['layer_m4'] not found.")
            return feature_dict["layer_m4"]

        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image], feature_mode: str = "final") -> torch.Tensor:
        tokens = self.forward_tokens(images, feature_mode=feature_mode)

        cls = tokens[:, 0, :]
        patch_start = 1 + self.reg_tokens
        patch_mean = tokens[:, patch_start:, :].mean(dim=1)

        feat = torch.cat([cls, patch_mean], dim=-1)
        return feat


# =========================================================
# Helper for building shared MoE config
# =========================================================
def _build_moe_cfg(args):
    return {
        "moe_layers": [args.target_block_1, args.target_block_2],
        "adapter_dim": args.adapter_dim,
        "adapter_hidden_dim": args.adapter_hidden_dim,
        "num_experts": args.num_experts,
        "shared_expert": args.shared_expert,
        "routing_strategy": args.routing_strategy,
        "top_k": args.top_k,
        "init_threshold": args.init_threshold,
        "min_experts": args.min_experts,
        "max_experts": args.max_experts,
        "gate_init_scale": args.gate_init_scale,
        "gate_noise_std": args.gate_noise_std,
        "shared_alpha": args.shared_alpha,
        "use_routing_proj": args.use_routing_proj,
        "routing_metric": args.routing_metric,
    }


def _build_stage2_layer_map(args):
    return {
        args.target_block_1: args.source_stage2_layer_1,
        args.target_block_2: args.source_stage2_layer_2,
    }


def _build_stage1_block_map(args):
    src1 = getattr(args, "source_stage1_block_1", None)
    src2 = getattr(args, "source_stage1_block_2", None)

    if src1 is None:
        src1 = args.target_block_1
    if src2 is None:
        src2 = args.target_block_2

    return {
        args.target_block_1: src1,
        args.target_block_2: src2,
    }


def _get_stage1_ckpt(args):
    stage1_ckpt = getattr(args, "stage1_ckpt", None)
    if stage1_ckpt is None or stage1_ckpt == "":
        stage1_ckpt = args.stage2_ckpt
    return stage1_ckpt


# =========================================================
# Factory
# =========================================================
def build_feature_extractor(args):
    name = args.encoder_name.lower()

    if name in ["dinov2", "dinov2_small", "dinov2-s"]:
        return FrozenDINOv2FeatureExtractor(
            model_name=getattr(args, "dinov2_model_name", "facebook/dinov2-small"),
            weight_path=getattr(args, "dinov2_weight", ""),
            device=args.device,
            cache_dir=getattr(args, "dinov2_cache_dir", "./pretrained_models"),
        )

    elif name in ["openclip_b16", "openclip", "open_clip"]:
        return OpenCLIPEncoder(
            model_name=getattr(args, "openclip_model_name", "ViT-B-16"),
            weight_path=getattr(args, "openclip_weight", ""),
            device=args.device,
            precision=getattr(args, "openclip_precision", "fp16"),
            normalize=not getattr(args, "no_openclip_normalize", False),
        )

    elif name in ["openclip_b16_moe", "openclip_moe", "open_clip_moe"]:
        return OpenCLIPMoEEncoder(
            model_name=getattr(args, "openclip_model_name", "ViT-B-16"),
            weight_path=getattr(args, "openclip_weight", ""),
            stage2_ckpt=args.stage2_ckpt,
            device=args.device,
            target_block_1=args.target_block_1,
            target_block_2=args.target_block_2,
            source_stage2_layer_1=args.source_stage2_layer_1,
            source_stage2_layer_2=args.source_stage2_layer_2,
            adapter_dim=args.adapter_dim,
            adapter_hidden_dim=args.adapter_hidden_dim,
            num_experts=args.num_experts,
            shared_expert=args.shared_expert,
            routing_strategy=args.routing_strategy,
            top_k=args.top_k,
            init_threshold=args.init_threshold,
            min_experts=args.min_experts,
            max_experts=args.max_experts,
            gate_init_scale=args.gate_init_scale,
            gate_noise_std=args.gate_noise_std,
            shared_alpha=args.shared_alpha,
            use_routing_proj=args.use_routing_proj,
            routing_metric=args.routing_metric,
            freeze_backbone_except_moe=args.freeze_backbone_except_moe,
            precision=getattr(args, "openclip_precision", "fp16"),
            normalize=not getattr(args, "no_openclip_normalize", False),
        )

    elif name == "virchow2":
        return FrozenVirchow2FeatureExtractor(
            weight_path=args.virchow2_weight,
            device=args.device,
        )

    elif name == "virchow2_moe":
        return Virchow2MoEFeatureExtractor(
            virchow2_weight=args.virchow2_weight,
            stage2_ckpt=args.stage2_ckpt,
            device=args.device,
            target_block_1=args.target_block_1,
            target_block_2=args.target_block_2,
            source_stage2_layer_1=args.source_stage2_layer_1,
            source_stage2_layer_2=args.source_stage2_layer_2,
            adapter_dim=args.adapter_dim,
            adapter_hidden_dim=args.adapter_hidden_dim,
            num_experts=args.num_experts,
            shared_expert=args.shared_expert,
            routing_strategy=args.routing_strategy,
            top_k=args.top_k,
            init_threshold=args.init_threshold,
            min_experts=args.min_experts,
            max_experts=args.max_experts,
            gate_init_scale=args.gate_init_scale,
            gate_noise_std=args.gate_noise_std,
            shared_alpha=args.shared_alpha,
            use_routing_proj=args.use_routing_proj,
            routing_metric=args.routing_metric,
            freeze_backbone_except_moe=args.freeze_backbone_except_moe,
        )

    elif name == "uni":
        enc = UNIEncoder(
            weight_path=args.uni_weight,
            device=args.device,
        )
        return FrozenTimmPathologyFeatureExtractor(
            enc,
            target_block_2=args.target_block_2,
        )

    elif name == "uni_moe":
        model = UNIMoEEncoder(
            uni_cfg={
                "weight_path": args.uni_weight,
                "device": args.device,
            },
            moe_cfg=_build_moe_cfg(args),
        )

        model.load_stage2_moe_from_ckpt(
            stage2_ckpt_path=args.stage2_ckpt,
            target_to_source_layer_map=_build_stage2_layer_map(args),
            strict=False,
        )

        return DirectBridgeMoEFeatureExtractor(
            model=model,
            freeze_backbone_except_moe=args.freeze_backbone_except_moe,
        )

    elif name in ["uni_moe_stage1", "uni_stage1_moe"]:
        model = UNIMoEEncoder(
            uni_cfg={
                "weight_path": args.uni_weight,
                "device": args.device,
            },
            moe_cfg=_build_moe_cfg(args),
        )

        model.load_stage1_moe_from_ckpt(
            stage1_ckpt_path=_get_stage1_ckpt(args),
            target_to_source_block_map=_build_stage1_block_map(args),
            strict=False,
        )

        return DirectBridgeMoEFeatureExtractor(
            model=model,
            freeze_backbone_except_moe=args.freeze_backbone_except_moe,
        )

    elif name in ["uni_moe_random", "uni_random_moe"]:
        model = UNIMoEEncoder(
            uni_cfg={
                "weight_path": args.uni_weight,
                "device": args.device,
            },
            moe_cfg=_build_moe_cfg(args),
        )

        print(
            "[Random-MoE] No stage1/stage2 checkpoint loaded. "
            "MoE adapter remains randomly initialized."
        )
        print(
            f"[Random-MoE] target blocks = "
            f"{args.target_block_1}, {args.target_block_2}"
        )

        return DirectBridgeMoEFeatureExtractor(
            model=model,
            freeze_backbone_except_moe=args.freeze_backbone_except_moe,
        )

    elif name == "uni2_h":
        enc = UNI2HEncoder(
            weight_path=args.uni2_weight,
            device=args.device,
        )
        return FrozenTimmPathologyFeatureExtractor(
            enc,
            target_block_2=args.target_block_2,
        )

    elif name == "uni2_h_moe":
        model = UNI2HMoEEncoder(
            uni2_cfg={
                "weight_path": args.uni2_weight,
                "device": args.device,
            },
            moe_cfg=_build_moe_cfg(args),
        )

        model.load_stage2_moe_from_ckpt(
            stage2_ckpt_path=args.stage2_ckpt,
            target_to_source_layer_map=_build_stage2_layer_map(args),
            strict=False,
        )

        return DirectBridgeMoEFeatureExtractor(
            model=model,
            freeze_backbone_except_moe=args.freeze_backbone_except_moe,
        )

    elif name == "hoptimus0":
        enc = HOptimus0Encoder(
            device=args.device,
            local_hf_hub_id=(
                args.hopt_local_hf_hub_id
                if getattr(args, "hopt_local_hf_hub_id", "") != ""
                else None
            ),
            manual_arch_name=(
                args.hopt_manual_arch_name
                if getattr(args, "hopt_manual_arch_name", "") != ""
                else None
            ),
            manual_create_kwargs=None,
            weight_path=(
                args.hopt_weight
                if getattr(args, "hopt_weight", "") != ""
                else None
            ),
        )
        return FrozenTimmPathologyFeatureExtractor(
            enc,
            target_block_2=args.target_block_2,
        )

    elif name == "hoptimus0_moe":
        model = HOptimus0MoEEncoder(
            hopt_cfg={
                "device": args.device,
                "local_hf_hub_id": (
                    args.hopt_local_hf_hub_id
                    if getattr(args, "hopt_local_hf_hub_id", "") != ""
                    else None
                ),
                "manual_arch_name": (
                    args.hopt_manual_arch_name
                    if getattr(args, "hopt_manual_arch_name", "") != ""
                    else None
                ),
                "manual_create_kwargs": None,
                "weight_path": (
                    args.hopt_weight
                    if getattr(args, "hopt_weight", "") != ""
                    else None
                ),
            },
            moe_cfg=_build_moe_cfg(args),
        )

        model.load_stage2_moe_from_ckpt(
            stage2_ckpt_path=args.stage2_ckpt,
            target_to_source_layer_map=_build_stage2_layer_map(args),
            strict=False,
        )

        return DirectBridgeMoEFeatureExtractor(
            model=model,
            freeze_backbone_except_moe=args.freeze_backbone_except_moe,
        )

    else:
        raise ValueError(f"Unknown encoder_name: {args.encoder_name}")