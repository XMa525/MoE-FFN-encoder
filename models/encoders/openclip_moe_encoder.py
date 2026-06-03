#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image

import open_clip

from models.encoders.moe_FFN import MoEFFN

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


# =========================================================
# Utils
# =========================================================
def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("model.", "")
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v
    return new_state_dict


def _load_openclip_local_state_dict(
    model: nn.Module,
    weight_path: str,
    model_name: str = "OpenCLIP",
):
    """
    Load OpenCLIP local checkpoint.

    Supported:
    - .safetensors
    - .bin
    - .pt
    - .pth

    Expected OpenCLIP HF file:
    - open_clip_model.safetensors
    """
    if weight_path is None or weight_path == "":
        raise ValueError(f"[{model_name}] weight_path is empty.")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"[{model_name}] local weight not found: {weight_path}")

    print(f"[{model_name}] loading local weight: {weight_path}")

    if weight_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError(
                "safetensors is required to load .safetensors weights. "
                "Please run: pip install safetensors"
            )
        state_dict = safe_load_file(weight_path, device="cpu")
    else:
        state_dict = torch.load(weight_path, map_location="cpu")

    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "encoder" in state_dict and isinstance(state_dict["encoder"], dict):
            state_dict = state_dict["encoder"]
        elif "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]

    if not isinstance(state_dict, dict):
        raise TypeError(f"[{model_name}] unsupported checkpoint format: {type(state_dict)}")

    state_dict = _clean_state_dict_keys(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[{model_name}] strict load success.")
    except Exception as e:
        print(f"[{model_name}] strict load failed, fallback strict=False: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[{model_name}] missing keys ({len(missing)}): {missing[:20]}")
        print(f"[{model_name}] unexpected keys ({len(unexpected)}): {unexpected[:20]}")


def _extract_stage2_layer_moe_state(
    ckpt_path: str,
    src_layer_idx: int,
) -> Dict[str, torch.Tensor]:
    """
    Extract one MoE layer from DINOv2 stage2 checkpoint.

    Returned keys are prefixed with:
        moe.xxx

    This matches OpenCLIPBridgeMoEFFN, whose MoE submodule is named `moe`.
    """
    if ckpt_path is None or ckpt_path == "":
        raise ValueError("stage2_ckpt is required for OpenCLIP-MoE.")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"stage2_ckpt not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "student_state_dict" in ckpt:
        student_state = ckpt["student_state_dict"]
    elif "state_dict" in ckpt:
        student_state = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        student_state = ckpt["model_state_dict"]
    else:
        student_state = ckpt

    candidate_prefixes = [
        f"blocks.{src_layer_idx}.mlp.",
        f"base_encoder.blocks.{src_layer_idx}.mlp.",
        f"base_encoder.model.encoder.layer.{src_layer_idx}.mlp.",
        f"student.blocks.{src_layer_idx}.mlp.",
        f"module.blocks.{src_layer_idx}.mlp.",
    ]

    prefix = None
    for p in candidate_prefixes:
        if any(k.startswith(p) for k in student_state.keys()):
            prefix = p
            break

    if prefix is None:
        example_keys = list(student_state.keys())[:50]
        raise KeyError(
            f"Cannot find MoE layer prefix for src_layer_idx={src_layer_idx}. "
            f"Example keys: {example_keys}"
        )

    out = {}
    for k, v in student_state.items():
        if k.startswith(prefix):
            sub_key = k[len(prefix):]
            out[f"moe.{sub_key}"] = v

    print(f"[Stage2->OpenCLIP] extracted {len(out)} tensors from prefix: {prefix}")
    return out


# =========================================================
# Frozen OpenCLIP
# =========================================================
class OpenCLIPEncoder(nn.Module):
    """
    Frozen OpenCLIP image encoder.

    This class is intentionally not a timm-style token encoder.
    It uses OpenCLIP's own visual encoder and returns image embedding.

    Output:
        ViT-B-16: [B, 512]
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        weight_path: str = "",
        device: str = "cuda",
        precision: str = "fp16",
        normalize: bool = True,
    ):
        super().__init__()

        if weight_path is None or weight_path == "":
            raise ValueError(
                "OpenCLIPEncoder requires local weight_path. "
                "Please provide --openclip_weight."
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.weight_path = weight_path
        self.precision = precision
        self.normalize = normalize

        print(f"[OpenCLIPEncoder] create architecture: {model_name}, pretrained=None")

        self.model, _, self.transforms = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None,
            device="cpu",
        )

        _load_openclip_local_state_dict(
            model=self.model,
            weight_path=weight_path,
            model_name="OpenCLIPEncoder",
        )

        self.model = self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.out_dim = getattr(self.model.visual, "output_dim", None)
        if self.out_dim is None:
            self.out_dim = getattr(self.model, "embed_dim", None)

        print(
            f"[OpenCLIPEncoder] loaded: model_name={model_name}, "
            f"out_dim={self.out_dim}, normalize={normalize}, precision={precision}"
        )

    def preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]

        x = torch.stack(
            [self.transforms(img.convert("RGB")) for img in images],
            dim=0,
        )
        return x.to(self.device, non_blocking=True)

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image]) -> torch.Tensor:
        x = self.preprocess_images(images)

        use_amp = self.device.type == "cuda" and self.precision in ["fp16", "bf16"]
        amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            feat = self.model.encode_image(x)

        feat = feat.float()

        if self.normalize:
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return feat


# =========================================================
# OpenCLIP Bridge MoE FFN
# =========================================================
class OpenCLIPBridgeMoEFFN(nn.Module):
    """
    Replace OpenCLIP visual transformer block MLP.

    OpenCLIP residual attention block normally uses token shape:
        [L, B, D]

    MoEFFN expects:
        [B, L, D]

    So this module converts:
        [L, B, D] -> [B, L, D] -> MoE -> [L, B, D]
    """

    def __init__(
        self,
        in_dim: int,
        moe_dim: int,
        moe_hidden_dim: int,
        moe_layer_idx_for_forward: int,
        moe_cfg: Dict,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.moe_dim = moe_dim
        self.moe_hidden_dim = moe_hidden_dim
        self.moe_layer_idx_for_forward = moe_layer_idx_for_forward

        self.proj_in = nn.Linear(in_dim, moe_dim)
        self.proj_out = nn.Linear(moe_dim, in_dim)

        self.moe = MoEFFN(
            dim=moe_dim,
            hidden_dim=moe_hidden_dim,
            num_experts=moe_cfg.get("num_experts", 4),
            num_layers=moe_cfg.get("num_layers", 2),
            shared_expert=moe_cfg.get("shared_expert", True),
            routing_strategy=moe_cfg.get("routing_strategy", "proto_topany"),
            top_k=moe_cfg.get("top_k", 2),
            init_threshold=moe_cfg.get("init_threshold", 0.0),
            min_experts=moe_cfg.get("min_experts", 1),
            max_experts=moe_cfg.get("max_experts", 2),
            gate_init_scale=moe_cfg.get("gate_init_scale", 2.0),
            gate_noise_std=moe_cfg.get("gate_noise_std", 0.02),
            shared_alpha=moe_cfg.get("shared_alpha", 0.05),
            use_cluster_bias=moe_cfg.get("use_cluster_bias", False),
            num_clusters=moe_cfg.get("num_clusters", None),
            cluster_bias_scale=moe_cfg.get("cluster_bias_scale", 1.0),
            background_cluster_ids=moe_cfg.get("background_cluster_ids", [5]),
            use_routing_proj=moe_cfg.get("use_routing_proj", False),
            routing_proj_dim=moe_cfg.get("routing_proj_dim", None),
            routing_proj_hidden_dim=moe_cfg.get("routing_proj_hidden_dim", None),
            routing_proj_dropout=moe_cfg.get("routing_proj_dropout", 0.0),
            routing_metric=moe_cfg.get("routing_metric", "cosine"),
            use_patch_guided_routing=moe_cfg.get("use_patch_guided_routing", False),
            patch_guided_mode=moe_cfg.get("patch_guided_mode", "none"),
            patch_context_alpha=moe_cfg.get("patch_context_alpha", 0.5),
            patch_summary_type=moe_cfg.get("patch_summary_type", "global"),
            patch_summary_kernel=moe_cfg.get("patch_summary_kernel", 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"OpenCLIPBridgeMoEFFN expects 3D tensor, got shape={x.shape}")

        # OpenCLIP ViT uses [L, B, D]
        # Under autocast, x may be fp16. MoEFFN aggregation/index_add is safer in fp32.
        orig_dtype = x.dtype
        x_bld = x.permute(1, 0, 2).contiguous()  # [B, L, D]

        # Disable autocast inside MoE bridge to avoid Half/Float mismatch in index_add_.
        if x_bld.device.type == "cuda":
            ctx = torch.autocast(device_type="cuda", enabled=False)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            x_bld_fp32 = x_bld.float()

            z = self.proj_in(x_bld_fp32)

            z_out = self.moe(
                z,
                layer_idx=self.moe_layer_idx_for_forward,
                return_gates=False,
                is_eval=True,
                offline_cluster_ids=None,
            )

            y_bld = self.proj_out(z_out)

        # Cast back to original dtype so the residual add in OpenCLIP block is consistent.
        y_bld = y_bld.to(dtype=orig_dtype)
        y = y_bld.permute(1, 0, 2).contiguous()  # [L, B, D]

        return y


# =========================================================
# OpenCLIP + MoE Adapter
# =========================================================
class OpenCLIPMoEEncoder(nn.Module):
    """
    OpenCLIP ViT + Direct Bridge MoE Adapter.

    Workflow:
    1. Create OpenCLIP architecture with pretrained=None.
    2. Load local OpenCLIP weights from --openclip_weight.
    3. Replace visual.transformer.resblocks[target_block].mlp with OpenCLIPBridgeMoEFFN.
    4. Load DINOv2-stage2 MoE weights into the bridge modules.
    5. Use model.encode_image(x) to obtain [B, D] image embeddings.

    For ViT-B-16:
        visual transformer depth = 12
        target_block_1=-3 -> block 9
        target_block_2=-2 -> block 10
        output dim = 512
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        weight_path: str = "",
        stage2_ckpt: str = "",
        device: str = "cuda",
        target_block_1: int = -3,
        target_block_2: int = -2,
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
        precision: str = "fp16",
        normalize: bool = True,
    ):
        super().__init__()

        if weight_path is None or weight_path == "":
            raise ValueError(
                "OpenCLIPMoEEncoder requires local weight_path. "
                "Please provide --openclip_weight."
            )
        if stage2_ckpt is None or stage2_ckpt == "":
            raise ValueError(
                "OpenCLIPMoEEncoder requires stage2_ckpt. "
                "Please provide --stage2_ckpt."
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.weight_path = weight_path
        self.stage2_ckpt = stage2_ckpt
        self.precision = precision
        self.normalize = normalize

        print(f"[OpenCLIPMoEEncoder] create architecture: {model_name}, pretrained=None")

        self.model, _, self.transforms = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None,
            device="cpu",
        )

        _load_openclip_local_state_dict(
            model=self.model,
            weight_path=weight_path,
            model_name="OpenCLIPMoEEncoder",
        )

        if not hasattr(self.model, "visual"):
            raise AttributeError("OpenCLIP model has no visual module.")
        if not hasattr(self.model.visual, "transformer"):
            raise AttributeError("OpenCLIP visual module has no transformer.")
        if not hasattr(self.model.visual.transformer, "resblocks"):
            raise AttributeError("OpenCLIP visual.transformer has no resblocks.")

        self.blocks = self.model.visual.transformer.resblocks
        depth = len(self.blocks)

        def resolve_idx(idx: int) -> int:
            return idx if idx >= 0 else depth + idx

        target_block_1 = resolve_idx(target_block_1)
        target_block_2 = resolve_idx(target_block_2)

        if target_block_1 < 0 or target_block_1 >= depth:
            raise ValueError(f"target_block_1 out of range: {target_block_1}, depth={depth}")
        if target_block_2 < 0 or target_block_2 >= depth:
            raise ValueError(f"target_block_2 out of range: {target_block_2}, depth={depth}")
        if target_block_1 == target_block_2:
            raise ValueError(f"target_block_1 and target_block_2 are the same: {target_block_1}")

        self.moe_layer_map = {
            target_block_1: 0,
            target_block_2: 1,
        }

        moe_cfg = {
            "num_layers": 2,
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
        }

        for block_idx, moe_layer_idx in self.moe_layer_map.items():
            block = self.blocks[block_idx]

            if not hasattr(block, "mlp"):
                raise AttributeError(f"OpenCLIP resblock {block_idx} has no mlp.")

            old_mlp = block.mlp

            # OpenCLIP ViT-B-16 old_mlp is usually Sequential:
            # OrderedDict([("c_fc", Linear(768, 3072)), ...])
            if hasattr(old_mlp, "c_fc"):
                in_dim = old_mlp.c_fc.in_features
            elif hasattr(old_mlp, "__getitem__") and hasattr(old_mlp[0], "in_features"):
                in_dim = old_mlp[0].in_features
            elif hasattr(self.model.visual, "width"):
                in_dim = self.model.visual.width
            else:
                raise RuntimeError(
                    f"Cannot infer OpenCLIP MLP input dim for block {block_idx}. "
                    f"old_mlp={old_mlp}"
                )

            print(
                f"[OpenCLIPMoEEncoder] replacing resblocks[{block_idx}].mlp "
                f"with OpenCLIPBridgeMoEFFN: {in_dim} -> {adapter_dim} -> {in_dim}"
            )

            block.mlp = OpenCLIPBridgeMoEFFN(
                in_dim=in_dim,
                moe_dim=adapter_dim,
                moe_hidden_dim=adapter_hidden_dim,
                moe_layer_idx_for_forward=moe_layer_idx,
                moe_cfg=moe_cfg,
            )

        target_to_source_layer_map = {
            target_block_1: source_stage2_layer_1,
            target_block_2: source_stage2_layer_2,
        }

        for target_block_idx, src_stage2_layer_idx in target_to_source_layer_map.items():
            layer_state = _extract_stage2_layer_moe_state(
                ckpt_path=stage2_ckpt,
                src_layer_idx=src_stage2_layer_idx,
            )

            msg = self.blocks[target_block_idx].mlp.load_state_dict(
                layer_state,
                strict=False,
            )
            print(
                f"[OpenCLIPMoEEncoder] loaded stage2 MoE -> "
                f"target_block={target_block_idx}, source_stage2_layer={src_stage2_layer_idx}"
            )
            print(msg)

        self.model = self.model.to(self.device)

        if freeze_backbone_except_moe:
            for p in self.model.parameters():
                p.requires_grad = False

            for block_idx in self.moe_layer_map.keys():
                for p in self.blocks[block_idx].mlp.parameters():
                    p.requires_grad = True

        self.model.eval()

        self.out_dim = getattr(self.model.visual, "output_dim", None)
        if self.out_dim is None:
            self.out_dim = getattr(self.model, "embed_dim", None)

        print(
            f"[OpenCLIPMoEEncoder] loaded: model_name={model_name}, "
            f"depth={depth}, moe_blocks={list(self.moe_layer_map.keys())}, "
            f"out_dim={self.out_dim}, normalize={normalize}, precision={precision}"
        )

    def preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        if isinstance(images, Image.Image):
            images = [images]

        x = torch.stack(
            [self.transforms(img.convert("RGB")) for img in images],
            dim=0,
        )
        return x.to(self.device, non_blocking=True)

    @torch.no_grad()
    def extract_features(self, images: List[Image.Image]) -> torch.Tensor:
        x = self.preprocess_images(images)

        use_amp = self.device.type == "cuda" and self.precision in ["fp16", "bf16"]
        amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            feat = self.model.encode_image(x)

        feat = feat.float()

        if self.normalize:
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)

        return feat