#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import timm
from timm.layers import SwiGLUPacked
from timm.data.transforms_factory import create_transform
from PIL import Image

from models.encoders.moe_FFN import MoEFFN


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


def _load_local_state_dict(model: nn.Module, weight_path: str, model_name: str):
    state_dict = torch.load(weight_path, map_location="cpu")

    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "encoder" in state_dict:
            state_dict = state_dict["encoder"]

    state_dict = _clean_state_dict_keys(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[{model_name}] strict load success: {weight_path}")
    except Exception as e:
        print(f"[{model_name}] strict load failed, fallback strict=False: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[{model_name}] missing keys ({len(missing)}): {missing[:20]}")
        print(f"[{model_name}] unexpected keys ({len(unexpected)}): {unexpected[:20]}")

    if not hasattr(model, "pos_embed") or model.pos_embed is None:
        num_patches = model.patch_embed.num_patches + 1
        embed_dim = model.embed_dim
        model.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        print(f"[{model_name}] pos_embed was missing, created fallback pos_embed.")


def _extract_stage2_layer_moe_state(
    ckpt_path: str,
    src_layer_idx: int,
) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    student_state = ckpt["student_state_dict"]

    candidate_prefixes = [
        f"blocks.{src_layer_idx}.mlp.",
        f"base_encoder.blocks.{src_layer_idx}.mlp.",
        f"base_encoder.model.encoder.layer.{src_layer_idx}.mlp.",
    ]

    prefix = None
    for p in candidate_prefixes:
        if any(k.startswith(p) for k in student_state.keys()):
            prefix = p
            break

    if prefix is None:
        raise KeyError(f"Cannot find MoE layer prefix for src_layer_idx={src_layer_idx}")

    out = {}
    for k, v in student_state.items():
        if k.startswith(prefix):
            sub_key = k[len(prefix):]
            out[f"moe.{sub_key}"] = v

    print(f"[Stage2->PathologyBackbone] extracted {len(out)} tensors from prefix: {prefix}")
    return out


def _unwrap_stage1_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise TypeError(f"Checkpoint should be dict, got {type(ckpt)}")

    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "encoder",
        "student_state_dict",
        "student",
        "model",
        "net",
        "module",
    ]

    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            print(f"[Stage1-MoE] use ckpt['{key}'] as state_dict, num_keys={len(state)}")
            return state

    if any(hasattr(v, "shape") for v in ckpt.values()):
        print(f"[Stage1-MoE] use checkpoint itself as raw state_dict, num_keys={len(ckpt)}")
        return ckpt

    raise KeyError(
        "Cannot find state_dict in stage1 checkpoint. "
        f"Top-level keys: {list(ckpt.keys())}"
    )


def _clean_stage1_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}

    prefixes = [
        "module.",
        "model.",
        "student.",
        "student_model.",
        "encoder.",
        "backbone.",
        "base_encoder.",
    ]

    for k, v in state_dict.items():
        new_k = k

        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
                    changed = True

        out[new_k] = v

    return out


def _extract_stage1_layer_moe_state(
    ckpt_path: str,
    src_block_idx: int,
) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = _unwrap_stage1_state_dict(ckpt)
    state = _clean_stage1_state_dict_keys(state)

    candidate_prefixes = [
        f"blocks.{src_block_idx}.mlp.",
        f"model.blocks.{src_block_idx}.mlp.",
        f"base_encoder.blocks.{src_block_idx}.mlp.",
    ]

    prefix = None
    for p in candidate_prefixes:
        if any(k.startswith(p) for k in state.keys()):
            prefix = p
            break

    if prefix is None:
        examples = [k for k in state.keys() if f"blocks.{src_block_idx}" in k][:30]
        raise KeyError(
            f"Cannot find stage1 MoE MLP prefix for src_block_idx={src_block_idx}. "
            f"Examples containing blocks.{src_block_idx}: {examples}"
        )

    out = {}
    for k, v in state.items():
        if not k.startswith(prefix):
            continue

        sub_key = k[len(prefix):]

        if (
            sub_key.startswith("experts.")
            or sub_key.startswith("shared_expert.")
            or sub_key.startswith("gate.")
        ):
            out[f"moe.{sub_key}"] = v

    if len(out) == 0:
        matched = [k for k in state.keys() if k.startswith(prefix)][:30]
        raise KeyError(
            f"Found prefix={prefix}, but no experts/shared_expert/gate tensors extracted. "
            f"Matched examples: {matched}"
        )

    print(
        f"[Stage1-MoE -> PathologyBackbone] extracted {len(out)} tensors "
        f"from prefix: {prefix}"
    )

    for i, k in enumerate(out.keys()):
        if i >= 10:
            break
        print(f"  [Stage1-MoE key] {k}: {tuple(out[k].shape)}")

    return out


# =========================================================
# Base timm ViT encoder
# =========================================================
class BaseTimmPathologyEncoder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        model_name: str = "Backbone",
        transform_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()

        self.patch_embed = self.model.patch_embed
        self.blocks = self.model.blocks
        self.norm = self.model.norm

        self.embed_dim = self.model.embed_dim
        self.num_layers = len(self.blocks)
        self.reg_tokens = getattr(self.model, "reg_tokens", 0)

        if transform_kwargs is None:
            transform_kwargs = dict(
                input_size=(3, 224, 224),
                interpolation="bicubic",
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                crop_pct=1.0,
            )
        self.transforms = create_transform(**transform_kwargs)

        print(f"[{model_name}] layers={self.num_layers}, embed_dim={self.embed_dim}, reg_tokens={self.reg_tokens}")

    def preprocess_images(self, images):
        if isinstance(images, Image.Image):
            images = [images]
        x = torch.stack([self.transforms(img) for img in images]).to(self.device)
        return x

    def patch_embed_forward(self, images_or_tensor):
        if isinstance(images_or_tensor, torch.Tensor):
            x = images_or_tensor.to(self.device)
        else:
            x = self.preprocess_images(images_or_tensor)

        x = self.model.patch_embed(x)

        if hasattr(self.model, "_pos_embed"):
            x = self.model._pos_embed(x)
        else:
            B = x.shape[0]
            tokens = []
            if getattr(self.model, "cls_token", None) is not None:
                cls_tok = self.model.cls_token.expand(B, -1, -1)
                tokens.append(cls_tok)
            if getattr(self.model, "reg_token", None) is not None:
                reg_tok = self.model.reg_token.expand(B, -1, -1)
                tokens.append(reg_tok)
            tokens.append(x)
            x = torch.cat(tokens, dim=1)

            if getattr(self.model, "pos_embed", None) is not None:
                x = x + self.model.pos_embed[:, : x.shape[1]]
            if hasattr(self.model, "pos_drop"):
                x = self.model.pos_drop(x)

        if hasattr(self.model, "patch_drop"):
            x = self.model.patch_drop(x)
        if hasattr(self.model, "norm_pre"):
            x = self.model.norm_pre(x)

        return x

    def forward(
        self,
        images,
        return_tokens: bool = False,
        return_features: bool = False,
    ):
        x = self.patch_embed_forward(images)

        feature_dict = {"block_outputs": {}}

        for i, blk in enumerate(self.blocks):
            x = blk(x)

            if return_features:
                feature_dict["block_outputs"][i] = x
                if i == len(self.blocks) - 4:
                    feature_dict["layer_m4"] = x
                if i == len(self.blocks) - 1:
                    feature_dict["layer_last"] = x

        x_out = self.norm(x)

        if return_tokens and return_features:
            return x_out, feature_dict
        elif return_tokens:
            return x_out
        elif return_features:
            return x_out[:, 0, :], feature_dict
        else:
            return x_out[:, 0, :]


# =========================================================
# UNI
# =========================================================
class UNIEncoder(BaseTimmPathologyEncoder):
    def __init__(
        self,
        weight_path: str,
        device: str = "cuda",
    ):
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        _load_local_state_dict(model, weight_path, "UNI")

        super().__init__(
            model=model,
            device=device,
            model_name="UNI",
            transform_kwargs=dict(
                input_size=(3, 224, 224),
                interpolation="bicubic",
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                crop_pct=1.0,
            ),
        )


# =========================================================
# UNI2-h
# =========================================================
class UNI2HEncoder(BaseTimmPathologyEncoder):
    def __init__(
        self,
        weight_path: str,
        device: str = "cuda",
    ):
        model = timm.create_model(
            "vit_giant_patch14_224",
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
        _load_local_state_dict(model, weight_path, "UNI2-h")

        super().__init__(
            model=model,
            device=device,
            model_name="UNI2-h",
            transform_kwargs=dict(
                input_size=(3, 224, 224),
                interpolation="bicubic",
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                crop_pct=1.0,
            ),
        )


# =========================================================
# H-optimus-0
# =========================================================
class HOptimus0Encoder(BaseTimmPathologyEncoder):
    def __init__(
        self,
        device: str = "cuda",
        local_hf_hub_id: Optional[str] = None,
        manual_arch_name: Optional[str] = None,
        manual_create_kwargs: Optional[Dict] = None,
        weight_path: Optional[str] = None,
    ):
        if local_hf_hub_id is not None:
            model = timm.create_model(
                local_hf_hub_id,
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=False,
                num_classes=0,
            )
        else:
            if manual_arch_name is None:
                raise ValueError("For H-optimus-0, provide either local_hf_hub_id or manual_arch_name.")
            if manual_create_kwargs is None:
                manual_create_kwargs = {}
            model = timm.create_model(
                manual_arch_name,
                pretrained=False,
                **manual_create_kwargs,
            )
            if weight_path is None:
                raise ValueError("manual_arch_name mode requires weight_path.")
            _load_local_state_dict(model, weight_path, "H-optimus-0")

        super().__init__(
            model=model,
            device=device,
            model_name="H-optimus-0",
            transform_kwargs=dict(
                input_size=(3, 224, 224),
                interpolation="bicubic",
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517),
                crop_pct=1.0,
            ),
        )



# =========================================================
# Direct BridgeMoEFFN
# =========================================================
class BridgeMoEFFN(nn.Module):
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

    def set_shared_alpha(self, shared_alpha: float):
        if not hasattr(self.moe, "shared_alpha"):
            raise AttributeError(
                "MoEFFN does not have attribute 'shared_alpha'. "
                "Please check models/encoders/moe_FFN.py."
            )

        old_alpha = self.moe.shared_alpha
        self.moe.shared_alpha = float(shared_alpha)

        print(
            f"[BridgeMoEFFN] layer_idx={self.moe_layer_idx_for_forward} "
            f"shared_alpha: {old_alpha} -> {self.moe.shared_alpha}"
        )

    def forward(
        self,
        x,
        return_gates: bool = False,
        is_eval: bool = False,
        offline_cluster_ids=None,
    ):
        z = self.proj_in(x)

        moe_out = self.moe(
            z,
            layer_idx=self.moe_layer_idx_for_forward,
            return_gates=return_gates,
            is_eval=is_eval,
            offline_cluster_ids=offline_cluster_ids,
        )

        if return_gates:
            z_out, gate_info = moe_out
        else:
            z_out = moe_out
            gate_info = None

        y = self.proj_out(z_out)

        if return_gates:
            return y, gate_info
        else:
            return y

# =========================================================
# Generic direct-bridge MoE encoder for timm pathology ViTs
# =========================================================
class TimmPathologyMoEEncoder(nn.Module):
    def __init__(
        self,
        base_encoder: BaseTimmPathologyEncoder,
        moe_cfg: Dict,
    ):
        super().__init__()

        self.base_encoder = base_encoder
        self.blocks = self.base_encoder.blocks
        self.norm = self.base_encoder.norm
        self.embed_dim = self.base_encoder.embed_dim
        self.device = self.base_encoder.device
        self.reg_tokens = self.base_encoder.reg_tokens

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.moe_layers_idx = moe_cfg.get("moe_layers", [-3, -2])
        self.num_layers = len(self.moe_layers_idx)

        self.adapter_dim = moe_cfg.get("adapter_dim", 384)
        self.adapter_hidden_dim = moe_cfg.get("adapter_hidden_dim", 1536)
        self.shared_alpha = float(moe_cfg.get("shared_alpha", 0.05))

        depth = len(self.blocks)
        self.moe_layer_map = {}

        bridge_cfg = dict(moe_cfg)
        bridge_cfg["num_layers"] = self.num_layers

        for moe_layer_idx, idx in enumerate(self.moe_layers_idx):
            real_idx = idx if idx >= 0 else depth + idx
            self.moe_layer_map[real_idx] = moe_layer_idx

            block = self.blocks[real_idx]
            in_dim = block.mlp.fc1.in_features

            print(
                f"[INFO] Replacing block {real_idx} MLP with BridgeMoEFFN: "
                f"{in_dim} -> {self.adapter_dim} -> {in_dim}"
            )

            block.mlp = BridgeMoEFFN(
                in_dim=in_dim,
                moe_dim=self.adapter_dim,
                moe_hidden_dim=self.adapter_hidden_dim,
                moe_layer_idx_for_forward=moe_layer_idx,
                moe_cfg=bridge_cfg,
            ).to(self.device)

    def load_stage2_moe_from_ckpt(
        self,
        stage2_ckpt_path: str,
        target_to_source_layer_map: Dict[int, int],
        strict: bool = False,
    ):
        for tgt_block_idx, src_stage2_layer_idx in target_to_source_layer_map.items():
            if tgt_block_idx not in self.moe_layer_map:
                raise ValueError(f"target block {tgt_block_idx} not in moe_layer_map={self.moe_layer_map}")

            layer_state = _extract_stage2_layer_moe_state(
                ckpt_path=stage2_ckpt_path,
                src_layer_idx=src_stage2_layer_idx,
            )

            msg = self.blocks[tgt_block_idx].mlp.load_state_dict(layer_state, strict=strict)
            print(f"[Load stage2 -> target block] tgt_block={tgt_block_idx}, src_stage2_layer={src_stage2_layer_idx}")
            print(msg)

    def load_stage1_moe_from_ckpt(
        self,
        stage1_ckpt_path: str,
        target_to_source_block_map: Dict[int, int],
        strict: bool = False,
    ):
        for tgt_block_idx, src_block_idx in target_to_source_block_map.items():
            if tgt_block_idx not in self.moe_layer_map:
                raise ValueError(
                    f"target block {tgt_block_idx} not in moe_layer_map={self.moe_layer_map}"
                )

            layer_state = _extract_stage1_layer_moe_state(
                ckpt_path=stage1_ckpt_path,
                src_block_idx=src_block_idx,
            )

            msg = self.blocks[tgt_block_idx].mlp.load_state_dict(layer_state, strict=strict)

            print(
                f"[Load stage1 -> target block] "
                f"tgt_block={tgt_block_idx}, src_block={src_block_idx}"
            )
            print(msg)

    def set_shared_alpha(self, shared_alpha: float):
        """
        Set shared_alpha for all inserted BridgeMoEFFN layers.
        This can be used after loading MoE checkpoints for inference-time
        hyper-parameter sensitivity.
        """
        n_updated = 0

        for block_idx in sorted(self.moe_layer_map.keys()):
            mlp = self.blocks[block_idx].mlp

            if not isinstance(mlp, BridgeMoEFFN):
                print(
                    f"[Warning] block {block_idx} mlp is not BridgeMoEFFN, "
                    f"got {type(mlp)}. Skip."
                )
                continue

            mlp.set_shared_alpha(shared_alpha)
            n_updated += 1

        self.shared_alpha = float(shared_alpha)

        print(
            f"[TimmPathologyMoEEncoder] updated shared_alpha={self.shared_alpha} "
            f"for {n_updated} MoE layers"
        )

    def _forward_moe_block(
        self,
        blk,
        x,
        moe_layer_idx,
        return_gates: bool = False,
        is_eval: bool = False,
        offline_cluster_ids=None,
    ):
        residual = x
        x_norm = blk.norm1(x)
        attn_out = blk.attn(x_norm)
        attn_out = blk.ls1(attn_out)
        x = residual + blk.drop_path1(attn_out)

        residual = x
        mlp_input = blk.norm2(x)

        if isinstance(blk.mlp, BridgeMoEFFN):
            mlp_out = blk.mlp(
                mlp_input,
                return_gates=return_gates,
                is_eval=is_eval,
                offline_cluster_ids=offline_cluster_ids,
            )
            if return_gates:
                mlp_out, gate_info = mlp_out
            else:
                gate_info = None
        else:
            mlp_out = blk.mlp(mlp_input)
            gate_info = None

        mlp_out = blk.ls2(mlp_out)
        x = residual + blk.drop_path2(mlp_out)

        return x, gate_info

    def forward(
        self,
        images,
        return_gates: bool = False,
        mask=None,
        is_eval: bool = False,
        return_features: bool = False,
        offline_cluster_ids=None,
    ):
        x = self.base_encoder.patch_embed_forward(images)

        gate_info_list = []
        feature_dict = {
            "block_outputs": {},
            "moe_block_indices": sorted(list(self.moe_layer_map.keys())),
        }
        moe_feature_list = []

        cls_plus_reg = 1 + self.reg_tokens
        prefix_tokens = x[:, :cls_plus_reg, :]
        patch_tokens = x[:, cls_plus_reg:, :]

        if mask is not None:
            B, N, D = patch_tokens.shape
            mask_expanded = mask.unsqueeze(-1).to(patch_tokens.device)
            mask_token = self.mask_token.expand(B, N, -1)
            patch_tokens = torch.where(mask_expanded, mask_token, patch_tokens)

        x = torch.cat([prefix_tokens, patch_tokens], dim=1)

        for i, blk in enumerate(self.blocks):
            if i in self.moe_layer_map:
                moe_layer_idx = self.moe_layer_map[i]
                x, gate_info = self._forward_moe_block(
                    blk,
                    x,
                    moe_layer_idx=moe_layer_idx,
                    return_gates=return_gates,
                    is_eval=is_eval,
                    offline_cluster_ids=offline_cluster_ids,
                )
                if return_gates:
                    gate_info_list.append(gate_info)
                if return_features:
                    moe_feature_list.append(x)
            else:
                x = blk(x)

            if return_features:
                feature_dict["block_outputs"][i] = x
                if i == len(self.blocks) - 4:
                    feature_dict["layer_m4"] = x
                if i == len(self.blocks) - 1:
                    feature_dict["layer_last"] = x

        x_out = self.norm(x)

        if return_gates and return_features:
            return x_out, gate_info_list, feature_dict, moe_feature_list
        elif return_gates:
            return x_out, gate_info_list
        elif return_features:
            return x_out, feature_dict, moe_feature_list
        else:
            return x_out


# =========================================================
# Convenience wrappers
# =========================================================
class UNIMoEEncoder(TimmPathologyMoEEncoder):
    def __init__(self, uni_cfg: Dict, moe_cfg: Dict):
        base_encoder = UNIEncoder(**uni_cfg)
        super().__init__(base_encoder=base_encoder, moe_cfg=moe_cfg)


class UNI2HMoEEncoder(TimmPathologyMoEEncoder):
    def __init__(self, uni2_cfg: Dict, moe_cfg: Dict):
        base_encoder = UNI2HEncoder(**uni2_cfg)
        super().__init__(base_encoder=base_encoder, moe_cfg=moe_cfg)


class HOptimus0MoEEncoder(TimmPathologyMoEEncoder):
    def __init__(self, hopt_cfg: Dict, moe_cfg: Dict):
        base_encoder = HOptimus0Encoder(**hopt_cfg)
        super().__init__(base_encoder=base_encoder, moe_cfg=moe_cfg)