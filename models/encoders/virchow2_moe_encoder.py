#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
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
def _load_local_virchow2_weights(model: nn.Module, weight_path: str):
    state_dict = torch.load(weight_path, map_location="cpu")

    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "encoder" in state_dict:
            state_dict = state_dict["encoder"]

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("model.", "")
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v
    state_dict = new_state_dict

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[Virchow2] strict load success: {weight_path}")
    except Exception as e:
        print(f"[Virchow2] strict load failed, fallback strict=False: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Virchow2] missing keys ({len(missing)}): {missing[:20]}")
        print(f"[Virchow2] unexpected keys ({len(unexpected)}): {unexpected[:20]}")

    if not hasattr(model, "pos_embed") or model.pos_embed is None:
        num_patches = model.patch_embed.num_patches + 1
        embed_dim = model.embed_dim
        model.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        print("[Virchow2] pos_embed was missing, created fallback pos_embed.")


def _extract_stage2_layer_moe_state(
    ckpt_path: str,
    src_layer_idx: int,
) -> Dict[str, torch.Tensor]:
    """
    从 stage2 checkpoint 中提取单层 MoE 权重，
    direct bridge 期望 key:
        moe.experts.0.ffn.0.weight
        moe.shared_expert.ffn.0.weight
        moe.gate.gate_vectors
        ...
    """
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

    print(f"[Stage2->Virchow2] extracted {len(out)} tensors from prefix: {prefix}")
    return out


# =========================================================
# 1) Virchow2Encoder
# =========================================================
class Virchow2Encoder(nn.Module):
    def __init__(
        self,
        weight_path: str = "models/distill_teacher/Virchow2/pytorch_model.bin",
        device: str = "cuda",
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
            init_values=1e-5,
        )

        _load_local_virchow2_weights(self.model, weight_path)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transforms = create_transform(
            input_size=(3, 224, 224),
            interpolation="bicubic",
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_pct=1.0,
        )

        self.patch_embed = self.model.patch_embed
        self.blocks = self.model.blocks
        self.norm = self.model.norm

        self.embed_dim = self.model.embed_dim
        self.num_layers = len(self.blocks)
        self.reg_tokens = getattr(self.model, "reg_tokens", 0)

        print(f"[Virchow2Encoder] layers={self.num_layers}, embed_dim={self.embed_dim}, reg_tokens={self.reg_tokens}")

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

    def forward(self, images, return_tokens: bool = False):
        x = self.patch_embed_forward(images)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if return_tokens:
            return x
        else:
            cls = x[:, 0, :]
            return cls


# =========================================================
# 2) Direct BridgeMoEFFN
# =========================================================
class BridgeMoEFFN(nn.Module):
    """
    d_model(如1280) -> 384 -> MoEFFN -> d_model
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
            use_routing_proj=moe_cfg.get("use_routing_proj", True),
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
# 3) Residual BridgeMoEFFN
# =========================================================
class ResidualBridgeMoEFFN(nn.Module):
    """
    output = base_mlp(x) + alpha * bridge_delta(x)
    """
    def __init__(
        self,
        base_mlp: nn.Module,
        in_dim: int,
        moe_dim: int,
        moe_hidden_dim: int,
        moe_layer_idx_for_forward: int,
        moe_cfg: Dict,
        alpha_init: float = 0.05,
        learnable_alpha: bool = True,
    ):
        super().__init__()
        self.base_mlp = base_mlp

        self.bridge = BridgeMoEFFN(
            in_dim=in_dim,
            moe_dim=moe_dim,
            moe_hidden_dim=moe_hidden_dim,
            moe_layer_idx_for_forward=moe_layer_idx_for_forward,
            moe_cfg=moe_cfg,
        )

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))

    def forward(
        self,
        x,
        return_gates: bool = False,
        is_eval: bool = False,
        offline_cluster_ids=None,
    ):
        base_out = self.base_mlp(x)

        bridge_out = self.bridge(
            x,
            return_gates=return_gates,
            is_eval=is_eval,
            offline_cluster_ids=offline_cluster_ids,
        )

        if return_gates:
            bridge_delta, gate_info = bridge_out
        else:
            bridge_delta = bridge_out
            gate_info = None

        y = base_out + self.alpha * bridge_delta

        if return_gates:
            return y, gate_info
        else:
            return y


# =========================================================
# 4) Virchow2MoEEncoder
# =========================================================
class Virchow2MoEEncoder(nn.Module):
    def __init__(
        self,
        virchow2_cfg: Dict,
        moe_cfg: Dict,
    ):
        super().__init__()

        self.base_encoder = Virchow2Encoder(**virchow2_cfg)

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

        self.num_experts = moe_cfg.get("num_experts", 4)
        self.shared_expert = moe_cfg.get("shared_expert", True)
        self.routing_strategy = moe_cfg.get("routing_strategy", "proto_topany")
        self.top_k = moe_cfg.get("top_k", 2)
        self.init_threshold = moe_cfg.get("init_threshold", 0.0)
        self.min_experts = moe_cfg.get("min_experts", 1)
        self.max_experts = moe_cfg.get("max_experts", 2)
        self.gate_init_scale = moe_cfg.get("gate_init_scale", 2.0)
        self.gate_noise_std = moe_cfg.get("gate_noise_std", 0.02)
        self.shared_alpha = moe_cfg.get("shared_alpha", 0.05)
        self.use_routing_proj = moe_cfg.get("use_routing_proj", False)
        self.routing_proj_dim = moe_cfg.get("routing_proj_dim", None)
        self.routing_proj_hidden_dim = moe_cfg.get("routing_proj_hidden_dim", None)
        self.routing_proj_dropout = moe_cfg.get("routing_proj_dropout", 0.0)

        self.use_cluster_bias = moe_cfg.get("use_cluster_bias", False)
        self.num_clusters = moe_cfg.get("num_clusters", None)
        self.cluster_bias_scale = moe_cfg.get("cluster_bias_scale", 1.0)
        self.background_cluster_ids = moe_cfg.get("background_cluster_ids", [5])
        self.routing_metric = moe_cfg.get("routing_metric", "cosine")

        self.use_patch_guided_routing = moe_cfg.get("use_patch_guided_routing", False)
        self.patch_guided_mode = moe_cfg.get("patch_guided_mode", "none")
        self.patch_context_alpha = moe_cfg.get("patch_context_alpha", 0.5)
        self.patch_summary_type = moe_cfg.get("patch_summary_type", "global")
        self.patch_summary_kernel = moe_cfg.get("patch_summary_kernel", 3)

        # 新增：direct / residual
        self.bridge_mode = moe_cfg.get("bridge_mode", "direct")
        self.residual_alpha_init = moe_cfg.get("residual_alpha_init", 0.05)
        self.learnable_residual_alpha = moe_cfg.get("learnable_residual_alpha", True)

        depth = len(self.blocks)
        self.moe_layer_map = {}

        bridge_cfg = dict(moe_cfg)
        bridge_cfg["num_layers"] = self.num_layers

        for moe_layer_idx, idx in enumerate(self.moe_layers_idx):
            real_idx = idx if idx >= 0 else depth + idx
            self.moe_layer_map[real_idx] = moe_layer_idx

            block = self.blocks[real_idx]
            in_dim = block.mlp.fc1.in_features
            old_mlp = block.mlp

            if self.bridge_mode == "direct":
                print(f"[INFO] Replacing Virchow2 block {real_idx} MLP with BridgeMoEFFN: {in_dim} -> {self.adapter_dim} -> {in_dim}")
                block.mlp = BridgeMoEFFN(
                    in_dim=in_dim,
                    moe_dim=self.adapter_dim,
                    moe_hidden_dim=self.adapter_hidden_dim,
                    moe_layer_idx_for_forward=moe_layer_idx,
                    moe_cfg=bridge_cfg,
                )

            elif self.bridge_mode == "residual":
                print(f"[INFO] Wrapping Virchow2 block {real_idx} MLP with ResidualBridgeMoEFFN: base_mlp + alpha * bridge")
                block.mlp = ResidualBridgeMoEFFN(
                    base_mlp=old_mlp,
                    in_dim=in_dim,
                    moe_dim=self.adapter_dim,
                    moe_hidden_dim=self.adapter_hidden_dim,
                    moe_layer_idx_for_forward=moe_layer_idx,
                    moe_cfg=bridge_cfg,
                    alpha_init=self.residual_alpha_init,
                    learnable_alpha=self.learnable_residual_alpha,
                )
            else:
                raise ValueError(f"Unsupported bridge_mode={self.bridge_mode}")

    def load_stage2_moe_from_ckpt(
        self,
        stage2_ckpt_path: str,
        target_to_source_layer_map: Dict[int, int],
        strict: bool = False,
    ):
        for tgt_block_idx, src_stage2_layer_idx in target_to_source_layer_map.items():
            if tgt_block_idx not in self.moe_layer_map:
                raise ValueError(f"Virchow2 block {tgt_block_idx} is not in current moe_layer_map={self.moe_layer_map}")

            layer_state = _extract_stage2_layer_moe_state(
                ckpt_path=stage2_ckpt_path,
                src_layer_idx=src_stage2_layer_idx,
            )

            blk_mlp = self.blocks[tgt_block_idx].mlp

            if isinstance(blk_mlp, BridgeMoEFFN):
                msg = blk_mlp.load_state_dict(layer_state, strict=strict)

            elif isinstance(blk_mlp, ResidualBridgeMoEFFN):
                residual_state = {}
                for k, v in layer_state.items():
                    if k.startswith("moe."):
                        residual_state[f"bridge.{k}"] = v
                    else:
                        residual_state[k] = v
                msg = blk_mlp.load_state_dict(residual_state, strict=strict)

            else:
                raise TypeError(f"Unsupported mlp type for loading stage2 weights: {type(blk_mlp)}")

            print(f"[Load stage2 -> Virchow2] tgt_block={tgt_block_idx}, src_stage2_layer={src_stage2_layer_idx}")
            print(msg)

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

        if isinstance(blk.mlp, (BridgeMoEFFN, ResidualBridgeMoEFFN)):
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
        feature_dict = {}
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
                moe_feature_list.append(x)
            else:
                x = blk(x)

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