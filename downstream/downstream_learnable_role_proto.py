# downstream/downstream_learnable_role_proto.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableRolePrototypeHead(nn.Module):
    """
    Lightweight downstream role-prototype adaptation head.

    Design goal:
    - freeze encoder / MoE experts / router by default
    - optionally freeze or unfreeze projection head
    - only adapt role prototypes (and optionally proj_l12)
    - produce patch-level role affinity logits in teacher space
    """
    def __init__(
        self,
        init_prototypes: np.ndarray,
        role_names: List[str],
        proj_head: nn.Module,
        normalize_proto: bool = True,
        learn_proj: bool = False,
        temperature: float = 0.07,
    ):
        super().__init__()
        assert init_prototypes.ndim == 2
        self.role_names = list(role_names)
        self.num_roles = len(self.role_names)
        self.temperature = float(temperature)
        self.normalize_proto = bool(normalize_proto)

        init_proto_t = torch.from_numpy(init_prototypes).float()
        if self.normalize_proto:
            init_proto_t = F.normalize(init_proto_t, dim=-1)

        self.role_prototypes = nn.Parameter(init_proto_t)  # [R, D_teacher]
        self.proj_head = proj_head

        for p in self.proj_head.parameters():
            p.requires_grad = bool(learn_proj)

    def get_normalized_prototypes(self) -> torch.Tensor:
        proto = self.role_prototypes
        if self.normalize_proto:
            proto = F.normalize(proto, dim=-1)
        return proto

    def forward_patch_roles(self, patch_feat_student: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        patch_feat_student: [B, N, D_student]
        return:
            feat_teacher: [B, N, D_teacher]
            role_logits:   [B, N, R]
        """
        feat_teacher = self.proj_head(patch_feat_student)
        feat_teacher = F.normalize(feat_teacher, dim=-1)
        proto = self.get_normalized_prototypes()  # [R, D]
        role_logits = torch.einsum("bnd,rd->bnr", feat_teacher, proto)
        role_logits = role_logits / self.temperature
        return feat_teacher, role_logits

    def patch_role_stats(self, role_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        probs = torch.softmax(role_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
        top1 = topk_vals[..., 0]
        if probs.shape[-1] >= 2:
            margin = topk_vals[..., 0] - topk_vals[..., 1]
        else:
            margin = torch.ones_like(top1)
        pred = topk_idx[..., 0]
        return {
            "role_probs": probs,
            "top1_conf": top1,
            "top1_margin": margin,
            "pred_role": pred,
        }


@dataclass
class ProtoAdaptOutput:
    total_loss: torch.Tensor
    loss_dict: Dict[str, float]
    bag_logits: torch.Tensor
    patch_role_logits: torch.Tensor


class DownstreamLearnableRoleProtoWrapper(nn.Module):
    """
    Wrap a trained stage2 MoE encoder for downstream slide-level learning with learnable role prototypes.

    Recommended phase-1 downstream adaptation:
    - freeze student encoder / experts / router
    - freeze projection head OR optionally unfreeze it
    - learn only role prototypes
    - aggregate patch evidence into slide prediction
    """
    def __init__(
        self,
        student_model: nn.Module,
        proj_head: nn.Module,
        init_prototypes: np.ndarray,
        role_names: List[str],
        tumor_role_name: str = "tumor",
        use_last_moe_output: bool = True,
        learn_proj: bool = False,
        proto_lr_scale: float = 1.0,
        role_temperature: float = 0.07,
        bag_topk_ratio: float = 0.1,
        bag_topk_min: int = 4,
        bag_topk_max: int = 32,
        tumor_margin_loss_weight: float = 0.0,
        tumor_margin_pos: float = 0.05,
        tumor_margin_neg: float = 0.05,
    ):
        super().__init__()
        self.student = student_model
        self.use_last_moe_output = bool(use_last_moe_output)
        self.role_head = LearnableRolePrototypeHead(
            init_prototypes=init_prototypes,
            role_names=role_names,
            proj_head=proj_head,
            normalize_proto=True,
            learn_proj=learn_proj,
            temperature=role_temperature,
        )
        self.role_names = list(role_names)
        self.role_to_idx = {n: i for i, n in enumerate(self.role_names)}
        self.tumor_role_id = self.role_to_idx[tumor_role_name]

        self.bag_topk_ratio = float(bag_topk_ratio)
        self.bag_topk_min = int(bag_topk_min)
        self.bag_topk_max = int(bag_topk_max)

        self.tumor_margin_loss_weight = float(tumor_margin_loss_weight)
        self.tumor_margin_pos = float(tumor_margin_pos)
        self.tumor_margin_neg = float(tumor_margin_neg)

        self.slide_classifier = nn.Linear(init_prototypes.shape[1], 1)
        nn.init.xavier_uniform_(self.slide_classifier.weight)
        nn.init.zeros_(self.slide_classifier.bias)

        self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.student.parameters():
            p.requires_grad = False
        self.student.eval()

    def extract_patch_student_features(self, images: torch.Tensor, is_eval: bool = False) -> torch.Tensor:
        student_out, gate_info_list, feature_dict, moe_feature_list = self.student(
            images,
            return_gates=True,
            mask=None,
            is_eval=is_eval,
            return_features=True,
            offline_cluster_ids=None,
        )
        if self.use_last_moe_output and len(moe_feature_list) > 0:
            feat = moe_feature_list[-1]   # [B, N+1, D]
        else:
            feat = feature_dict["layer_12"]
        return feat[:, 1:, :]  # [B, N, D_student]

    def compute_tumor_minus_other(self, role_logits: torch.Tensor) -> torch.Tensor:
        # role_logits: [B, N, R]
        tumor = role_logits[..., self.tumor_role_id]
        if role_logits.shape[-1] == 1:
            return tumor
        other_mask = torch.ones(role_logits.shape[-1], device=role_logits.device, dtype=torch.bool)
        other_mask[self.tumor_role_id] = False
        other_max = role_logits[..., other_mask].max(dim=-1).values
        return tumor - other_max

    def topk_bag_pool(self, feat_teacher: torch.Tensor, tumor_margin: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # feat_teacher: [B, N, D], tumor_margin: [B, N]
        B, N, D = feat_teacher.shape
        bag_repr_list = []
        stats = {}
        for b in range(B):
            score_b = tumor_margin[b]
            feat_b = feat_teacher[b]
            k = max(self.bag_topk_min, int(round(N * self.bag_topk_ratio)))
            k = min(k, self.bag_topk_max, N)
            idx = torch.topk(score_b, k=k, largest=True).indices
            bag_repr = feat_b[idx].mean(dim=0)
            bag_repr = F.normalize(bag_repr, dim=-1)
            bag_repr_list.append(bag_repr)
        bag_repr = torch.stack(bag_repr_list, dim=0)
        stats["bag_topk_k"] = float(k)
        return bag_repr, stats

    def compute_optional_margin_loss(self, tumor_margin: torch.Tensor, slide_label: torch.Tensor) -> torch.Tensor:
        if self.tumor_margin_loss_weight <= 0:
            return tumor_margin.new_tensor(0.0)
        # tumor_margin: [B, N], slide_label: [B]
        topk = max(1, min(16, tumor_margin.shape[1] // 10))
        topk_vals = torch.topk(tumor_margin, k=topk, dim=1, largest=True).values.mean(dim=1)
        pos_mask = slide_label > 0.5
        neg_mask = ~pos_mask
        losses = []
        if pos_mask.any():
            losses.append(F.relu(self.tumor_margin_pos - topk_vals[pos_mask]).mean())
        if neg_mask.any():
            losses.append(F.relu(topk_vals[neg_mask] + self.tumor_margin_neg).mean())
        if len(losses) == 0:
            return tumor_margin.new_tensor(0.0)
        return torch.stack(losses).mean()

    def forward(self, images: torch.Tensor, slide_label: Optional[torch.Tensor] = None, is_eval: bool = False) -> ProtoAdaptOutput:
        with torch.set_grad_enabled(any(p.requires_grad for p in self.student.parameters())):
            patch_student = self.extract_patch_student_features(images, is_eval=is_eval)  # [B,N,D]

        feat_teacher, patch_role_logits = self.role_head.forward_patch_roles(patch_student)
        role_stats = self.role_head.patch_role_stats(patch_role_logits)
        tumor_margin = self.compute_tumor_minus_other(patch_role_logits)
        bag_repr, bag_stats = self.topk_bag_pool(feat_teacher, tumor_margin)
        bag_logits = self.slide_classifier(bag_repr).squeeze(-1)

        total_loss = bag_logits.new_tensor(0.0)
        loss_dict = {
            "bag_topk_k": bag_stats["bag_topk_k"],
            "role_top1_conf_mean": float(role_stats["top1_conf"].mean().detach().cpu()),
            "role_top1_margin_mean": float(role_stats["top1_margin"].mean().detach().cpu()),
            "tumor_margin_mean": float(tumor_margin.mean().detach().cpu()),
            "tumor_margin_top1pct": float(torch.quantile(tumor_margin.flatten(), 0.99).detach().cpu()),
        }

        if slide_label is not None:
            slide_label = slide_label.float().view(-1).to(bag_logits.device)
            bag_loss = F.binary_cross_entropy_with_logits(bag_logits, slide_label)
            margin_loss = self.compute_optional_margin_loss(tumor_margin, slide_label)
            total_loss = bag_loss + self.tumor_margin_loss_weight * margin_loss
            loss_dict.update({
                "total_loss": float(total_loss.detach().cpu()),
                "bag_bce": float(bag_loss.detach().cpu()),
                "tumor_margin_aux": float(margin_loss.detach().cpu()),
                "bag_prob_mean": float(torch.sigmoid(bag_logits).mean().detach().cpu()),
            })
        else:
            loss_dict.update({
                "total_loss": 0.0,
                "bag_bce": 0.0,
                "tumor_margin_aux": 0.0,
                "bag_prob_mean": float(torch.sigmoid(bag_logits).mean().detach().cpu()),
            })

        return ProtoAdaptOutput(
            total_loss=total_loss,
            loss_dict=loss_dict,
            bag_logits=bag_logits,
            patch_role_logits=patch_role_logits,
        )


def load_role_proto_dir(role_proto_dir: str) -> Tuple[np.ndarray, List[str]]:
    protos = np.load(os.path.join(role_proto_dir, "role_prototypes_init.npy")).astype(np.float32)
    with open(os.path.join(role_proto_dir, "role_names.json"), "r", encoding="utf-8") as f:
        role_names = json.load(f)
    return protos, role_names


def build_downstream_proto_model(
    student_model: nn.Module,
    proj_head: nn.Module,
    role_proto_dir: str,
    learn_proj: bool = False,
    use_last_moe_output: bool = True,
    tumor_role_name: str = "tumor",
    role_temperature: float = 0.07,
    bag_topk_ratio: float = 0.1,
    bag_topk_min: int = 4,
    bag_topk_max: int = 32,
    tumor_margin_loss_weight: float = 0.0,
) -> DownstreamLearnableRoleProtoWrapper:
    protos, role_names = load_role_proto_dir(role_proto_dir)
    return DownstreamLearnableRoleProtoWrapper(
        student_model=student_model,
        proj_head=proj_head,
        init_prototypes=protos,
        role_names=role_names,
        tumor_role_name=tumor_role_name,
        use_last_moe_output=use_last_moe_output,
        learn_proj=learn_proj,
        role_temperature=role_temperature,
        bag_topk_ratio=bag_topk_ratio,
        bag_topk_min=bag_topk_min,
        bag_topk_max=bag_topk_max,
        tumor_margin_loss_weight=tumor_margin_loss_weight,
    )
