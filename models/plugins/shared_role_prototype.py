import json
import os
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedRolePrototype(nn.Module):
    """
    共享主 role prototype
    """

    def __init__(
        self,
        init_prototypes: torch.Tensor,
        role_names: List[str],
        normalize: bool = True,
        learnable: bool = True,
    ):
        super().__init__()

        if init_prototypes.ndim != 2:
            raise ValueError(
                f"init_prototypes must be [R, D], got {tuple(init_prototypes.shape)}"
            )
        if init_prototypes.shape[0] != len(role_names):
            raise ValueError(
                f"num prototypes {init_prototypes.shape[0]} != num role names {len(role_names)}"
            )

        self.role_names = list(role_names)
        self.num_roles = int(init_prototypes.shape[0])
        self.proto_dim = int(init_prototypes.shape[1])
        self.normalize = bool(normalize)

        init_prototypes = init_prototypes.float()

        self.init_prototypes = nn.Parameter(
            init_prototypes.clone(),
            requires_grad=False,
        )

        self.prototypes = nn.Parameter(
            init_prototypes.clone(),
            requires_grad=learnable,
        )

    @classmethod
    def from_files(
        cls,
        role_proto_dir: str,
        normalize: bool = True,
        learnable: bool = True,
        device: Optional[str] = None,
    ):
        proto_path = os.path.join(role_proto_dir, "role_prototypes_init.npy")
        names_path = os.path.join(role_proto_dir, "role_names.json")

        if not os.path.exists(proto_path):
            raise FileNotFoundError(f"Missing prototype file: {proto_path}")
        if not os.path.exists(names_path):
            raise FileNotFoundError(f"Missing role names file: {names_path}")

        protos = np.load(proto_path).astype(np.float32)
        with open(names_path, "r", encoding="utf-8") as f:
            role_names = json.load(f)

        protos = torch.from_numpy(protos)
        model = cls(
            init_prototypes=protos,
            role_names=role_names,
            normalize=normalize,
            learnable=learnable,
        )
        if device is not None:
            model = model.to(device)
        return model

    def get_prototypes(self) -> torch.Tensor:
        protos = self.prototypes
        if self.normalize:
            protos = F.normalize(protos, dim=-1)
        return protos

    def get_init_prototypes(self) -> torch.Tensor:
        protos = self.init_prototypes
        if self.normalize:
            protos = F.normalize(protos, dim=-1)
        return protos

    def forward(self) -> torch.Tensor:
        return self.get_prototypes()

    def role_name_to_id(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.role_names)}


class PatchRoleSummaryFromSharedProto(nn.Module):
    """
    用共享主 prototype 直接从 patch-level feature 计算 role summary
    """

    def __init__(
        self,
        shared_role_proto: SharedRolePrototype,
        tau: float = 1.0,
        use_softmax: bool = True,
    ):
        super().__init__()
        self.shared_role_proto = shared_role_proto
        self.tau = float(tau)
        self.use_softmax = bool(use_softmax)

    @staticmethod
    def compute_one_vs_rest_gaps(role_logits: torch.Tensor) -> torch.Tensor:
        if role_logits.ndim != 3:
            raise ValueError(f"role_logits must be [B, N, R], got {tuple(role_logits.shape)}")

        B, N, R = role_logits.shape
        gaps = []

        for r in range(R):
            cur = role_logits[..., r]
            if R == 1:
                other_max = torch.zeros_like(cur)
            else:
                other_ids = [i for i in range(R) if i != r]
                other_max = role_logits[..., other_ids].max(dim=-1).values
            gaps.append(cur - other_max)

        return torch.stack(gaps, dim=-1)

    @staticmethod
    def compute_top1_gap(role_logits: torch.Tensor) -> torch.Tensor:
        R = role_logits.shape[-1]
        top2 = torch.topk(role_logits, k=min(2, R), dim=-1).values
        if R >= 2:
            return (top2[..., 0] - top2[..., 1]).unsqueeze(-1)
        return torch.ones_like(top2[..., 0:1])

    def forward(self, patch_feat_teacher_space: torch.Tensor) -> dict:
        squeeze_back = False
        if patch_feat_teacher_space.ndim == 2:
            patch_feat_teacher_space = patch_feat_teacher_space.unsqueeze(0)
            squeeze_back = True

        if patch_feat_teacher_space.ndim != 3:
            raise ValueError(
                f"patch_feat_teacher_space must be [B, N, D] or [N, D], got {tuple(patch_feat_teacher_space.shape)}"
            )

        protos = self.shared_role_proto.get_prototypes()
        feats = F.normalize(patch_feat_teacher_space, dim=-1)
        logits = feats @ protos.t()

        if self.use_softmax:
            probs = torch.softmax(logits / self.tau, dim=-1)
        else:
            probs = logits

        gaps = self.compute_one_vs_rest_gaps(logits)
        top1_gap = self.compute_top1_gap(logits)

        out = {
            "patch_role_logits": logits,
            "patch_role_probs": probs,
            "patch_role_gaps": gaps,
            "patch_top1_gap": top1_gap,
        }

        if squeeze_back:
            out = {k: v.squeeze(0) for k, v in out.items()}

        return out
