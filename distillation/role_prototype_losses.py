import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RolePrototypeLossOutput:
    total_loss: torch.Tensor
    attraction_loss: torch.Tensor
    separation_loss: torch.Tensor
    role_target_loss: torch.Tensor
    stats: Dict[str, float]


class RolePrototypeBank(nn.Module):
    """
    Static role prototype bank for the new role-prior branch.

    Current role design:
    - prototype 0: tumor
    - prototype 1: stroma
    - prototype 2: ambiguous
    - prototype 3: free expert (no prototype target)

    Notes
    -----
    1. This class only stores semantic prototypes for the first 3 roles.
    2. The free expert is handled outside by masking / not assigning a semantic target.
    3. Prototypes are expected to be loaded from role_prototypes_init.npy with shape [3, D].
    """

    def __init__(self, prototype_path: str, role_names_path: Optional[str] = None, normalize: bool = True):
        super().__init__()
        protos = np.load(prototype_path).astype(np.float32)
        if protos.ndim != 2:
            raise ValueError(f"Expected prototypes with shape [R, D], got {protos.shape}")

        proto_tensor = torch.from_numpy(protos)
        if normalize:
            proto_tensor = F.normalize(proto_tensor, dim=-1)
        self.register_buffer("prototypes", proto_tensor, persistent=True)

        if role_names_path is not None:
            with open(role_names_path, "r", encoding="utf-8") as f:
                role_names = json.load(f)
            if len(role_names) != protos.shape[0]:
                raise ValueError("role_names length does not match prototype count")
            self.role_names = role_names
        else:
            self.role_names = [f"role_{i}" for i in range(protos.shape[0])]

    @property
    def num_roles(self) -> int:
        return int(self.prototypes.shape[0])

    @property
    def dim(self) -> int:
        return int(self.prototypes.shape[1])

    def cosine_to_prototypes(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, D]
        returns: [N, R]
        """
        x = F.normalize(x, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        return x @ p.t()


class StaticRolePrototypeLoss(nn.Module):
    """
    Static role prototype loss for the first post-prototype experiment.

    Intended usage
    --------------
    You already have:
      P1 tumor prototype
      P2 stroma prototype
      P3 ambiguous prototype
      P4 free expert

    This loss supports two complementary supervision sources:

    A. Explicit role targets (recommended for the first static-prototype experiment)
       Example: tokens/patches routed to expert 0 should approach tumor prototype, etc.

    B. Soft semantic target distribution over the 3 prototypes
       Example: a token may be partly tumor and partly stroma.

    For the very first run, the cleanest setup is:
    - apply this loss only to expert outputs from the 3 semantic experts
    - do not apply it to the free expert
    - use a small weight so it acts as a shaping prior rather than a dominant objective
    """

    def __init__(
        self,
        prototype_bank: RolePrototypeBank,
        tau: float = 0.07,
        attraction_weight: float = 1.0,
        separation_weight: float = 0.5,
        target_weight: float = 1.0,
        margin: float = 0.05,
    ):
        super().__init__()
        self.prototype_bank = prototype_bank
        self.tau = tau
        self.attraction_weight = attraction_weight
        self.separation_weight = separation_weight
        self.target_weight = target_weight
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,
        role_indices: Optional[torch.Tensor] = None,
        soft_role_targets: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> RolePrototypeLossOutput:
        """
        Parameters
        ----------
        features:
            [N, D] semantic-expert output features.

        role_indices:
            [N] long tensor with values in [0, R-1].
            Hard role targets. Use this when expert 0/1/2 correspond to tumor/stroma/ambiguous.

        soft_role_targets:
            [N, R] float tensor. Optional soft semantic target distribution.
            This is useful later if you define soft role membership for boundary tokens.

        valid_mask:
            [N] bool tensor. Optional mask selecting rows to supervise.
            Use this to exclude the free expert or low-confidence rows.

        Returns
        -------
        RolePrototypeLossOutput
        """
        if features.ndim != 2:
            raise ValueError(f"features must be [N, D], got {tuple(features.shape)}")

        if valid_mask is None:
            valid_mask = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)

        if valid_mask.sum() == 0:
            zero = features.new_tensor(0.0)
            return RolePrototypeLossOutput(
                total_loss=zero,
                attraction_loss=zero,
                separation_loss=zero,
                role_target_loss=zero,
                stats={
                    "num_valid": 0.0,
                    "mean_self_sim": 0.0,
                    "mean_other_max_sim": 0.0,
                },
            )

        x = features[valid_mask]
        sims = self.prototype_bank.cosine_to_prototypes(x)  # [Nv, R]

        attraction_loss = x.new_tensor(0.0)
        separation_loss = x.new_tensor(0.0)
        role_target_loss = x.new_tensor(0.0)

        self_sim = None
        other_max = None

        if role_indices is not None:
            hard_targets = role_indices[valid_mask]
            if hard_targets.min() < 0 or hard_targets.max() >= self.prototype_bank.num_roles:
                raise ValueError("role_indices out of prototype range")

            row_idx = torch.arange(sims.shape[0], device=sims.device)
            self_sim = sims[row_idx, hard_targets]

            # attraction: maximize similarity to assigned prototype
            attraction_loss = (1.0 - self_sim).mean()

            # separation: assigned prototype should exceed best non-assigned by margin
            masked = sims.clone()
            masked[row_idx, hard_targets] = -1e9
            other_max = masked.max(dim=1).values
            separation_loss = F.relu(self.margin - self_sim + other_max).mean()

            # role target loss: prototype classification CE
            role_target_loss = F.cross_entropy(sims / self.tau, hard_targets)

        elif soft_role_targets is not None:
            soft = soft_role_targets[valid_mask]
            if soft.ndim != 2 or soft.shape[1] != self.prototype_bank.num_roles:
                raise ValueError("soft_role_targets must be [N, R]")
            soft = soft / (soft.sum(dim=1, keepdim=True) + 1e-8)

            probs = F.log_softmax(sims / self.tau, dim=1)
            role_target_loss = -(soft * probs).sum(dim=1).mean()

            # soft attraction uses expectation of similarity under target distribution
            self_sim = (soft * sims).sum(dim=1)
            attraction_loss = (1.0 - self_sim).mean()

            # soft separation: compare target-weighted sim vs best non-target-ish sim
            soft_mask = soft > 0.5
            masked = sims.clone()
            masked[soft_mask] = -1e9
            other_max = masked.max(dim=1).values
            separation_loss = F.relu(self.margin - self_sim + other_max).mean()

        else:
            raise ValueError("Either role_indices or soft_role_targets must be provided")

        total_loss = (
            self.attraction_weight * attraction_loss
            + self.separation_weight * separation_loss
            + self.target_weight * role_target_loss
        )

        stats = {
            "num_valid": float(valid_mask.sum().item()),
            "mean_self_sim": float(self_sim.mean().item()) if self_sim is not None else 0.0,
            "mean_other_max_sim": float(other_max.mean().item()) if other_max is not None else 0.0,
            "mean_proto_logit": float(sims.mean().item()),
        }

        return RolePrototypeLossOutput(
            total_loss=total_loss,
            attraction_loss=attraction_loss,
            separation_loss=separation_loss,
            role_target_loss=role_target_loss,
            stats=stats,
        )


def build_semantic_expert_mask(active_expert_ids: torch.Tensor, free_expert_id: int = 3) -> torch.Tensor:
    """
    active_expert_ids: [N] long tensor.
    Returns mask selecting rows that belong to semantic experts rather than the free expert.
    """
    return active_expert_ids != free_expert_id


def build_hard_role_indices_from_expert_ids(
    active_expert_ids: torch.Tensor,
    expert_to_role: Optional[Dict[int, int]] = None,
) -> torch.Tensor:
    """
    Default mapping for current mainline:
      expert 0 -> tumor prototype 0
      expert 1 -> stroma prototype 1
      expert 2 -> ambiguous prototype 2
      expert 3 -> free expert (should be masked out separately)
    """
    if expert_to_role is None:
        expert_to_role = {0: 0, 1: 1, 2: 2}

    out = torch.full_like(active_expert_ids, fill_value=-1)
    for expert_id, role_id in expert_to_role.items():
        out[active_expert_ids == expert_id] = role_id
    return out


# -----------------------------
# Minimal example
# -----------------------------
def demo_usage() -> None:
    bank = RolePrototypeBank(
        prototype_path="role_prototypes_init.npy",
        role_names_path="role_names.json",
        normalize=True,
    )
    criterion = StaticRolePrototypeLoss(
        prototype_bank=bank,
        tau=0.07,
        attraction_weight=1.0,
        separation_weight=0.5,
        target_weight=1.0,
        margin=0.05,
    )

    # Example batch: semantic-expert outputs from your MoE layer.
    features = torch.randn(128, bank.dim)
    active_expert_ids = torch.randint(low=0, high=4, size=(128,))

    valid_mask = build_semantic_expert_mask(active_expert_ids, free_expert_id=3)
    role_indices = build_hard_role_indices_from_expert_ids(active_expert_ids)

    # Only semantic experts should be supervised.
    safe_role_indices = role_indices.clone()
    safe_role_indices[~valid_mask] = 0

    out = criterion(
        features=features,
        role_indices=safe_role_indices,
        valid_mask=valid_mask,
    )
    print(out.total_loss.item(), out.stats)


if __name__ == "__main__":
    demo_usage()
