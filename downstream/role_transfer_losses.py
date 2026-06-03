#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# basic utils
# =========================================================
def safe_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def weighted_mean(x: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros((), device=x.device)

    if weight is None:
        return x.mean()

    w = weight.float()
    return (x * w).sum() / w.sum().clamp_min(1e-6)


def normalize_score(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.numel() == 0:
        return x
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


def zscore_score(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if x.numel() == 0:
        return x
    mean = x.mean()
    std = x.std(unbiased=False)
    if std.detach().item() < eps:
        return x - mean
    return (x - mean) / (std + eps)


# =========================================================
# role summary
# =========================================================
def build_role_summary_from_feats(
    patch_feat: torch.Tensor,
    proj_l12: nn.Module,
    summary_builder: nn.Module,
):
    patch_feat_teacher = proj_l12(patch_feat)
    patch_feat_teacher = F.normalize(patch_feat_teacher, dim=-1)
    return summary_builder(patch_feat_teacher.unsqueeze(0))


def compute_tumor_gap_from_role_dict(
    role_dict: Dict[str, torch.Tensor],
    role_names: List[str],
    tumor_name: str,
    negative_role_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    role_logits = role_dict["patch_role_logits"][0]
    role_probs = role_dict["patch_role_probs"][0]
    top1_gap = role_dict["patch_top1_gap"][0].squeeze(-1)

    role_to_idx = {n: i for i, n in enumerate(role_names)}
    if tumor_name not in role_to_idx:
        raise KeyError(f"tumor role '{tumor_name}' not found in role_names={role_names}")

    tumor_idx = role_to_idx[tumor_name]
    neg_ids = [role_to_idx[n] for n in negative_role_names if n in role_to_idx]
    if len(neg_ids) == 0:
        raise ValueError(
            f"No valid negative_role_names found in role_names. "
            f"got={negative_role_names}, role_names={role_names}"
        )

    tumor_prob = role_probs[:, tumor_idx]
    tumor_logit = role_logits[:, tumor_idx]
    neg_logit = role_logits[:, neg_ids].max(dim=-1).values
    tumor_gap = tumor_logit - neg_logit

    return tumor_gap, tumor_prob, top1_gap, role_logits


def compute_online_role_scores(
    patch_feat_adapt: torch.Tensor,
    proj_l12: nn.Module,
    summary_builder: nn.Module,
    role_names: List[str],
    tumor_name: str,
    negative_role_names: List[str],
):
    role_dict = build_role_summary_from_feats(
        patch_feat=patch_feat_adapt,
        proj_l12=proj_l12,
        summary_builder=summary_builder,
    )
    tumor_gap, tumor_prob, top1_gap, role_logits = compute_tumor_gap_from_role_dict(
        role_dict=role_dict,
        role_names=role_names,
        tumor_name=tumor_name,
        negative_role_names=negative_role_names,
    )
    return {
        "tumor_gap": tumor_gap,
        "tumor_prob": tumor_prob,
        "top1_gap": top1_gap,
        "role_logits": role_logits,
    }


# =========================================================
# asymmetric gap losses
# =========================================================
def compute_positive_gap_loss(
    tumor_gap: torch.Tensor,
    margin_pos: float,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.relu(margin_pos - tumor_gap)
    return weighted_mean(loss, weight)


def compute_negative_gap_loss(
    tumor_gap: torch.Tensor,
    margin_neg: float,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.relu(margin_neg + tumor_gap)
    return weighted_mean(loss, weight)


# =========================================================
# ranking loss
# =========================================================
def compute_pairwise_ranking_loss(
    pos_gap: torch.Tensor,
    neg_gap: torch.Tensor,
    margin_rank: float,
    pos_weight: Optional[torch.Tensor] = None,
    neg_weight: Optional[torch.Tensor] = None,
    mode: str = "topk_mean",
    topk: int = 8,
) -> torch.Tensor:
    if pos_gap.numel() == 0 or neg_gap.numel() == 0:
        device = pos_gap.device if pos_gap.numel() > 0 else neg_gap.device
        return torch.zeros((), device=device)

    if mode == "topk_mean":
        kpos = min(topk, len(pos_gap))
        kneg = min(topk, len(neg_gap))
        pos_score = torch.topk(pos_gap, k=kpos).values.mean()
        neg_score = torch.topk(neg_gap, k=kneg).values.mean()
        return F.relu(margin_rank - (pos_score - neg_score))

    if mode == "all_pairs":
        diff = pos_gap[:, None] - neg_gap[None, :]
        loss = F.relu(margin_rank - diff)

        if pos_weight is not None or neg_weight is not None:
            pw = torch.ones_like(pos_gap) if pos_weight is None else pos_weight.float()
            nw = torch.ones_like(neg_gap) if neg_weight is None else neg_weight.float()
            w = pw[:, None] * nw[None, :]
            return (loss * w).sum() / w.sum().clamp_min(1e-6)

        return loss.mean()

    raise ValueError(f"Unsupported ranking mode: {mode}")


# =========================================================
# context-aware weighting
# =========================================================
def build_positive_context_weight(
    tumor_prob: torch.Tensor,
    top1_gap: torch.Tensor,
    pos_context_score: Optional[torch.Tensor] = None,
    alpha_tumor_prob: float = 1.0,
    alpha_top1_gap: float = 1.0,
    alpha_context: float = 1.0,
    detach_weight: bool = True,
) -> torch.Tensor:
    w = (
        alpha_tumor_prob * normalize_score(tumor_prob)
        + alpha_top1_gap * normalize_score(top1_gap)
    )

    if pos_context_score is not None:
        w = w + alpha_context * normalize_score(pos_context_score)

    w = torch.clamp(w, min=1e-6)
    if detach_weight:
        w = w.detach()
    return w


def build_negative_context_weight(
    tumor_prob: torch.Tensor,
    top1_gap: torch.Tensor,
    neg_context_score: Optional[torch.Tensor] = None,
    alpha_tumor_prob: float = 1.0,
    alpha_top1_gap: float = 1.0,
    alpha_context: float = 1.0,
    detach_weight: bool = True,
) -> torch.Tensor:
    w = (
        alpha_tumor_prob * normalize_score(tumor_prob)
        + alpha_top1_gap * normalize_score(top1_gap)
    )

    if neg_context_score is not None:
        w = w + alpha_context * normalize_score(neg_context_score)

    w = torch.clamp(w, min=1e-6)
    if detach_weight:
        w = w.detach()
    return w


def build_context_weight(
    tumor_prob: torch.Tensor,
    top1_gap: torch.Tensor,
    context_score: Optional[torch.Tensor] = None,
    mode: str = "none",
    alpha_tumor_prob: float = 1.0,
    alpha_top1_gap: float = 1.0,
    alpha_context: float = 1.0,
    detach_weight: bool = True,
) -> Optional[torch.Tensor]:
    if mode == "none":
        return None

    w = (
        alpha_tumor_prob * normalize_score(tumor_prob)
        + alpha_top1_gap * normalize_score(top1_gap)
    )

    if mode == "prob_gap_context":
        if context_score is None:
            raise ValueError("context_score is required when mode='prob_gap_context'")
        w = w + alpha_context * normalize_score(context_score)

    w = torch.clamp(w, min=1e-6)
    if detach_weight:
        w = w.detach()
    return w


# =========================================================
# online proposal selection
# =========================================================
def build_positive_support_mask(
    tumor_prob: torch.Tensor,
    tumor_gap: torch.Tensor,
    top1_gap: torch.Tensor,
    pos_context_score: Optional[torch.Tensor] = None,
    pos_neighbor_gap_mean: Optional[torch.Tensor] = None,
    pos_neighbor_gap_max: Optional[torch.Tensor] = None,
    pos_neighbor_prob_mean: Optional[torch.Tensor] = None,
    pos_neighbor_prob_max: Optional[torch.Tensor] = None,
    min_tumor_prob: float = -1e6,
    min_center_gap: float = -1e6,
    min_top1_gap: float = -1e6,
    min_pos_context_score: float = -1e6,
    min_neighbor_gap_mean: float = -1e6,
    min_neighbor_gap_max: float = -1e6,
    min_neighbor_prob_mean: float = -1e6,
    min_neighbor_prob_max: float = -1e6,
) -> torch.Tensor:
    min_tumor_prob = float(min_tumor_prob)
    min_center_gap = float(min_center_gap)
    min_top1_gap = float(min_top1_gap)
    min_pos_context_score = float(min_pos_context_score)
    min_neighbor_gap_mean = float(min_neighbor_gap_mean)
    min_neighbor_gap_max = float(min_neighbor_gap_max)
    min_neighbor_prob_mean = float(min_neighbor_prob_mean)
    min_neighbor_prob_max = float(min_neighbor_prob_max)

    mask = torch.ones_like(tumor_gap, dtype=torch.bool)

    if min_tumor_prob > -1e5:
        mask &= (tumor_prob >= min_tumor_prob)
    if min_center_gap > -1e5:
        mask &= (tumor_gap >= min_center_gap)
    if min_top1_gap > -1e5:
        mask &= (top1_gap >= min_top1_gap)
    if pos_context_score is not None and min_pos_context_score > -1e5:
        mask &= (pos_context_score >= min_pos_context_score)
    if pos_neighbor_gap_mean is not None and min_neighbor_gap_mean > -1e5:
        mask &= (pos_neighbor_gap_mean >= min_neighbor_gap_mean)
    if pos_neighbor_gap_max is not None and min_neighbor_gap_max > -1e5:
        mask &= (pos_neighbor_gap_max >= min_neighbor_gap_max)
    if pos_neighbor_prob_mean is not None and min_neighbor_prob_mean > -1e5:
        mask &= (pos_neighbor_prob_mean >= min_neighbor_prob_mean)
    if pos_neighbor_prob_max is not None and min_neighbor_prob_max > -1e5:
        mask &= (pos_neighbor_prob_max >= min_neighbor_prob_max)

    return mask


def build_positive_support_mask_weak(
    tumor_prob: torch.Tensor,
    tumor_gap: torch.Tensor,
    top1_gap: torch.Tensor,
    pos_context_score: Optional[torch.Tensor] = None,
    pos_neighbor_gap_mean: Optional[torch.Tensor] = None,
    pos_neighbor_gap_max: Optional[torch.Tensor] = None,
    pos_neighbor_prob_mean: Optional[torch.Tensor] = None,
    pos_neighbor_prob_max: Optional[torch.Tensor] = None,
    fallback_min_tumor_prob: float = -1e6,
    fallback_min_center_gap: float = -1e6,
    fallback_min_top1_gap: float = -1e6,
    fallback_min_pos_context_score: float = -1e6,
    fallback_min_neighbor_gap_mean: float = -1e6,
    fallback_min_neighbor_gap_max: float = -1e6,
    fallback_min_neighbor_prob_mean: float = -1e6,
    fallback_min_neighbor_prob_max: float = -1e6,
) -> torch.Tensor:
    return build_positive_support_mask(
        tumor_prob=tumor_prob,
        tumor_gap=tumor_gap,
        top1_gap=top1_gap,
        pos_context_score=pos_context_score,
        pos_neighbor_gap_mean=pos_neighbor_gap_mean,
        pos_neighbor_gap_max=pos_neighbor_gap_max,
        pos_neighbor_prob_mean=pos_neighbor_prob_mean,
        pos_neighbor_prob_max=pos_neighbor_prob_max,
        min_tumor_prob=fallback_min_tumor_prob,
        min_center_gap=fallback_min_center_gap,
        min_top1_gap=fallback_min_top1_gap,
        min_pos_context_score=fallback_min_pos_context_score,
        min_neighbor_gap_mean=fallback_min_neighbor_gap_mean,
        min_neighbor_gap_max=fallback_min_neighbor_gap_max,
        min_neighbor_prob_mean=fallback_min_neighbor_prob_mean,
        min_neighbor_prob_max=fallback_min_neighbor_prob_max,
    )


def filter_positive_candidates_by_support(
    tumor_gap: torch.Tensor,
    tumor_prob: torch.Tensor,
    top1_gap: torch.Tensor,
    pos_context_score: Optional[torch.Tensor] = None,
    pos_neighbor_gap_mean: Optional[torch.Tensor] = None,
    pos_neighbor_gap_max: Optional[torch.Tensor] = None,
    pos_neighbor_prob_mean: Optional[torch.Tensor] = None,
    pos_neighbor_prob_max: Optional[torch.Tensor] = None,
    use_strong_pos_support: bool = True,
    allow_pos_fallback: bool = True,
    min_pos_keep: int = 1,
    min_tumor_prob: float = -1e6,
    min_center_gap: float = -1e6,
    min_top1_gap: float = -1e6,
    min_pos_context_score: float = -1e6,
    min_neighbor_gap_mean: float = -1e6,
    min_neighbor_gap_max: float = -1e6,
    min_neighbor_prob_mean: float = -1e6,
    min_neighbor_prob_max: float = -1e6,
    fallback_min_tumor_prob: float = -1e6,
    fallback_min_center_gap: float = -1e6,
    fallback_min_top1_gap: float = -1e6,
    fallback_min_pos_context_score: float = -1e6,
    fallback_min_neighbor_gap_mean: float = -1e6,
    fallback_min_neighbor_gap_max: float = -1e6,
    fallback_min_neighbor_prob_mean: float = -1e6,
    fallback_min_neighbor_prob_max: float = -1e6,
):
    n = int(tumor_gap.numel())
    if n == 0:
        empty = torch.zeros_like(tumor_gap, dtype=torch.bool)
        return empty, {
            "pos_support_num_before": 0.0,
            "pos_support_num_after_strong": 0.0,
            "pos_support_num_after_final": 0.0,
            "pos_support_ratio_strong": 0.0,
            "pos_support_ratio_final": 0.0,
            "pos_support_used_fallback": 0.0,
        }

    if not use_strong_pos_support:
        base_mask = torch.ones_like(tumor_gap, dtype=torch.bool)
        return base_mask, {
            "pos_support_num_before": float(n),
            "pos_support_num_after_strong": float(n),
            "pos_support_num_after_final": float(n),
            "pos_support_ratio_strong": 1.0,
            "pos_support_ratio_final": 1.0,
            "pos_support_used_fallback": 0.0,
        }

    strong_mask = build_positive_support_mask(
        tumor_prob=tumor_prob,
        tumor_gap=tumor_gap,
        top1_gap=top1_gap,
        pos_context_score=pos_context_score,
        pos_neighbor_gap_mean=pos_neighbor_gap_mean,
        pos_neighbor_gap_max=pos_neighbor_gap_max,
        pos_neighbor_prob_mean=pos_neighbor_prob_mean,
        pos_neighbor_prob_max=pos_neighbor_prob_max,
        min_tumor_prob=min_tumor_prob,
        min_center_gap=min_center_gap,
        min_top1_gap=min_top1_gap,
        min_pos_context_score=min_pos_context_score,
        min_neighbor_gap_mean=min_neighbor_gap_mean,
        min_neighbor_gap_max=min_neighbor_gap_max,
        min_neighbor_prob_mean=min_neighbor_prob_mean,
        min_neighbor_prob_max=min_neighbor_prob_max,
    )

    num_after_strong = int(strong_mask.sum().item())
    final_mask = strong_mask
    used_fallback = 0.0

    if num_after_strong < min_pos_keep and allow_pos_fallback:
        weak_mask = build_positive_support_mask_weak(
            tumor_prob=tumor_prob,
            tumor_gap=tumor_gap,
            top1_gap=top1_gap,
            pos_context_score=pos_context_score,
            pos_neighbor_gap_mean=pos_neighbor_gap_mean,
            pos_neighbor_gap_max=pos_neighbor_gap_max,
            pos_neighbor_prob_mean=pos_neighbor_prob_mean,
            pos_neighbor_prob_max=pos_neighbor_prob_max,
            fallback_min_tumor_prob=fallback_min_tumor_prob,
            fallback_min_center_gap=fallback_min_center_gap,
            fallback_min_top1_gap=fallback_min_top1_gap,
            fallback_min_pos_context_score=fallback_min_pos_context_score,
            fallback_min_neighbor_gap_mean=fallback_min_neighbor_gap_mean,
            fallback_min_neighbor_gap_max=fallback_min_neighbor_gap_max,
            fallback_min_neighbor_prob_mean=fallback_min_neighbor_prob_mean,
            fallback_min_neighbor_prob_max=fallback_min_neighbor_prob_max,
        )
        if int(weak_mask.sum().item()) > 0:
            final_mask = weak_mask
            used_fallback = 1.0

    num_after_final = int(final_mask.sum().item())
    return final_mask, {
        "pos_support_num_before": float(n),
        "pos_support_num_after_strong": float(num_after_strong),
        "pos_support_num_after_final": float(num_after_final),
        "pos_support_ratio_strong": float(num_after_strong / max(n, 1)),
        "pos_support_ratio_final": float(num_after_final / max(n, 1)),
        "pos_support_used_fallback": float(used_fallback),
    }


def select_online_positive_proposals(
    tumor_gap: torch.Tensor,
    tumor_prob: torch.Tensor,
    top1_gap: torch.Tensor,
    pos_context_score: Optional[torch.Tensor] = None,
    pos_neighbor_gap_mean: Optional[torch.Tensor] = None,
    pos_neighbor_gap_max: Optional[torch.Tensor] = None,
    pos_neighbor_prob_mean: Optional[torch.Tensor] = None,
    pos_neighbor_prob_max: Optional[torch.Tensor] = None,
    use_strong_pos_support: bool = True,
    allow_pos_support_fallback: bool = True,
    min_pos_keep: int = 4,
    select_topk: Optional[int] = None,
    pos_support_min_tumor_prob: float = -1e6,
    pos_support_min_center_gap: float = -1e6,
    pos_support_min_top1_gap: float = -1e6,
    pos_support_min_context_score: float = -1e6,
    pos_support_min_neighbor_gap_mean: float = -1e6,
    pos_support_min_neighbor_gap_max: float = -1e6,
    pos_support_min_neighbor_prob_mean: float = -1e6,
    pos_support_min_neighbor_prob_max: float = -1e6,
    pos_fallback_min_tumor_prob: float = -1e6,
    pos_fallback_min_center_gap: float = -1e6,
    pos_fallback_min_top1_gap: float = -1e6,
    pos_fallback_min_context_score: float = -1e6,
    pos_fallback_min_neighbor_gap_mean: float = -1e6,
    pos_fallback_min_neighbor_gap_max: float = -1e6,
    pos_fallback_min_neighbor_prob_mean: float = -1e6,
    pos_fallback_min_neighbor_prob_max: float = -1e6,
):
    keep_mask, support_stats = filter_positive_candidates_by_support(
        tumor_gap=tumor_gap,
        tumor_prob=tumor_prob,
        top1_gap=top1_gap,
        pos_context_score=pos_context_score,
        pos_neighbor_gap_mean=pos_neighbor_gap_mean,
        pos_neighbor_gap_max=pos_neighbor_gap_max,
        pos_neighbor_prob_mean=pos_neighbor_prob_mean,
        pos_neighbor_prob_max=pos_neighbor_prob_max,
        use_strong_pos_support=use_strong_pos_support,
        allow_pos_fallback=allow_pos_support_fallback,
        min_pos_keep=min_pos_keep,
        min_tumor_prob=pos_support_min_tumor_prob,
        min_center_gap=pos_support_min_center_gap,
        min_top1_gap=pos_support_min_top1_gap,
        min_pos_context_score=pos_support_min_context_score,
        min_neighbor_gap_mean=pos_support_min_neighbor_gap_mean,
        min_neighbor_gap_max=pos_support_min_neighbor_gap_max,
        min_neighbor_prob_mean=pos_support_min_neighbor_prob_mean,
        min_neighbor_prob_max=pos_support_min_neighbor_prob_max,
        fallback_min_tumor_prob=pos_fallback_min_tumor_prob,
        fallback_min_center_gap=pos_fallback_min_center_gap,
        fallback_min_top1_gap=pos_fallback_min_top1_gap,
        fallback_min_pos_context_score=pos_fallback_min_context_score,
        fallback_min_neighbor_gap_mean=pos_fallback_min_neighbor_gap_mean,
        fallback_min_neighbor_gap_max=pos_fallback_min_neighbor_gap_max,
        fallback_min_neighbor_prob_mean=pos_fallback_min_neighbor_prob_mean,
        fallback_min_neighbor_prob_max=pos_fallback_min_neighbor_prob_max,
    )

    valid_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        valid_idx = torch.arange(len(tumor_gap), device=tumor_gap.device)

    score = (
        1.2 * zscore_score(tumor_gap[valid_idx])
        + 0.8 * zscore_score(tumor_prob[valid_idx])
        + 0.5 * zscore_score(top1_gap[valid_idx])
    )
    if pos_context_score is not None:
        score = score + 0.8 * zscore_score(pos_context_score[valid_idx])
    if pos_neighbor_gap_mean is not None:
        score = score + 0.6 * zscore_score(pos_neighbor_gap_mean[valid_idx])
    if pos_neighbor_gap_max is not None:
        score = score + 0.4 * zscore_score(pos_neighbor_gap_max[valid_idx])

    if select_topk is not None:
        k = min(int(select_topk), int(valid_idx.numel()))
        local = torch.topk(score, k=k, dim=0).indices
        selected_idx = valid_idx[local]
    else:
        selected_idx = valid_idx

    selected_idx = selected_idx[torch.argsort(tumor_gap[selected_idx], descending=True)]

    sel_stats = dict(support_stats)
    sel_stats["pos_selected_num"] = float(int(selected_idx.numel()))
    sel_stats["pos_selected_ratio"] = float(int(selected_idx.numel()) / max(int(tumor_gap.numel()), 1))
    return selected_idx, sel_stats


def select_online_negative_proposals(
    tumor_gap: torch.Tensor,
    tumor_prob: torch.Tensor,
    top1_gap: torch.Tensor,
    neg_context_score: Optional[torch.Tensor] = None,
    neighbor_gap_mean: Optional[torch.Tensor] = None,
    neighbor_gap_max: Optional[torch.Tensor] = None,
    select_topk: Optional[int] = None,
):
    idx = torch.arange(len(tumor_gap), device=tumor_gap.device)
    if idx.numel() == 0:
        return idx, {
            "neg_selected_num": 0.0,
            "neg_selected_ratio": 0.0,
        }

    score = (
        1.3 * zscore_score(tumor_gap)
        + 0.8 * zscore_score(tumor_prob)
        + 0.4 * zscore_score(top1_gap)
    )
    if neg_context_score is not None:
        score = score + 0.8 * zscore_score(neg_context_score)
    if neighbor_gap_mean is not None:
        score = score - 0.6 * zscore_score(neighbor_gap_mean)
    if neighbor_gap_max is not None:
        score = score - 0.3 * zscore_score(neighbor_gap_max)

    if select_topk is not None:
        k = min(int(select_topk), len(tumor_gap))
        local = torch.topk(score, k=k, dim=0).indices
        idx = idx[local]

    idx = idx[torch.argsort(tumor_gap[idx], descending=True)]
    return idx, {
        "neg_selected_num": float(int(idx.numel())),
        "neg_selected_ratio": float(int(idx.numel()) / max(int(tumor_gap.numel()), 1)),
    }


# =========================================================
# main asymmetric role proto loss
# =========================================================
def compute_asymmetric_role_proto_loss(
    patch_feat_adapt: torch.Tensor,
    slide_label: int,
    candidate_type: str,
    proj_l12: nn.Module,
    summary_builder: nn.Module,
    role_names: List[str],
    tumor_name: str,
    negative_role_names: List[str],
    margin_pos: float,
    margin_neg: float,
    pos_context_score: Optional[torch.Tensor] = None,
    neg_context_score: Optional[torch.Tensor] = None,
    context_score: Optional[torch.Tensor] = None,
    context_weight_mode: str = "none",
    alpha_tumor_prob: float = 1.0,
    alpha_top1_gap: float = 1.0,
    alpha_context: float = 1.0,
    detach_weight: bool = True,
    selected_idx: Optional[torch.Tensor] = None,
):
    role_dict = build_role_summary_from_feats(
        patch_feat=patch_feat_adapt,
        proj_l12=proj_l12,
        summary_builder=summary_builder,
    )

    tumor_gap, tumor_prob, top1_gap, role_logits = compute_tumor_gap_from_role_dict(
        role_dict=role_dict,
        role_names=role_names,
        tumor_name=tumor_name,
        negative_role_names=negative_role_names,
    )

    if selected_idx is not None and selected_idx.numel() > 0:
        tumor_gap_used = tumor_gap[selected_idx]
        tumor_prob_used = tumor_prob[selected_idx]
        top1_gap_used = top1_gap[selected_idx]
        role_logits_used = role_logits[selected_idx]
        pos_context_used = pos_context_score[selected_idx] if pos_context_score is not None else None
        neg_context_used = neg_context_score[selected_idx] if neg_context_score is not None else None
    else:
        tumor_gap_used = tumor_gap
        tumor_prob_used = tumor_prob
        top1_gap_used = top1_gap
        role_logits_used = role_logits
        pos_context_used = pos_context_score
        neg_context_used = neg_context_score

    if slide_label == 1:
        used_context = pos_context_used if pos_context_used is not None else context_score

        generic_weight = None
        if context_weight_mode != "none":
            generic_weight = build_context_weight(
                tumor_prob=tumor_prob_used,
                top1_gap=top1_gap_used,
                context_score=used_context,
                mode=context_weight_mode,
                alpha_tumor_prob=alpha_tumor_prob,
                alpha_top1_gap=alpha_top1_gap,
                alpha_context=alpha_context,
                detach_weight=detach_weight,
            )

        final_weight = (
            build_positive_context_weight(
                tumor_prob=tumor_prob_used,
                top1_gap=top1_gap_used,
                pos_context_score=used_context,
                alpha_tumor_prob=alpha_tumor_prob,
                alpha_top1_gap=alpha_top1_gap,
                alpha_context=alpha_context,
                detach_weight=detach_weight,
            )
            if used_context is not None else generic_weight
        )

        proto_loss = compute_positive_gap_loss(
            tumor_gap=tumor_gap_used,
            margin_pos=margin_pos,
            weight=final_weight,
        )
        loss_mode = "pos"
        pos_loss = proto_loss
        neg_loss = torch.zeros((), device=proto_loss.device)

    else:
        used_context = neg_context_used if neg_context_used is not None else context_score

        generic_weight = None
        if context_weight_mode != "none":
            generic_weight = build_context_weight(
                tumor_prob=tumor_prob_used,
                top1_gap=top1_gap_used,
                context_score=used_context,
                mode=context_weight_mode,
                alpha_tumor_prob=alpha_tumor_prob,
                alpha_top1_gap=alpha_top1_gap,
                alpha_context=alpha_context,
                detach_weight=detach_weight,
            )

        final_weight = (
            build_negative_context_weight(
                tumor_prob=tumor_prob_used,
                top1_gap=top1_gap_used,
                neg_context_score=used_context,
                alpha_tumor_prob=alpha_tumor_prob,
                alpha_top1_gap=alpha_top1_gap,
                alpha_context=alpha_context,
                detach_weight=detach_weight,
            )
            if used_context is not None else generic_weight
        )

        proto_loss = compute_negative_gap_loss(
            tumor_gap=tumor_gap_used,
            margin_neg=margin_neg,
            weight=final_weight,
        )
        loss_mode = "neg"
        pos_loss = torch.zeros((), device=proto_loss.device)
        neg_loss = proto_loss

    stats = {
        "candidate_type": str(candidate_type),
        "loss_mode": loss_mode,
        "num_candidates": int(len(tumor_gap)),
        "num_selected": int(len(tumor_gap_used)),
        "mean_tumor_gap": safe_float(tumor_gap_used.mean()) if tumor_gap_used.numel() > 0 else float("nan"),
        "mean_tumor_prob": safe_float(tumor_prob_used.mean()) if tumor_prob_used.numel() > 0 else float("nan"),
        "mean_top1_gap": safe_float(top1_gap_used.mean()) if top1_gap_used.numel() > 0 else float("nan"),
        "mean_context_score": safe_float(used_context.mean()) if used_context is not None and used_context.numel() > 0 else float("nan"),
        "mean_weight": safe_float(final_weight.mean()) if final_weight is not None else float("nan"),
        "pos_proto_loss": safe_float(pos_loss),
        "neg_proto_loss": safe_float(neg_loss),
        "pos_mean_gap": safe_float(tumor_gap_used.mean()) if slide_label == 1 and tumor_gap_used.numel() > 0 else float("nan"),
        "neg_mean_gap": safe_float(tumor_gap_used.mean()) if slide_label == 0 and tumor_gap_used.numel() > 0 else float("nan"),
    }

    aux = {
        "tumor_gap_all": tumor_gap,
        "tumor_prob_all": tumor_prob,
        "top1_gap_all": top1_gap,
        "tumor_gap_used": tumor_gap_used,
        "tumor_prob_used": tumor_prob_used,
        "top1_gap_used": top1_gap_used,
        "weight": final_weight,
        "used_context_score": used_context,
        "role_logits": role_logits_used,
    }

    return proto_loss, stats, tumor_gap_used, aux


# =========================================================
# weak slide proxy
# =========================================================
def compute_slide_proxy_loss(
    tumor_gap: torch.Tensor,
    slide_label: int,
    topk: int,
):
    if tumor_gap.numel() == 0:
        zero = torch.zeros((), device=tumor_gap.device)
        return zero, 0.0

    k = min(topk, len(tumor_gap))
    score = torch.topk(tumor_gap, k=k, dim=0).values.mean().view(1)
    label = torch.tensor([float(slide_label)], device=score.device)
    loss = F.binary_cross_entropy_with_logits(score, label)
    return loss, safe_float(score)


# =========================================================
# preserve loss
# =========================================================
def compute_preserve_loss(
    patch_feat_adapt: torch.Tensor,
    patch_feat_frozen: torch.Tensor,
):
    z1 = F.normalize(patch_feat_adapt, dim=-1)
    z2 = F.normalize(patch_feat_frozen, dim=-1)
    cos = (z1 * z2).sum(dim=-1)
    return (1.0 - cos).mean()