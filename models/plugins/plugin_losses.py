from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def compute_role_proto_anchor_loss(
    current_proto: torch.Tensor,
    init_proto: torch.Tensor,
    normalize: bool = True,
    mode: str = "cosine",
):
    if current_proto.shape != init_proto.shape:
        raise ValueError(
            f"prototype shape mismatch: current={tuple(current_proto.shape)}, init={tuple(init_proto.shape)}"
        )

    if normalize:
        current_proto = F.normalize(current_proto, dim=-1)
        init_proto = F.normalize(init_proto, dim=-1)

    if mode == "cosine":
        sim = (current_proto * init_proto).sum(dim=-1)
        loss = (1.0 - sim).mean()
    elif mode == "l2":
        loss = ((current_proto - init_proto) ** 2).mean()
    else:
        raise ValueError(f"Unsupported anchor mode: {mode}")

    return loss


def compute_plugin_feature_shift_stats(
    patch_feat_raw: torch.Tensor,
    patch_feat_plugin: torch.Tensor,
) -> Dict[str, float]:
    with torch.no_grad():
        raw_n = F.normalize(patch_feat_raw, dim=-1)
        plg_n = F.normalize(patch_feat_plugin, dim=-1)

        cosine = (raw_n * plg_n).sum(dim=-1)
        l2 = ((patch_feat_plugin - patch_feat_raw) ** 2).sum(dim=-1).sqrt()

        return {
            "plugin_feat_cos_mean": float(cosine.mean().detach().cpu()),
            "plugin_feat_cos_min": float(cosine.min().detach().cpu()),
            "plugin_feat_cos_max": float(cosine.max().detach().cpu()),
            "plugin_feat_l2_mean": float(l2.mean().detach().cpu()),
            "plugin_feat_l2_max": float(l2.max().detach().cpu()),
        }


def summarize_plugin_outputs(enc_out: dict) -> Dict[str, float]:
    stats = {}

    if "plugin_aux" in enc_out and enc_out["plugin_aux"] is not None:
        stats.update(enc_out["plugin_aux"])

    if "patch_role_probs" in enc_out:
        stats["patch_role_prob_mean"] = float(
            enc_out["patch_role_probs"].mean().detach().cpu()
        )

    if "patch_top1_gap" in enc_out:
        stats["patch_top1_gap_mean"] = float(
            enc_out["patch_top1_gap"].mean().detach().cpu()
        )

    if "patch_feat_raw" in enc_out and "patch_feat_plugin" in enc_out:
        stats.update(
            compute_plugin_feature_shift_stats(
                patch_feat_raw=enc_out["patch_feat_raw"],
                patch_feat_plugin=enc_out["patch_feat_plugin"],
            )
        )

    return stats


def compute_plugin_feature_preserve_loss(
    patch_feat_raw: torch.Tensor,
    patch_feat_plugin: torch.Tensor,
    mode: str = "cosine",
):
    if patch_feat_raw.shape != patch_feat_plugin.shape:
        raise ValueError(
            f"shape mismatch: raw={tuple(patch_feat_raw.shape)}, plugin={tuple(patch_feat_plugin.shape)}"
        )

    if mode == "l2":
        return ((patch_feat_plugin - patch_feat_raw) ** 2).mean()
    if mode == "cosine":
        raw_n = F.normalize(patch_feat_raw, dim=-1)
        plg_n = F.normalize(patch_feat_plugin, dim=-1)
        return (1.0 - (raw_n * plg_n).sum(dim=-1)).mean()

    raise ValueError(f"Unsupported preserve mode: {mode}")
