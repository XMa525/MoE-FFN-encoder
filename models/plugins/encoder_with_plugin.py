from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.plugins.shared_role_prototype import PatchRoleSummaryFromSharedProto
from models.plugins.role_aware_tail_plugin import RoleAwareTailWithSharedSummary


def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def set_plugin_train_mode(
    encoder: nn.Module,
    role_proj_head: nn.Module,
    shared_role_proto: nn.Module,
    plugin: nn.Module,
    aggregator: Optional[nn.Module] = None,
    train_encoder: bool = False,
    train_role_proj: bool = False,
    train_shared_proto: bool = False,
    train_plugin: bool = True,
    train_aggregator: bool = True,
):
    set_requires_grad(encoder, train_encoder)
    set_requires_grad(role_proj_head, train_role_proj)
    set_requires_grad(shared_role_proto, train_shared_proto)
    set_requires_grad(plugin, train_plugin)
    if aggregator is not None:
        set_requires_grad(aggregator, train_aggregator)


class EncoderWithRoleAwarePlugin(nn.Module):
    """
    encoder -> proj_l12 -> shared role proto summary -> plugin
    """

    def __init__(
        self,
        encoder: nn.Module,
        role_proj_head: nn.Module,
        shared_role_proto: nn.Module,
        plugin: RoleAwareTailWithSharedSummary,
        use_last_moe_output: bool = True,
        freeze_encoder: bool = True,
        freeze_role_proj: bool = True,
        role_summary_tau: float = 1.0,
        use_plugin: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.role_proj_head = role_proj_head
        self.shared_role_proto = shared_role_proto
        self.plugin = plugin
        self.use_last_moe_output = bool(use_last_moe_output)
        self.use_plugin = bool(use_plugin)

        self.summary_builder = PatchRoleSummaryFromSharedProto(
            shared_role_proto=self.shared_role_proto,
            tau=role_summary_tau,
            use_softmax=True,
        )

        if freeze_encoder:
            set_requires_grad(self.encoder, False)
        if freeze_role_proj:
            set_requires_grad(self.role_proj_head, False)

    def extract_patch_feat(
        self,
        images: torch.Tensor,
        is_eval: bool = False,
        offline_cluster_ids=None,
    ):
        student_out, gate_info_list, feature_dict, moe_feature_list = self.encoder(
            images,
            return_gates=True,
            mask=None,
            is_eval=is_eval,
            return_features=True,
            offline_cluster_ids=offline_cluster_ids,
        )

        if self.use_last_moe_output and len(moe_feature_list) > 0:
            feat = moe_feature_list[-1]
        else:
            feat = feature_dict["layer_12"]

        patch_feat = feat[:, 1:, :]
        return patch_feat, gate_info_list, feature_dict, moe_feature_list

    def forward(
        self,
        images: torch.Tensor,
        is_eval: bool = False,
        offline_cluster_ids=None,
        return_aux: bool = True,
        use_plugin: Optional[bool] = None,
    ) -> Dict[str, Any]:
        patch_feat_raw, gate_info_list, feature_dict, moe_feature_list = self.extract_patch_feat(
            images=images,
            is_eval=is_eval,
            offline_cluster_ids=offline_cluster_ids,
        )

        patch_feat_teacher_space = self.role_proj_head(patch_feat_raw)
        patch_feat_teacher_space = F.normalize(patch_feat_teacher_space, dim=-1)
        role_dict = self.summary_builder(patch_feat_teacher_space)

        apply_plugin = self.use_plugin if use_plugin is None else bool(use_plugin)
        if apply_plugin:
            plugin_out = self.plugin(
                patch_feat=patch_feat_raw,
                patch_role_probs=role_dict["patch_role_probs"],
                patch_role_gaps=role_dict["patch_role_gaps"],
                patch_role_logits=role_dict["patch_role_logits"],
                patch_top1_gap=role_dict["patch_top1_gap"],
                return_aux=return_aux,
            )
            if return_aux:
                patch_feat_plugin, plugin_aux = plugin_out
            else:
                patch_feat_plugin = plugin_out
                plugin_aux = None
        else:
            patch_feat_plugin = patch_feat_raw
            plugin_aux = {"plugin_bypassed": 1.0} if return_aux else None

        return {
            "patch_feat_raw": patch_feat_raw,
            "patch_feat_plugin": patch_feat_plugin,
            "patch_feat_out": patch_feat_plugin,
            "patch_feat_teacher_space": patch_feat_teacher_space,
            "patch_role_logits": role_dict["patch_role_logits"],
            "patch_role_probs": role_dict["patch_role_probs"],
            "patch_role_gaps": role_dict["patch_role_gaps"],
            "patch_top1_gap": role_dict["patch_top1_gap"],
            "plugin_aux": plugin_aux,
            "feature_dict": feature_dict,
            "gate_info_list": gate_info_list,
            "moe_feature_list": moe_feature_list,
            "plugin_enabled": apply_plugin,
        }
