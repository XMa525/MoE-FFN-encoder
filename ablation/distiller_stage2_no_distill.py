"""
No-distillation Stage2 distiller for ablation.

Purpose:
    w/o distillation = random MoE initialization + role-prototype / WSI / target-style
    objectives, with teacher-student alignment disabled.

Usage:
    Put this file at:
        distillation/distiller_stage2_no_distill.py

    Then import:
        from distillation.distiller_stage2_no_distill import MoEDistillerStage2NoDistill

Notes:
    - This class reuses all role-prototype, WSI bag, hard-negative, context, and other
      non-distillation losses implemented in MoEDistillerStage2.
    - It overrides forward() so the teacher is not used during training.
    - It passes a dummy teacher feature tensor only because the inherited
      compute_stage2_loss() function still calls compute_alignment_loss(); the
      corresponding alignment weights are forced to zero.
"""

from __future__ import annotations

import torch

from distillation.distiller_stage2 import MoEDistillerStage2


class MoEDistillerStage2NoDistill(MoEDistillerStage2):
    """
    Stage2 ablation distiller without teacher-student distillation.

    Compared with the full Stage2:
        - cls_align_weight = 0
        - align_weight = 0
        - teacher forward is skipped
        - all role-prototype / WSI / auxiliary objectives remain controlled by cfg

    This is intended for the paper setting:
        w/o distillation: MoE ✓, distillation ×, role prototype ✓, target FT ✓
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        old_cls = float(self.cls_align_weight)
        old_patch = float(self.align_weight)
        self.cls_align_weight = 0.0
        self.align_weight = 0.0
        self.cfg["disable_distillation"] = True
        self.cfg["disable_distill_loss"] = True

        print(
            "[MoEDistillerStage2NoDistill] teacher-student distillation disabled: "
            f"cls_align_weight {old_cls} -> 0.0, align_weight {old_patch} -> 0.0"
        )

    def forward(
        self,
        images,
        offline_cluster_ids=None,
        is_eval=False,
        wsi_images=None,
        wsi_slide_label=None,
        slide_label_batch=None,
        neighbor_images_list=None,
        slide_id_batch=None,
    ):
        """
        Same interface as MoEDistillerStage2.forward(), but without teacher forward.
        """
        B = images.shape[0]
        mask = self.get_random_mask(B, images.device) if self.use_stage2_mask else None

        student_out, gate_info_list, feature_dict, moe_feature_list = self.student(
            images,
            return_gates=True,
            mask=mask,
            is_eval=is_eval,
            return_features=True,
            offline_cluster_ids=offline_cluster_ids,
        )

        # Student alignment projection is still used by some downstream helper logic.
        # Alignment losses are computed but multiplied by zero in total_loss.
        s_feat_12 = feature_dict["layer_12"]             # [B, N+1, 384]
        s_proj_12 = self.proj_l12(s_feat_12)             # [B, N+1, 1280]

        # compute_alignment_loss() expects teacher tokens with 1 CLS + 4 register tokens + N patches.
        # Student has 1 CLS + N patches, so create a dummy teacher-shaped tensor [B, N+5, 1280].
        B2, n_student_tokens, tea_dim = s_proj_12.shape
        n_patch = n_student_tokens - 1
        t_feat_32 = s_proj_12.new_zeros(B2, n_patch + 5, tea_dim)
        # Make CLS roughly comparable to avoid NaN-like edge cases; weight is still zero.
        t_feat_32[:, 0:1, :] = s_proj_12[:, 0:1, :].detach()
        t_feat_32[:, 5:, :] = s_proj_12[:, 1:, :].detach()

        if self.use_last_moe_output and len(moe_feature_list) > 0:
            spec_feat = moe_feature_list[-1]
        else:
            spec_feat = feature_dict["layer_12"]

        spec_patch = spec_feat[:, 1:, :]
        B3, N, D = spec_patch.shape

        dispatch_weight = self.get_last_dispatch_weight(gate_info_list, B3, N)
        dispatch_mask = self.get_last_dispatch_mask(gate_info_list, B3, N)

        total_loss, loss_dict = self.compute_stage2_loss(
            s_proj_12=s_proj_12,
            t_feat_32=t_feat_32,
            spec_patch=spec_patch,
            dispatch_weight=dispatch_weight,
            dispatch_mask=dispatch_mask,
            gate_info_list=gate_info_list,
            offline_cluster_ids=offline_cluster_ids,
            wsi_images=wsi_images,
            wsi_slide_label=wsi_slide_label,
            slide_label_batch=slide_label_batch,
            is_eval=is_eval,
            center_images=images,
            neighbor_images_list=neighbor_images_list,
            slide_id_batch=slide_id_batch,
        )

        # Make logging explicit.
        loss_dict["distillation_disabled"] = 1.0
        loss_dict["cls_align_weight"] = float(self.cls_align_weight)
        loss_dict["patch_align_weight"] = float(self.align_weight)
        loss_dict["cls_align_weighted"] = 0.0
        loss_dict["patch_align_weighted"] = 0.0

        return total_loss, loss_dict, gate_info_list
