import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .masking import generate_block_mask
from distillation.role_prototype_losses import (
    RolePrototypeBank,
    StaticRolePrototypeLoss,
    build_semantic_expert_mask,
    build_hard_role_indices_from_expert_ids,
)

class MoEDistillerStage2(nn.Module):
    """
    Stage-2 specialization refinement

    核心思想：
    - teacher alignment 只是护栏，不再是主任务
    - 主任务变成 expert output representation separation
    - 复用现有 MoEEncoder / MoEFFN / GatingNetwork 接口
    """

    def __init__(
        self,
        student_model,
        teacher_model,
        stu_dim=384,
        tea_dim=1280,
        grid_size=16,
        stage2_cfg=None,
    ):
        super().__init__()

        self.student = student_model
        self.teacher = teacher_model
        self.grid_size = grid_size
        self.cfg = stage2_cfg or {}

        # ---------- stage2 loss weights ----------
        self.align_weight = self.cfg.get("align_weight", 0.10)
        self.cls_align_weight = self.cfg.get("cls_align_weight", 0.02)

        self.proto_sep_weight = self.cfg.get("proto_sep_weight", 1.0)
        self.proto_margin = self.cfg.get("proto_margin", 0.10)

        self.cluster_sep_weight = self.cfg.get("cluster_sep_weight", 0.0)
        self.cluster_margin = self.cfg.get("cluster_margin", 0.10)

        self.intra_compact_weight = self.cfg.get("intra_compact_weight", 0.0)

        self.routing_stability_weight = self.cfg.get("routing_stability_weight", 0.01)
        self.z_loss_weight = float(self.cfg.get("z_loss_weight", 1e-4))

        self.valid_cluster_ids = self.cfg.get("valid_cluster_ids", [0, 1, 4])
        self.use_last_moe_output = self.cfg.get("use_last_moe_output", True)

        self.sp_loss_weight = self.cfg.get("sp_loss_weight", 0.0)
        self.sp_layers = self.cfg.get("sp_layers", [-1])   # 默认只用最后一个 MoE block
        self.sp_use_soft_weight = self.cfg.get("sp_use_soft_weight", False)
        self.sp_min_active_experts = self.cfg.get("sp_min_active_experts", 2)

        self.use_expert_floor = self.cfg.get("use_expert_floor", False)
        self.expert_floor_weight = self.cfg.get("expert_floor_weight", 0.0)
        self.expert_floor_tau = self.cfg.get("expert_floor_tau", 0.03)
        self.expert_floor_layers = self.cfg.get("expert_floor_layers", [0])

        self.role_proto_weight = self.cfg.get("role_proto_weight", 0.0)
        self.role_tau = self.cfg.get("role_tau", 0.07)
        self.role_attraction_weight = self.cfg.get("role_attraction_weight", 1.0)
        self.role_separation_weight = self.cfg.get("role_separation_weight", 0.5)
        self.role_target_weight = self.cfg.get("role_target_weight", 1.0)
        self.role_margin = self.cfg.get("role_margin", 0.05)

        self.enable_role_proto = self.role_proto_weight > 0
        self.free_expert_id = self.cfg.get("free_expert_id", 3)
        self.role_proto_dir = self.cfg.get("role_proto_dir", None)

        self.role_use_conf_mask = bool(self.cfg.get("role_use_conf_mask", True))
        self.role_conf_thresh = float(self.cfg.get("role_conf_thresh", 0.45))
        self.max_role_tokens_per_batch = int(self.cfg.get("max_role_tokens_per_batch", 0))  # 0 = no cap

        self.role_use_margin_mask = bool(self.cfg.get("role_use_margin_mask", False))
        self.role_margin_conf_thresh = float(self.cfg.get("role_margin_conf_thresh", 0.08))     

        self.use_stage2_mask = bool(self.cfg.get("use_stage2_mask", False))

        # # ---------- B: ambiguous role weakening ----------
        self.use_ambiguous_role_weakening = bool(
            self.cfg.get("use_ambiguous_role_weakening", False)
        )
        self.ambiguous_role_name = self.cfg.get("ambiguous_role_name", "ambiguous")
        self.ambiguous_target_scale = float(self.cfg.get("ambiguous_target_scale", 1.0))
        self.ambiguous_attraction_scale = float(self.cfg.get("ambiguous_attraction_scale", 1.0))
        self.ambiguous_separation_scale = float(self.cfg.get("ambiguous_separation_scale", 1.0))
        self.ambiguous_only_target_ce = bool(
            self.cfg.get("ambiguous_only_target_ce", False)
        )

        # ---------- C1: balanced semantic sampling ----------
        self.role_balance_semantic_sampling = bool(
            self.cfg.get("role_balance_semantic_sampling", False)
        )
        self.max_role_tokens_per_expert = int(
            self.cfg.get("max_role_tokens_per_expert", 0)
        )  # 0 = disabled

        # ---------- C2: free expert floor ----------
        self.use_free_expert_floor = bool(self.cfg.get("use_free_expert_floor", False))
        self.free_expert_floor_weight = float(self.cfg.get("free_expert_floor_weight", 0.0))
        self.free_expert_floor_tau = float(self.cfg.get("free_expert_floor_tau", 0.03))

        # ---------- D: tumor-aware asymmetric role weighting ----------
        self.use_role_asymmetric_weighting = bool(
            self.cfg.get("use_role_asymmetric_weighting", False)
        )

        self.tumor_role_name = self.cfg.get("tumor_role_name", "tumor")
        self.stroma_role_name = self.cfg.get("stroma_role_name", "stroma")

        self.tumor_target_scale = float(self.cfg.get("tumor_target_scale", 1.5))
        self.tumor_attraction_scale = float(self.cfg.get("tumor_attraction_scale", 1.5))
        self.tumor_separation_scale = float(self.cfg.get("tumor_separation_scale", 0.5))

        self.stroma_target_scale = float(self.cfg.get("stroma_target_scale", 1.0))
        self.stroma_attraction_scale = float(self.cfg.get("stroma_attraction_scale", 1.0))
        self.stroma_separation_scale = float(self.cfg.get("stroma_separation_scale", 0.5))

        # ---------- E: tumor-over-ambiguous preference ----------
        self.use_tumor_preference_loss = bool(
            self.cfg.get("use_tumor_preference_loss", False)
        )
        self.tumor_preference_weight = float(
            self.cfg.get("tumor_preference_weight", 0.2)
        )
        self.tumor_pref_margin = float(
            self.cfg.get("tumor_pref_margin", 0.05)
        )
        self.tumor_candidate_min_sim = float(
            self.cfg.get("tumor_candidate_min_sim", 0.45)
        )
        self.tumor_amb_near_margin = float(
            self.cfg.get("tumor_amb_near_margin", 0.10)
        )

        # ---------- F: WSI bag-level tumor evidence constraint ----------
        self.use_wsi_bag_loss = bool(self.cfg.get("use_wsi_bag_loss", False))
        self.wsi_bag_loss_weight = float(self.cfg.get("wsi_bag_loss_weight", 0.1))

        self.use_wsi_bag_margin_loss = bool(self.cfg.get("use_wsi_bag_margin_loss", False))
        self.wsi_bag_margin_weight = float(self.cfg.get("wsi_bag_margin_weight", 0.1))
        self.wsi_bag_margin = float(self.cfg.get("wsi_bag_margin", 0.10))

        self.wsi_topk_ratio = float(self.cfg.get("wsi_topk_ratio", 0.1))
        self.wsi_topk_min = int(self.cfg.get("wsi_topk_min", 4))
        self.wsi_topk_max = int(self.cfg.get("wsi_topk_max", 16))
        self.wsi_patch_batch_size = int(self.cfg.get("wsi_patch_batch_size", 8))

        # ---------- F2: asymmetric WSI bag loss ----------
        self.use_wsi_asymmetric_loss = bool(
            self.cfg.get("use_wsi_asymmetric_loss", False)
        )

        # BCE weights
        self.wsi_pos_bce_weight = float(
            self.cfg.get("wsi_pos_bce_weight", self.wsi_bag_loss_weight)
        )
        self.wsi_neg_bce_weight = float(
            self.cfg.get("wsi_neg_bce_weight", self.wsi_bag_loss_weight)
        )

        # margin weights
        self.wsi_pos_margin_weight = float(
            self.cfg.get("wsi_pos_margin_weight", self.wsi_bag_margin_weight)
        )
        self.wsi_neg_margin_weight = float(
            self.cfg.get("wsi_neg_margin_weight", self.wsi_bag_margin_weight)
        )

        # margin targets
        self.wsi_pos_margin = float(
            self.cfg.get("wsi_pos_margin", self.wsi_bag_margin)
        )
        self.wsi_neg_margin = float(
            self.cfg.get("wsi_neg_margin", self.wsi_bag_margin)
        )

        self.use_wsi_pos_tail_protect = bool(self.cfg.get("use_wsi_pos_tail_protect", False))
        self.wsi_pos_tail_protect_weight = float(self.cfg.get("wsi_pos_tail_protect_weight", 0.0))
        self.wsi_pos_tail_ratio = float(self.cfg.get("wsi_pos_tail_ratio", 0.01))   # 先试 top1%
        self.wsi_pos_tail_min = int(self.cfg.get("wsi_pos_tail_min", 2))
        self.wsi_pos_tail_floor = float(self.cfg.get("wsi_pos_tail_floor", 0.00))   # 很宽松，先从 0 开始

        # ---------- G: hard-negative repulsion ----------
        self.use_hn_repulsion_loss = bool(self.cfg.get("use_hn_repulsion_loss", False))
        self.hn_repulsion_weight = float(self.cfg.get("hn_repulsion_weight", 0.0))
        self.hn_repulsion_margin = float(self.cfg.get("hn_repulsion_margin", 0.05))

        self.hn_bank_dir = self.cfg.get("hn_bank_dir", None)
        self.hn_use_classes = list(self.cfg.get("hn_use_classes", ["gland_like", "fibrous_dense"]))
        self.hn_batch_size_per_class = int(self.cfg.get("hn_batch_size_per_class", 64))
        self.hn_l2_normalize_bank = bool(self.cfg.get("hn_l2_normalize_bank", True))

        # ---------- G2: online hard-negative repulsion ----------
        self.use_online_hn_repulsion_loss = bool(
            self.cfg.get("use_online_hn_repulsion_loss", False)
        )
        self.online_hn_repulsion_weight = float(
            self.cfg.get("online_hn_repulsion_weight", 0.0)
        )
        self.online_hn_repulsion_margin = float(
            self.cfg.get("online_hn_repulsion_margin", 0.05)
        )

        self.online_hn_use_classes = list(
            self.cfg.get("online_hn_use_classes", self.hn_use_classes)
        )
        self.online_hn_topk_per_class = int(
            self.cfg.get("online_hn_topk_per_class", 128)
        )
        self.online_hn_min_sim_to_center = float(
            self.cfg.get("online_hn_min_sim_to_center", -1.0)
        )
        self.online_hn_require_tumor_dominant = bool(
            self.cfg.get("online_hn_require_tumor_dominant", True)
        )

        self.online_hn_max_delta = float(
            self.cfg.get("online_hn_max_delta", 1e6)
        )
        self.online_hn_gland_like_min_sim_to_center = float(
            self.cfg.get("online_hn_gland_like_min_sim_to_center", self.online_hn_min_sim_to_center)
        )
        self.online_hn_fibrous_dense_min_sim_to_center = float(
            self.cfg.get("online_hn_fibrous_dense_min_sim_to_center", self.online_hn_min_sim_to_center)
        )

        # ---------- G3: negative-only context inconsistency HN ----------
        self.use_negative_context_hn_loss = bool(
            self.cfg.get("use_negative_context_hn_loss", False)
        )
        self.negative_context_hn_weight = float(
            self.cfg.get("negative_context_hn_weight", 0.0)
        )

        # negative trigger
        self.neg_ctx_trigger_score = float(
            self.cfg.get("neg_ctx_trigger_score", 0.10)
        )
        self.neg_ctx_neighbor_support_max = float(
            self.cfg.get("neg_ctx_neighbor_support_max", 0.00)
        )
        self.neg_ctx_gap_margin = float(
            self.cfg.get("neg_ctx_gap_margin", 0.10)
        )

        # neighbor reduce mode
        self.neg_ctx_use_topk_neighbor = bool(
            self.cfg.get("neg_ctx_use_topk_neighbor", True)
        )
        self.neg_ctx_topk_ratio = float(
            self.cfg.get("neg_ctx_topk_ratio", 0.25)
        )
        self.neg_ctx_topk_min = int(
            self.cfg.get("neg_ctx_topk_min", 2)
        )
        self.neg_ctx_topk_max = int(
            self.cfg.get("neg_ctx_topk_max", 8)
        )
        self.neg_ctx_min_neighbors = int(
            self.cfg.get("neg_ctx_min_neighbors", 1)
        )

        # ---------- G4: positive context protection / monitoring ----------
        self.use_positive_context_protection = bool(
            self.cfg.get("use_positive_context_protection", True)
        )
        self.positive_context_protect_weight = float(
            self.cfg.get("positive_context_protect_weight", 0.0)
        )

        self.pos_ctx_trigger_score = float(
            self.cfg.get("pos_ctx_trigger_score", 0.10)
        )
        self.pos_ctx_neighbor_support_min = float(
            self.cfg.get("pos_ctx_neighbor_support_min", 0.05)
        )
        self.pos_ctx_score_floor = float(
            self.cfg.get("pos_ctx_score_floor", 0.05)
        )
        # ---------- G4.5: batch context-guided loss (主线邻域机制) ----------
        self.use_batch_context_guided_loss = bool(
            self.cfg.get("use_batch_context_guided_loss", True)
        )
        self.batch_context_loss_weight = float(
            self.cfg.get("batch_context_loss_weight", 0.06)
        )

        # negative: center高但neighbor低 -> 压
        self.batch_ctx_neg_center_min = float(
            self.cfg.get("batch_ctx_neg_center_min", 0.08)
        )
        self.batch_ctx_neg_neighbor_max = float(
            self.cfg.get("batch_ctx_neg_neighbor_max", 0.00)
        )
        self.batch_ctx_neg_margin = float(
            self.cfg.get("batch_ctx_neg_margin", 0.08)
        )

        # positive: center高且neighbor支持 -> 保
        self.batch_ctx_pos_center_min = float(
            self.cfg.get("batch_ctx_pos_center_min", 0.00)
        )
        self.batch_ctx_pos_neighbor_min = float(
            self.cfg.get("batch_ctx_pos_neighbor_min", 0.03)
        )
        self.batch_ctx_pos_margin = float(
            self.cfg.get("batch_ctx_pos_margin", 0.10)
        )

        # neighbor aggregation
        self.batch_ctx_use_topk_neighbor = bool(
            self.cfg.get("batch_ctx_use_topk_neighbor", True)
        )
        self.batch_ctx_topk_ratio = float(
            self.cfg.get("batch_ctx_topk_ratio", 0.25)
        )
        self.batch_ctx_topk_min = int(
            self.cfg.get("batch_ctx_topk_min", 2)
        )
        self.batch_ctx_topk_max = int(
            self.cfg.get("batch_ctx_topk_max", 8)
        )

        # soft context score:
        # negative 用 center - neighbor_ref
        # positive 用 center + pos_neighbor_scale * neighbor_ref
        self.batch_ctx_pos_neighbor_scale = float(
            self.cfg.get("batch_ctx_pos_neighbor_scale", 0.7)
        )
        # ---------- G5: negative-slide global top-k suppression ----------
        self.use_neg_global_topk_suppression = bool(
            self.cfg.get("use_neg_global_topk_suppression", False)
        )
        self.neg_global_topk_weight = float(
            self.cfg.get("neg_global_topk_weight", 0.0)
        )
        self.neg_global_topk_margin = float(
            self.cfg.get("neg_global_topk_margin", 0.0)
        )
        # ---------- H: asymmetric conditional ranking ----------
        self.use_conditional_pairwise_ranking = bool(
            self.cfg.get("use_conditional_pairwise_ranking", False)
        )

        # overall weights
        self.cond_rank_neg_weight = float(
            self.cfg.get("cond_rank_neg_weight", 0.05)
        )
        self.cond_rank_pos_weight = float(
            self.cfg.get("cond_rank_pos_weight", 0.02)
        )

        # margins
        self.cond_rank_neg_margin = float(
            self.cfg.get("cond_rank_neg_margin", 0.10)
        )
        self.cond_rank_pos_margin = float(
            self.cfg.get("cond_rank_pos_margin", 0.05)
        )

        # top-k selection
        self.cond_rank_neg_topk = int(
            self.cfg.get("cond_rank_neg_topk", 8)
        )
        self.cond_rank_pos_topk = int(
            self.cfg.get("cond_rank_pos_topk", 4)
        )

        # optional ratio fallback / compatibility
        self.cond_rank_neg_topk_ratio = float(
            self.cfg.get("cond_rank_neg_topk_ratio", 0.0)
        )
        self.cond_rank_pos_topk_ratio = float(
            self.cfg.get("cond_rank_pos_topk_ratio", 0.0)
        )

        # positive token selection mode:
        # "tumor_minus_other" | "tumor"
        self.cond_rank_pos_select_mode = str(
            self.cfg.get("cond_rank_pos_select_mode", "tumor_minus_other")
        )

        # extra confidence gate for positive tokens
        self.cond_rank_pos_min_tumor_score = float(
            self.cfg.get("cond_rank_pos_min_tumor_score", -1e6)
        )
        self.cond_rank_pos_min_gap = float(
            self.cfg.get("cond_rank_pos_min_gap", -1e6)
        )

        # whether to skip positive ranking when no valid tokens remain
        self.cond_rank_allow_empty_pos = bool(
            self.cfg.get("cond_rank_allow_empty_pos", True)
        )

        # ---------- H1.4: positive expert-aware selection ----------
        self.use_pos_expert_balanced_selection = bool(
            self.cfg.get("use_pos_expert_balanced_selection", True)
        )

        # 只对 semantic experts 做 positive balanced selection
        self.pos_semantic_expert_ids = list(
            self.cfg.get("pos_semantic_expert_ids", [0, 1, 2])
        )

        # 每个 expert 最多选多少个 positive token
        self.cond_rank_pos_topk_per_expert = int(
            self.cfg.get("cond_rank_pos_topk_per_expert", 2)
        )

        # 如果某些 expert 没 token，是否允许只用有 token 的 expert
        self.cond_rank_pos_allow_partial_experts = bool(
            self.cfg.get("cond_rank_pos_allow_partial_experts", True)
        )

        # ---------- H1.5: negative preselection (front-loaded selection) ----------
        self.use_neg_preselection = bool(
            self.cfg.get("use_neg_preselection", True)
        )
        self.neg_preselect_target_expert = self.cfg.get("neg_preselect_target_expert", 0)
        if self.neg_preselect_target_expert is not None:
            self.neg_preselect_target_expert = int(self.neg_preselect_target_expert)

        self.neg_preselect_min_tumor_score = float(
            self.cfg.get("neg_preselect_min_tumor_score", 0.15)
        )
        self.neg_preselect_min_gap = float(
            self.cfg.get("neg_preselect_min_gap", -0.05)
        )
        self.neg_preselect_max_gap = float(
            self.cfg.get("neg_preselect_max_gap", 0.10)
        )
        self.neg_preselect_require_tumor_dominant = bool(
            self.cfg.get("neg_preselect_require_tumor_dominant", False)
        )
        self.neg_preselect_allow_fallback = bool(
            self.cfg.get("neg_preselect_allow_fallback", True)
        )

        # ---------- H1.6: context-aware token selection ----------
        self.use_context_guided_selection = bool(
            self.cfg.get("use_context_guided_selection", True)
        )

        # context score = alpha * center_gap + beta * nb_topk_mean + gamma * nb_topk_max
        self.ctx_score_alpha = float(self.cfg.get("ctx_score_alpha", 1.0))
        self.ctx_score_beta = float(self.cfg.get("ctx_score_beta", 0.8))
        self.ctx_score_gamma = float(self.cfg.get("ctx_score_gamma", 0.3))

        # positive 侧可额外奖励“中心-邻域一致”
        self.ctx_pos_consistency_weight = float(
            self.cfg.get("ctx_pos_consistency_weight", 0.20)
        )

        # negative 侧可额外奖励“中心高、邻域低”的孤立性（更像伪阳）
        self.ctx_neg_isolation_weight = float(
            self.cfg.get("ctx_neg_isolation_weight", 0.25)
        )

        # 选择邻域聚合方式
        self.ctx_use_topk_neighbor = bool(
            self.cfg.get("ctx_use_topk_neighbor", True)
        )
        self.ctx_topk_ratio = float(
            self.cfg.get("ctx_topk_ratio", 0.25)
        )
        self.ctx_topk_min = int(
            self.cfg.get("ctx_topk_min", 2)
        )
        self.ctx_topk_max = int(
            self.cfg.get("ctx_topk_max", 8)
        )

        # 若没有邻域，则是否退化为单点分数
        self.ctx_allow_no_neighbor_fallback = bool(
            self.cfg.get("ctx_allow_no_neighbor_fallback", True)
        )    

        # ---------- H1.7: positive context ranking ----------
        self.use_positive_context_ranking = bool(
            self.cfg.get("use_positive_context_ranking", False)
        )

        self.pos_ctx_rank_center_weight = float(
            self.cfg.get("pos_ctx_rank_center_weight", 1.0)
        )
        self.pos_ctx_rank_neighbor_weight = float(
            self.cfg.get("pos_ctx_rank_neighbor_weight", 1.0)
        )
        self.pos_ctx_rank_neighbor_max_weight = float(
            self.cfg.get("pos_ctx_rank_neighbor_max_weight", 0.3)
        )
        self.pos_ctx_rank_support_weight = float(
            self.cfg.get("pos_ctx_rank_support_weight", 0.3)
        )
        self.pos_ctx_rank_consistency_weight = float(
            self.cfg.get("pos_ctx_rank_consistency_weight", 0.5)
        )

        # ---------- H1.8: positive support mask ----------
        self.use_pos_support_mask = bool(
            self.cfg.get("use_pos_support_mask", False)
        )
        self.pos_support_center_min = float(
            self.cfg.get("pos_support_center_min", -1e6)
        )
        self.pos_support_neighbor_min = float(
            self.cfg.get("pos_support_neighbor_min", -1e6)
        )
        self.pos_support_neighbor_max_min = float(
            self.cfg.get("pos_support_neighbor_max_min", -1e6)
        )
        self.pos_support_min_tumor_score = float(
            self.cfg.get("pos_support_min_tumor_score", -1e6)
        )
        self.pos_support_allow_fallback = bool(
            self.cfg.get("pos_support_allow_fallback", True)
        )
        self.pos_min_selected_experts = int(
            self.cfg.get("pos_min_selected_experts", 2)
        )
        # ---------- H1.8b: per-expert positive quota ----------
        # 用于限制 positive supervision 不要再次过度流向 E0
        # 默认：E0 更严格，E1/E2 正常
        default_pos_quota_map = {
            0: 1,
            1: self.cond_rank_pos_topk_per_expert,
            2: self.cond_rank_pos_topk_per_expert,
        }
        raw_pos_quota_map = self.cfg.get(
            "pos_topk_per_expert_map",
            default_pos_quota_map,
        )
        self.pos_topk_per_expert_map = {
            int(k): int(v) for k, v in raw_pos_quota_map.items()
        }

        # 当 expert 数不足时，是否直接阻断 positive cond-rank
        # 你当前主线建议开着
        self.pos_require_min_experts = bool(
            self.cfg.get("pos_require_min_experts", True)
        )

        # ---------- H1.10: positive slide dedup ----------
        self.use_pos_slide_dedup = bool(
            self.cfg.get("use_pos_slide_dedup", True)
        )
        self.pos_max_tokens_per_slide_per_expert = int(
            self.cfg.get("pos_max_tokens_per_slide_per_expert", 1)
        )

        # ---------- H2: free-expert preference for negative ranking tokens ----------
        self.use_neg_rank_free_preference = bool(
            self.cfg.get("use_neg_rank_free_preference", False)
        )
        self.neg_rank_free_pref_weight = float(
            self.cfg.get("neg_rank_free_pref_weight", 0.0)
        )
        self.neg_rank_free_pref_eps = float(
            self.cfg.get("neg_rank_free_pref_eps", 1e-6)
        )

        # 可选：只对足够“边界”的 negative selected token 生效
        # 若 <= -1e5 则视为不启用过滤
        self.neg_rank_free_pref_min_gap = float(
            self.cfg.get("neg_rank_free_pref_min_gap", -1e6)
        )
        self.neg_rank_free_pref_max_gap = float(
            self.cfg.get("neg_rank_free_pref_max_gap", 1e6)
        )

        # ---------- H3: residual HN push ----------
        self.use_residual_hn_push = bool(
            self.cfg.get("use_residual_hn_push", False)
        )
        self.residual_hn_push_weight = float(
            self.cfg.get("residual_hn_push_weight", 0.0)
        )
        self.residual_hn_push_margin = float(
            self.cfg.get("residual_hn_push_margin", 0.05)
        )

        # 只打 still-positive residual HN
        self.residual_hn_min_gap = float(
            self.cfg.get("residual_hn_min_gap", 0.0)
        )
        self.residual_hn_max_gap = float(
            self.cfg.get("residual_hn_max_gap", 1e6)
        )

        self.residual_hn_target_expert = self.cfg.get("residual_hn_target_expert", None)
        if self.residual_hn_target_expert is not None:
            self.residual_hn_target_expert = int(self.residual_hn_target_expert)

        # ---------- H4: positive anchor boost ----------
        self.use_positive_anchor_boost = bool(
            self.cfg.get("use_positive_anchor_boost", False)
        )
        self.positive_anchor_weight = float(
            self.cfg.get("positive_anchor_weight", 0.0)
        )
        self.positive_anchor_margin = float(
            self.cfg.get("positive_anchor_margin", 0.08)
        )

        self.positive_anchor_topk = int(
            self.cfg.get("positive_anchor_topk", 4)
        )
        self.positive_anchor_topk_ratio = float(
            self.cfg.get("positive_anchor_topk_ratio", 0.0)
        )

        self.positive_anchor_select_mode = str(
            self.cfg.get("positive_anchor_select_mode", "tumor_minus_other")
        )

        self.positive_anchor_min_tumor_score = float(
            self.cfg.get("positive_anchor_min_tumor_score", -1e6)
        )
        self.positive_anchor_min_gap = float(
            self.cfg.get("positive_anchor_min_gap", 0.0)
        )

        self.positive_anchor_allow_empty = bool(
            self.cfg.get("positive_anchor_allow_empty", True)
        )

        # 可选：正样本只看 top-k tumor evidence；
        # 负样本也同样看 top-k，但会通过 BCE / margin 压低 tumor evidence
        self.wsi_use_prob_head = bool(self.cfg.get("wsi_use_prob_head", True))

        if self.wsi_use_prob_head:
            self.wsi_bag_classifier = nn.Linear(tea_dim, 1)
            nn.init.xavier_uniform_(self.wsi_bag_classifier.weight)
            nn.init.zeros_(self.wsi_bag_classifier.bias)
        else:
            self.wsi_bag_classifier = None

        if self.enable_role_proto:
            assert self.role_proto_dir is not None, "role_proto_dir must be set when role_proto_weight > 0"

            self.role_bank = RolePrototypeBank(
                prototype_path=os.path.join(self.role_proto_dir, "role_prototypes_init.npy"),
                role_names_path=os.path.join(self.role_proto_dir, "role_names.json"),
                normalize=True,
            )

            self.role_criterion = StaticRolePrototypeLoss(
                prototype_bank=self.role_bank,
                tau=self.role_tau,
                attraction_weight=self.role_attraction_weight,
                separation_weight=self.role_separation_weight,
                target_weight=self.role_target_weight,
                margin=self.role_margin,
            )
        else:
            self.role_bank = None
            self.role_criterion = None

        if self.enable_role_proto and self.role_bank is not None:
            role_names = list(self.role_bank.role_names)

            self.tumor_role_id = role_names.index(self.tumor_role_name) if self.tumor_role_name in role_names else None
            self.stroma_role_id = role_names.index(self.stroma_role_name) if self.stroma_role_name in role_names else None
            self.ambiguous_role_id = role_names.index(self.ambiguous_role_name) if self.ambiguous_role_name in role_names else None
        else:
            self.tumor_role_id = None
            self.stroma_role_id = None
            self.ambiguous_role_id = None
        # ---------- mask bank ----------
        self.mask_pool_size = self.cfg.get("mask_pool_size", 4096)
        self.mask_ratio = self.cfg.get("mask_ratio", 0.3)
        self.register_buffer(
            "mask_bank",
            generate_block_mask(
                batch_size=self.mask_pool_size,
                grid_size=self.grid_size,
                mask_ratio=self.mask_ratio,
            )
        )

        # ---------- teacher frozen ----------
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # ---------- stage2 freeze policy ----------
        self._apply_stage2_freeze_strategy()

        # ---------- light alignment head ----------
        self.proj_l12 = nn.Linear(stu_dim, tea_dim)
        nn.init.xavier_uniform_(self.proj_l12.weight)

        self.tea_features = {}

        # ---------- hard-negative bank ----------
        self.hn_banks = {}
        self.hn_bank_sizes = {}
        self.hn_centers = {}

        if self.use_hn_repulsion_loss or self.use_online_hn_repulsion_loss:
            assert self.hn_bank_dir is not None, (
                "hn_bank_dir must be set when use_hn_repulsion_loss=True "
                "or use_online_hn_repulsion_loss=True"
            )
            self._load_hn_repulsion_banks()

        self._register_hooks()

    # =========================================================
    # setup
    # =========================================================
    def _load_hn_repulsion_banks(self):
        """
        从 analysis_outputs/hn_repulsion_bank_v1 读取:
            gland_like_features.npy
            fibrous_dense_features.npy
            ...
        存到:
            self.hn_banks[class_name] = [N, D_teacher]
            self.hn_centers[class_name] = [D_teacher]
        """
        self.hn_centers = {}

        for cls_name in self.hn_use_classes:
            feat_path = os.path.join(self.hn_bank_dir, f"{cls_name}_features.npy")
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"HN bank feature file not found: {feat_path}")

            arr = np.load(feat_path)   # [N, D]
            if arr.ndim != 2:
                raise ValueError(f"HN bank must be 2D, got shape {arr.shape} for {feat_path}")

            bank = torch.from_numpy(arr).float()
            if self.hn_l2_normalize_bank:
                bank = F.normalize(bank, dim=-1)

            center = bank.mean(dim=0)
            center = F.normalize(center, dim=-1)

            self.register_buffer(f"hn_bank_{cls_name}", bank)
            self.register_buffer(f"hn_center_{cls_name}", center)

            self.hn_banks[cls_name] = getattr(self, f"hn_bank_{cls_name}")
            self.hn_centers[cls_name] = getattr(self, f"hn_center_{cls_name}")
            self.hn_bank_sizes[cls_name] = int(bank.shape[0])

            print(
                f"[HN bank] loaded {cls_name}: "
                f"bank_shape={tuple(bank.shape)}, center_shape={tuple(center.shape)}"
            )

    def _register_hooks(self):
        def get_tea_hook(name):
            def hook(module, input, output):
                self.tea_features[name] = output[0] if isinstance(output, tuple) else output
            return hook

        # 仍然沿用你一阶段 teacher 的高层特征
        self.teacher.blocks[31].register_forward_hook(get_tea_hook("layer_32"))

    def _apply_stage2_freeze_strategy(self):
        """
        第二阶段只训练：
        - experts
        - shared_expert
        - gate routing_proj
        - gate_vectors
        - expert_threshold
        - logit_scale
        - 少量 norm
        """
        # for n, p in self.student.named_parameters():
        #     if "routing_proj" in n:
        #         print("ROUTING_PROJ:", n)

        # for n, p in self.student.named_parameters():
        #     if "norm" in n.lower():
        #         print("NORM:", n)

        # for n, p in self.student.named_parameters():
        #     if ".mlp.gate." in n:
        #         print("GATE:", n)
        # 总层数
        num_blocks = len(self.student.base_encoder.model.encoder.layer)

        # 把 [-3, -2] 这种相对索引转成真实层号 [9, 10]
        moe_block_ids = set()
        for idx in self.student.moe_layers_idx:
            real_idx = idx if idx >= 0 else num_blocks + idx
            moe_block_ids.add(real_idx)
        for name, p in self.student.named_parameters():
            p.requires_grad = False

            # expert FFN
            if ".mlp.experts." in name:
                p.requires_grad = True

            if ".mlp.shared_expert." in name:
                p.requires_grad = True

            # gate params
            if ".mlp.gate.routing_proj." in name:
                p.requires_grad = True
            if ".mlp.gate.gate_vectors" in name:
                p.requires_grad = True
            if ".mlp.gate.expert_threshold" in name:
                p.requires_grad = True
            if ".mlp.gate.logit_scale" in name:
                p.requires_grad = True
            if ".mlp.patch_context_proj." in name:
                p.requires_grad = True

            # 3) 只解冻 MoE block 自己的 norm1 / norm2
            for block_id in moe_block_ids:
                if f"base_encoder.model.encoder.layer.{block_id}.norm1." in name or \
                    f"base_encoder.model.encoder.layer.{block_id}.norm2." in name:
                        p.requires_grad = True
            if p.requires_grad and "norm" in name.lower():
                print("[Unfrozen norm]", name)        

    # =========================================================
    # helpers
    # =========================================================
    def split_tokens(self, feat, is_teacher=False):
        cls_token = feat[:, 0:1, :]
        patch_tokens = feat[:, 5:, :] if is_teacher else feat[:, 1:, :]
        return cls_token, patch_tokens

    def get_random_mask(self, B, device):
        rand_indices = torch.randint(0, self.mask_pool_size, (B,), device=device)
        return self.mask_bank[rand_indices]

    def weighted_smooth_l1(self, pred, target, token_mask):
        diff = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)  # [B, N]
        token_mask = token_mask.float()
        return (diff * token_mask).sum() / (token_mask.sum() + 1e-6)

    def build_valid_cluster_mask(self, cluster_ids):
        valid_ids = torch.tensor(self.valid_cluster_ids, device=cluster_ids.device)
        return (cluster_ids.unsqueeze(-1) == valid_ids.view(1, 1, -1)).any(dim=-1)

    def get_last_dispatch_weight(self, gate_info_list, B, N):
        dispatch_weight = gate_info_list[-1]["dispatch_weight"]   # [B*(N+1), E]
        E = dispatch_weight.shape[-1]
        dispatch_weight = dispatch_weight.view(B, N + 1, E)
        dispatch_weight = dispatch_weight[:, 1:, :]  # remove CLS
        return dispatch_weight

    def get_last_dispatch_mask(self, gate_info_list, B, N):
        dispatch_mask = gate_info_list[-1]["dispatch_mask"].float()
        E = dispatch_mask.shape[-1]
        dispatch_mask = dispatch_mask.view(B, N + 1, E)
        dispatch_mask = dispatch_mask[:, 1:, :]
        return dispatch_mask

    def get_last_score(self, gate_info_list, B, N):
        score = gate_info_list[-1]["score"]   # [B*(N+1), E]
        E = score.shape[-1]
        score = score.view(B, N + 1, E)
        score = score[:, 1:, :]
        return score
    
    def get_layer_dispatch_weight(self, gate_info_list, layer_idx, B, N):
        """
        return:
            dispatch_weight: [B, N, E]
        """
        dispatch_weight = gate_info_list[layer_idx]["dispatch_weight"]   # [B*(N+1), E]
        E = dispatch_weight.shape[-1]
        dispatch_weight = dispatch_weight.view(B, N + 1, E)
        dispatch_weight = dispatch_weight[:, 1:, :]   # remove CLS
        return dispatch_weight
    
    def get_layer_expert_outputs(self, gate_info_list, layer_idx, B, N):
        """
        return:
            expert_outputs: [B, N, E, D]
        """
        expert_outputs = gate_info_list[layer_idx]["expert_outputs"]   # [B*(N+1), E, D]
        E = expert_outputs.shape[1]
        D = expert_outputs.shape[2]

        expert_outputs = expert_outputs.view(B, N + 1, E, D)
        expert_outputs = expert_outputs[:, 1:, :, :]   # remove CLS
        return expert_outputs

    # =========================================================
    # loss: guardrail alignment
    # =========================================================
    def compute_alignment_loss(self, s_proj_12, t_feat_32):
        s_cls, s_patch = self.split_tokens(s_proj_12, is_teacher=False)
        t_cls, t_patch = self.split_tokens(t_feat_32, is_teacher=True)

        target = torch.ones(s_cls.shape[0], device=s_cls.device)
        loss_cls = F.cosine_embedding_loss(
            s_cls.squeeze(1), t_cls.squeeze(1), target
        )

        token_mask = torch.ones(
            s_patch.shape[:2], device=s_patch.device, dtype=s_patch.dtype
        )
        loss_patch = self.weighted_smooth_l1(s_patch, t_patch, token_mask)

        return loss_cls, loss_patch

    # =========================================================
    # loss: output representation specialization
    # =========================================================
    def compute_expert_output_prototypes(self, feat_patch, dispatch_weight):
        """
        feat_patch: [B, N, D]
        dispatch_weight: [B, N, E]
        """
        B, N, D = feat_patch.shape
        E = dispatch_weight.shape[-1]

        feat_flat = feat_patch.reshape(B * N, D)
        weight_flat = dispatch_weight.reshape(B * N, E)

        protos = []
        masses = []

        for e in range(E):
            w = weight_flat[:, e:e+1]                      # [BN, 1]
            mass = w.sum().clamp_min(1e-6)
            proto = (feat_flat * w).sum(dim=0) / mass
            protos.append(proto)
            masses.append(mass)

        protos = torch.stack(protos, dim=0)               # [E, D]
        masses = torch.stack(masses, dim=0)               # [E]
        return protos, masses

    def compute_proto_sep_loss(self, feat_patch, dispatch_weight):
        protos, masses = self.compute_expert_output_prototypes(feat_patch, dispatch_weight)
        protos = F.normalize(protos, dim=-1)
        sim_mat = protos @ protos.t()   # [E, E]

        E = sim_mat.shape[0]
        losses = []
        sims = []

        for i in range(E):
            for j in range(E):
                if i == j:
                    continue
                sims.append(sim_mat[i, j].detach())
                losses.append(F.relu(sim_mat[i, j] - self.proto_margin))

        if len(losses) == 0:
            zero = torch.tensor(0.0, device=feat_patch.device)
            return zero, {
                "proto_avg_cos": 0.0,
                "proto_min_mass": float(masses.min().detach().cpu()),
            }

        loss = torch.stack(losses).mean()
        avg_cos = torch.stack(sims).mean()

        stats = {
            "proto_avg_cos": float(avg_cos.detach().cpu()),
            "proto_min_mass": float(masses.min().detach().cpu()),
        }
        return loss, stats

    def compute_cluster_sep_loss(self, feat_patch, dispatch_weight, cluster_ids):
        if cluster_ids is None:
            zero = torch.tensor(0.0, device=feat_patch.device)
            return zero, {"cluster_sep_pairs": 0}

        B, N, D = feat_patch.shape
        valid_mask = self.build_valid_cluster_mask(cluster_ids)
        hard_expert = dispatch_weight.argmax(dim=-1)   # [B, N]
        E = dispatch_weight.shape[-1]

        losses = []
        pair_count = 0

        for b in range(B):
            feat_b = feat_patch[b]         # [N, D]
            cid_b = cluster_ids[b]         # [N]
            exp_b = hard_expert[b]         # [N]
            vmask_b = valid_mask[b]        # [N]

            for cid in self.valid_cluster_ids:
                idx = ((cid_b == cid) & vmask_b).nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() < 2:
                    continue

                feat_c = feat_b[idx]
                exp_c = exp_b[idx]

                centers = []
                for e in range(E):
                    idx_e = (exp_c == e).nonzero(as_tuple=False).squeeze(-1)
                    if idx_e.numel() == 0:
                        continue
                    centers.append(feat_c[idx_e].mean(dim=0))

                if len(centers) < 2:
                    continue

                centers = torch.stack(centers, dim=0)
                centers = F.normalize(centers, dim=-1)
                sim = centers @ centers.t()

                K = sim.shape[0]
                for i in range(K):
                    for j in range(K):
                        if i == j:
                            continue
                        losses.append(F.relu(sim[i, j] - self.cluster_margin))
                        pair_count += 1

        if len(losses) == 0:
            zero = torch.tensor(0.0, device=feat_patch.device)
            return zero, {"cluster_sep_pairs": 0}

        return torch.stack(losses).mean(), {"cluster_sep_pairs": pair_count}

    def compute_intra_compact_loss(self, feat_patch, dispatch_weight):
        hard_expert = dispatch_weight.argmax(dim=-1)
        B, N, D = feat_patch.shape
        E = dispatch_weight.shape[-1]

        losses = []
        for b in range(B):
            feat_b = feat_patch[b]
            exp_b = hard_expert[b]
            for e in range(E):
                idx = (exp_b == e).nonzero(as_tuple=False).squeeze(-1)
                if idx.numel() < 2:
                    continue
                feat_e = feat_b[idx]
                center = feat_e.mean(dim=0, keepdim=True)
                losses.append(((feat_e - center) ** 2).mean())

        if len(losses) == 0:
            zero = torch.tensor(0.0, device=feat_patch.device)
            return zero
        return torch.stack(losses).mean()

    def compute_sp_loss(self, expert_outputs, dispatch_mask, dispatch_weight=None):
        """
        expert_outputs: [B, N, E, D]
        dispatch_mask:  [B, N, E]  hard active mask
        dispatch_weight:[B, N, E] or None

        只对同一 token 上 active experts 的 pair 做 cosine similarity penalty
        """
        B, N, E, D = expert_outputs.shape
        device = expert_outputs.device

        expert_outputs = F.normalize(expert_outputs, dim=-1)

        losses = []
        pair_count = 0
        token_count = 0

        for b in range(B):
            for n in range(N):
                active_idx = torch.nonzero(dispatch_mask[b, n] > 0, as_tuple=False).squeeze(-1)

                if active_idx.numel() < self.sp_min_active_experts:
                    continue

                token_count += 1

                feats = expert_outputs[b, n, active_idx, :]   # [K, D]
                sim_mat = feats @ feats.t()                   # [K, K]

                K = sim_mat.shape[0]
                token_losses = []

                for i in range(K):
                    for j in range(i + 1, K):
                        sim_ij = sim_mat[i, j]

                        if self.sp_use_soft_weight and dispatch_weight is not None:
                            ei = active_idx[i]
                            ej = active_idx[j]
                            w = dispatch_weight[b, n, ei] * dispatch_weight[b, n, ej]
                            token_losses.append(w * (sim_ij ** 2))
                        else:
                            token_losses.append(sim_ij ** 2)

                        pair_count += 1

                if len(token_losses) > 0:
                    losses.append(torch.stack(token_losses).mean())

        if len(losses) == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, {
                "sp_pair_count": 0,
                "sp_token_count": 0,
                "sp_avg_pair_cos": 0.0,
            }

        loss = torch.stack(losses).mean()

        # 额外统计：active pair 的平均 cosine
        with torch.no_grad():
            pair_sims = []
            for b in range(B):
                for n in range(N):
                    active_idx = torch.nonzero(dispatch_mask[b, n] > 0, as_tuple=False).squeeze(-1)
                    if active_idx.numel() < self.sp_min_active_experts:
                        continue
                    feats = expert_outputs[b, n, active_idx, :]
                    sim_mat = feats @ feats.t()
                    K = sim_mat.shape[0]
                    for i in range(K):
                        for j in range(i + 1, K):
                            pair_sims.append(sim_mat[i, j])
            avg_pair_cos = torch.stack(pair_sims).mean() if len(pair_sims) > 0 else torch.tensor(0.0, device=device)

        return loss, {
            "sp_pair_count": pair_count,
            "sp_token_count": token_count,
            "sp_avg_pair_cos": float(avg_pair_cos.detach().cpu()),
        }

    def compute_expert_floor_loss(self, dispatch_weight):
        """
        dispatch_weight: [B, N, E]

        防止某个 expert 的平均 dispatch mass 掉到接近 0
        """
        # 每个 expert 的平均 dispatch mass
        expert_mass = dispatch_weight.mean(dim=(0, 1))   # [E]

        floor_gap = F.relu(self.expert_floor_tau - expert_mass)   # [E]
        loss = floor_gap.mean()

        stats = {
            "expert_floor_loss_raw": float(loss.detach().cpu()),
            "expert_floor_min_mass": float(expert_mass.min().detach().cpu()),
            "expert_floor_max_mass": float(expert_mass.max().detach().cpu()),
        }

        for i, v in enumerate(expert_mass):
            stats[f"expert_floor_mass_e{i}"] = float(v.detach().cpu())

        return loss, stats

    def compute_role_affinity_logits(self, features_teacher_space):
        """
        features_teacher_space: [N, D_teacher]
        return:
            logits: [N, R]
        """
        protos = self.role_bank.prototypes.to(features_teacher_space.device)   # [R, D]
        protos = F.normalize(protos, dim=-1)

        feats = F.normalize(features_teacher_space, dim=-1)
        logits = feats @ protos.t()
        return logits
    
    def sample_hn_bank_features(self):
        """
        return:
            sampled: dict[class_name] -> [K, D_teacher]
        """
        sampled = {}
        device = next(self.parameters()).device

        for cls_name in self.hn_use_classes:
            if cls_name not in self.hn_banks:
                continue

            bank = self.hn_banks[cls_name].to(device)   # [N, D]
            N = bank.shape[0]
            if N == 0:
                continue

            k = min(self.hn_batch_size_per_class, N)
            idx = torch.randint(0, N, (k,), device=device)
            sampled[cls_name] = bank[idx]

        return sampled
    
    def encode_center_patch_score_from_image(self, images, is_eval=False):
        """
        images: [B, 3, H, W]
        return:
            patch_repr: [B, D_teacher]
            tumor_score: [B]
        """
        patch_repr, tumor_score = self.encode_patch_batch_for_wsi_bag(
            images=images,
            is_eval=is_eval,
        )
        return patch_repr, tumor_score
    
    def _reduce_context_neighbor_scores(self, neighbor_scores: torch.Tensor):
        """
        neighbor_scores: [K]
        return:
            ref_score: scalar
            stats: dict
        """
        if neighbor_scores.numel() == 0:
            zero = neighbor_scores.new_tensor(0.0)
            return zero, {
                "neighbor_mean_score": 0.0,
                "neighbor_topk_mean_score": 0.0,
                "neighbor_max_score": 0.0,
                "neighbor_num": 0.0,
            }

        mean_score = neighbor_scores.mean()
        max_score = neighbor_scores.max()

        if self.neg_ctx_use_topk_neighbor:
            K = neighbor_scores.numel()
            topk = max(self.neg_ctx_topk_min, int(round(K * self.neg_ctx_topk_ratio)))
            topk = min(topk, self.neg_ctx_topk_max, K)
            topk_vals = torch.topk(neighbor_scores, k=topk, largest=True).values
            ref_score = topk_vals.mean()
        else:
            topk = min(self.neg_ctx_topk_max, neighbor_scores.numel())
            topk_vals = torch.topk(neighbor_scores, k=topk, largest=True).values
            ref_score = mean_score

        return ref_score, {
            "neighbor_mean_score": float(mean_score.detach().cpu()),
            "neighbor_topk_mean_score": float(topk_vals.mean().detach().cpu()),
            "neighbor_max_score": float(max_score.detach().cpu()),
            "neighbor_num": float(neighbor_scores.numel()),
        }

    def _check_negative_context_trigger(
        self,
        center_score: torch.Tensor,
        neighbor_ref_score: torch.Tensor,
        num_neighbors: int,
        label_value: int,
    ):
        if label_value != 0:
            return False

        if num_neighbors < self.neg_ctx_min_neighbors:
            return False

        center_val = float(center_score.detach().cpu())
        neighbor_val = float(neighbor_ref_score.detach().cpu())
        gap_val = center_val - neighbor_val

        if center_val < self.neg_ctx_trigger_score:
            return False
        if neighbor_val > self.neg_ctx_neighbor_support_max:
            return False
        if gap_val < self.neg_ctx_gap_margin:
            return False

        return True

    def _check_positive_context_supported(
        self,
        center_score: torch.Tensor,
        neighbor_ref_score: torch.Tensor,
        num_neighbors: int,
        label_value: int,
    ):
        if label_value != 1:
            return False

        if num_neighbors < self.neg_ctx_min_neighbors:
            return False

        center_val = float(center_score.detach().cpu())
        neighbor_val = float(neighbor_ref_score.detach().cpu())

        if center_val < self.pos_ctx_trigger_score:
            return False
        if neighbor_val < self.pos_ctx_neighbor_support_min:
            return False

        return True

    def compute_negative_context_hn_and_positive_protection(
        self,
        center_images,
        slide_label_batch,
        neighbor_images_list=None,
        is_eval=False,
    ):
        """
        主线邻域版：
        - negative：如果 center 高、neighbor 低，则压制 center-neighbor 的context gap
        - positive：如果 center 有一定tumor倾向、neighbor 也支持，则提升 center + alpha*neighbor 的context score

        return:
            neg_ctx_loss: scalar
            pos_protect_loss: scalar
            stats: dict
        """
        device = center_images.device
        zero = next(self.parameters()).new_tensor(0.0)

        stats = {
            "neg_ctx_loss_raw": 0.0,
            "neg_ctx_num_checked": 0.0,
            "neg_ctx_num_triggered": 0.0,
            "neg_ctx_trigger_ratio": 0.0,
            "neg_ctx_center_score_mean": 0.0,
            "neg_ctx_neighbor_ref_score_mean": 0.0,
            "neg_ctx_gap_mean": 0.0,

            "pos_ctx_num_checked": 0.0,
            "pos_ctx_num_supported": 0.0,
            "pos_ctx_supported_ratio": 0.0,
            "pos_ctx_center_score_mean": 0.0,
            "pos_ctx_neighbor_ref_score_mean": 0.0,
            "pos_ctx_gap_mean": 0.0,
            "pos_ctx_protect_raw": 0.0,

            # 新增
            "batch_ctx_neg_context_score_mean": 0.0,
            "batch_ctx_pos_context_score_mean": 0.0,
        }

        if center_images is None or neighbor_images_list is None or slide_label_batch is None:
            return zero, zero, stats

        if (not self.use_batch_context_guided_loss) and \
           (not self.use_negative_context_hn_loss) and \
           (not self.use_positive_context_protection):
            return zero, zero, stats

        B = center_images.shape[0]
        if len(neighbor_images_list) != B:
            raise ValueError(
                f"neighbor_images_list length {len(neighbor_images_list)} != batch size {B}"
            )

        neg_losses = []
        pos_losses = []

        neg_center_scores = []
        neg_neighbor_scores = []
        neg_gaps = []
        neg_context_scores = []

        pos_center_scores = []
        pos_neighbor_scores = []
        pos_gaps = []
        pos_context_scores = []

        neg_checked = 0
        neg_triggered = 0
        pos_checked = 0
        pos_supported = 0

        for i in range(B):
            nb_imgs_i = neighbor_images_list[i]

            if nb_imgs_i is None:
                continue
            if (not torch.is_tensor(nb_imgs_i)) or nb_imgs_i.ndim != 4 or nb_imgs_i.shape[0] == 0:
                continue

            center_img_i = center_images[i:i+1]
            _, center_score_i = self.encode_center_patch_score_from_image(
                center_img_i,
                is_eval=is_eval,
            )
            center_score_i = center_score_i.view(-1)[0]

            nb_imgs_i = nb_imgs_i.to(device, non_blocking=True)
            _, neighbor_scores_i = self.encode_center_patch_score_from_image(
                nb_imgs_i,
                is_eval=is_eval,
            )

            # ---------- neighbor reduce ----------
            if self.batch_ctx_use_topk_neighbor:
                K = int(neighbor_scores_i.numel())
                topk = max(self.batch_ctx_topk_min, int(round(K * self.batch_ctx_topk_ratio)))
                topk = min(topk, self.batch_ctx_topk_max, K)
                neighbor_ref_i = torch.topk(neighbor_scores_i, k=topk, largest=True).values.mean()
            else:
                neighbor_ref_i = neighbor_scores_i.mean()

            gap_i = center_score_i - neighbor_ref_i
            y = int(slide_label_batch[i].item())
            if y < 0:
                continue

            # =========================================================
            # negative: center高 / neighbor低 -> 压制 context gap
            # =========================================================
            if y == 0:
                neg_checked += 1
                neg_center_scores.append(float(center_score_i.detach().cpu()))
                neg_neighbor_scores.append(float(neighbor_ref_i.detach().cpu()))
                neg_gaps.append(float(gap_i.detach().cpu()))

                neg_valid = (
                    (center_score_i >= self.batch_ctx_neg_center_min) and
                    (neighbor_ref_i <= self.batch_ctx_neg_neighbor_max)
                )

                if neg_valid:
                    neg_triggered += 1
                    context_score_i = center_score_i - neighbor_ref_i
                    loss_i = F.relu(context_score_i - self.batch_ctx_neg_margin)
                    neg_losses.append(loss_i)
                    neg_context_scores.append(float(context_score_i.detach().cpu()))

            # =========================================================
            # positive: center有一定tumor倾向 + neighbor支持 -> 保护
            # =========================================================
            if y == 1:
                pos_checked += 1
                pos_center_scores.append(float(center_score_i.detach().cpu()))
                pos_neighbor_scores.append(float(neighbor_ref_i.detach().cpu()))
                pos_gaps.append(float(gap_i.detach().cpu()))

                pos_valid = (
                    (center_score_i >= self.batch_ctx_pos_center_min) and
                    (neighbor_ref_i >= self.batch_ctx_pos_neighbor_min)
                )

                if pos_valid:
                    pos_supported += 1
                    context_score_i = center_score_i + self.batch_ctx_pos_neighbor_scale * neighbor_ref_i
                    loss_i = F.relu(self.batch_ctx_pos_margin - context_score_i)
                    pos_losses.append(loss_i)
                    pos_context_scores.append(float(context_score_i.detach().cpu()))

        neg_ctx_loss = torch.stack(neg_losses).mean() if len(neg_losses) > 0 else zero
        pos_protect_loss = torch.stack(pos_losses).mean() if len(pos_losses) > 0 else zero

        stats["neg_ctx_loss_raw"] = float(neg_ctx_loss.detach().cpu())
        stats["pos_ctx_protect_raw"] = float(pos_protect_loss.detach().cpu())

        stats["neg_ctx_num_checked"] = float(neg_checked)
        stats["neg_ctx_num_triggered"] = float(neg_triggered)
        stats["neg_ctx_trigger_ratio"] = float(neg_triggered / max(neg_checked, 1))

        stats["pos_ctx_num_checked"] = float(pos_checked)
        stats["pos_ctx_num_supported"] = float(pos_supported)
        stats["pos_ctx_supported_ratio"] = float(pos_supported / max(pos_checked, 1))

        if len(neg_center_scores) > 0:
            stats["neg_ctx_center_score_mean"] = float(np.mean(neg_center_scores))
            stats["neg_ctx_neighbor_ref_score_mean"] = float(np.mean(neg_neighbor_scores))
            stats["neg_ctx_gap_mean"] = float(np.mean(neg_gaps))

        if len(pos_center_scores) > 0:
            stats["pos_ctx_center_score_mean"] = float(np.mean(pos_center_scores))
            stats["pos_ctx_neighbor_ref_score_mean"] = float(np.mean(pos_neighbor_scores))
            stats["pos_ctx_gap_mean"] = float(np.mean(pos_gaps))

        if len(neg_context_scores) > 0:
            stats["batch_ctx_neg_context_score_mean"] = float(np.mean(neg_context_scores))

        if len(pos_context_scores) > 0:
            stats["batch_ctx_pos_context_score_mean"] = float(np.mean(pos_context_scores))

        return neg_ctx_loss, pos_protect_loss, stats
    def compute_single_hn_repulsion_loss(self, hn_features_teacher_space):
        """
        hn_features_teacher_space: [K, D_teacher], already in teacher space
        loss form:
            relu(sim_tumor - sim_other_max + margin)
        """
        if hn_features_teacher_space is None or hn_features_teacher_space.numel() == 0:
            zero = next(self.parameters()).new_tensor(0.0)
            return zero, {
                "hn_mean_sim_tumor": 0.0,
                "hn_mean_sim_other_max": 0.0,
                "hn_mean_delta": 0.0,
                "hn_frac_delta_gt_0": 0.0,
            }

        hn_features_teacher_space = F.normalize(hn_features_teacher_space, dim=-1)
        logits = self.compute_role_affinity_logits(hn_features_teacher_space)   # [K, R]

        sim_tumor = logits[:, self.tumor_role_id]

        other_ids = [i for i in range(logits.shape[1]) if i != self.tumor_role_id]
        sim_other_max = logits[:, other_ids].max(dim=1).values

        delta = sim_tumor - sim_other_max
        loss = F.relu(delta + self.hn_repulsion_margin).mean()

        stats = {
            "hn_mean_sim_tumor": float(sim_tumor.mean().detach().cpu()),
            "hn_mean_sim_other_max": float(sim_other_max.mean().detach().cpu()),
            "hn_mean_delta": float(delta.mean().detach().cpu()),
            "hn_frac_delta_gt_0": float((delta > 0).float().mean().detach().cpu()),
        }
        return loss, stats
    
    def compute_hn_repulsion_loss(self):
        """
        对多个 superclass bank 分别采样并平均
        """
        zero = next(self.parameters()).new_tensor(0.0)

        if (not self.use_hn_repulsion_loss) or (self.hn_repulsion_weight <= 0):
            return zero, {
                "hn_repulsion_raw": 0.0,
                "hn_num_classes_used": 0.0,
            }

        assert self.role_bank is not None, "role_bank is required for HN repulsion"
        assert self.tumor_role_id is not None, "tumor_role_id is required for HN repulsion"

        sampled = self.sample_hn_bank_features()

        losses = []
        stats_all = {}
        num_used = 0

        for cls_name, feat in sampled.items():
            loss_cls, stats_cls = self.compute_single_hn_repulsion_loss(feat)
            losses.append(loss_cls)
            num_used += 1

            stats_all[f"hn_{cls_name}_raw"] = float(loss_cls.detach().cpu())
            stats_all[f"hn_{cls_name}_mean_sim_tumor"] = stats_cls["hn_mean_sim_tumor"]
            stats_all[f"hn_{cls_name}_mean_sim_other_max"] = stats_cls["hn_mean_sim_other_max"]
            stats_all[f"hn_{cls_name}_mean_delta"] = stats_cls["hn_mean_delta"]
            stats_all[f"hn_{cls_name}_frac_delta_gt_0"] = stats_cls["hn_frac_delta_gt_0"]

        if len(losses) == 0:
            stats_all["hn_repulsion_raw"] = 0.0
            stats_all["hn_num_classes_used"] = 0.0
            return zero, stats_all

        loss = torch.stack(losses).mean()
        stats_all["hn_repulsion_raw"] = float(loss.detach().cpu())
        stats_all["hn_num_classes_used"] = float(num_used)
        return loss, stats_all
    def _get_online_hn_min_sim_threshold(self, cls_name):
        if cls_name in ["gland_like", "gland_like_sub3"]:
            return self.online_hn_gland_like_min_sim_to_center
        if cls_name == "fibrous_dense":
            return self.online_hn_fibrous_dense_min_sim_to_center
        return self.online_hn_min_sim_to_center
    def build_online_patch_teacher_features(self, spec_patch):
        """
        spec_patch: [B, N, D_student]
        return:
            online_feat: [B*N, D_teacher], normalized
        """
        B, N, D = spec_patch.shape
        feat = spec_patch.reshape(B * N, D)          # [BN, D_student]
        feat = self.proj_l12(feat)                   # [BN, D_teacher]
        feat = F.normalize(feat, dim=-1)
        return feat
    
    def compute_single_online_hn_repulsion_loss(self, online_feat, cls_name):
        """
        online_feat: [BN, D_teacher], normalized
        cls_name: e.g. gland_like / fibrous_dense

        逻辑：
        1) 用 superclass centroid 匹配当前 batch 在线 token
        2) 取 top-k 最像该 superclass 的 token
        3) 可选：只保留当前 tumor-dominant 的 token
        4) 对这些 token 做 repulsion:
               relu(sim_tumor - sim_other_max + margin)
        """
        zero = online_feat.new_tensor(0.0)

        if cls_name not in self.hn_centers:
            return zero, {
                "online_hn_num_selected": 0.0,
                "online_hn_mean_center_sim": 0.0,
                "online_hn_mean_delta": 0.0,
                "online_hn_frac_delta_gt_0": 0.0,
            }

        center = self.hn_centers[cls_name].to(online_feat.device)   # [D]
        sim_center = online_feat @ center                            # [BN]

        # 先按与 superclass centroid 的相似度取 top-k
        k = min(self.online_hn_topk_per_class, sim_center.shape[0])
        if k <= 0:
            return zero, {
                "online_hn_num_selected": 0.0,
                "online_hn_mean_center_sim": 0.0,
                "online_hn_mean_delta": 0.0,
                "online_hn_frac_delta_gt_0": 0.0,
            }

        topk_vals, topk_idx = torch.topk(sim_center, k=k, largest=True)
        sel_feat = online_feat[topk_idx]   # [k, D]

        # role affinity
        logits = self.compute_role_affinity_logits(sel_feat)   # [k, R]
        sim_tumor = logits[:, self.tumor_role_id]

        other_ids = [i for i in range(logits.shape[1]) if i != self.tumor_role_id]
        sim_other_max = logits[:, other_ids].max(dim=1).values
        delta = sim_tumor - sim_other_max

        valid_mask = torch.ones_like(delta, dtype=torch.bool)

        min_sim_thr = self._get_online_hn_min_sim_threshold(cls_name)
        if min_sim_thr > -1.0:
            valid_mask = valid_mask & (topk_vals >= min_sim_thr)

        if self.online_hn_require_tumor_dominant:
            valid_mask = valid_mask & (delta > 0)

        # 关键：只保留轻度 tumor-dominant 的边界危险样本
        valid_mask = valid_mask & (delta < self.online_hn_max_delta)

        num_valid = int(valid_mask.sum().item())
        if num_valid == 0:
            return zero, {
                "online_hn_num_selected": 0.0,
                "online_hn_mean_center_sim": float(topk_vals.mean().detach().cpu()),
                "online_hn_mean_delta": float(delta.mean().detach().cpu()),
                "online_hn_frac_delta_gt_0": float((delta > 0).float().mean().detach().cpu()),
            }

        loss = F.relu(delta[valid_mask] + self.online_hn_repulsion_margin).mean()

        stats = {
            "online_hn_num_selected": float(num_valid),
            "online_hn_mean_center_sim": float(topk_vals[valid_mask].mean().detach().cpu()),
            "online_hn_mean_delta": float(delta[valid_mask].mean().detach().cpu()),
            "online_hn_frac_delta_gt_0": float((delta[valid_mask] > 0).float().mean().detach().cpu()),
        }
        return loss, stats

    def compute_online_hn_repulsion_loss(self, spec_patch):
        """
        spec_patch: [B, N, D_student]

        在当前 batch 在线 token 上做 HN repulsion
        """
        zero = spec_patch.new_tensor(0.0)

        if (not self.use_online_hn_repulsion_loss) or (self.online_hn_repulsion_weight <= 0):
            return zero, {
                "online_hn_repulsion_raw": 0.0,
                "online_hn_num_classes_used": 0.0,
            }

        assert self.role_bank is not None, "role_bank is required for online HN repulsion"
        assert self.tumor_role_id is not None, "tumor_role_id is required for online HN repulsion"

        online_feat = self.build_online_patch_teacher_features(spec_patch)   # [BN, D_teacher]

        losses = []
        stats_all = {}
        num_used = 0

        for cls_name in self.online_hn_use_classes:
            loss_cls, stats_cls = self.compute_single_online_hn_repulsion_loss(
                online_feat=online_feat,
                cls_name=cls_name,
            )
            losses.append(loss_cls)
            num_used += 1

            stats_all[f"online_hn_{cls_name}_raw"] = float(loss_cls.detach().cpu())
            stats_all[f"online_hn_{cls_name}_num_selected"] = stats_cls["online_hn_num_selected"]
            stats_all[f"online_hn_{cls_name}_mean_center_sim"] = stats_cls["online_hn_mean_center_sim"]
            stats_all[f"online_hn_{cls_name}_mean_delta"] = stats_cls["online_hn_mean_delta"]
            stats_all[f"online_hn_{cls_name}_frac_delta_gt_0"] = stats_cls["online_hn_frac_delta_gt_0"]

        if len(losses) == 0:
            stats_all["online_hn_repulsion_raw"] = 0.0
            stats_all["online_hn_num_classes_used"] = 0.0
            return zero, stats_all

        loss = torch.stack(losses).mean()
        stats_all["online_hn_repulsion_raw"] = float(loss.detach().cpu())
        stats_all["online_hn_num_classes_used"] = float(num_used)
        return loss, stats_all

    def compute_role_conf_and_margin(self, features_teacher_space):
        """
        features_teacher_space: [N, D_teacher]
        return:
            role_probs:   [N, R]
            main_role_id: [N]
            main_conf:    [N]
            role_margin:  [N]   # top1 - top2 over role probs
        """
        logits = self.compute_role_affinity_logits(features_teacher_space)   # [N, R]
        probs = torch.softmax(logits / self.role_tau, dim=-1)

        topk_vals, topk_idx = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1)
        main_role_id = topk_idx[:, 0]
        main_conf = topk_vals[:, 0]

        if probs.shape[-1] >= 2:
            role_margin = topk_vals[:, 0] - topk_vals[:, 1]
        else:
            role_margin = torch.ones_like(main_conf)

        return probs, main_role_id, main_conf, role_margin

    def _compose_role_loss_with_scales(
        self,
        out,
        target_scale: float = 1.0,
        attraction_scale: float = 1.0,
        separation_scale: float = 1.0,
    ):
        total = (
            target_scale * out.role_target_loss
            + attraction_scale * out.attraction_loss
            + separation_scale * out.separation_loss
        )
        return total

    def compute_free_expert_floor_loss(self, dispatch_weight):
        """
        dispatch_weight: [B, N, E]
        encourage free expert to keep a minimum average dispatch mass
        """
        free_mass = dispatch_weight[..., self.free_expert_id].mean()
        loss = F.relu(self.free_expert_floor_tau - free_mass)

        stats = {
            "free_expert_floor_raw": float(loss.detach().cpu()),
            "free_expert_mass": float(free_mass.detach().cpu()),
        }
        return loss, stats

    def encode_patch_batch_for_wsi_bag(self, images, is_eval=False):
        """
        images: [B_img, 3, H, W]，这里 B_img 是一张 WSI 里抽出来的一批 patch
        return:
            patch_repr: [B_img, D_teacher]
            tumor_score: [B_img]
        """
        student_out, gate_info_list, feature_dict, moe_feature_list = self.student(
            images,
            return_gates=True,
            mask=None,
            is_eval=is_eval,
            return_features=True,
            offline_cluster_ids=None,
        )

        # 取最后一个 MoE block token feature
        if self.use_last_moe_output and len(moe_feature_list) > 0:
            feat = moe_feature_list[-1]          # [B_img, T+1, 384]
        else:
            feat = feature_dict["layer_12"]      # [B_img, T+1, 384]

        patch_tokens = feat[:, 1:, :]            # [B_img, T, 384]
        patch_tokens_proj = self.proj_l12(patch_tokens)   # [B_img, T, 1280]

        # 一张 patch-image 的表示：mean pooling over patch tokens
        patch_repr = patch_tokens_proj.mean(dim=1)        # [B_img, 1280]
        patch_repr = F.normalize(patch_repr, dim=-1)

        tumor_score = self.compute_wsi_tumor_evidence_score(patch_repr)
        return patch_repr, tumor_score

    def encode_patch_batch_for_wsi_bag_with_dispatch(self, images, is_eval=False):
        """
        images: [B_img, 3, H, W]
        return:
            patch_repr: [B_img, D_teacher]
            tumor_score: [B_img]
            dispatch_weight_img: [B_img, E]   # image-level aggregated dispatch weight
        """
        student_out, gate_info_list, feature_dict, moe_feature_list = self.student(
            images,
            return_gates=True,
            mask=None,
            is_eval=is_eval,
            return_features=True,
            offline_cluster_ids=None,
        )

        if self.use_last_moe_output and len(moe_feature_list) > 0:
            feat = moe_feature_list[-1]          # [B_img, T+1, 384]
        else:
            feat = feature_dict["layer_12"]      # [B_img, T+1, 384]

        patch_tokens = feat[:, 1:, :]            # [B_img, T, 384]
        patch_tokens_proj = self.proj_l12(patch_tokens)   # [B_img, T, 1280]

        patch_repr = patch_tokens_proj.mean(dim=1)        # [B_img, 1280]
        patch_repr = F.normalize(patch_repr, dim=-1)
        tumor_score = self.compute_wsi_tumor_evidence_score(patch_repr)

        # last-layer token dispatch -> image-level dispatch
        B_img = images.shape[0]
        N_tok = patch_tokens.shape[1]
        dispatch_weight = self.get_last_dispatch_weight(gate_info_list, B_img, N_tok)   # [B_img, T, E]
        dispatch_weight_img = dispatch_weight.mean(dim=1)                                # [B_img, E]

        return patch_repr, tumor_score, dispatch_weight_img

    def compute_wsi_tumor_evidence_score(self, patch_repr):
        sim_tumor, sim_other_max, gap, _ = self.compute_role_scores_for_patch_repr(patch_repr)
        return gap

    def compute_role_scores_for_patch_repr(self, patch_repr):
        """
        patch_repr: [N, D_teacher]
        return:
            sim_tumor: [N]
            sim_other_max: [N]
            gap: [N] = sim_tumor - sim_other_max
            role_logits: [N, R]
        """
        assert self.role_bank is not None, "role_bank is required"
        assert self.tumor_role_id is not None, "tumor_role_id is required"

        role_logits = self.compute_role_affinity_logits(patch_repr)   # [N, R]
        sim_tumor = role_logits[:, self.tumor_role_id]

        num_roles = role_logits.shape[1]
        other_role_ids = [i for i in range(num_roles) if i != self.tumor_role_id]
        assert len(other_role_ids) > 0, "At least one non-tumor role is required"

        sim_other_max = role_logits[:, other_role_ids].max(dim=1).values
        gap = sim_tumor - sim_other_max
        return sim_tumor, sim_other_max, gap, role_logits
    
    def _reduce_neighbor_values(self, values: torch.Tensor):
        """
        values: [K]
        return:
            mean_val: scalar
            max_val: scalar
            topk_mean_val: scalar
        """
        if values is None or values.numel() == 0:
            zero = next(self.parameters()).new_tensor(0.0)
            return zero, zero, zero

        mean_val = values.mean()
        max_val = values.max()

        if self.ctx_use_topk_neighbor:
            K = values.numel()
            topk = max(self.ctx_topk_min, int(round(K * self.ctx_topk_ratio)))
            topk = min(topk, self.ctx_topk_max, K)
            topk_vals = torch.topk(values, k=topk, largest=True).values
            topk_mean_val = topk_vals.mean()
        else:
            topk_mean_val = mean_val

        return mean_val, max_val, topk_mean_val

    def build_context_guided_score_from_center_and_neighbors(
        self,
        center_patch_repr,
        neighbor_patch_repr=None,
        mode="negative",
    ):
        """
        center_patch_repr: [D_teacher]
        neighbor_patch_repr: [K, D_teacher] or None

        return:
            out: dict with
                center_gap
                nb_gap_mean
                nb_gap_max
                nb_gap_topk_mean
                ctx_score
                isolation_score
                consistency_score
        """
        zero = center_patch_repr.new_tensor(0.0)

        center_patch_repr = center_patch_repr.view(1, -1)
        _, _, center_gap, _ = self.compute_role_scores_for_patch_repr(center_patch_repr)
        center_gap = center_gap.view(-1)[0]

        if neighbor_patch_repr is None or neighbor_patch_repr.numel() == 0:
            nb_gap_mean = zero
            nb_gap_max = zero
            nb_gap_topk_mean = zero
        else:
            _, _, nb_gap, _ = self.compute_role_scores_for_patch_repr(neighbor_patch_repr)
            nb_gap_mean, nb_gap_max, nb_gap_topk_mean = self._reduce_neighbor_values(nb_gap)

        # 基础 context score
        ctx_score = (
            self.ctx_score_alpha * center_gap
            + self.ctx_score_beta * nb_gap_topk_mean
            + self.ctx_score_gamma * nb_gap_max
        )

        # 用于 negative：中心高但邻域低 => 更像孤立伪阳
        isolation_score = center_gap - nb_gap_topk_mean

        # 用于 positive：中心与邻域一起高 => 更像真实 tumor evidence
        consistency_score = 0.5 * (center_gap + nb_gap_topk_mean)

        if mode == "negative":
            ctx_score = ctx_score + self.ctx_neg_isolation_weight * isolation_score
        else:
            ctx_score = ctx_score + self.ctx_pos_consistency_weight * consistency_score

        return {
            "center_gap": center_gap,
            "nb_gap_mean": nb_gap_mean,
            "nb_gap_max": nb_gap_max,
            "nb_gap_topk_mean": nb_gap_topk_mean,
            "ctx_score": ctx_score,
            "isolation_score": isolation_score,
            "consistency_score": consistency_score,
        }

    def build_positive_context_ranking_signal(
        self,
        center_gap,
        nb_gap_topk_mean,
        nb_gap_max,
    ):
        """
        positive专用 ranking signal

        目标：
        - center 自己要有 tumor evidence
        - neighbor 也要支持
        - center / neighbor 越一致越好
        """
        support = 0.5 * (nb_gap_topk_mean + nb_gap_max)
        consistency = -torch.abs(center_gap - nb_gap_topk_mean)

        signal = (
            self.pos_ctx_rank_center_weight * center_gap
            + self.pos_ctx_rank_neighbor_weight * nb_gap_topk_mean
            + self.pos_ctx_rank_neighbor_max_weight * nb_gap_max
            + self.pos_ctx_rank_support_weight * support
            + self.pos_ctx_rank_consistency_weight * consistency
        )
        return signal

    def apply_positive_support_mask(
        self,
        valid_mask,
        sim_tumor,
        gap,
        context_dict=None,
    ):
        """
        在 positive candidate formation 阶段前移 support 逻辑。
        """
        if (not self.use_pos_support_mask):
            return valid_mask

        mask = valid_mask.clone()

        if self.pos_support_min_tumor_score > -1e5:
            mask = mask & (sim_tumor >= self.pos_support_min_tumor_score)

        if context_dict is not None:
            center_gap = context_dict["center_gap"]
            nb_gap_topk_mean = context_dict["nb_gap_topk_mean"]
            nb_gap_max = context_dict["nb_gap_max"]

            if self.pos_support_center_min > -1e5:
                mask = mask & (center_gap >= self.pos_support_center_min)

            if self.pos_support_neighbor_min > -1e5:
                mask = mask & (nb_gap_topk_mean >= self.pos_support_neighbor_min)

            if self.pos_support_neighbor_max_min > -1e5:
                mask = mask & (nb_gap_max >= self.pos_support_neighbor_max_min)
        else:
            # 没有 context 时，只用 center gap 退化
            if self.pos_support_center_min > -1e5:
                mask = mask & (gap >= self.pos_support_center_min)

        return mask

    def build_context_scores_for_batch_centers(
        self,
        center_images,
        neighbor_images_list=None,
        is_eval=False,
        mode="negative",
    ):
        """
        center_images: [B,3,H,W]
        neighbor_images_list: list[Tensor[K_i,3,H,W] or None]

        return:
            dict of tensors, each [B]
        """
        device = center_images.device
        B = center_images.shape[0]

        center_repr, _ = self.encode_center_patch_score_from_image(center_images, is_eval=is_eval)

        ctx_score_all = []
        center_gap_all = []
        nb_gap_mean_all = []
        nb_gap_max_all = []
        nb_gap_topk_mean_all = []
        isolation_all = []
        consistency_all = []

        if neighbor_images_list is None:
            neighbor_images_list = [None] * B

        for i in range(B):
            nb_imgs_i = neighbor_images_list[i]

            if nb_imgs_i is not None and torch.is_tensor(nb_imgs_i) and nb_imgs_i.ndim == 4 and nb_imgs_i.shape[0] > 0:
                nb_imgs_i = nb_imgs_i.to(device, non_blocking=True)
                nb_repr_i, _ = self.encode_center_patch_score_from_image(nb_imgs_i, is_eval=is_eval)
            else:
                nb_repr_i = None

            out_i = self.build_context_guided_score_from_center_and_neighbors(
                center_patch_repr=center_repr[i],
                neighbor_patch_repr=nb_repr_i,
                mode=mode,
            )

            ctx_score_all.append(out_i["ctx_score"])
            center_gap_all.append(out_i["center_gap"])
            nb_gap_mean_all.append(out_i["nb_gap_mean"])
            nb_gap_max_all.append(out_i["nb_gap_max"])
            nb_gap_topk_mean_all.append(out_i["nb_gap_topk_mean"])
            isolation_all.append(out_i["isolation_score"])
            consistency_all.append(out_i["consistency_score"])

        return {
            "ctx_score": torch.stack(ctx_score_all, dim=0),
            "center_gap": torch.stack(center_gap_all, dim=0),
            "nb_gap_mean": torch.stack(nb_gap_mean_all, dim=0),
            "nb_gap_max": torch.stack(nb_gap_max_all, dim=0),
            "nb_gap_topk_mean": torch.stack(nb_gap_topk_mean_all, dim=0),
            "isolation_score": torch.stack(isolation_all, dim=0),
            "consistency_score": torch.stack(consistency_all, dim=0),
        }

    def encode_wsi_patch_repr_with_optional_context(
        self,
        images,
        neighbor_images_list=None,
        patch_batch_size=None,
        is_eval=False,
        mode="negative",
    ):
        """
        images: [N,3,H,W] or list[tensor]
        neighbor_images_list: list[Tensor[K_i,3,H,W] or None] with len=N

        return:
            patch_repr: [N, D_teacher]
            dispatch_weight_img: [N, E]
            context_dict: dict of [N] tensors
        """
        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        device = next(self.parameters()).device
        images = images.to(device, non_blocking=True)

        if patch_batch_size is None:
            patch_batch_size = self.wsi_patch_batch_size

        N_total = images.shape[0]
        if neighbor_images_list is None:
            neighbor_images_list = [None] * N_total

        all_repr = []
        all_dispatch = []

        all_ctx_score = []
        all_center_gap = []
        all_nb_gap_mean = []
        all_nb_gap_max = []
        all_nb_gap_topk_mean = []
        all_isolation = []
        all_consistency = []

        for start in range(0, N_total, patch_batch_size):
            end = min(start + patch_batch_size, N_total)
            patch_batch = images[start:end]
            nb_batch = neighbor_images_list[start:end]

            patch_repr, _, dispatch_weight_img = self.encode_patch_batch_for_wsi_bag_with_dispatch(
                patch_batch,
                is_eval=is_eval,
            )
            all_repr.append(patch_repr)
            all_dispatch.append(dispatch_weight_img)

            if self.use_context_guided_selection:
                ctx_dict = self.build_context_scores_for_batch_centers(
                    center_images=patch_batch,
                    neighbor_images_list=nb_batch,
                    is_eval=is_eval,
                    mode=mode,
                )
                all_ctx_score.append(ctx_dict["ctx_score"])
                all_center_gap.append(ctx_dict["center_gap"])
                all_nb_gap_mean.append(ctx_dict["nb_gap_mean"])
                all_nb_gap_max.append(ctx_dict["nb_gap_max"])
                all_nb_gap_topk_mean.append(ctx_dict["nb_gap_topk_mean"])
                all_isolation.append(ctx_dict["isolation_score"])
                all_consistency.append(ctx_dict["consistency_score"])

        patch_repr = torch.cat(all_repr, dim=0)
        dispatch_weight_img = torch.cat(all_dispatch, dim=0)

        if self.use_context_guided_selection:
            context_dict = {
                "ctx_score": torch.cat(all_ctx_score, dim=0),
                "center_gap": torch.cat(all_center_gap, dim=0),
                "nb_gap_mean": torch.cat(all_nb_gap_mean, dim=0),
                "nb_gap_max": torch.cat(all_nb_gap_max, dim=0),
                "nb_gap_topk_mean": torch.cat(all_nb_gap_topk_mean, dim=0),
                "isolation_score": torch.cat(all_isolation, dim=0),
                "consistency_score": torch.cat(all_consistency, dim=0),
            }
        else:
            context_dict = None

        return patch_repr, dispatch_weight_img, context_dict

    def _resolve_cond_rank_topk(self, num_tokens, topk_fixed, topk_ratio=0.0):
        """
        num_tokens: int
        return: int >= 1
        """
        if num_tokens <= 0:
            return 0

        k = int(topk_fixed)
        if topk_ratio is not None and topk_ratio > 0:
            k_ratio = int(round(num_tokens * topk_ratio))
            k = max(k, k_ratio)

        k = max(1, min(k, num_tokens))
        return k
    
    
    def get_negative_rank_selected_token_indices(
        self,
        patch_repr,
        dispatch_weight_wsi=None,
        context_dict=None,
    ):
        """
        negative selection:
        - 如果有 context_dict，则按 ctx_score 选
        - 否则退化到 sim_tumor
        """
        device = patch_repr.device
        sim_tumor, sim_other_max, gap, _ = self.compute_role_scores_for_patch_repr(patch_repr)
        N = patch_repr.shape[0]

        zero_idx = torch.empty(0, dtype=torch.long, device=device)

        if N == 0:
            return zero_idx, {
                "neg_preselect_num_all": 0.0,
                "neg_preselect_num_valid": 0.0,
                "neg_preselect_used_fallback": 0.0,
                "neg_rank_sel_num_before_filter": 0.0,
                "neg_rank_sel_num_after_filter": 0.0,
                "neg_rank_sel_gap_mean": 0.0,
                "neg_rank_sel_gap_max": 0.0,
                "neg_rank_sel_gap_min": 0.0,
                "neg_rank_ctx_score_mean": 0.0,
                "neg_rank_sel_e0": 0.0,
                "neg_rank_sel_e1": 0.0,
                "neg_rank_sel_e2": 0.0,
            }

        if context_dict is not None and self.use_context_guided_selection:
            ranking_signal = context_dict["ctx_score"]
            center_gap_for_filter = context_dict["center_gap"]
        else:
            ranking_signal = sim_tumor
            center_gap_for_filter = gap

        valid_mask = torch.ones(N, dtype=torch.bool, device=device)

        if self.use_neg_preselection:
            if self.neg_preselect_min_tumor_score > -1e5:
                valid_mask = valid_mask & (sim_tumor >= self.neg_preselect_min_tumor_score)

            if self.neg_preselect_min_gap > -1e5:
                valid_mask = valid_mask & (center_gap_for_filter >= self.neg_preselect_min_gap)

            if self.neg_preselect_max_gap < 1e5:
                valid_mask = valid_mask & (center_gap_for_filter <= self.neg_preselect_max_gap)

            if self.neg_preselect_require_tumor_dominant:
                valid_mask = valid_mask & (gap > 0)

            if dispatch_weight_wsi is not None and self.neg_preselect_target_expert is not None:
                hard_expert = dispatch_weight_wsi.argmax(dim=-1)
                valid_mask = valid_mask & (hard_expert == self.neg_preselect_target_expert)

        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        used_fallback = 0.0

        if valid_idx.numel() == 0:
            if self.neg_preselect_allow_fallback:
                used_fallback = 1.0
                valid_idx = torch.arange(N, device=device)
            else:
                return zero_idx, {
                    "neg_preselect_num_all": float(N),
                    "neg_preselect_num_valid": 0.0,
                    "neg_preselect_used_fallback": 0.0,
                    "neg_rank_sel_num_before_filter": 0.0,
                    "neg_rank_sel_num_after_filter": 0.0,
                    "neg_rank_sel_gap_mean": 0.0,
                    "neg_rank_sel_gap_max": 0.0,
                    "neg_rank_sel_gap_min": 0.0,
                    "neg_rank_ctx_score_mean": 0.0,
                }

        ranking_signal_valid = ranking_signal[valid_idx]

        k = self._resolve_cond_rank_topk(
            num_tokens=int(valid_idx.numel()),
            topk_fixed=self.cond_rank_neg_topk,
            topk_ratio=self.cond_rank_neg_topk_ratio,
        )

        topk_local = torch.topk(ranking_signal_valid, k=k, largest=True).indices
        topk_idx = valid_idx[topk_local]

        gap_sel = gap[topk_idx]
        ctx_sel = ranking_signal[topk_idx]

        valid_mask_2 = torch.ones_like(gap_sel, dtype=torch.bool)

        if self.neg_rank_free_pref_min_gap > -1e5:
            valid_mask_2 = valid_mask_2 & (gap_sel >= self.neg_rank_free_pref_min_gap)
        if self.neg_rank_free_pref_max_gap < 1e5:
            valid_mask_2 = valid_mask_2 & (gap_sel <= self.neg_rank_free_pref_max_gap)

        topk_idx_filtered = topk_idx[valid_mask_2]
        gap_sel_filtered = gap[topk_idx_filtered] if topk_idx_filtered.numel() > 0 else gap_sel.new_empty(0)
        ctx_sel_filtered = ranking_signal[topk_idx_filtered] if topk_idx_filtered.numel() > 0 else ctx_sel.new_empty(0)

        if gap_sel_filtered.numel() == 0:
            return topk_idx_filtered, {
                "neg_preselect_num_all": float(N),
                "neg_preselect_num_valid": float(valid_idx.numel()),
                "neg_preselect_used_fallback": float(used_fallback),
                "neg_rank_sel_num_before_filter": float(k),
                "neg_rank_sel_num_after_filter": 0.0,
                "neg_rank_sel_gap_mean": 0.0,
                "neg_rank_sel_gap_max": 0.0,
                "neg_rank_sel_gap_min": 0.0,
                "neg_rank_ctx_score_mean": 0.0,
            }

        neg_sel_e0 = neg_sel_e1 = neg_sel_e2 = 0.0
        if dispatch_weight_wsi is not None and topk_idx_filtered.numel() > 0:
            hard_expert = dispatch_weight_wsi.argmax(dim=-1)
            sel_expert = hard_expert[topk_idx_filtered]
            neg_sel_e0 = float((sel_expert == 0).sum().item())
            neg_sel_e1 = float((sel_expert == 1).sum().item())
            neg_sel_e2 = float((sel_expert == 2).sum().item())

        return topk_idx_filtered, {
            "neg_preselect_num_all": float(N),
            "neg_preselect_num_valid": float(valid_idx.numel()),
            "neg_preselect_used_fallback": float(used_fallback),
            "neg_rank_sel_num_before_filter": float(k),
            "neg_rank_sel_num_after_filter": float(topk_idx_filtered.numel()),
            "neg_rank_sel_gap_mean": float(gap_sel_filtered.mean().detach().cpu()),
            "neg_rank_sel_gap_max": float(gap_sel_filtered.max().detach().cpu()),
            "neg_rank_sel_gap_min": float(gap_sel_filtered.min().detach().cpu()),
            "neg_rank_ctx_score_mean": float(ctx_sel_filtered.mean().detach().cpu()),
            "neg_rank_sel_e0": neg_sel_e0,
            "neg_rank_sel_e1": neg_sel_e1,
            "neg_rank_sel_e2": neg_sel_e2,
        }

    def apply_positive_support_mask_weak(
        self,
        valid_mask,
        sim_tumor,
        gap,
        context_dict=None,
    ):
        mask = valid_mask.clone()

        weak_center_min = min(self.pos_support_center_min, -0.02)
        weak_nb_min = min(self.pos_support_neighbor_min, 0.0)
        weak_nb_max_min = min(self.pos_support_neighbor_max_min, 0.02)
        weak_tumor_min = min(self.pos_support_min_tumor_score, 0.0)

        if weak_tumor_min > -1e5:
            mask = mask & (sim_tumor >= weak_tumor_min)

        if context_dict is not None:
            center_gap = context_dict["center_gap"]
            nb_gap_topk_mean = context_dict["nb_gap_topk_mean"]
            nb_gap_max = context_dict["nb_gap_max"]

            if weak_center_min > -1e5:
                mask = mask & (center_gap >= weak_center_min)
            if weak_nb_min > -1e5:
                mask = mask & (nb_gap_topk_mean >= weak_nb_min)
            if weak_nb_max_min > -1e5:
                mask = mask & (nb_gap_max >= weak_nb_max_min)
        else:
            if weak_center_min > -1e5:
                mask = mask & (gap >= weak_center_min)

        return mask

    def get_positive_rank_selected_token_indices(
        self,
        patch_repr,
        dispatch_weight_wsi=None,
        context_dict=None,
        slide_id_batch=None,
    ):
        """
        positive selection:
        1) base candidate filter
        2) positive support mask
        3) expert-aware balanced selection + per-expert quota + slide dedup
        4) require min selected experts if configured
        5) fallback to global top-k only when expert-balanced path is unavailable
        """
        device = patch_repr.device
        zero_idx = torch.empty(0, dtype=torch.long, device=device)

        base_stats = {
            "cond_rank_pos_num_candidates": 0.0,
            "cond_rank_pos_num_selected": 0.0,
            "cond_rank_pos_selected_gap_mean": 0.0,
            "cond_rank_pos_selected_gap_max": 0.0,
            "cond_rank_pos_selected_gap_min": 0.0,
            "cond_rank_pos_ctx_score_mean": 0.0,
            "cond_rank_pos_e0_selected": 0.0,
            "cond_rank_pos_e1_selected": 0.0,
            "cond_rank_pos_e2_selected": 0.0,
            "cond_rank_pos_use_expert_balanced": float(self.use_pos_expert_balanced_selection),
            "cond_rank_pos_support_num_before": 0.0,
            "cond_rank_pos_support_num_after": 0.0,
            "cond_rank_pos_support_ratio": 0.0,
            "cond_rank_pos_dedup_dropped": 0.0,
            "cond_rank_pos_num_selected_experts": 0.0,
            "cond_rank_pos_failed_min_experts": 0.0,
        }

        if patch_repr is None or patch_repr.numel() == 0:
            return zero_idx, base_stats

        sim_tumor, sim_other_max, gap, _ = self.compute_role_scores_for_patch_repr(patch_repr)

        # -------------------------------------------------
        # ranking signal
        # -------------------------------------------------
        if context_dict is not None and self.use_context_guided_selection:
            if self.cond_rank_pos_select_mode == "tumor":
                ranking_signal = sim_tumor + 0.2 * context_dict["nb_gap_topk_mean"]
            else:
                if self.use_positive_context_ranking:
                    ranking_signal = self.build_positive_context_ranking_signal(
                        center_gap=context_dict["center_gap"],
                        nb_gap_topk_mean=context_dict["nb_gap_topk_mean"],
                        nb_gap_max=context_dict["nb_gap_max"],
                    )
                else:
                    ranking_signal = context_dict["ctx_score"]
        else:
            if self.cond_rank_pos_select_mode == "tumor":
                ranking_signal = sim_tumor
            else:
                ranking_signal = gap

        # -------------------------------------------------
        # step1: base valid mask
        # -------------------------------------------------
        valid_mask = torch.ones_like(ranking_signal, dtype=torch.bool)

        if self.cond_rank_pos_min_tumor_score > -1e5:
            valid_mask = valid_mask & (sim_tumor >= self.cond_rank_pos_min_tumor_score)

        if self.cond_rank_pos_min_gap > -1e5:
            valid_mask = valid_mask & (gap >= self.cond_rank_pos_min_gap)

        num_before_support = int(valid_mask.sum().item())

        # -------------------------------------------------
        # step2: strong support mask
        # -------------------------------------------------
        valid_mask = self.apply_positive_support_mask(
            valid_mask=valid_mask,
            sim_tumor=sim_tumor,
            gap=gap,
            context_dict=context_dict,
        )

        num_after_support = int(valid_mask.sum().item())

        # -------------------------------------------------
        # step2b: weak support fallback
        # -------------------------------------------------
        if num_after_support == 0 and self.pos_support_allow_fallback:
            valid_mask = self.apply_positive_support_mask_weak(
                valid_mask=torch.ones_like(ranking_signal, dtype=torch.bool),
                sim_tumor=sim_tumor,
                gap=gap,
                context_dict=context_dict,
            )

            if self.cond_rank_pos_min_tumor_score > -1e5:
                valid_mask = valid_mask & (sim_tumor >= self.cond_rank_pos_min_tumor_score)
            if self.cond_rank_pos_min_gap > -1e5:
                valid_mask = valid_mask & (gap >= self.cond_rank_pos_min_gap)

            num_after_support = int(valid_mask.sum().item())

        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        base_stats["cond_rank_pos_support_num_before"] = float(num_before_support)
        base_stats["cond_rank_pos_support_num_after"] = float(num_after_support)
        base_stats["cond_rank_pos_support_ratio"] = float(num_after_support / max(num_before_support, 1))
        base_stats["cond_rank_pos_num_candidates"] = float(valid_idx.numel())

        if valid_idx.numel() == 0:
            return zero_idx, base_stats

        # -------------------------------------------------
        # normalize slide ids
        # -------------------------------------------------
        slide_ids = None
        if slide_id_batch is not None:
            if torch.is_tensor(slide_id_batch):
                slide_ids = [str(x.item()) for x in slide_id_batch.view(-1)]
            else:
                slide_ids = [str(x) for x in slide_id_batch]

            if len(slide_ids) != patch_repr.shape[0]:
                raise ValueError(
                    f"slide_id_batch length {len(slide_ids)} != num tokens {patch_repr.shape[0]}"
                )

        # -------------------------------------------------
        # A) expert-aware balanced positive selection
        # -------------------------------------------------
        if self.use_pos_expert_balanced_selection and dispatch_weight_wsi is not None:
            hard_expert = dispatch_weight_wsi.argmax(dim=-1)   # [N]

            selected_idx_list = []
            per_expert_counts = {0: 0, 1: 0, 2: 0}
            dedup_dropped = 0

            for expert_id in self.pos_semantic_expert_ids:
                expert_id = int(expert_id)

                expert_mask = valid_mask & (hard_expert == expert_id)
                expert_idx = torch.nonzero(expert_mask, as_tuple=False).squeeze(-1)

                if expert_idx.numel() == 0:
                    continue

                expert_signal = ranking_signal[expert_idx]
                sort_local = torch.argsort(expert_signal, descending=True)
                expert_idx_sorted = expert_idx[sort_local]

                # slide dedup within expert
                if (
                    self.use_pos_slide_dedup
                    and slide_ids is not None
                    and self.pos_max_tokens_per_slide_per_expert > 0
                ):
                    kept = []
                    per_slide_counter = {}

                    for idx_t in expert_idx_sorted.tolist():
                        sid = slide_ids[idx_t]
                        cur = per_slide_counter.get(sid, 0)
                        if cur < self.pos_max_tokens_per_slide_per_expert:
                            kept.append(idx_t)
                            per_slide_counter[sid] = cur + 1
                        else:
                            dedup_dropped += 1

                    if len(kept) == 0:
                        continue

                    expert_idx_sorted = torch.tensor(
                        kept,
                        device=device,
                        dtype=torch.long,
                    )

                if expert_idx_sorted.numel() == 0:
                    continue

                # per-expert quota, especially for E0
                expert_cap = int(
                    self.pos_topk_per_expert_map.get(
                        expert_id,
                        self.cond_rank_pos_topk_per_expert,
                    )
                )
                if expert_cap <= 0:
                    continue

                k_e = min(expert_cap, expert_idx_sorted.numel())
                if k_e <= 0:
                    continue

                sel_idx_e = expert_idx_sorted[:k_e]
                selected_idx_list.append(sel_idx_e)

                if expert_id in per_expert_counts:
                    per_expert_counts[expert_id] = int(sel_idx_e.numel())

            experts_with_selection = sum(
                1 for _, cnt in per_expert_counts.items() if cnt > 0
            )

            # 没选到任何 expert
            if len(selected_idx_list) == 0:
                base_stats["cond_rank_pos_use_expert_balanced"] = 1.0
                base_stats["cond_rank_pos_dedup_dropped"] = float(dedup_dropped)
                base_stats["cond_rank_pos_num_selected_experts"] = 0.0
                return zero_idx, base_stats

            # # expert coverage 不够，直接阻断 positive cond-rank
            # if self.pos_require_min_experts and experts_with_selection < self.pos_min_selected_experts:
            #     base_stats["cond_rank_pos_use_expert_balanced"] = 1.0
            #     base_stats["cond_rank_pos_dedup_dropped"] = float(dedup_dropped)
            #     base_stats["cond_rank_pos_num_selected_experts"] = float(experts_with_selection)
            #     return zero_idx, base_stats
            failed_min_experts = 0.0
            if self.pos_require_min_experts and experts_with_selection < self.pos_min_selected_experts:
                failed_min_experts = 1.0

            topk_idx = torch.cat(selected_idx_list, dim=0)

            gap_sel = gap[topk_idx]
            ctx_sel = ranking_signal[topk_idx]

            return topk_idx, {
                "cond_rank_pos_num_candidates": float(valid_idx.numel()),
                "cond_rank_pos_num_selected": float(topk_idx.numel()),
                "cond_rank_pos_selected_gap_mean": float(gap_sel.mean().detach().cpu()),
                "cond_rank_pos_selected_gap_max": float(gap_sel.max().detach().cpu()),
                "cond_rank_pos_selected_gap_min": float(gap_sel.min().detach().cpu()),
                "cond_rank_pos_ctx_score_mean": float(ctx_sel.mean().detach().cpu()),
                "cond_rank_pos_e0_selected": float(per_expert_counts[0]),
                "cond_rank_pos_e1_selected": float(per_expert_counts[1]),
                "cond_rank_pos_e2_selected": float(per_expert_counts[2]),
                "cond_rank_pos_use_expert_balanced": 1.0,
                "cond_rank_pos_support_num_before": float(num_before_support),
                "cond_rank_pos_support_num_after": float(num_after_support),
                "cond_rank_pos_support_ratio": float(num_after_support / max(num_before_support, 1)),
                "cond_rank_pos_dedup_dropped": float(dedup_dropped),
                "cond_rank_pos_num_selected_experts": float(experts_with_selection),
                "cond_rank_pos_failed_min_experts": float(failed_min_experts),
            
            }

        # -------------------------------------------------
        # B) fallback: global top-k positive selection
        # only used when expert-balanced path is unavailable
        # -------------------------------------------------
        ranking_signal_valid = ranking_signal[valid_idx]
        k = self._resolve_cond_rank_topk(
            num_tokens=int(valid_idx.numel()),
            topk_fixed=self.cond_rank_pos_topk,
            topk_ratio=self.cond_rank_pos_topk_ratio,
        )
        topk_local = torch.topk(ranking_signal_valid, k=k, largest=True).indices
        topk_idx = valid_idx[topk_local]

        gap_sel = gap[topk_idx]
        ctx_sel = ranking_signal[topk_idx]

        e0_sel = e1_sel = e2_sel = 0.0
        num_selected_experts = 0.0

        if dispatch_weight_wsi is not None and topk_idx.numel() > 0:
            hard_expert = dispatch_weight_wsi.argmax(dim=-1)
            sel_expert = hard_expert[topk_idx]
            e0_sel = float((sel_expert == 0).sum().item())
            e1_sel = float((sel_expert == 1).sum().item())
            e2_sel = float((sel_expert == 2).sum().item())
            num_selected_experts = float(
                ((e0_sel > 0) + (e1_sel > 0) + (e2_sel > 0))
            )

        return topk_idx, {
            "cond_rank_pos_num_candidates": float(valid_idx.numel()),
            "cond_rank_pos_num_selected": float(topk_idx.numel()),
            "cond_rank_pos_selected_gap_mean": float(gap_sel.mean().detach().cpu()),
            "cond_rank_pos_selected_gap_max": float(gap_sel.max().detach().cpu()),
            "cond_rank_pos_selected_gap_min": float(gap_sel.min().detach().cpu()),
            "cond_rank_pos_ctx_score_mean": float(ctx_sel.mean().detach().cpu()),
            "cond_rank_pos_e0_selected": e0_sel,
            "cond_rank_pos_e1_selected": e1_sel,
            "cond_rank_pos_e2_selected": e2_sel,
            "cond_rank_pos_use_expert_balanced": 0.0,
            "cond_rank_pos_support_num_before": float(num_before_support),
            "cond_rank_pos_support_num_after": float(num_after_support),
            "cond_rank_pos_support_ratio": float(num_after_support / max(num_before_support, 1)),
            "cond_rank_pos_dedup_dropped": 0.0,
            "cond_rank_pos_num_selected_experts": num_selected_experts,
            "cond_rank_pos_failed_min_experts": 0.0,
        }
         
    def get_positive_anchor_token_indices(self, patch_repr, context_dict=None):
        device = patch_repr.device
        zero_idx = torch.empty(0, dtype=torch.long, device=device)

        if patch_repr is None or patch_repr.numel() == 0:
            return zero_idx, {
                "pos_anchor_num_candidates": 0.0,
                "pos_anchor_num_selected": 0.0,
                "pos_anchor_gap_mean": 0.0,
                "pos_anchor_gap_max": 0.0,
                "pos_anchor_gap_min": 0.0,
                "pos_anchor_tumor_mean": 0.0,
                "pos_anchor_ctx_score_mean": 0.0,
            }

        sim_tumor, sim_other_max, gap, _ = self.compute_role_scores_for_patch_repr(patch_repr)

        if context_dict is not None and self.use_context_guided_selection:
            if self.positive_anchor_select_mode == "tumor":
                ranking_signal = sim_tumor + 0.2 * context_dict["nb_gap_topk_mean"]
            else:
                if self.use_positive_context_ranking:
                    ranking_signal = self.build_positive_context_ranking_signal(
                        center_gap=context_dict["center_gap"],
                        nb_gap_topk_mean=context_dict["nb_gap_topk_mean"],
                        nb_gap_max=context_dict["nb_gap_max"],
                    )
                else:
                    ranking_signal = context_dict["ctx_score"]
        else:
            if self.positive_anchor_select_mode == "tumor":
                ranking_signal = sim_tumor
            else:
                ranking_signal = gap

        valid_mask = torch.ones_like(ranking_signal, dtype=torch.bool)

        if self.positive_anchor_min_tumor_score > -1e5:
            valid_mask = valid_mask & (sim_tumor >= self.positive_anchor_min_tumor_score)

        if self.positive_anchor_min_gap > -1e5:
            valid_mask = valid_mask & (gap >= self.positive_anchor_min_gap)

        valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

        if valid_idx.numel() == 0:
            return zero_idx, {
                "pos_anchor_num_candidates": 0.0,
                "pos_anchor_num_selected": 0.0,
                "pos_anchor_gap_mean": 0.0,
                "pos_anchor_gap_max": 0.0,
                "pos_anchor_gap_min": 0.0,
                "pos_anchor_tumor_mean": 0.0,
                "pos_anchor_ctx_score_mean": 0.0,
            }

        ranking_signal_valid = ranking_signal[valid_idx]
        k = self._resolve_cond_rank_topk(
            num_tokens=int(valid_idx.numel()),
            topk_fixed=self.positive_anchor_topk,
            topk_ratio=self.positive_anchor_topk_ratio,
        )

        topk_local = torch.topk(ranking_signal_valid, k=k, largest=True).indices
        topk_idx = valid_idx[topk_local]

        gap_sel = gap[topk_idx]
        sim_tumor_sel = sim_tumor[topk_idx]
        ctx_sel = ranking_signal[topk_idx]

        return topk_idx, {
            "pos_anchor_num_candidates": float(valid_idx.numel()),
            "pos_anchor_num_selected": float(topk_idx.numel()),
            "pos_anchor_gap_mean": float(gap_sel.mean().detach().cpu()),
            "pos_anchor_gap_max": float(gap_sel.max().detach().cpu()),
            "pos_anchor_gap_min": float(gap_sel.min().detach().cpu()),
            "pos_anchor_tumor_mean": float(sim_tumor_sel.mean().detach().cpu()),
            "pos_anchor_ctx_score_mean": float(ctx_sel.mean().detach().cpu()),
        }

    
    def get_residual_hn_token_indices(
        self,
        patch_repr,
        dispatch_weight_wsi,
        context_dict=None,
    ):
        device = patch_repr.device
        zero_idx = torch.empty(0, dtype=torch.long, device=device)

        if patch_repr is None or patch_repr.numel() == 0:
            return zero_idx, {
                "residual_hn_num_before_expert_filter": 0.0,
                "residual_hn_num_after_expert_filter": 0.0,
                "residual_hn_gap_mean": 0.0,
                "residual_hn_gap_max": 0.0,
                "residual_hn_gap_min": 0.0,
            }

        neg_idx, _ = self.get_negative_rank_selected_token_indices(
            patch_repr=patch_repr,
            dispatch_weight_wsi=dispatch_weight_wsi,
            context_dict=context_dict,
        )
        if neg_idx.numel() == 0:
            return zero_idx, {
                "residual_hn_num_before_expert_filter": 0.0,
                "residual_hn_num_after_expert_filter": 0.0,
                "residual_hn_gap_mean": 0.0,
                "residual_hn_gap_max": 0.0,
                "residual_hn_gap_min": 0.0,
            }

        _, _, gap, _ = self.compute_role_scores_for_patch_repr(patch_repr)
        hard_expert = dispatch_weight_wsi.argmax(dim=-1)

        gap_sel = gap[neg_idx]
        expert_sel = hard_expert[neg_idx]

        valid_mask = torch.ones_like(gap_sel, dtype=torch.bool)

        if self.residual_hn_target_expert is not None:
            valid_mask = valid_mask & (expert_sel == self.residual_hn_target_expert)

        if self.residual_hn_min_gap > -1e5:
            valid_mask = valid_mask & (gap_sel >= self.residual_hn_min_gap)
        if self.residual_hn_max_gap < 1e5:
            valid_mask = valid_mask & (gap_sel <= self.residual_hn_max_gap)

        residual_idx = neg_idx[valid_mask]
        gap_res = gap[residual_idx] if residual_idx.numel() > 0 else gap_sel.new_empty(0)

        if gap_res.numel() == 0:
            return residual_idx, {
                "residual_hn_num_before_expert_filter": float(neg_idx.numel()),
                "residual_hn_num_after_expert_filter": 0.0,
                "residual_hn_gap_mean": 0.0,
                "residual_hn_gap_max": 0.0,
                "residual_hn_gap_min": 0.0,
            }

        return residual_idx, {
            "residual_hn_num_before_expert_filter": float(neg_idx.numel()),
            "residual_hn_num_after_expert_filter": float(residual_idx.numel()),
            "residual_hn_gap_mean": float(gap_res.mean().detach().cpu()),
            "residual_hn_gap_max": float(gap_res.max().detach().cpu()),
            "residual_hn_gap_min": float(gap_res.min().detach().cpu()),
        }

    def compute_single_wsi_conditional_pairwise_ranking_loss(
        self,
        patch_repr,
        slide_label,
        dispatch_weight_wsi=None,
        context_dict=None,
    ):
        zero = patch_repr.new_tensor(0.0)

        stats = {
            "cond_rank_raw": 0.0,
            "cond_rank_num_tokens": 0.0,
            "cond_rank_num_selected": 0.0,
            "cond_rank_mean_tumor": 0.0,
            "cond_rank_mean_other_max": 0.0,
            "cond_rank_mean_gap": 0.0,
            "cond_rank_selected_gap_mean": 0.0,
            "cond_rank_selected_gap_max": 0.0,
            "cond_rank_selected_gap_min": 0.0,
            "cond_rank_mode": -1.0,

            "cond_rank_neg_raw": 0.0,
            "cond_rank_neg_num_tokens": 0.0,
            "cond_rank_neg_num_selected": 0.0,
            "cond_rank_neg_mean_tumor": 0.0,
            "cond_rank_neg_mean_other_max": 0.0,
            "cond_rank_neg_mean_gap": 0.0,
            "cond_rank_neg_selected_gap_mean": 0.0,
            "cond_rank_neg_selected_gap_max": 0.0,
            "cond_rank_neg_selected_gap_min": 0.0,

            "cond_rank_pos_raw": 0.0,
            "cond_rank_pos_num_tokens": 0.0,
            "cond_rank_pos_num_selected": 0.0,
            "cond_rank_pos_mean_tumor": 0.0,
            "cond_rank_pos_mean_other_max": 0.0,
            "cond_rank_pos_mean_gap": 0.0,
            "cond_rank_pos_selected_gap_mean": 0.0,
            "cond_rank_pos_selected_gap_max": 0.0,
            "cond_rank_pos_selected_gap_min": 0.0,
            "cond_rank_pos_ctx_score_mean": 0.0,
            
        }

        if patch_repr is None or patch_repr.numel() == 0:
            return zero, stats

        y = self._get_wsi_label_value(slide_label)

        sim_tumor, sim_other_max, gap, _ = self.compute_role_scores_for_patch_repr(patch_repr)

        N = patch_repr.shape[0]
        stats["cond_rank_num_tokens"] = float(N)
        stats["cond_rank_mean_tumor"] = float(sim_tumor.mean().detach().cpu())
        stats["cond_rank_mean_other_max"] = float(sim_other_max.mean().detach().cpu())
        stats["cond_rank_mean_gap"] = float(gap.mean().detach().cpu())
        stats["cond_rank_mode"] = float(y)

        if y == 0:
            topk_idx, aux = self.get_negative_rank_selected_token_indices(
                patch_repr=patch_repr,
                dispatch_weight_wsi=dispatch_weight_wsi,
                context_dict=context_dict,
            )
            stats.update(aux)

            if topk_idx.numel() == 0:
                stats["cond_rank_neg_num_tokens"] = float(N)
                stats["cond_rank_neg_num_selected"] = 0.0
                stats["cond_rank_neg_mean_tumor"] = float(sim_tumor.mean().detach().cpu())
                stats["cond_rank_neg_mean_other_max"] = float(sim_other_max.mean().detach().cpu())
                stats["cond_rank_neg_mean_gap"] = float(gap.mean().detach().cpu())
                return zero, stats

            gap_sel = gap[topk_idx]
            loss = F.relu(self.cond_rank_neg_margin + gap_sel).mean()

            stats["cond_rank_num_selected"] = float(topk_idx.numel())
            stats["cond_rank_selected_gap_mean"] = float(gap_sel.mean().detach().cpu())
            stats["cond_rank_selected_gap_max"] = float(gap_sel.max().detach().cpu())
            stats["cond_rank_selected_gap_min"] = float(gap_sel.min().detach().cpu())
            stats["cond_rank_raw"] = float(loss.detach().cpu())

            stats["cond_rank_neg_raw"] = float(loss.detach().cpu())
            stats["cond_rank_neg_num_tokens"] = float(N)
            stats["cond_rank_neg_num_selected"] = float(topk_idx.numel())
            stats["cond_rank_neg_mean_tumor"] = float(sim_tumor.mean().detach().cpu())
            stats["cond_rank_neg_mean_other_max"] = float(sim_other_max.mean().detach().cpu())
            stats["cond_rank_neg_mean_gap"] = float(gap.mean().detach().cpu())
            stats["cond_rank_neg_selected_gap_mean"] = float(gap_sel.mean().detach().cpu())
            stats["cond_rank_neg_selected_gap_max"] = float(gap_sel.max().detach().cpu())
            stats["cond_rank_neg_selected_gap_min"] = float(gap_sel.min().detach().cpu())

            return loss, stats

        else:
                
            topk_idx, aux = self.get_positive_rank_selected_token_indices(
                patch_repr=patch_repr,
                dispatch_weight_wsi=dispatch_weight_wsi,
                context_dict=context_dict,
            )
            stats.update(aux)

            if topk_idx.numel() == 0:
                if self.cond_rank_allow_empty_pos:
                    stats["cond_rank_pos_raw"] = 0.0
                    stats["cond_rank_pos_num_tokens"] = float(N)
                    stats["cond_rank_pos_num_selected"] = 0.0
                    stats["cond_rank_pos_mean_tumor"] = float(sim_tumor.mean().detach().cpu())
                    stats["cond_rank_pos_mean_other_max"] = float(sim_other_max.mean().detach().cpu())
                    stats["cond_rank_pos_mean_gap"] = float(gap.mean().detach().cpu())
                    return zero, stats
                else:
                    topk_idx = torch.arange(N, device=patch_repr.device)

            gap_sel = gap[topk_idx]

            loss = F.relu(self.cond_rank_pos_margin - gap_sel).mean()

            stats["cond_rank_num_selected"] = float(topk_idx.numel())
            stats["cond_rank_selected_gap_mean"] = float(gap_sel.mean().detach().cpu())
            stats["cond_rank_selected_gap_max"] = float(gap_sel.max().detach().cpu())
            stats["cond_rank_selected_gap_min"] = float(gap_sel.min().detach().cpu())
            stats["cond_rank_raw"] = float(loss.detach().cpu())

            stats["cond_rank_pos_raw"] = float(loss.detach().cpu())
            stats["cond_rank_pos_num_tokens"] = float(N)
            stats["cond_rank_pos_num_selected"] = float(topk_idx.numel())
            stats["cond_rank_pos_mean_tumor"] = float(sim_tumor.mean().detach().cpu())
            stats["cond_rank_pos_mean_other_max"] = float(sim_other_max.mean().detach().cpu())
            stats["cond_rank_pos_mean_gap"] = float(gap.mean().detach().cpu())

            # 注意：下面这些会优先用 helper 里算好的 expert-aware stats
            # 如果 helper 没提供，也不会报错
            stats["cond_rank_pos_selected_gap_mean"] = stats.get(
                "cond_rank_pos_selected_gap_mean",
                float(gap_sel.mean().detach().cpu())
            )
            stats["cond_rank_pos_selected_gap_max"] = stats.get(
                "cond_rank_pos_selected_gap_max",
                float(gap_sel.max().detach().cpu())
            )
            stats["cond_rank_pos_selected_gap_min"] = stats.get(
                "cond_rank_pos_selected_gap_min",
                float(gap_sel.min().detach().cpu())
            )

            return loss, stats
            # if context_dict is not None and self.use_context_guided_selection:
            #     if self.cond_rank_pos_select_mode == "tumor":
            #         ranking_signal = sim_tumor + 0.2 * context_dict["nb_gap_topk_mean"]
            #     else:
            #         if self.use_positive_context_ranking:
            #             ranking_signal = self.build_positive_context_ranking_signal(
            #                 center_gap=context_dict["center_gap"],
            #                 nb_gap_topk_mean=context_dict["nb_gap_topk_mean"],
            #                 nb_gap_max=context_dict["nb_gap_max"],
            #             )
            #         else:
            #             ranking_signal = context_dict["ctx_score"]
            # else:
            #     if self.cond_rank_pos_select_mode == "tumor":
            #         ranking_signal = sim_tumor
            #     else:
            #         ranking_signal = gap

            # valid_mask = torch.ones_like(ranking_signal, dtype=torch.bool)

            # if self.cond_rank_pos_min_tumor_score > -1e5:
            #     valid_mask = valid_mask & (sim_tumor >= self.cond_rank_pos_min_tumor_score)

            # if self.cond_rank_pos_min_gap > -1e5:
            #     valid_mask = valid_mask & (gap >= self.cond_rank_pos_min_gap)

            # valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)

            # if valid_idx.numel() == 0:
            #     if self.cond_rank_allow_empty_pos:
            #         stats["cond_rank_pos_raw"] = 0.0
            #         stats["cond_rank_pos_num_tokens"] = float(N)
            #         stats["cond_rank_pos_num_selected"] = 0.0
            #         stats["cond_rank_pos_mean_tumor"] = float(sim_tumor.mean().detach().cpu())
            #         stats["cond_rank_pos_mean_other_max"] = float(sim_other_max.mean().detach().cpu())
            #         stats["cond_rank_pos_mean_gap"] = float(gap.mean().detach().cpu())
            #         return zero, stats
            #     else:
            #         valid_idx = torch.arange(N, device=patch_repr.device)

            # ranking_signal_valid = ranking_signal[valid_idx]

            # k = self._resolve_cond_rank_topk(
            #     num_tokens=int(valid_idx.numel()),
            #     topk_fixed=self.cond_rank_pos_topk,
            #     topk_ratio=self.cond_rank_pos_topk_ratio,
            # )

            # topk_local = torch.topk(ranking_signal_valid, k=k, largest=True).indices
            # topk_idx = valid_idx[topk_local]

            # gap_sel = gap[topk_idx]
            # ctx_sel = ranking_signal[topk_idx]
            # loss = F.relu(self.cond_rank_pos_margin - gap_sel).mean()

            # stats["cond_rank_num_selected"] = float(k)
            # stats["cond_rank_selected_gap_mean"] = float(gap_sel.mean().detach().cpu())
            # stats["cond_rank_selected_gap_max"] = float(gap_sel.max().detach().cpu())
            # stats["cond_rank_selected_gap_min"] = float(gap_sel.min().detach().cpu())
            # stats["cond_rank_raw"] = float(loss.detach().cpu())

            # stats["cond_rank_pos_raw"] = float(loss.detach().cpu())
            # stats["cond_rank_pos_num_tokens"] = float(N)
            # stats["cond_rank_pos_num_selected"] = float(k)
            # stats["cond_rank_pos_mean_tumor"] = float(sim_tumor.mean().detach().cpu())
            # stats["cond_rank_pos_mean_other_max"] = float(sim_other_max.mean().detach().cpu())
            # stats["cond_rank_pos_mean_gap"] = float(gap.mean().detach().cpu())
            # stats["cond_rank_pos_selected_gap_mean"] = float(gap_sel.mean().detach().cpu())
            # stats["cond_rank_pos_selected_gap_max"] = float(gap_sel.max().detach().cpu())
            # stats["cond_rank_pos_selected_gap_min"] = float(gap_sel.min().detach().cpu())
            # stats["cond_rank_pos_ctx_score_mean"] = float(ctx_sel.mean().detach().cpu())

            # return loss, stats

    def compute_wsi_conditional_pairwise_ranking_loss(
        self,
        images,
        slide_label,
        patch_batch_size=None,
        is_eval=False,
        neighbor_images_list=None,
    ):
        zero = next(self.parameters()).new_tensor(0.0)

        stats = {
            "cond_rank_raw": 0.0,
            "cond_rank_weighted": 0.0,
            "cond_rank_num_tokens": 0.0,
            "cond_rank_num_selected": 0.0,
            "cond_rank_mean_tumor": 0.0,
            "cond_rank_mean_other_max": 0.0,
            "cond_rank_mean_gap": 0.0,
            "cond_rank_selected_gap_mean": 0.0,
            "cond_rank_selected_gap_max": 0.0,
            "cond_rank_selected_gap_min": 0.0,
            "cond_rank_mode": -1.0,

            "cond_rank_neg_raw": 0.0,
            "cond_rank_neg_weighted": 0.0,
            "cond_rank_neg_num_tokens": 0.0,
            "cond_rank_neg_num_selected": 0.0,
            "cond_rank_neg_mean_tumor": 0.0,
            "cond_rank_neg_mean_other_max": 0.0,
            "cond_rank_neg_mean_gap": 0.0,
            "cond_rank_neg_selected_gap_mean": 0.0,
            "cond_rank_neg_selected_gap_max": 0.0,
            "cond_rank_neg_selected_gap_min": 0.0,

            "cond_rank_pos_raw": 0.0,
            "cond_rank_pos_weighted": 0.0,
            "cond_rank_pos_num_tokens": 0.0,
            "cond_rank_pos_num_selected": 0.0,
            "cond_rank_pos_mean_tumor": 0.0,
            "cond_rank_pos_mean_other_max": 0.0,
            "cond_rank_pos_mean_gap": 0.0,
            "cond_rank_pos_selected_gap_mean": 0.0,
            "cond_rank_pos_selected_gap_max": 0.0,
            "cond_rank_pos_selected_gap_min": 0.0,
        }

        if (not self.use_conditional_pairwise_ranking):
            return zero, stats

        if images is None or slide_label is None:
            return zero, stats

        y = self._get_wsi_label_value(slide_label)
        mode = "positive" if y == 1 else "negative"

        patch_repr, dispatch_weight_img, context_dict = self.encode_wsi_patch_repr_with_optional_context(
            images=images,
            neighbor_images_list=neighbor_images_list,
            patch_batch_size=patch_batch_size,
            is_eval=is_eval,
            mode=mode,
        )

        loss, raw_stats = self.compute_single_wsi_conditional_pairwise_ranking_loss(
            patch_repr=patch_repr,
            slide_label=slide_label,
            dispatch_weight_wsi=dispatch_weight_img,
            context_dict=context_dict,
        )

        stats.update(raw_stats)

        cur_weight = self.cond_rank_pos_weight if y == 1 else self.cond_rank_neg_weight
        weighted_val = cur_weight * loss
        stats["cond_rank_weighted"] = float(weighted_val.detach().cpu())

        if y == 0:
            stats["cond_rank_neg_weighted"] = float(weighted_val.detach().cpu())
        else:
            stats["cond_rank_pos_weighted"] = float(weighted_val.detach().cpu())

        if context_dict is not None:
            stats["ctx_center_gap_mean"] = float(context_dict["center_gap"].mean().detach().cpu())
            stats["ctx_nb_gap_topk_mean"] = float(context_dict["nb_gap_topk_mean"].mean().detach().cpu())
            stats["ctx_score_mean"] = float(context_dict["ctx_score"].mean().detach().cpu())

        return loss, stats

    def compute_neg_rank_free_preference_loss(
        self,
        patch_repr,
        dispatch_weight_wsi,
        slide_label,
    ):
        """
        patch_repr: [N, D_teacher]
        dispatch_weight_wsi: [N, E]
        slide_label: scalar / tensor scalar in {0,1}

        return:
            loss: scalar
            stats: dict
        """
        zero = patch_repr.new_tensor(0.0)

        stats = {
            "neg_rank_free_pref_raw": 0.0,
            "neg_rank_free_pref_weighted": 0.0,
            "neg_rank_free_pref_num_selected": 0.0,
            "neg_rank_free_pref_free_mass_mean": 0.0,
            "neg_rank_free_pref_free_mass_min": 0.0,
            "neg_rank_free_pref_free_mass_max": 0.0,
            "neg_rank_sel_num_before_filter": 0.0,
            "neg_rank_sel_num_after_filter": 0.0,
            "neg_rank_sel_gap_mean": 0.0,
            "neg_rank_sel_gap_max": 0.0,
            "neg_rank_sel_gap_min": 0.0,
        }

        if (not self.use_neg_rank_free_preference) or (self.neg_rank_free_pref_weight <= 0):
            return zero, stats

        if patch_repr is None or dispatch_weight_wsi is None:
            return zero, stats

        y = self._get_wsi_label_value(slide_label)
        if y != 0:
            return zero, stats

        if patch_repr.shape[0] == 0 or dispatch_weight_wsi.shape[0] == 0:
            return zero, stats

        topk_idx, aux = self.get_negative_rank_selected_token_indices(
            patch_repr=patch_repr,
            dispatch_weight_wsi=dispatch_weight_wsi,
        )
        stats.update(aux)

        if topk_idx.numel() == 0:
            return zero, stats

        free_mass = dispatch_weight_wsi[topk_idx, self.free_expert_id]   # [K]
        loss = -torch.log(free_mass + self.neg_rank_free_pref_eps).mean()

        stats["neg_rank_free_pref_raw"] = float(loss.detach().cpu())
        stats["neg_rank_free_pref_num_selected"] = float(topk_idx.numel())
        stats["neg_rank_free_pref_free_mass_mean"] = float(free_mass.mean().detach().cpu())
        stats["neg_rank_free_pref_free_mass_min"] = float(free_mass.min().detach().cpu())
        stats["neg_rank_free_pref_free_mass_max"] = float(free_mass.max().detach().cpu())

        return loss, stats

    def compute_residual_hn_push_loss(
        self,
        patch_repr,
        dispatch_weight_wsi,
        slide_label,
    ):
        """
        patch_repr: [N, D_teacher]
        dispatch_weight_wsi: [N, E]
        slide_label: scalar / tensor scalar in {0,1}

        return:
            loss: scalar
            stats: dict
        """
        zero = patch_repr.new_tensor(0.0)

        stats = {
            "residual_hn_push_raw": 0.0,
            "residual_hn_push_weighted": 0.0,
            "residual_hn_num_before_expert_filter": 0.0,
            "residual_hn_num_after_expert_filter": 0.0,
            "residual_hn_gap_mean": 0.0,
            "residual_hn_gap_max": 0.0,
            "residual_hn_gap_min": 0.0,
        }

        if (not self.use_residual_hn_push) or (self.residual_hn_push_weight <= 0):
            return zero, stats

        if patch_repr is None or dispatch_weight_wsi is None:
            return zero, stats

        y = self._get_wsi_label_value(slide_label)
        if y != 0:
            return zero, stats

        residual_idx, aux = self.get_residual_hn_token_indices(
            patch_repr=patch_repr,
            dispatch_weight_wsi=dispatch_weight_wsi,
        )
        stats.update(aux)

        if residual_idx.numel() == 0:
            return zero, stats

        _, _, gap, _ = self.compute_role_scores_for_patch_repr(patch_repr)
        gap_res = gap[residual_idx]   # [K]

        loss = F.relu(self.residual_hn_push_margin + gap_res).mean()

        stats["residual_hn_push_raw"] = float(loss.detach().cpu())
        return loss, stats

    def compute_positive_anchor_boost_loss(
        self,
        patch_repr,
        slide_label,
        context_dict=None,
    ):
        zero = patch_repr.new_tensor(0.0)

        stats = {
            "positive_anchor_boost_raw": 0.0,
            "positive_anchor_boost_weighted": 0.0,
            "pos_anchor_num_candidates": 0.0,
            "pos_anchor_num_selected": 0.0,
            "pos_anchor_gap_mean": 0.0,
            "pos_anchor_gap_max": 0.0,
            "pos_anchor_gap_min": 0.0,
            "pos_anchor_tumor_mean": 0.0,
        }

        if (not self.use_positive_anchor_boost) or (self.positive_anchor_weight <= 0):
            return zero, stats

        if patch_repr is None or patch_repr.numel() == 0:
            return zero, stats

        y = self._get_wsi_label_value(slide_label)
        if y != 1:
            return zero, stats

        topk_idx, aux = self.get_positive_anchor_token_indices(
            patch_repr=patch_repr,
            context_dict=context_dict,
        )
        stats.update(aux)

        if topk_idx.numel() == 0:
            if self.positive_anchor_allow_empty:
                return zero, stats
            else:
                sim_tumor, sim_other_max, gap, _ = self.compute_role_scores_for_patch_repr(patch_repr)
                ranking_signal = gap
                k = self._resolve_cond_rank_topk(
                    num_tokens=patch_repr.shape[0],
                    topk_fixed=self.positive_anchor_topk,
                    topk_ratio=self.positive_anchor_topk_ratio,
                )
                topk_idx = torch.topk(ranking_signal, k=k, largest=True).indices

        _, _, gap, _ = self.compute_role_scores_for_patch_repr(patch_repr)
        gap_sel = gap[topk_idx]

        loss = F.relu(self.positive_anchor_margin - gap_sel).mean()
        stats["positive_anchor_boost_raw"] = float(loss.detach().cpu())
        return loss, stats

    def compute_wsi_neg_rank_free_preference_loss(
        self,
        images,
        slide_label,
        patch_batch_size=None,
        is_eval=False,
    ):
        """
        images:
            [N, 3, H, W] or list[tensor]
        slide_label:
            scalar / tensor scalar in {0,1}

        return:
            loss: scalar
            stats: dict
        """
        zero = next(self.parameters()).new_tensor(0.0)

        stats = {
            "neg_rank_free_pref_raw": 0.0,
            "neg_rank_free_pref_weighted": 0.0,
            "neg_rank_free_pref_num_selected": 0.0,
            "neg_rank_free_pref_free_mass_mean": 0.0,
            "neg_rank_free_pref_free_mass_min": 0.0,
            "neg_rank_free_pref_free_mass_max": 0.0,
            "neg_rank_sel_num_before_filter": 0.0,
            "neg_rank_sel_num_after_filter": 0.0,
            "neg_rank_sel_gap_mean": 0.0,
            "neg_rank_sel_gap_max": 0.0,
            "neg_rank_sel_gap_min": 0.0,
        }

        if (not self.use_neg_rank_free_preference) or (self.neg_rank_free_pref_weight <= 0):
            return zero, stats

        if images is None or slide_label is None:
            return zero, stats

        y = self._get_wsi_label_value(slide_label)
        if y != 0:
            return zero, stats

        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        device = next(self.parameters()).device
        images = images.to(device, non_blocking=True)

        if patch_batch_size is None:
            patch_batch_size = self.wsi_patch_batch_size

        all_repr = []
        all_dispatch = []

        for start in range(0, images.shape[0], patch_batch_size):
            end = min(start + patch_batch_size, images.shape[0])
            patch_batch = images[start:end]

            patch_repr, _, dispatch_weight_img = self.encode_patch_batch_for_wsi_bag_with_dispatch(
                patch_batch,
                is_eval=is_eval,
            )
            all_repr.append(patch_repr)
            all_dispatch.append(dispatch_weight_img)

        patch_repr = torch.cat(all_repr, dim=0)               # [N, D_teacher]
        dispatch_weight_img = torch.cat(all_dispatch, dim=0)  # [N, E]

        loss, raw_stats = self.compute_neg_rank_free_preference_loss(
            patch_repr=patch_repr,
            dispatch_weight_wsi=dispatch_weight_img,
            slide_label=slide_label,
        )
        stats.update(raw_stats)
        stats["neg_rank_free_pref_weighted"] = float(
            (self.neg_rank_free_pref_weight * loss).detach().cpu()
        )
        return loss, stats

    def compute_wsi_residual_hn_push_loss(
        self,
        images,
        slide_label,
        patch_batch_size=None,
        is_eval=False,
        neighbor_images_list=None,
    ):
        zero = next(self.parameters()).new_tensor(0.0)

        stats = {
            "residual_hn_push_raw": 0.0,
            "residual_hn_push_weighted": 0.0,
            "residual_hn_num_before_expert_filter": 0.0,
            "residual_hn_num_after_expert_filter": 0.0,
            "residual_hn_gap_mean": 0.0,
            "residual_hn_gap_max": 0.0,
            "residual_hn_gap_min": 0.0,
        }

        if (not self.use_residual_hn_push) or (self.residual_hn_push_weight <= 0):
            return zero, stats

        if images is None or slide_label is None:
            return zero, stats

        y = self._get_wsi_label_value(slide_label)
        if y != 0:
            return zero, stats

        patch_repr, dispatch_weight_img, context_dict = self.encode_wsi_patch_repr_with_optional_context(
            images=images,
            neighbor_images_list=neighbor_images_list,
            patch_batch_size=patch_batch_size,
            is_eval=is_eval,
            mode="negative",
        )

        loss, raw_stats = self.compute_residual_hn_push_loss(
            patch_repr=patch_repr,
            dispatch_weight_wsi=dispatch_weight_img,
            slide_label=slide_label,
        )
        stats.update(raw_stats)
        stats["residual_hn_push_weighted"] = float(
            (self.residual_hn_push_weight * loss).detach().cpu()
        )

        return loss, stats

    def compute_wsi_positive_anchor_boost_loss(
        self,
        images,
        slide_label,
        patch_batch_size=None,
        is_eval=False,
        neighbor_images_list=None,
    ):
        zero = next(self.parameters()).new_tensor(0.0)

        stats = {
            "positive_anchor_boost_raw": 0.0,
            "positive_anchor_boost_weighted": 0.0,
            "pos_anchor_num_candidates": 0.0,
            "pos_anchor_num_selected": 0.0,
            "pos_anchor_gap_mean": 0.0,
            "pos_anchor_gap_max": 0.0,
            "pos_anchor_gap_min": 0.0,
            "pos_anchor_tumor_mean": 0.0,
        }

        if (not self.use_positive_anchor_boost) or (self.positive_anchor_weight <= 0):
            return zero, stats

        if images is None or slide_label is None:
            return zero, stats

        y = self._get_wsi_label_value(slide_label)
        if y != 1:
            return zero, stats

        patch_repr, _, context_dict = self.encode_wsi_patch_repr_with_optional_context(
            images=images,
            neighbor_images_list=neighbor_images_list,
            patch_batch_size=patch_batch_size,
            is_eval=is_eval,
            mode="positive",
        )

        loss, raw_stats = self.compute_positive_anchor_boost_loss(
            patch_repr=patch_repr,
            slide_label=slide_label,
            context_dict=context_dict,
        )
        stats.update(raw_stats)
        stats["positive_anchor_boost_weighted"] = float(
            (self.positive_anchor_weight * loss).detach().cpu()
        )

        return loss, stats
    
    def compute_batch_conditional_pairwise_ranking_loss(
        self,
        center_images,
        slide_label_batch,
        neighbor_images_list=None,
        is_eval=False,
        slide_id_batch=None,
    ):
        """
        batch-level conditional ranking:
        - 对整个 batch 的 center patch 一起做 selection
        - negative centers: 全局选高风险伪阳
        - positive centers: 全局做 support mask + expert-balanced + slide dedup
        """
        zero = next(self.parameters()).new_tensor(0.0)

        stats = {
            "cond_rank_raw": 0.0,
            "cond_rank_weighted": 0.0,
            "cond_rank_num_tokens": 0.0,
            "cond_rank_num_selected": 0.0,
            "cond_rank_mean_tumor": 0.0,
            "cond_rank_mean_other_max": 0.0,
            "cond_rank_mean_gap": 0.0,
            "cond_rank_selected_gap_mean": 0.0,
            "cond_rank_selected_gap_max": 0.0,
            "cond_rank_selected_gap_min": 0.0,
            "cond_rank_mode": -1.0,

            "cond_rank_neg_raw": 0.0,
            "cond_rank_neg_weighted": 0.0,
            "cond_rank_neg_num_tokens": 0.0,
            "cond_rank_neg_num_selected": 0.0,
            "cond_rank_neg_mean_tumor": 0.0,
            "cond_rank_neg_mean_other_max": 0.0,
            "cond_rank_neg_mean_gap": 0.0,
            "cond_rank_neg_selected_gap_mean": 0.0,
            "cond_rank_neg_selected_gap_max": 0.0,
            "cond_rank_neg_selected_gap_min": 0.0,
            "neg_rank_sel_e0": 0.0,
            "neg_rank_sel_e1": 0.0,
            "neg_rank_sel_e2": 0.0,

            "cond_rank_pos_raw": 0.0,
            "cond_rank_pos_weighted": 0.0,
            "cond_rank_pos_num_tokens": 0.0,
            "cond_rank_pos_num_selected": 0.0,
            "cond_rank_pos_mean_tumor": 0.0,
            "cond_rank_pos_mean_other_max": 0.0,
            "cond_rank_pos_mean_gap": 0.0,
            "cond_rank_pos_selected_gap_mean": 0.0,
            "cond_rank_pos_selected_gap_max": 0.0,
            "cond_rank_pos_selected_gap_min": 0.0,
            "cond_rank_pos_ctx_score_mean": 0.0,

            "cond_rank_pos_num_candidates": 0.0,
            "cond_rank_pos_e0_selected": 0.0,
            "cond_rank_pos_e1_selected": 0.0,
            "cond_rank_pos_e2_selected": 0.0,
            "cond_rank_pos_use_expert_balanced": 0.0,
            "cond_rank_pos_support_num_before": 0.0,
            "cond_rank_pos_support_num_after": 0.0,
            "cond_rank_pos_support_ratio": 0.0,
            "cond_rank_pos_dedup_dropped": 0.0,
            "cond_rank_pos_num_selected_experts": 0.0,
            "cond_rank_pos_failed_min_experts": 0.0,

            "ctx_center_gap_mean": 0.0,
            "ctx_nb_gap_topk_mean": 0.0,
            "ctx_score_mean": 0.0,
        }

        if (not self.use_conditional_pairwise_ranking):
            return zero, stats

        if center_images is None or slide_label_batch is None:
            return zero, stats

        B = center_images.shape[0]
        if B == 0:
            return zero, stats

        device = center_images.device
        slide_label_batch = slide_label_batch.to(device).view(-1).long()

        if neighbor_images_list is None:
            neighbor_images_list = [None] * B

        # -------------------------------------------------
        # 1) encode all centers jointly
        # -------------------------------------------------
        center_repr, _, dispatch_weight_img = self.encode_patch_batch_for_wsi_bag_with_dispatch(
            center_images,
            is_eval=is_eval,
        )   # center_repr: [B, D_teacher], dispatch_weight_img: [B, E]

        # 先统一算全 batch role score，避免后面作用域问题
        sim_tumor, sim_other_max, gap, _ = self.compute_role_scores_for_patch_repr(center_repr)

        stats["cond_rank_num_tokens"] = float(B)
        stats["cond_rank_mean_tumor"] = float(sim_tumor.mean().detach().cpu())
        stats["cond_rank_mean_other_max"] = float(sim_other_max.mean().detach().cpu())
        stats["cond_rank_mean_gap"] = float(gap.mean().detach().cpu())

        # -------------------------------------------------
        # 2) build batch context
        # -------------------------------------------------
        neg_context_dict = None
        pos_context_dict = None

        if self.use_context_guided_selection:
            neg_idx_all = torch.nonzero(slide_label_batch == 0, as_tuple=False).squeeze(-1)
            pos_idx_all = torch.nonzero(slide_label_batch == 1, as_tuple=False).squeeze(-1)

            if neg_idx_all.numel() > 0:
                neg_center_images = center_images[neg_idx_all]
                neg_neighbor_images_list = [neighbor_images_list[i] for i in neg_idx_all.tolist()]
                neg_context_dict = self.build_context_scores_for_batch_centers(
                    center_images=neg_center_images,
                    neighbor_images_list=neg_neighbor_images_list,
                    is_eval=is_eval,
                    mode="negative",
                )

            if pos_idx_all.numel() > 0:
                pos_center_images = center_images[pos_idx_all]
                pos_neighbor_images_list = [neighbor_images_list[i] for i in pos_idx_all.tolist()]
                pos_context_dict = self.build_context_scores_for_batch_centers(
                    center_images=pos_center_images,
                    neighbor_images_list=pos_neighbor_images_list,
                    is_eval=is_eval,
                    mode="positive",
                )

            ctx_score_vals = []
            ctx_center_vals = []
            ctx_nb_vals = []

            if neg_context_dict is not None:
                ctx_score_vals.append(neg_context_dict["ctx_score"])
                ctx_center_vals.append(neg_context_dict["center_gap"])
                ctx_nb_vals.append(neg_context_dict["nb_gap_topk_mean"])

            if pos_context_dict is not None:
                ctx_score_vals.append(pos_context_dict["ctx_score"])
                ctx_center_vals.append(pos_context_dict["center_gap"])
                ctx_nb_vals.append(pos_context_dict["nb_gap_topk_mean"])

            if len(ctx_score_vals) > 0:
                stats["ctx_center_gap_mean"] = float(torch.cat(ctx_center_vals).mean().detach().cpu())
                stats["ctx_nb_gap_topk_mean"] = float(torch.cat(ctx_nb_vals).mean().detach().cpu())
                stats["ctx_score_mean"] = float(torch.cat(ctx_score_vals).mean().detach().cpu())

        # -------------------------------------------------
        # 3) negative selection on all negative centers
        # -------------------------------------------------
        neg_idx_all = torch.nonzero(slide_label_batch == 0, as_tuple=False).squeeze(-1)
        neg_loss = zero

        if neg_idx_all.numel() > 0:
            neg_repr = center_repr[neg_idx_all]
            neg_dispatch = dispatch_weight_img[neg_idx_all]
            neg_context = neg_context_dict

            neg_sel_local, neg_aux = self.get_negative_rank_selected_token_indices(
                patch_repr=neg_repr,
                dispatch_weight_wsi=neg_dispatch,
                context_dict=neg_context,
            )

            stats.update(neg_aux)
            stats["cond_rank_neg_num_tokens"] = float(neg_repr.shape[0])
            stats["cond_rank_neg_mean_tumor"] = float(sim_tumor[neg_idx_all].mean().detach().cpu())
            stats["cond_rank_neg_mean_other_max"] = float(sim_other_max[neg_idx_all].mean().detach().cpu())
            stats["cond_rank_neg_mean_gap"] = float(gap[neg_idx_all].mean().detach().cpu())

            if neg_sel_local.numel() > 0:
                neg_gap_sel = self.compute_role_scores_for_patch_repr(neg_repr)[2][neg_sel_local]
                neg_loss = F.relu(self.cond_rank_neg_margin + neg_gap_sel).mean()

                stats["cond_rank_neg_raw"] = float(neg_loss.detach().cpu())
                stats["cond_rank_neg_num_selected"] = float(neg_sel_local.numel())
                stats["cond_rank_neg_selected_gap_mean"] = float(neg_gap_sel.mean().detach().cpu())
                stats["cond_rank_neg_selected_gap_max"] = float(neg_gap_sel.max().detach().cpu())
                stats["cond_rank_neg_selected_gap_min"] = float(neg_gap_sel.min().detach().cpu())

        # -------------------------------------------------
        # 4) positive selection on all positive centers
        # -------------------------------------------------
        pos_idx_all = torch.nonzero(slide_label_batch == 1, as_tuple=False).squeeze(-1)
        pos_loss = zero

        if pos_idx_all.numel() > 0:
            pos_repr = center_repr[pos_idx_all]
            pos_dispatch = dispatch_weight_img[pos_idx_all]
            pos_context = pos_context_dict

            pos_slide_ids = None
            if slide_id_batch is not None:
                if torch.is_tensor(slide_id_batch):
                    slide_id_batch_list = [str(x.item()) for x in slide_id_batch.view(-1)]
                else:
                    slide_id_batch_list = [str(x) for x in slide_id_batch]
                pos_slide_ids = [slide_id_batch_list[i] for i in pos_idx_all.tolist()]

            pos_sel_local, pos_aux = self.get_positive_rank_selected_token_indices(
                patch_repr=pos_repr,
                dispatch_weight_wsi=pos_dispatch,
                context_dict=pos_context,
                slide_id_batch=pos_slide_ids,
            )

            stats.update(pos_aux)
            stats["cond_rank_pos_num_tokens"] = float(pos_repr.shape[0])
            stats["cond_rank_pos_mean_tumor"] = float(sim_tumor[pos_idx_all].mean().detach().cpu())
            stats["cond_rank_pos_mean_other_max"] = float(sim_other_max[pos_idx_all].mean().detach().cpu())
            stats["cond_rank_pos_mean_gap"] = float(gap[pos_idx_all].mean().detach().cpu())

            if pos_sel_local.numel() > 0:
                pos_gap_sel = self.compute_role_scores_for_patch_repr(pos_repr)[2][pos_sel_local]
                pos_loss = F.relu(self.cond_rank_pos_margin - pos_gap_sel).mean()

                stats["cond_rank_pos_raw"] = float(pos_loss.detach().cpu())
                stats["cond_rank_pos_num_selected"] = float(pos_sel_local.numel())
                stats["cond_rank_pos_selected_gap_mean"] = float(pos_gap_sel.mean().detach().cpu())
                stats["cond_rank_pos_selected_gap_max"] = float(pos_gap_sel.max().detach().cpu())
                stats["cond_rank_pos_selected_gap_min"] = float(pos_gap_sel.min().detach().cpu())

        # -------------------------------------------------
        # 5) merge
        # -------------------------------------------------
        total_loss = zero
        total_weight = 0.0

        if neg_idx_all.numel() > 0 and stats["cond_rank_neg_num_selected"] > 0:
            total_loss = total_loss + self.cond_rank_neg_weight * neg_loss
            total_weight += self.cond_rank_neg_weight

        if pos_idx_all.numel() > 0 and stats["cond_rank_pos_num_selected"] > 0:
            total_loss = total_loss + self.cond_rank_pos_weight * pos_loss
            total_weight += self.cond_rank_pos_weight

        if total_weight > 0:
            total_loss = total_loss / total_weight

        stats["cond_rank_raw"] = float(total_loss.detach().cpu())
        stats["cond_rank_num_selected"] = (
            stats["cond_rank_neg_num_selected"] + stats["cond_rank_pos_num_selected"]
        )

        return total_loss, stats
    def compute_batch_residual_hn_push_loss(
        self,
        center_images,
        slide_label_batch,
        neighbor_images_list=None,
        is_eval=False,
    ):
        """
        center-level residual HN push:
        - 只对 negative center 生效
        - 把仍然保持正 tumor gap 的危险 center 再往下压
        """
        zero = next(self.parameters()).new_tensor(0.0)

        stats = {
            "residual_hn_push_raw": 0.0,
            "residual_hn_push_weighted": 0.0,
            "residual_hn_num_before_expert_filter": 0.0,
            "residual_hn_num_after_expert_filter": 0.0,
            "residual_hn_gap_mean": 0.0,
            "residual_hn_gap_max": 0.0,
            "residual_hn_gap_min": 0.0,
        }

        if (not self.use_residual_hn_push) or (self.residual_hn_push_weight <= 0):
            return zero, stats

        if center_images is None or slide_label_batch is None:
            return zero, stats

        B = center_images.shape[0]
        if B == 0:
            return zero, stats

        device = center_images.device
        if neighbor_images_list is None:
            neighbor_images_list = [None] * B

        losses = []
        gap_after = []
        num_before = []
        num_after = []

        for i in range(B):
            y = int(slide_label_batch[i].item())
            if y != 0:
                continue

            center_img_i = center_images[i:i+1]
            nb_imgs_i = neighbor_images_list[i]

            center_repr_i, _, dispatch_weight_img_i = \
                self.encode_patch_batch_for_wsi_bag_with_dispatch(center_img_i, is_eval=is_eval)

            if (
                self.use_context_guided_selection
                and nb_imgs_i is not None
                and torch.is_tensor(nb_imgs_i)
                and nb_imgs_i.ndim == 4
                and nb_imgs_i.shape[0] > 0
            ):
                nb_imgs_i = nb_imgs_i.to(device, non_blocking=True)
                nb_repr_i, _, _ = self.encode_patch_batch_for_wsi_bag_with_dispatch(nb_imgs_i, is_eval=is_eval)
                ctx_i = self.build_context_guided_score_from_center_and_neighbors(
                    center_patch_repr=center_repr_i[0],
                    neighbor_patch_repr=nb_repr_i,
                    mode="negative",
                )
                context_dict_i = {
                    "ctx_score": ctx_i["ctx_score"].view(1),
                    "center_gap": ctx_i["center_gap"].view(1),
                    "nb_gap_mean": ctx_i["nb_gap_mean"].view(1),
                    "nb_gap_max": ctx_i["nb_gap_max"].view(1),
                    "nb_gap_topk_mean": ctx_i["nb_gap_topk_mean"].view(1),
                    "isolation_score": ctx_i["isolation_score"].view(1),
                    "consistency_score": ctx_i["consistency_score"].view(1),
                }
            else:
                context_dict_i = None

            loss_i, stats_i = self.compute_residual_hn_push_loss(
                patch_repr=center_repr_i,
                dispatch_weight_wsi=dispatch_weight_img_i,
                slide_label=0,
            )

            num_before.append(stats_i["residual_hn_num_before_expert_filter"])
            num_after.append(stats_i["residual_hn_num_after_expert_filter"])

            if stats_i["residual_hn_num_after_expert_filter"] > 0:
                losses.append(loss_i)
                gap_after.append(stats_i["residual_hn_gap_mean"])

        loss = torch.stack(losses).mean() if len(losses) > 0 else zero

        stats["residual_hn_push_raw"] = float(loss.detach().cpu())
        if len(num_before) > 0:
            stats["residual_hn_num_before_expert_filter"] = float(np.sum(num_before))
            stats["residual_hn_num_after_expert_filter"] = float(np.sum(num_after))
        if len(gap_after) > 0:
            stats["residual_hn_gap_mean"] = float(np.mean(gap_after))
            stats["residual_hn_gap_max"] = float(np.max(gap_after))
            stats["residual_hn_gap_min"] = float(np.min(gap_after))

        return loss, stats


    def compute_batch_positive_anchor_boost_loss(
        self,
        center_images,
        slide_label_batch,
        neighbor_images_list=None,
        is_eval=False,
    ):
        """
        center-level positive anchor boost:
        - 只对 positive center 生效
        - 用 center 自己或 center+neighbor 的 context score 选 anchor
        - 提升这些 high-value positive center 的 tumor gap
        """
        zero = next(self.parameters()).new_tensor(0.0)

        stats = {
            "positive_anchor_boost_raw": 0.0,
            "positive_anchor_boost_weighted": 0.0,
            "pos_anchor_num_candidates": 0.0,
            "pos_anchor_num_selected": 0.0,
            "pos_anchor_gap_mean": 0.0,
            "pos_anchor_gap_max": 0.0,
            "pos_anchor_gap_min": 0.0,
            "pos_anchor_tumor_mean": 0.0,
        }

        if (not self.use_positive_anchor_boost) or (self.positive_anchor_weight <= 0):
            return zero, stats

        if center_images is None or slide_label_batch is None:
            return zero, stats

        B = center_images.shape[0]
        if B == 0:
            return zero, stats

        device = center_images.device
        if neighbor_images_list is None:
            neighbor_images_list = [None] * B

        losses = []
        cand_nums = []
        sel_nums = []
        gap_means = []
        gap_maxs = []
        gap_mins = []
        tumor_means = []

        for i in range(B):
            y = int(slide_label_batch[i].item())
            if y != 1:
                continue

            center_img_i = center_images[i:i+1]
            nb_imgs_i = neighbor_images_list[i]

            center_repr_i, _, _ = self.encode_patch_batch_for_wsi_bag_with_dispatch(
                center_img_i,
                is_eval=is_eval,
            )

            if (
                self.use_context_guided_selection
                and nb_imgs_i is not None
                and torch.is_tensor(nb_imgs_i)
                and nb_imgs_i.ndim == 4
                and nb_imgs_i.shape[0] > 0
            ):
                nb_imgs_i = nb_imgs_i.to(device, non_blocking=True)
                nb_repr_i, _, _ = self.encode_patch_batch_for_wsi_bag_with_dispatch(
                    nb_imgs_i,
                    is_eval=is_eval,
                )
                ctx_i = self.build_context_guided_score_from_center_and_neighbors(
                    center_patch_repr=center_repr_i[0],
                    neighbor_patch_repr=nb_repr_i,
                    mode="positive",
                )
                context_dict_i = {
                    "ctx_score": ctx_i["ctx_score"].view(1),
                    "center_gap": ctx_i["center_gap"].view(1),
                    "nb_gap_mean": ctx_i["nb_gap_mean"].view(1),
                    "nb_gap_max": ctx_i["nb_gap_max"].view(1),
                    "nb_gap_topk_mean": ctx_i["nb_gap_topk_mean"].view(1),
                    "isolation_score": ctx_i["isolation_score"].view(1),
                    "consistency_score": ctx_i["consistency_score"].view(1),
                }
            else:
                context_dict_i = None

            loss_i, stats_i = self.compute_positive_anchor_boost_loss(
                patch_repr=center_repr_i,
                slide_label=1,
                context_dict=context_dict_i,
            )

            cand_nums.append(stats_i["pos_anchor_num_candidates"])
            sel_nums.append(stats_i["pos_anchor_num_selected"])

            if stats_i["pos_anchor_num_selected"] > 0:
                losses.append(loss_i)
                gap_means.append(stats_i["pos_anchor_gap_mean"])
                gap_maxs.append(stats_i["pos_anchor_gap_max"])
                gap_mins.append(stats_i["pos_anchor_gap_min"])
                tumor_means.append(stats_i["pos_anchor_tumor_mean"])

        loss = torch.stack(losses).mean() if len(losses) > 0 else zero

        stats["positive_anchor_boost_raw"] = float(loss.detach().cpu())
        if len(cand_nums) > 0:
            stats["pos_anchor_num_candidates"] = float(np.sum(cand_nums))
            stats["pos_anchor_num_selected"] = float(np.sum(sel_nums))
        if len(gap_means) > 0:
            stats["pos_anchor_gap_mean"] = float(np.mean(gap_means))
            stats["pos_anchor_gap_max"] = float(np.max(gap_maxs))
            stats["pos_anchor_gap_min"] = float(np.min(gap_mins))
            stats["pos_anchor_tumor_mean"] = float(np.mean(tumor_means))

        return loss, stats



    def topk_pool_wsi_tumor_evidence(self, patch_repr, tumor_score):
        """
        patch_repr:  [N, D_teacher]
        tumor_score: [N]
        return:
            bag_repr: [1, D_teacher]
            aux_stats: dict
        """
        N = patch_repr.shape[0]
        k = max(self.wsi_topk_min, int(round(N * self.wsi_topk_ratio)))
        k = min(k, self.wsi_topk_max, N)

        topk_vals, topk_idx = torch.topk(tumor_score, k=k, largest=True)
        topk_repr = patch_repr[topk_idx]                     # [k, D]
        bag_repr = topk_repr.mean(dim=0, keepdim=True)      # [1, D]
        bag_repr = F.normalize(bag_repr, dim=-1)

        aux_stats = {
            "wsi_num_patches": float(N),
            "wsi_topk_k": float(k),
            "wsi_topk_score_mean": float(topk_vals.mean().detach().cpu()),
            "wsi_topk_score_max": float(topk_vals.max().detach().cpu()),
            "wsi_topk_score_min": float(topk_vals.min().detach().cpu()),
        }
        return bag_repr, aux_stats
    
    def compute_wsi_topk_mean_score(self, tumor_score):
        """
        tumor_score: [N]
        return:
            mean_score: scalar
            aux_stats: dict
        """
        N = tumor_score.shape[0]
        k = max(self.wsi_topk_min, int(round(N * self.wsi_topk_ratio)))
        k = min(k, self.wsi_topk_max, N)

        topk_vals, topk_idx = torch.topk(tumor_score, k=k, largest=True)
        mean_score = topk_vals.mean()

        aux_stats = {
            "wsi_num_patches": float(N),
            "wsi_topk_k": float(k),
            "wsi_topk_score_mean": float(topk_vals.mean().detach().cpu()),
            "wsi_topk_score_max": float(topk_vals.max().detach().cpu()),
            "wsi_topk_score_min": float(topk_vals.min().detach().cpu()),
        }
        return mean_score, aux_stats

    def _get_wsi_label_value(self, slide_label):
        """
        slide_label: tensor([0.]) / tensor([1.]) / scalar
        return: int in {0,1}
        """
        if torch.is_tensor(slide_label):
            val = int(slide_label.view(-1)[0].item())
        else:
            val = int(slide_label)
        return val

    def _get_wsi_asym_weights_and_margins(self, slide_label):
        """
        return:
            bce_weight: float
            margin_weight: float
            margin_target: float
            label_name: str
        """
        y = self._get_wsi_label_value(slide_label)

        if (not self.use_wsi_asymmetric_loss):
            return (
                self.wsi_bag_loss_weight,
                self.wsi_bag_margin_weight,
                self.wsi_bag_margin,
                "symmetric",
            )

        if y == 1:
            return (
                self.wsi_pos_bce_weight,
                self.wsi_pos_margin_weight,
                self.wsi_pos_margin,
                "positive",
            )
        else:
            return (
                self.wsi_neg_bce_weight,
                self.wsi_neg_margin_weight,
                self.wsi_neg_margin,
                "negative",
            )

    def compute_wsi_top_tail_mean_score(self, tumor_score, ratio=0.01, min_k=2):
        N = tumor_score.shape[0]
        k = max(min_k, int(round(N * ratio)))
        k = min(k, N)

        top_vals, _ = torch.topk(tumor_score, k=k, largest=True)
        return top_vals.mean(), {
            "wsi_pos_tail_k": float(k),
            "wsi_pos_tail_mean_score": float(top_vals.mean().detach().cpu()),
            "wsi_pos_tail_max_score": float(top_vals.max().detach().cpu()),
            "wsi_pos_tail_min_score": float(top_vals.min().detach().cpu()),
        }

    def compute_negative_global_topk_suppression_loss(
        self,
        images,
        slide_label,
        patch_batch_size=None,
        is_eval=False,
    ):
        """
        只对 negative slide 生效：
        压制该负片中 tumor evidence 最高的一批 patch 的 top-k mean score

        images:
            [N, 3, H, W] 或 list[tensor]
        slide_label:
            scalar / tensor scalar, must be 0 or 1

        return:
            loss: scalar
            stats: dict
        """
        zero = next(self.parameters()).new_tensor(0.0)

        stats = {
            "neg_global_topk_raw": 0.0,
            "neg_global_topk_mean_score": 0.0,
            "neg_global_topk_margin": float(self.neg_global_topk_margin),
            "neg_global_topk_active": 0.0,
            "neg_global_num_patches": 0.0,
            "neg_global_topk_k": 0.0,
            "neg_global_topk_score_max": 0.0,
            "neg_global_topk_score_min": 0.0,
        }

        if (not self.use_neg_global_topk_suppression) or (self.neg_global_topk_weight <= 0):
            return zero, stats

        if images is None or slide_label is None:
            return zero, stats

        y = self._get_wsi_label_value(slide_label)
        if y != 0:
            return zero, stats

        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        device = next(self.parameters()).device
        images = images.to(device, non_blocking=True)

        if patch_batch_size is None:
            patch_batch_size = self.wsi_patch_batch_size

        all_score = []

        for start in range(0, images.shape[0], patch_batch_size):
            end = min(start + patch_batch_size, images.shape[0])
            patch_batch = images[start:end]

            _, tumor_score = self.encode_patch_batch_for_wsi_bag(
                patch_batch,
                is_eval=is_eval,
            )
            all_score.append(tumor_score)

        tumor_score = torch.cat(all_score, dim=0)   # [N]
        mean_score, aux_stats = self.compute_wsi_topk_mean_score(tumor_score)

        # negative slide: 希望 top-k mean score <= margin
        loss = F.relu(mean_score - self.neg_global_topk_margin)

        stats["neg_global_topk_raw"] = float(loss.detach().cpu())
        stats["neg_global_topk_mean_score"] = float(mean_score.detach().cpu())
        stats["neg_global_topk_active"] = 1.0
        stats["neg_global_num_patches"] = aux_stats["wsi_num_patches"]
        stats["neg_global_topk_k"] = aux_stats["wsi_topk_k"]
        stats["neg_global_topk_score_max"] = aux_stats["wsi_topk_score_max"]
        stats["neg_global_topk_score_min"] = aux_stats["wsi_topk_score_min"]

        return loss, stats
    
    def compute_wsi_bag_loss(self, images, slide_label, patch_batch_size=None, is_eval=False):
        """
        images:
            [N, 3, H, W] 或 list[tensor]
        slide_label:
            0 / 1
        return:
            total_bag_loss: scalar
            stats: dict
        """
        zero = next(self.parameters()).new_tensor(0.0)

        if (not self.use_wsi_bag_loss) and (not self.use_wsi_bag_margin_loss):
            return zero, {
                "wsi_bag_loss_raw": 0.0,
                "wsi_bag_margin_raw": 0.0,
                "wsi_num_patches": 0.0,
                "wsi_topk_k": 0.0,
                "wsi_topk_score_mean": 0.0,
                "wsi_topk_score_max": 0.0,
                "wsi_topk_score_min": 0.0,
                "wsi_prob": 0.0,
                "wsi_topk_mean_score": 0.0,
            }

        if images is None:
            return zero, {
                "wsi_bag_loss_raw": 0.0,
                "wsi_bag_margin_raw": 0.0,
                "wsi_num_patches": 0.0,
                "wsi_topk_k": 0.0,
                "wsi_topk_score_mean": 0.0,
                "wsi_topk_score_max": 0.0,
                "wsi_topk_score_min": 0.0,
                "wsi_prob": 0.0,
                "wsi_topk_mean_score": 0.0,
            }

        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        device = next(self.parameters()).device
        images = images.to(device, non_blocking=True)

        if patch_batch_size is None:
            patch_batch_size = self.wsi_patch_batch_size

        all_repr = []
        all_score = []

        for start in range(0, images.shape[0], patch_batch_size):
            end = min(start + patch_batch_size, images.shape[0])
            patch_batch = images[start:end]

            patch_repr, tumor_score = self.encode_patch_batch_for_wsi_bag(
                patch_batch,
                is_eval=is_eval,
            )
            all_repr.append(patch_repr)
            all_score.append(tumor_score)

        patch_repr = torch.cat(all_repr, dim=0)       # [N, D]
        tumor_score = torch.cat(all_score, dim=0)     # [N]

        if not torch.is_tensor(slide_label):
            slide_label = torch.tensor([slide_label], device=device, dtype=torch.float32)
        else:
            slide_label = slide_label.to(device).float().view(1)

        total_bag_loss = zero
        stats = {}

        bce_weight_cur, margin_weight_cur, margin_target_cur, label_mode = \
            self._get_wsi_asym_weights_and_margins(slide_label)

        # -------------------------------------------------
        # 1) BCE bag classifier loss
        # -------------------------------------------------
        if self.use_wsi_bag_loss:
            assert self.wsi_bag_classifier is not None, "wsi_bag_classifier is None but use_wsi_bag_loss=True"

            bag_repr, aux_stats = self.topk_pool_wsi_tumor_evidence(
                patch_repr=patch_repr,
                tumor_score=tumor_score,
            )
            # 注意：这里不能 detach
            logit = self.wsi_bag_classifier(bag_repr).view(1)
            loss_bce = F.binary_cross_entropy_with_logits(logit, slide_label)
            prob = torch.sigmoid(logit).item()

            total_bag_loss = total_bag_loss + bce_weight_cur * loss_bce

            stats.update(aux_stats)
            stats["wsi_bag_loss_raw"] = float(loss_bce.detach().cpu())
            stats["wsi_prob"] = float(prob)
            stats["wsi_bce_weight_cur"] = float(bce_weight_cur)
        else:
            stats["wsi_bag_loss_raw"] = 0.0
            stats["wsi_prob"] = 0.0
            stats["wsi_bce_weight_cur"] = 0.0

        # -------------------------------------------------
        # 2) top-k tumor evidence margin loss
        # positive slide: want score >= margin
        # negative slide: want score <= -margin
        # -------------------------------------------------
        if self.use_wsi_bag_margin_loss:
            mean_score, aux_stats2 = self.compute_wsi_topk_mean_score(tumor_score)

            # y in {0,1} -> target_sign in {-1,+1}
            target_sign = slide_label * 2.0 - 1.0
            # positive: want mean_score >= +margin_target_cur
            # negative: want mean_score <= -margin_target_cur
            loss_margin = F.relu(margin_target_cur - target_sign * mean_score).mean()

            total_bag_loss = total_bag_loss + margin_weight_cur * loss_margin

            # 若前面 BCE 分支没写 aux_stats，这里补上
            for k, v in aux_stats2.items():
                if k not in stats:
                    stats[k] = v

            stats["wsi_bag_margin_raw"] = float(loss_margin.detach().cpu())
            stats["wsi_topk_mean_score"] = float(mean_score.detach().cpu())
            stats["wsi_margin_weight_cur"] = float(margin_weight_cur)
           
        else:
            stats["wsi_bag_margin_raw"] = 0.0
            stats["wsi_topk_mean_score"] = 0.0
            stats["wsi_margin_weight_cur"] = 0.0
        
        if self.use_wsi_pos_tail_protect and self._get_wsi_label_value(slide_label) == 1:
            pos_tail_mean, pos_tail_stats = self.compute_wsi_top_tail_mean_score(
                tumor_score,
                ratio=self.wsi_pos_tail_ratio,
                min_k=self.wsi_pos_tail_min,
            )

            loss_pos_tail = F.relu(self.wsi_pos_tail_floor - pos_tail_mean)
            total_bag_loss = total_bag_loss + self.wsi_pos_tail_protect_weight * loss_pos_tail

            stats["wsi_pos_tail_protect_raw"] = float(loss_pos_tail.detach().cpu())
            stats.update(pos_tail_stats)
        else:
            stats["wsi_pos_tail_protect_raw"] = 0.0

        stats["wsi_label"] = float(self._get_wsi_label_value(slide_label))
        stats["wsi_bag_total"] = float(total_bag_loss.detach().cpu())
        return total_bag_loss, stats

    def compute_role_proto_loss(self, spec_patch, dispatch_weight):
        """
        spec_patch: [B, N, D_student]
        dispatch_weight: [B, N, E]

        设计说明：
        1) role proto 主损失：
        - 继续使用 high-confidence + balanced semantic sampling
        - 支持 asymmetric role weighting
        2) tumor preference 辅助损失：
        - 单独走一条更宽松的 candidate pool
        - 不复用 role proto 主损失那批 top-k token
        - 只抓“当前更像 ambiguous，但已接近 tumor”的 token
        """
        if (not self.enable_role_proto) or self.role_criterion is None:
            zero = spec_patch.new_tensor(0.0)
            return zero, {
                "role_proto": 0.0,
                "role_attract": 0.0,
                "role_separate": 0.0,
                "role_target": 0.0,
                "role_mean_self_sim": 0.0,
                "role_mean_other_max_sim": 0.0,
                "role_num_valid": 0.0,
                "role_valid_ratio": 0.0,
                "role_main_conf_mean": 0.0,
                "tumor_pref_raw": 0.0,
                "tumor_candidate_ratio": 0.0,
            }

        B, N, D = spec_patch.shape

        active_expert_ids = dispatch_weight.argmax(dim=-1)      # [B, N]
        main_conf = dispatch_weight.max(dim=-1).values          # [B, N]

        spec_flat = spec_patch.reshape(B * N, D)
        active_expert_ids = active_expert_ids.reshape(B * N)
        main_conf = main_conf.reshape(B * N)

        # =========================================================
        # A) role proto 主损失用 token 池（严格）
        # =========================================================
        proto_valid_mask = build_semantic_expert_mask(
            active_expert_ids,
            free_expert_id=self.free_expert_id,
        )
        candidate_idx = torch.nonzero(proto_valid_mask, as_tuple=False).squeeze(-1)
        role_main_conf = torch.zeros_like(main_conf)
        role_margin_conf = torch.zeros_like(main_conf)

        if candidate_idx.numel() > 0:
            cand_spec = spec_flat[candidate_idx]                 # [Nc, D_student]
            cand_feat = self.proj_l12(cand_spec)                 # [Nc, D_teacher]
            cand_feat = F.normalize(cand_feat, dim=-1)

            _, _, cand_role_conf, cand_role_margin = self.compute_role_conf_and_margin(cand_feat)

            role_main_conf[candidate_idx] = cand_role_conf
            role_margin_conf[candidate_idx] = cand_role_margin

        if self.role_use_conf_mask:
            proto_valid_mask = proto_valid_mask & (role_main_conf >= self.role_conf_thresh)

        if self.role_use_margin_mask:
            proto_valid_mask = proto_valid_mask & (role_margin_conf >= self.role_margin_conf_thresh)

        role_valid_ratio = proto_valid_mask.float().mean()
        num_valid = int(proto_valid_mask.sum().item())

        if num_valid == 0:
            zero = spec_patch.new_tensor(0.0)
            return zero, {
                "role_proto": 0.0,
                "role_attract": 0.0,
                "role_separate": 0.0,
                "role_target": 0.0,
                "role_mean_self_sim": 0.0,
                "role_mean_other_max_sim": 0.0,
                "role_num_valid": 0.0,
                "role_valid_ratio": float(role_valid_ratio.detach().cpu()),
                "role_main_conf_mean": 0.0,
                "tumor_pref_raw": 0.0,
                "tumor_candidate_ratio": 0.0,
            }

        proto_idx = torch.nonzero(proto_valid_mask, as_tuple=False).squeeze(-1)

        # ---------- balanced semantic sampling ----------
        if self.role_balance_semantic_sampling and self.max_role_tokens_per_expert > 0:
            selected_idx_list = []
            for expert_id in [0, 1, 2]:  # semantic experts only
                expert_mask = (active_expert_ids == expert_id) & proto_valid_mask
                expert_idx = torch.nonzero(expert_mask, as_tuple=False).squeeze(-1)
                if expert_idx.numel() == 0:
                    continue

                conf_e = role_main_conf[expert_idx]
                k = min(self.max_role_tokens_per_expert, expert_idx.numel())
                topk_local = torch.topk(conf_e, k=k, largest=True).indices
                selected_idx_list.append(expert_idx[topk_local])

            if len(selected_idx_list) > 0:
                proto_idx = torch.cat(selected_idx_list, dim=0)

        elif self.max_role_tokens_per_batch > 0 and proto_idx.numel() > self.max_role_tokens_per_batch:
            conf_valid = role_main_conf[proto_idx]
            topk_idx = torch.topk(
                conf_valid,
                k=self.max_role_tokens_per_batch,
                largest=True,
            ).indices
            proto_idx = proto_idx[topk_idx]

        # select first, then project
        spec_sel = spec_flat[proto_idx]                 # [Nv, D_student]
        features = self.proj_l12(spec_sel)              # [Nv, D_teacher]
        features = F.normalize(features, dim=-1)

        active_expert_sel = active_expert_ids[proto_idx]
        main_conf_sel = main_conf[proto_idx]

        role_main_conf_sel = role_main_conf[proto_idx]
        role_margin_conf_sel = role_margin_conf[proto_idx]

        role_indices = build_hard_role_indices_from_expert_ids(active_expert_sel)
        safe_role_indices = role_indices.clone()
        safe_role_indices[safe_role_indices < 0] = 0

        total_num = float(features.shape[0])
        total_loss_parts = []

        agg_stats = {
            "role_proto": 0.0,
            "role_attract": 0.0,
            "role_separate": 0.0,
            "role_target": 0.0,
            "role_mean_self_sim": 0.0,
            "role_mean_other_max_sim": 0.0,
            "role_num_valid": float(features.shape[0]),
        }

        def _run_group(mask, target_scale, attraction_scale, separation_scale):
            if mask.sum() == 0:
                return None, None

            out = self.role_criterion(
                features=features[mask],
                role_indices=safe_role_indices[mask],
                valid_mask=torch.ones(
                    int(mask.sum().item()),
                    dtype=torch.bool,
                    device=features.device,
                ),
            )

            group_total = self._compose_role_loss_with_scales(
                out,
                target_scale=target_scale,
                attraction_scale=attraction_scale,
                separation_scale=separation_scale,
            )
            return out, group_total

        # -------------------------
        # role-specific asymmetric weighting
        # -------------------------
        if self.use_role_asymmetric_weighting:
            tumor_mask = (
                safe_role_indices == self.tumor_role_id
                if self.tumor_role_id is not None
                else torch.zeros_like(safe_role_indices, dtype=torch.bool)
            )
            stroma_mask = (
                safe_role_indices == self.stroma_role_id
                if self.stroma_role_id is not None
                else torch.zeros_like(safe_role_indices, dtype=torch.bool)
            )
            amb_mask = (
                safe_role_indices == self.ambiguous_role_id
                if self.ambiguous_role_id is not None
                else torch.zeros_like(safe_role_indices, dtype=torch.bool)
            )

            covered_mask = tumor_mask | stroma_mask | amb_mask
            other_mask = ~covered_mask

            group_cfgs = [
                (
                    tumor_mask,
                    self.tumor_target_scale,
                    self.tumor_attraction_scale,
                    self.tumor_separation_scale,
                ),
                (
                    stroma_mask,
                    self.stroma_target_scale,
                    self.stroma_attraction_scale,
                    self.stroma_separation_scale,
                ),
                (
                    amb_mask,
                    self.ambiguous_target_scale,
                    self.ambiguous_attraction_scale,
                    self.ambiguous_separation_scale,
                ),
                (
                    other_mask,
                    1.0,
                    1.0,
                    self.role_separation_weight,
                ),
            ]

            for mask, ts, ats, ss in group_cfgs:
                if mask.sum() == 0:
                    continue

                out, group_total = _run_group(mask, ts, ats, ss)
                w = float(mask.sum().item()) / total_num
                total_loss_parts.append(w * group_total)

                agg_stats["role_proto"] += w * float(group_total.detach().cpu())
                agg_stats["role_attract"] += w * float(out.attraction_loss.detach().cpu())
                agg_stats["role_separate"] += w * float(out.separation_loss.detach().cpu())
                agg_stats["role_target"] += w * float(out.role_target_loss.detach().cpu())
                agg_stats["role_mean_self_sim"] += w * float(out.stats["mean_self_sim"])
                agg_stats["role_mean_other_max_sim"] += w * float(out.stats["mean_other_max_sim"])

            role_proto_total = (
                torch.stack(total_loss_parts).sum()
                if len(total_loss_parts) > 0
                else spec_patch.new_tensor(0.0)
            )
        else:
            out = self.role_criterion(
                features=features,
                role_indices=safe_role_indices,
                valid_mask=torch.ones(
                    features.shape[0],
                    dtype=torch.bool,
                    device=features.device,
                ),
            )
            role_proto_total = out.total_loss
            agg_stats["role_proto"] = float(out.total_loss.detach().cpu())
            agg_stats["role_attract"] = float(out.attraction_loss.detach().cpu())
            agg_stats["role_separate"] = float(out.separation_loss.detach().cpu())
            agg_stats["role_target"] = float(out.role_target_loss.detach().cpu())
            agg_stats["role_mean_self_sim"] = float(out.stats["mean_self_sim"])
            agg_stats["role_mean_other_max_sim"] = float(out.stats["mean_other_max_sim"])

        # =========================================================
        # B) tumor-over-ambiguous preference（单独 candidate pool，更宽松）
        # =========================================================
        tumor_pref_loss = spec_patch.new_tensor(0.0)
        tumor_candidate_ratio = 0.0

        if (
            self.use_tumor_preference_loss
            and self.tumor_role_id is not None
            and self.ambiguous_role_id is not None
        ):
            # --- 单独 candidate pool，不复用 proto_idx ---
            pref_valid_mask = build_semantic_expert_mask(
                active_expert_ids,
                free_expert_id=self.free_expert_id,
            )

            # 更宽松的 confidence 条件
            # 若 role_conf_thresh=0.45，这里自动放宽到 0.25；也可以以后单独加配置
            pref_conf_thresh = min(self.role_conf_thresh, 0.25)
            pref_valid_mask = pref_valid_mask & (main_conf >= pref_conf_thresh)

            # 更聚焦：优先从 ambiguous expert token 里找
            # 如果你后面发现当前 ambiguous 对应的并不是 expert 2，再单独改这里
            pref_valid_mask = pref_valid_mask & (active_expert_ids == 2)

            pref_idx = torch.nonzero(pref_valid_mask, as_tuple=False).squeeze(-1)

            if pref_idx.numel() > 0:
                pref_spec = spec_flat[pref_idx]                 # [Np, D_student]
                pref_features = self.proj_l12(pref_spec)        # [Np, D_teacher]
                pref_features = F.normalize(pref_features, dim=-1)

                pref_role_logits = self.compute_role_affinity_logits(pref_features)  # [Np, R]
                sim_tumor = pref_role_logits[:, self.tumor_role_id]
                sim_amb = pref_role_logits[:, self.ambiguous_role_id]
                pred_role = torch.argmax(pref_role_logits, dim=-1)

                # 只抓：当前更像 ambiguous，但已开始接近 tumor 的 token
                tumor_candidate_mask = (
                    (pred_role == self.ambiguous_role_id)
                    & (sim_tumor >= self.tumor_candidate_min_sim)
                    & ((sim_amb - sim_tumor) <= self.tumor_amb_near_margin)
                )

                if tumor_candidate_mask.sum() > 0:
                    tumor_pref_loss = F.relu(
                        sim_amb[tumor_candidate_mask]
                        - sim_tumor[tumor_candidate_mask]
                        + self.tumor_pref_margin
                    ).mean()

                    tumor_candidate_ratio = float(
                        tumor_candidate_mask.float().mean().detach().cpu()
                    )

        total_loss = role_proto_total + self.tumor_preference_weight * tumor_pref_loss

        stats = {
            "role_proto": agg_stats["role_proto"],
            "role_attract": agg_stats["role_attract"],
            "role_separate": agg_stats["role_separate"],
            "role_target": agg_stats["role_target"],
            "role_mean_self_sim": agg_stats["role_mean_self_sim"],
            "role_mean_other_max_sim": agg_stats["role_mean_other_max_sim"],
            "role_num_valid": agg_stats["role_num_valid"],
            "role_valid_ratio": float(role_valid_ratio.detach().cpu()),
            "role_main_conf_mean": float(role_main_conf_sel.mean().detach().cpu()),
            "role_margin_conf_mean": float(role_margin_conf_sel.mean().detach().cpu()),
            "tumor_pref_raw": float(tumor_pref_loss.detach().cpu()),
            "tumor_candidate_ratio": tumor_candidate_ratio,
        }
        return total_loss, stats

    # =========================================================
    # loss: routing stability regularization
    # =========================================================
    def compute_routing_stability_loss(self, gate_info_list):
        """
        这里只做一个很轻的稳定项：
        - score 不要过大爆炸
        - 只是 regularizer，不是主任务
        """
        losses = []
        for gate_info in gate_info_list:
            score = gate_info["score"]
            losses.append((score ** 2).mean())

        if len(losses) == 0:
            return torch.tensor(0.0, device=next(self.student.parameters()).device)
        return torch.stack(losses).mean()

    def compute_z_loss(self):
        z_losses = []
        for module in self.student.modules():
            if hasattr(module, "gate") and hasattr(module.gate, "last_sim"):
                if module.gate.last_sim is not None:
                    z_losses.append((module.gate.last_sim ** 2).mean())

        if len(z_losses) == 0:
            return torch.tensor(0.0, device=next(self.student.parameters()).device)
        return torch.stack(z_losses).mean()

    # =========================================================
    # forward
    # =========================================================
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
        self.tea_features.clear()

        B = images.shape[0]
        mask = self.get_random_mask(B, images.device) if self.use_stage2_mask else None

        with torch.no_grad():
            _ = self.teacher(images)
            for k, v in self.tea_features.items():
                self.tea_features[k] = v.detach()

        student_out, gate_info_list, feature_dict, moe_feature_list = self.student(
            images,
            return_gates=True,
            mask=mask,
            is_eval=is_eval,
            return_features=True,
            offline_cluster_ids=offline_cluster_ids,
        )

        # ---- alignment feature ----
        s_feat_12 = feature_dict["layer_12"]             # [B, N+1, 384]
        t_feat_32 = self.tea_features["layer_32"]        # [B, ?, 1280]
        s_proj_12 = self.proj_l12(s_feat_12)

        # ---- specialization feature ----
        # 推荐直接用最后一个 MoE block 输出，而不是 final norm 输出
        if self.use_last_moe_output and len(moe_feature_list) > 0:
            spec_feat = moe_feature_list[-1]             # [B, N+1, D]
        else:
            spec_feat = feature_dict["layer_12"]

        spec_patch = spec_feat[:, 1:, :]                 # [B, N, D]
        B2, N, D = spec_patch.shape

        dispatch_weight = self.get_last_dispatch_weight(gate_info_list, B2, N)
        dispatch_mask = self.get_last_dispatch_mask(gate_info_list, B2, N)

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

        return total_loss, loss_dict, gate_info_list

    # =========================================================
    # total stage2 loss
    # =========================================================
    def compute_stage2_loss(
        self,
        s_proj_12,
        t_feat_32,
        spec_patch,
        dispatch_weight,
        dispatch_mask,
        gate_info_list,
        offline_cluster_ids=None,
        wsi_images=None,
        wsi_slide_label=None,
        slide_label_batch=None,
        is_eval=False,
        center_images=None,
        neighbor_images_list=None,
        slide_id_batch=None,
    ):
        # 1) alignment guardrail
        loss_cls_align, loss_patch_align = self.compute_alignment_loss(s_proj_12, t_feat_32)

        # 2) main: output prototype separation
        loss_proto_sep, proto_stats = self.compute_proto_sep_loss(spec_patch, dispatch_weight)

        # 2.5) expert response specialization (SP loss)
        if self.sp_loss_weight > 0:
            B, N, _ = spec_patch.shape

            sp_losses = []
            sp_stats_list = []

            for layer_idx in self.sp_layers:
                real_idx = layer_idx if layer_idx >= 0 else len(gate_info_list) + layer_idx
                if real_idx < 0 or real_idx >= len(gate_info_list):
                    continue

                if "expert_outputs" not in gate_info_list[real_idx]:
                    raise KeyError(
                        f"gate_info_list[{real_idx}] does not contain 'expert_outputs'. "
                        f"Please cache expert_outputs in MoE block forward."
                    )

                expert_outputs_l = self.get_layer_expert_outputs(gate_info_list, real_idx, B, N)  # [B,N,E,D]

                dispatch_mask_l = gate_info_list[real_idx]["dispatch_mask"].float()
                E = dispatch_mask_l.shape[-1]
                dispatch_mask_l = dispatch_mask_l.view(B, N + 1, E)[:, 1:, :]

                dispatch_weight_l = gate_info_list[real_idx]["dispatch_weight"]
                dispatch_weight_l = dispatch_weight_l.view(B, N + 1, E)[:, 1:, :]

                sp_l, sp_stats_l = self.compute_sp_loss(
                    expert_outputs=expert_outputs_l,
                    dispatch_mask=dispatch_mask_l,
                    dispatch_weight=dispatch_weight_l,
                )
                sp_losses.append(sp_l)
                sp_stats_list.append(sp_stats_l)

            if len(sp_losses) > 0:
                loss_sp = torch.stack(sp_losses).mean()
                sp_stats = {
                    "sp_pair_count": max(x["sp_pair_count"] for x in sp_stats_list),
                    "sp_token_count": max(x["sp_token_count"] for x in sp_stats_list),
                    "sp_avg_pair_cos": sum(x["sp_avg_pair_cos"] for x in sp_stats_list) / len(sp_stats_list),
                }
            else:
                loss_sp = torch.tensor(0.0, device=spec_patch.device)
                sp_stats = {
                    "sp_pair_count": 0,
                    "sp_token_count": 0,
                    "sp_avg_pair_cos": 0.0,
                }
        else:
            loss_sp = torch.tensor(0.0, device=spec_patch.device)
            sp_stats = {
                "sp_pair_count": 0,
                "sp_token_count": 0,
                "sp_avg_pair_cos": 0.0,
            }

        # 2.6) expert anti-drop floor loss
        if self.use_expert_floor and self.expert_floor_weight > 0:
            B, N, _ = spec_patch.shape

            floor_losses = []
            floor_stats_list = []

            for layer_idx in self.expert_floor_layers:
                real_idx = layer_idx if layer_idx >= 0 else len(gate_info_list) + layer_idx
                if real_idx < 0 or real_idx >= len(gate_info_list):
                    continue

                dispatch_weight_l = self.get_layer_dispatch_weight(gate_info_list, real_idx, B, N)
                floor_l, floor_stats_l = self.compute_expert_floor_loss(dispatch_weight_l)
                floor_losses.append(floor_l)
                floor_stats_list.append(floor_stats_l)

            if len(floor_losses) > 0:
                loss_floor = torch.stack(floor_losses).mean()

                # 汇总 stats：取所有层里最坏的 min_mass，以及平均 raw loss
                floor_stats = {
                    "expert_floor_loss_raw": sum(x["expert_floor_loss_raw"] for x in floor_stats_list) / len(floor_stats_list),
                    "expert_floor_min_mass": min(x["expert_floor_min_mass"] for x in floor_stats_list),
                    "expert_floor_max_mass": max(x["expert_floor_max_mass"] for x in floor_stats_list),
                }

                # 只记录第一层的各 expert mass，方便你看 layer9 是否把 E0 救回来
                first_stats = floor_stats_list[0]
                for k, v in first_stats.items():
                    if k.startswith("expert_floor_mass_e"):
                        floor_stats[k] = v
            else:
                loss_floor = torch.tensor(0.0, device=spec_patch.device)
                floor_stats = {
                    "expert_floor_loss_raw": 0.0,
                    "expert_floor_min_mass": 0.0,
                    "expert_floor_max_mass": 0.0,
                }
        else:
            loss_floor = torch.tensor(0.0, device=spec_patch.device)
            floor_stats = {
                "expert_floor_loss_raw": 0.0,
                "expert_floor_min_mass": 0.0,
                "expert_floor_max_mass": 0.0,
            }

        # 3) optional: cluster-aware output separation

        if self.cluster_sep_weight > 0:
            loss_cluster_sep, cluster_stats = self.compute_cluster_sep_loss(
                spec_patch, dispatch_weight, offline_cluster_ids
            )
        else:
            loss_cluster_sep = torch.tensor(0.0, device=spec_patch.device)
            cluster_stats = {"cluster_sep_pairs": 0}

        # 4) optional: intra expert compactness
        if self.intra_compact_weight > 0:
            loss_intra = self.compute_intra_compact_loss(spec_patch, dispatch_weight)
        else:
            loss_intra = torch.tensor(0.0, device=spec_patch.device)
        

        # 5) routing stability
        loss_route_reg = self.compute_routing_stability_loss(gate_info_list)

        # 6) z loss
        loss_z = self.compute_z_loss()

        #role proto loss
        loss_role_proto = spec_patch.new_tensor(0.0)
        
        role_stats = {
            "role_proto": 0.0,
            "role_attract": 0.0,
            "role_separate": 0.0,
            "role_target": 0.0,
            "role_mean_self_sim": 0.0,
            "role_mean_other_max_sim": 0.0,
            "role_num_valid": 0.0,
            "role_valid_ratio":0.0,
            "role_main_conf_mean": 0.0,
            "tumor_pref_raw": 0.0,
            "tumor_candidate_ratio": 0.0,
        }
        loss_free_expert_floor = spec_patch.new_tensor(0.0)
        free_expert_stats = {
            "free_expert_floor_raw": 0.0,
            "free_expert_mass": 0.0,
        }
        # 7) optional: WSI-level bag constraint
        loss_wsi_bag = spec_patch.new_tensor(0.0)
        wsi_stats = {
            "wsi_bag_loss_raw": 0.0,
            "wsi_bag_margin_raw": 0.0,
            "wsi_bag_total": 0.0,
            "wsi_num_patches": 0.0,
            "wsi_topk_k": 0.0,
            "wsi_topk_score_mean": 0.0,
            "wsi_topk_score_max": 0.0,
            "wsi_topk_score_min": 0.0,
            "wsi_prob": 0.0,
            "wsi_topk_mean_score": 0.0,
        }

        if (self.use_wsi_bag_loss or self.use_wsi_bag_margin_loss) and (wsi_images is not None) and (wsi_slide_label is not None):
            loss_wsi_bag, wsi_stats = self.compute_wsi_bag_loss(
                images=wsi_images,
                slide_label=wsi_slide_label,
                patch_batch_size=self.wsi_patch_batch_size,
                is_eval=is_eval,
            )
    
        # 7.5) optional: asymmetric conditional pairwise ranking
        loss_cond_rank = spec_patch.new_tensor(0.0)
        cond_rank_stats = {
            # shared
            "cond_rank_raw": 0.0,
            "cond_rank_weighted": 0.0,
            "cond_rank_num_tokens": 0.0,
            "cond_rank_num_selected": 0.0,
            "cond_rank_mean_tumor": 0.0,
            "cond_rank_mean_other_max": 0.0,
            "cond_rank_mean_gap": 0.0,
            "cond_rank_selected_gap_mean": 0.0,
            "cond_rank_selected_gap_max": 0.0,
            "cond_rank_selected_gap_min": 0.0,
            "cond_rank_mode": -1.0,

            # negative-only
            "cond_rank_neg_raw": 0.0,
            "cond_rank_neg_weighted": 0.0,
            "cond_rank_neg_num_tokens": 0.0,
            "cond_rank_neg_num_selected": 0.0,
            "cond_rank_neg_mean_tumor": 0.0,
            "cond_rank_neg_mean_other_max": 0.0,
            "cond_rank_neg_mean_gap": 0.0,
            "cond_rank_neg_selected_gap_mean": 0.0,
            "cond_rank_neg_selected_gap_max": 0.0,
            "cond_rank_neg_selected_gap_min": 0.0,
            "neg_rank_sel_e0": 0.0,
            "neg_rank_sel_e1": 0.0,
            "neg_rank_sel_e2": 0.0,

            # positive-only
            "cond_rank_pos_raw": 0.0,
            "cond_rank_pos_weighted": 0.0,
            "cond_rank_pos_num_tokens": 0.0,
            "cond_rank_pos_num_selected": 0.0,
            "cond_rank_pos_mean_tumor": 0.0,
            "cond_rank_pos_mean_other_max": 0.0,
            "cond_rank_pos_mean_gap": 0.0,
            "cond_rank_pos_selected_gap_mean": 0.0,
            "cond_rank_pos_selected_gap_max": 0.0,
            "cond_rank_pos_selected_gap_min": 0.0,
            "cond_rank_pos_ctx_score_mean": 0.0,

            "cond_rank_pos_num_candidates": 0.0,
            "cond_rank_pos_e0_selected": 0.0,
            "cond_rank_pos_e1_selected": 0.0,
            "cond_rank_pos_e2_selected": 0.0,
            "cond_rank_pos_use_expert_balanced": 0.0,
            "cond_rank_pos_support_num_before": 0.0,
            "cond_rank_pos_support_num_after": 0.0,
            "cond_rank_pos_support_ratio": 0.0,
            "cond_rank_pos_dedup_dropped": 0.0,
            "cond_rank_pos_num_selected_experts": 0.0,
            "cond_rank_pos_failed_min_experts": 0.0,
        }
        if (
            self.use_conditional_pairwise_ranking
            and (center_images is not None)
            and (slide_label_batch is not None)
        ):
            loss_cond_rank, cond_rank_stats = self.compute_batch_conditional_pairwise_ranking_loss(
                center_images=center_images,
                slide_label_batch=slide_label_batch,
                neighbor_images_list=neighbor_images_list if self.use_context_guided_selection else None,
                is_eval=is_eval,
                slide_id_batch=slide_id_batch,
            )
        # 7.6) optional: free-expert preference on negative ranking-selected tokens
        loss_neg_rank_free_pref = spec_patch.new_tensor(0.0)
        neg_rank_free_pref_stats = {
            "neg_rank_free_pref_raw": 0.0,
            "neg_rank_free_pref_weighted": 0.0,
            "neg_rank_free_pref_num_selected": 0.0,
            "neg_rank_free_pref_free_mass_mean": 0.0,
            "neg_rank_free_pref_free_mass_min": 0.0,
            "neg_rank_free_pref_free_mass_max": 0.0,
            "neg_rank_sel_num_before_filter": 0.0,
            "neg_rank_sel_num_after_filter": 0.0,
            "neg_rank_sel_gap_mean": 0.0,
            "neg_rank_sel_gap_max": 0.0,
            "neg_rank_sel_gap_min": 0.0,
        }
        if (
            self.use_neg_rank_free_preference
            and self.neg_rank_free_pref_weight > 0
            and (wsi_images is not None)
            and (wsi_slide_label is not None)
        ):
            loss_neg_rank_free_pref, neg_rank_free_pref_stats = self.compute_wsi_neg_rank_free_preference_loss(
                images=wsi_images,
                slide_label=wsi_slide_label,
                patch_batch_size=self.wsi_patch_batch_size,
                is_eval=is_eval,
            )

        # 7.7) optional: residual HN push
        loss_residual_hn_push = spec_patch.new_tensor(0.0)
        residual_hn_push_stats = {
            "residual_hn_push_raw": 0.0,
            "residual_hn_push_weighted": 0.0,
            "residual_hn_num_before_expert_filter": 0.0,
            "residual_hn_num_after_expert_filter": 0.0,
            "residual_hn_gap_mean": 0.0,
            "residual_hn_gap_max": 0.0,
            "residual_hn_gap_min": 0.0,
        }
        if (
            self.use_residual_hn_push
            and self.residual_hn_push_weight > 0
            and (center_images is not None)
            and (slide_label_batch is not None)
        ):
            loss_residual_hn_push, residual_hn_push_stats = self.compute_batch_residual_hn_push_loss(
                center_images=center_images,
                slide_label_batch=slide_label_batch,
                neighbor_images_list=neighbor_images_list if self.use_context_guided_selection else None,
                is_eval=is_eval,
            )

        # 7.8) optional: positive anchor boost
        loss_positive_anchor_boost = spec_patch.new_tensor(0.0)
        positive_anchor_stats = {
            "positive_anchor_boost_raw": 0.0,
            "positive_anchor_boost_weighted": 0.0,
            "pos_anchor_num_candidates": 0.0,
            "pos_anchor_num_selected": 0.0,
            "pos_anchor_gap_mean": 0.0,
            "pos_anchor_gap_max": 0.0,
            "pos_anchor_gap_min": 0.0,
            "pos_anchor_tumor_mean": 0.0,
        }
        
        if (
            self.use_positive_anchor_boost
            and self.positive_anchor_weight > 0
            and (center_images is not None)
            and (slide_label_batch is not None)
        ):
            loss_positive_anchor_boost, positive_anchor_stats = self.compute_batch_positive_anchor_boost_loss(
                center_images=center_images,
                slide_label_batch=slide_label_batch,
                neighbor_images_list=neighbor_images_list if self.use_context_guided_selection else None,
                is_eval=is_eval,
            )
        # 8) optional: hard-negative repulsion
        loss_hn_repulsion = spec_patch.new_tensor(0.0)
        hn_stats = {
            "hn_repulsion_raw": 0.0,
            "hn_num_classes_used": 0.0,
        }

        if self.use_hn_repulsion_loss and self.hn_repulsion_weight > 0:
            loss_hn_repulsion, hn_stats = self.compute_hn_repulsion_loss()

        # 8.5) optional: online hard-negative repulsion
        loss_online_hn_repulsion = spec_patch.new_tensor(0.0)
        online_hn_stats = {
            "online_hn_repulsion_raw": 0.0,
            "online_hn_num_classes_used": 0.0,
        }

        if self.use_online_hn_repulsion_loss and self.online_hn_repulsion_weight > 0:
            loss_online_hn_repulsion, online_hn_stats = self.compute_online_hn_repulsion_loss(
                spec_patch=spec_patch
            )

        # 8.6) negative-only context inconsistency + positive protection
        loss_neg_ctx_hn = spec_patch.new_tensor(0.0)
        loss_pos_ctx_protect = spec_patch.new_tensor(0.0)
        context_stats = {
            "neg_ctx_loss_raw": 0.0,
            "neg_ctx_num_checked": 0.0,
            "neg_ctx_num_triggered": 0.0,
            "neg_ctx_trigger_ratio": 0.0,
            "neg_ctx_center_score_mean": 0.0,
            "neg_ctx_neighbor_ref_score_mean": 0.0,
            "neg_ctx_gap_mean": 0.0,

            "pos_ctx_num_checked": 0.0,
            "pos_ctx_num_supported": 0.0,
            "pos_ctx_supported_ratio": 0.0,
            "pos_ctx_center_score_mean": 0.0,
            "pos_ctx_neighbor_ref_score_mean": 0.0,
            "pos_ctx_gap_mean": 0.0,
            "pos_ctx_protect_raw": 0.0,
        }

        if (
            (self.use_batch_context_guided_loss or
             self.use_negative_context_hn_loss or
             self.use_positive_context_protection)
            and (center_images is not None)
            and (neighbor_images_list is not None)
            and (slide_label_batch is not None)
        ):
            loss_neg_ctx_hn, loss_pos_ctx_protect, context_stats = \
                self.compute_negative_context_hn_and_positive_protection(
                    center_images=center_images,
                    slide_label_batch=slide_label_batch,
                    neighbor_images_list=neighbor_images_list,
                    is_eval=is_eval,
                )

        # 8.7) negative-slide global top-k suppression
        loss_neg_global_topk = spec_patch.new_tensor(0.0)
        neg_global_stats = {
            "neg_global_topk_raw": 0.0,
            "neg_global_topk_mean_score": 0.0,
            "neg_global_topk_margin": float(self.neg_global_topk_margin),
            "neg_global_topk_active": 0.0,
            "neg_global_num_patches": 0.0,
            "neg_global_topk_k": 0.0,
            "neg_global_topk_score_max": 0.0,
            "neg_global_topk_score_min": 0.0,
        }
        if (
            self.use_neg_global_topk_suppression
            and self.neg_global_topk_weight > 0
            and (wsi_images is not None)
            and (wsi_slide_label is not None)
        ):
            loss_neg_global_topk, neg_global_stats = self.compute_negative_global_topk_suppression_loss(
                images=wsi_images,
                slide_label=wsi_slide_label,
                patch_batch_size=self.wsi_patch_batch_size,
                is_eval=is_eval,
            )

        if self.enable_role_proto:
            loss_role_proto, role_stats = self.compute_role_proto_loss(
                spec_patch=spec_patch,
                dispatch_weight=dispatch_weight,
            )
        if self.use_free_expert_floor and self.free_expert_floor_weight > 0:
            loss_free_expert_floor, free_expert_stats = self.compute_free_expert_floor_loss(
                dispatch_weight=dispatch_weight
            )
        loss_cond_rank_weighted = spec_patch.new_tensor(0.0)
        if self.use_conditional_pairwise_ranking:
            loss_cond_rank_weighted = loss_cond_rank
            
        total_loss = (
            self.cls_align_weight * loss_cls_align
            + self.align_weight * loss_patch_align
            + self.proto_sep_weight * loss_proto_sep
            + self.sp_loss_weight * loss_sp
            + self.expert_floor_weight * loss_floor
            + self.cluster_sep_weight * loss_cluster_sep
            + self.intra_compact_weight * loss_intra
            + self.routing_stability_weight * loss_route_reg
            + self.z_loss_weight * loss_z
            + self.role_proto_weight * loss_role_proto
            + self.free_expert_floor_weight * loss_free_expert_floor
            + loss_wsi_bag
            + self.hn_repulsion_weight * loss_hn_repulsion
            + self.online_hn_repulsion_weight * loss_online_hn_repulsion
            + self.batch_context_loss_weight * (loss_neg_ctx_hn + loss_pos_ctx_protect)
            + self.neg_global_topk_weight * loss_neg_global_topk
            + loss_cond_rank_weighted
            + self.neg_rank_free_pref_weight * loss_neg_rank_free_pref
            + self.residual_hn_push_weight * loss_residual_hn_push
            + self.positive_anchor_weight * loss_positive_anchor_boost
        )

        # stats
        soft_frac = dispatch_weight.sum(dim=(0, 1))
        soft_frac = soft_frac / soft_frac.sum().clamp_min(1e-8)

        hard_frac = dispatch_mask.sum(dim=(0, 1))
        hard_frac = hard_frac / hard_frac.sum().clamp_min(1e-8)

        loss_dict = {
            "total_loss": float(total_loss.detach().cpu()),
            "cls_align": float(loss_cls_align.detach().cpu()),
            "patch_align": float(loss_patch_align.detach().cpu()),
            "proto_sep": float(loss_proto_sep.detach().cpu()),
            "sp_loss": float(loss_sp.detach().cpu()),
            "expert_floor": float(loss_floor.detach().cpu()),
            "cluster_sep": float(loss_cluster_sep.detach().cpu()),
            "intra_compact": float(loss_intra.detach().cpu()),
            "route_reg": float(loss_route_reg.detach().cpu()),
            "z_loss": float(loss_z.detach().cpu()),
            "soft_frac_max": float(soft_frac.max().detach().cpu()),
            "soft_frac_min": float(soft_frac.min().detach().cpu()),
            "hard_frac_max": float(hard_frac.max().detach().cpu()),
            "hard_frac_min": float(hard_frac.min().detach().cpu()),
        }

        loss_dict.update(proto_stats)
        loss_dict.update(sp_stats)
        loss_dict.update(floor_stats)
        loss_dict.update(cluster_stats)
        loss_dict.update(wsi_stats)
        loss_dict.update(hn_stats)
        loss_dict.update(online_hn_stats)
        loss_dict.update(context_stats)
        loss_dict.update(neg_global_stats)
        loss_dict.update(cond_rank_stats)
        loss_dict.update(neg_rank_free_pref_stats)
        loss_dict.update(residual_hn_push_stats)
        loss_dict.update(positive_anchor_stats)
        
        loss_dict["role_proto_raw"] = role_stats["role_proto"]
        loss_dict["role_proto_weighted"] = float((self.role_proto_weight * loss_role_proto).detach().cpu())
        loss_dict["role_attract"] = role_stats["role_attract"]
        loss_dict["role_separate"] = role_stats["role_separate"]
        loss_dict["role_target"] = role_stats["role_target"]
        loss_dict["role_mean_self_sim"] = role_stats["role_mean_self_sim"]
        loss_dict["role_mean_other_max_sim"] = role_stats["role_mean_other_max_sim"]
        loss_dict["role_num_valid"] = role_stats["role_num_valid"]
        loss_dict["role_valid_ratio"] = role_stats["role_valid_ratio"]
        loss_dict["role_main_conf_mean"] = role_stats["role_main_conf_mean"]
        loss_dict["free_expert_floor"] = float(
            (self.free_expert_floor_weight * loss_free_expert_floor).detach().cpu()
        )
        loss_dict["free_expert_floor_raw"] = free_expert_stats["free_expert_floor_raw"]
        loss_dict["free_expert_mass"] = free_expert_stats["free_expert_mass"]
        loss_dict["tumor_pref_raw"] = role_stats["tumor_pref_raw"]
        loss_dict["tumor_pref_weighted"] = float(
            (self.role_proto_weight * self.tumor_preference_weight * torch.tensor(role_stats["tumor_pref_raw"], device=spec_patch.device)).detach().cpu()
        )
        loss_dict["tumor_candidate_ratio"] = role_stats["tumor_candidate_ratio"]
        loss_dict["wsi_enabled"] = float(self.use_wsi_bag_loss or self.use_wsi_bag_margin_loss)
        loss_dict["hn_repulsion_weighted"] = float(
            (self.hn_repulsion_weight * loss_hn_repulsion).detach().cpu()
        )
        loss_dict["online_hn_repulsion_weighted"] = float(
            (self.online_hn_repulsion_weight * loss_online_hn_repulsion).detach().cpu()
        )
        loss_dict["batch_context_loss_weight"] = float(self.batch_context_loss_weight)
        loss_dict["neg_ctx_loss_weighted"] = float(
            (self.batch_context_loss_weight * loss_neg_ctx_hn).detach().cpu()
        )
        loss_dict["pos_ctx_protect_weighted"] = float(
            (self.batch_context_loss_weight * loss_pos_ctx_protect).detach().cpu()
        )
        loss_dict["batch_context_total_weighted"] = float(
            (self.batch_context_loss_weight * (loss_neg_ctx_hn + loss_pos_ctx_protect)).detach().cpu()
        )
        loss_dict["neg_global_topk_weighted"] = float(
            (self.neg_global_topk_weight * loss_neg_global_topk).detach().cpu()
        )
        loss_dict["cond_rank_enabled"] = float(self.use_conditional_pairwise_ranking)
        loss_dict["cond_rank_neg_weight"] = float(self.cond_rank_neg_weight)
        loss_dict["cond_rank_pos_weight"] = float(self.cond_rank_pos_weight)
        loss_dict["cond_rank_neg_margin"] = float(self.cond_rank_neg_margin)
        loss_dict["cond_rank_pos_margin"] = float(self.cond_rank_pos_margin)
        loss_dict["cond_rank_weighted"] = float(loss_cond_rank_weighted.detach().cpu())
        loss_dict["neg_rank_free_pref_weight"] = float(self.neg_rank_free_pref_weight)
        loss_dict["neg_rank_free_pref_weighted"] = float(
            (self.neg_rank_free_pref_weight * loss_neg_rank_free_pref).detach().cpu()
        )
        loss_dict["residual_hn_push_weight"] = float(self.residual_hn_push_weight)
        loss_dict["residual_hn_target_expert"] = (
            -1.0 if self.residual_hn_target_expert is None
            else float(self.residual_hn_target_expert)
        )
        loss_dict["residual_hn_push_weighted"] = float(
            (self.residual_hn_push_weight * loss_residual_hn_push).detach().cpu()
        )
        loss_dict["positive_anchor_weight"] = float(self.positive_anchor_weight)
        loss_dict["positive_anchor_margin"] = float(self.positive_anchor_margin)
        loss_dict["positive_anchor_boost_weighted"] = float(
            (self.positive_anchor_weight * loss_positive_anchor_boost).detach().cpu()
        )

        for i, v in enumerate(soft_frac):
            loss_dict[f"soft_frac_e{i}"] = float(v.detach().cpu())
        for i, v in enumerate(hard_frac):
            loss_dict[f"hard_frac_e{i}"] = float(v.detach().cpu())

        return total_loss, loss_dict