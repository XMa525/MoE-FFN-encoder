import torch
import torch.nn as nn
import torch.nn.functional as F
from .experts import FFNExpert
from .gating import GatingNetwork


class MoEFFN(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        num_layers,              # 保留兼容，不实际使用
        num_experts,
        shared_expert,
        routing_strategy,
        top_k,
        min_experts,
        last_routing_logits=None,
        shared_alpha=0.05,
        init_threshold=0.0,
        max_experts=2,
        gate_init_scale=2.0,
        gate_noise_std=0.02,
        top2_ratio_threshold=0.5,
        top2_abs_threshold=-1e9,
        num_clusters=None,
        cluster_bias_scale=1.0,
        use_cluster_bias=False,
        background_cluster_ids=None,
        use_cluster_candidate_mask=False,
        cluster_candidate_k=2,
        soft_candidate_mask=False,
        candidate_penalty=1.5,
        use_routing_proj=True,
        routing_proj_dim=None,
        routing_proj_hidden_dim=None,
        routing_proj_dropout=0.0,
        routing_metric="cosine",
        semi_dot_min_scale=0.5,
        semi_dot_max_scale=2.0,
        use_patch_guided_routing=False,
        patch_guided_mode="none",          # "none" | "residual" | "concat"
        patch_context_alpha=0.5,
        patch_summary_type="global",       # "global" | "local"
        patch_summary_kernel=3,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.shared_expert_enabled = shared_expert
        self.shared_alpha = shared_alpha
        self.max_experts = max_experts
        self.routing_metric = routing_metric

        self.experts = nn.ModuleList([
            FFNExpert(dim, hidden_dim) for _ in range(num_experts)
        ])

        if shared_expert:
            self.shared_expert = FFNExpert(dim, hidden_dim)

        self.gate = GatingNetwork(
            dim=dim,
            input_dim=dim,
            num_experts=num_experts,
            routing_strategy=routing_strategy,
            top_k=top_k,
            min_experts=min_experts,
            max_experts=max_experts,
            init_scale=gate_init_scale,
            init_threshold=init_threshold,
            noise_std=gate_noise_std,
            use_routing_proj=use_routing_proj,
            proj_dim=routing_proj_dim if routing_proj_dim is not None else dim,
            proj_hidden_dim=routing_proj_hidden_dim if routing_proj_hidden_dim is not None else dim,
            proj_dropout=routing_proj_dropout,
            routing_metric=routing_metric,
            semi_dot_min_scale=semi_dot_min_scale,
            semi_dot_max_scale=semi_dot_max_scale,
        )

        self.top2_ratio_threshold = top2_ratio_threshold
        self.top2_abs_threshold = top2_abs_threshold

        self.use_cluster_bias = use_cluster_bias
        self.num_clusters = num_clusters
        self.cluster_bias_scale = cluster_bias_scale
        self.background_cluster_ids = background_cluster_ids or []
        self.use_cluster_candidate_mask = use_cluster_candidate_mask
        self.cluster_candidate_k = cluster_candidate_k
        self.soft_candidate_mask = soft_candidate_mask
        self.candidate_penalty = candidate_penalty

        self.use_patch_guided_routing = use_patch_guided_routing
        self.patch_guided_mode = patch_guided_mode
        self.patch_context_alpha = patch_context_alpha
        self.patch_summary_type = patch_summary_type
        self.patch_summary_kernel = patch_summary_kernel

        if self.patch_summary_type not in ["global", "local"]:
            raise ValueError(f"Unsupported patch_summary_type: {self.patch_summary_type}")

        if not isinstance(self.patch_summary_kernel, int) or self.patch_summary_kernel < 1:
            raise ValueError(f"patch_summary_kernel must be positive int, got {self.patch_summary_kernel}")

        if self.patch_summary_kernel % 2 == 0:
            raise ValueError(f"patch_summary_kernel should be odd, got {self.patch_summary_kernel}")

        if self.use_patch_guided_routing:
            if self.patch_guided_mode == "residual":
                self.patch_context_proj = nn.Linear(dim, dim)
                nn.init.normal_(self.patch_context_proj.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(self.patch_context_proj.bias)
            elif self.patch_guided_mode == "concat":
                self.patch_context_proj = None
            elif self.patch_guided_mode in ["none", None]:
                self.patch_context_proj = None
            else:
                raise ValueError(f"Unsupported patch_guided_mode: {self.patch_guided_mode}")
        else:
            self.patch_context_proj = None

        if self.use_cluster_bias:
            assert self.num_clusters is not None
            init_bias = torch.zeros(self.num_clusters, self.num_experts)

            # 只对有效 cluster 给轻微偏好
            init_bias[0, [0, 3]] = 1.0
            init_bias[1, [1]] = 1.0
            init_bias[4, [2, 3]] = 1.0

            self.register_buffer("cluster_expert_bias", init_bias)

        if self.use_cluster_candidate_mask:
            assert self.num_clusters is not None
            self.cluster_expert_logits = nn.Parameter(
                0.01 * torch.randn(self.num_clusters, self.num_experts)
            )

        # debug / analysis cache
        self.last_gate_info = None
        self.last_dispatch_weight = None
        self.last_dispatch_mask = None
        self.unassigned_token_idx = None
        self.last_unassigned_ratio = 0.0

    def _normalize_sparse_weights(self, weights):
        row_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return weights / row_sum

    def build_candidate_mask(self, offline_cluster_ids, B, seq_len, device):
        """
        offline_cluster_ids: [B, N]
        return:
            candidate_mask: [B*(N+1), E] bool
        """
        assert offline_cluster_ids.shape[0] == B
        assert offline_cluster_ids.shape[1] == seq_len - 1

        patch_prior_logits = self.cluster_expert_logits[offline_cluster_ids]   # [B, N, E]
        patch_prior = torch.softmax(patch_prior_logits, dim=-1)

        k = min(self.cluster_candidate_k, self.num_experts)
        _, topk_idx = torch.topk(patch_prior, k=k, dim=-1)

        patch_candidate_mask = torch.zeros_like(patch_prior, dtype=torch.bool)
        patch_candidate_mask.scatter_(-1, topk_idx, True)

        if len(self.background_cluster_ids) > 0:
            bg_mask = torch.zeros_like(offline_cluster_ids, dtype=torch.bool)
            for bg_id in self.background_cluster_ids:
                bg_mask |= (offline_cluster_ids == bg_id)
            patch_candidate_mask[bg_mask] = True

        cls_mask = torch.ones(B, 1, self.num_experts, device=device, dtype=torch.bool)

        full_candidate_mask = torch.cat([cls_mask, patch_candidate_mask], dim=1)  # [B, N+1, E]
        candidate_mask = full_candidate_mask.reshape(B * seq_len, self.num_experts)
        return candidate_mask

    def build_patch_summary(self, x_tokens):
        """
        x_tokens: [B, N, D]  (no CLS)
        return:
            patch_summary: [B, N, D]

        modes:
            - global: whole-patch mean expanded to every token
            - local : each token gets a local neighborhood summary from 2D token grid
        """
        B, N, D = x_tokens.shape

        if self.patch_summary_type == "global":
            patch_summary = x_tokens.mean(dim=1, keepdim=True)   # [B, 1, D]
            patch_summary = patch_summary.expand(-1, N, -1)      # [B, N, D]
            return patch_summary

        elif self.patch_summary_type == "local":
            side = int(round(N ** 0.5))
            if side * side != N:
                raise ValueError(
                    f"Local patch summary requires square token grid, but got N={N}"
                )

            # [B, N, D] -> [B, D, H, W]
            x_2d = x_tokens.view(B, side, side, D).permute(0, 3, 1, 2).contiguous()

            k = self.patch_summary_kernel
            pad = k // 2

            local_summary = F.avg_pool2d(
                x_2d,
                kernel_size=k,
                stride=1,
                padding=pad,
                count_include_pad=False,
            )   # [B, D, H, W]

            # 避免 summary 过平滑，保留一部分 token identity
            local_summary = 0.5 * x_2d + 0.5 * local_summary

            # [B, D, H, W] -> [B, N, D]
            local_summary = local_summary.permute(0, 2, 3, 1).contiguous().view(B, N, D)
            return local_summary

        else:
            raise ValueError(f"Unsupported patch_summary_type: {self.patch_summary_type}")

    def build_gate_input(self, x):
        """
        x: [B, N+1, D], with CLS
        return:
            gate_input: [B, N+1, D]
        """
        if (not self.use_patch_guided_routing) or self.patch_guided_mode in ["none", None]:
            return x

        cls_token = x[:, :1, :]         # [B, 1, D]
        patch_tokens = x[:, 1:, :]      # [B, N, D]

        patch_summary = self.build_patch_summary(patch_tokens)   # [B, N, D]

        if self.patch_guided_mode == "residual":
            if self.patch_context_proj is None:
                raise RuntimeError("patch_context_proj is None in residual mode")

            patch_ctx = self.patch_context_proj(patch_summary)   # [B, N, D]
            guided_patch_tokens = patch_tokens + self.patch_context_alpha * patch_ctx

            gate_input = torch.cat([cls_token, guided_patch_tokens], dim=1)
            return gate_input

        elif self.patch_guided_mode == "concat":
            raise NotImplementedError(
                "concat mode not implemented yet in this minimal version. "
                "Please use patch_guided_mode='residual' or 'none'."
            )

        else:
            raise ValueError(f"Unsupported patch_guided_mode: {self.patch_guided_mode}")

    def route(self, score):
        """
        score: [T, E]
        return:
            dispatch_mask:   [T, E] bool
            dispatch_weight: [T, E] float
            aux_stats: dict
        """
        T, E = score.shape
        device = score.device
        rows = torch.arange(T, device=device)

        k = min(2, E)
        top_vals, top_idx = torch.topk(score, k=k, dim=-1)

        top1_val = top_vals[:, 0]
        top1_idx = top_idx[:, 0]

        dispatch_mask = torch.zeros_like(score, dtype=torch.bool)
        dispatch_mask[rows, top1_idx] = True
        top2_val = top1_val
        select_two = torch.zeros(T, dtype=torch.bool, device=device)

        if E >= 2 and self.max_experts is not None and self.max_experts >= 2:
            top2_val = top_vals[:, 1]
            top2_idx = top_idx[:, 1]

            alpha = self.top2_ratio_threshold
            beta = self.top2_abs_threshold

            select_two = (top2_val >= alpha * top1_val) & (top2_val >= beta)
            dispatch_mask[rows[select_two], top2_idx[select_two]] = True

        if self.gate.min_experts > 1:
            active_counts = dispatch_mask.sum(dim=-1)
            need_more = active_counts < self.gate.min_experts
            if need_more.any():
                k = min(self.gate.min_experts, E)
                fill_idx = torch.topk(score[need_more], k=k, dim=-1).indices
                new_mask = dispatch_mask.clone()
                new_mask[need_more] = False
                new_mask[need_more] = new_mask[need_more].scatter(1, fill_idx, True)
                dispatch_mask = new_mask

        masked_score = score.masked_fill(~dispatch_mask, float("-inf"))
        dispatch_weight = F.softmax(masked_score, dim=-1)
        dispatch_weight = torch.where(dispatch_mask, dispatch_weight, torch.zeros_like(dispatch_weight))
        dispatch_weight = self._normalize_sparse_weights(dispatch_weight)

        active_counts = dispatch_mask.sum(dim=-1)
        pre_fallback_unassigned = torch.zeros(T, dtype=torch.bool, device=device)

        aux_stats = {
            "active_counts": active_counts,
            "pre_fallback_unassigned": pre_fallback_unassigned,
            "pre_fallback_unassigned_ratio": pre_fallback_unassigned.float().mean(),
            "best_score": top1_val,
            "top2_score": top_vals[:, 1] if E >= 2 else top1_val,
            "top1_score_mean": top1_val.mean().detach(),
            "top2_score_mean": top2_val.mean().detach() if E >= 2 else top1_val.mean().detach(),
            "select_two_ratio": select_two.float().mean().detach(),
        }

        return dispatch_mask, dispatch_weight, aux_stats

    def forward(self, x, layer_idx=0, return_gates=False, is_eval=False, offline_cluster_ids=None):
        """
        x: [B, N+1, D]
        """
        B, seq_len, D = x.shape

        # expert 真正处理的输入仍然保留原始 token
        x_flat = x.reshape(B * seq_len, D)

        # gate 用的输入可按开关切换为 patch-guided
        gate_input = self.build_gate_input(x)                    # [B, N+1, D]
        gate_input_flat = gate_input.reshape(B * seq_len, D)

        cluster_bias = None
        if self.use_cluster_bias and offline_cluster_ids is not None:
            assert offline_cluster_ids.shape[0] == B
            assert offline_cluster_ids.shape[1] == seq_len - 1

            patch_bias = self.cluster_expert_bias[offline_cluster_ids]  # [B, N, E]

            if len(self.background_cluster_ids) > 0:
                bg_mask = torch.zeros_like(offline_cluster_ids, dtype=torch.bool)
                for bg_id in self.background_cluster_ids:
                    bg_mask |= (offline_cluster_ids == bg_id)
                patch_bias = patch_bias.masked_fill(bg_mask.unsqueeze(-1), 0.0)

            cls_bias = torch.zeros(B, 1, self.num_experts, device=x.device, dtype=patch_bias.dtype)
            full_bias = torch.cat([cls_bias, patch_bias], dim=1)   # [B, N+1, E]
            cluster_bias = full_bias.reshape(B * seq_len, self.num_experts)

        if self.use_cluster_bias and cluster_bias is not None and torch.rand(1).item() < 0.001:
            print("use_cluster_bias =", self.use_cluster_bias)
            print("cluster_bias shape =", cluster_bias.shape)
            print("cluster_bias abs mean =", cluster_bias.abs().mean().item())

        sim, score = self.gate(
            gate_input_flat,
            cluster_bias=cluster_bias,
            bias_scale=self.cluster_bias_scale,
        )

        candidate_mask = None
        if self.use_cluster_candidate_mask and offline_cluster_ids is not None:
            candidate_mask = self.build_candidate_mask(
                offline_cluster_ids=offline_cluster_ids,
                B=B,
                seq_len=seq_len,
                device=x.device,
            )

            if self.soft_candidate_mask:
                score = score - (~candidate_mask).float() * self.candidate_penalty
            else:
                score = score.masked_fill(~candidate_mask, -1e9)

        if self.use_cluster_candidate_mask and candidate_mask is not None and torch.rand(1).item() < 0.002:
            candidate_per_token = candidate_mask.sum(dim=1).float().mean().item()
            print("candidate_per_token =", candidate_per_token)

        dispatch_mask, dispatch_weight, aux_stats = self.route(score)

        self.last_dispatch_mask = dispatch_mask
        self.last_dispatch_weight = dispatch_weight

        output = torch.zeros_like(x_flat)

        # [T, E, D] 保存每个 token 在每个 expert 下的单独输出
        expert_outputs = torch.zeros(
            x_flat.shape[0], self.num_experts, x_flat.shape[1],
            device=x_flat.device, dtype=torch.float32
        )

        for i in range(self.num_experts):
            token_mask = dispatch_mask[:, i]
            if token_mask.sum() == 0:
                continue

            selected_tokens = x_flat[token_mask]
            expert_out = self.experts[i](selected_tokens)
            weight = dispatch_weight[token_mask, i].unsqueeze(-1)

            indices = token_mask.nonzero(as_tuple=True)[0]

            expert_outputs[indices, i, :] = expert_out.to(expert_outputs.dtype)
            output = output.index_add(
                0,
                indices,
                expert_out.float() * weight.float()
            )

        if self.shared_expert_enabled:
            shared_out = self.shared_expert(x_flat).float()
            output = output + self.shared_alpha * shared_out

        output = output.reshape(B, seq_len, D)

        gate_info = {
            "sim": sim,                                                # [B*(N+1), E]
            "score": score,                                            # [B*(N+1), E]
            "dispatch_mask": dispatch_mask,                            # [B*(N+1), E]
            "dispatch_weight": dispatch_weight,                        # [B*(N+1), E]
            "candidate_mask": candidate_mask,
            "select_two_ratio": aux_stats["select_two_ratio"],
            "active_counts": aux_stats["active_counts"],               # [B*(N+1)]
            "best_score": aux_stats["best_score"],                     # [B*(N+1)]
            "pre_fallback_unassigned": aux_stats["pre_fallback_unassigned"],
            "pre_fallback_unassigned_ratio": aux_stats["pre_fallback_unassigned_ratio"],
            "routing_z": self.gate.last_proj,                          # [B*(N+1), D_r]
            "expert_outputs": expert_outputs,                          # [B*(N+1), E, D]
            "routing_input_mode": (
                self.patch_guided_mode if self.use_patch_guided_routing else "token"
            ),
            "patch_summary_type": (
                self.patch_summary_type if self.use_patch_guided_routing else "none"
            ),
            "patch_summary_kernel": (
                int(self.patch_summary_kernel) if self.use_patch_guided_routing else 0
            ),
        }

        self.last_gate_info = gate_info

        if self.training and torch.rand(1).item() < 0.005:
            experts_per_token = dispatch_mask.sum(dim=1).float().mean().item()
            print(
                f"experts_per_token={experts_per_token:.4f}, "
                f"unassigned_ratio={self.last_unassigned_ratio:.4f}"
            )

        if self.training and torch.rand(1).item() < 0.002:
            nonzero_ratio = (expert_outputs.abs().sum(dim=-1) > 0).float().mean().item()
            print(f"[MoEFFN] expert_outputs nonzero ratio = {nonzero_ratio:.4f}")

        if self.training and self.use_patch_guided_routing and torch.rand(1).item() < 0.002:
            print(
                f"[MoEFFN] patch-guided routing enabled | "
                f"mode={self.patch_guided_mode} | alpha={self.patch_context_alpha} | "
                f"summary={self.patch_summary_type} | kernel={self.patch_summary_kernel}"
            )

        if self.training and self.use_patch_guided_routing and torch.rand(1).item() < 0.002:
            delta = (gate_input_flat - x_flat).norm(dim=-1).mean().item()
            base = x_flat.norm(dim=-1).mean().item()
            print(f"[MoEFFN] gate_input_delta_norm={delta:.6f}, base_token_norm={base:.6f}")

        if return_gates:
            return output, gate_info
        return output

    @torch.no_grad()
    def update_dynamic_experts(self, x_flat, max_new_experts=1):
        # 暂时禁用真实动态增删专家，避免结构继续复杂化
        return