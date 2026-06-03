import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchRoleSummary(nn.Module):
    """
    兼容/分析辅助模块：
    将 token-level role logits 聚合成 patch-level role descriptor。

    注意：
    - 主线更推荐：直接由共享主 prototype 在 patch-level 计算 role summary
    - 这里只作为兼容已有 token-role-logit 场景的工具
    """

    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = float(tau)

    @staticmethod
    def compute_one_vs_rest_gaps(patch_role_logits: torch.Tensor) -> torch.Tensor:
        if patch_role_logits.ndim != 3:
            raise ValueError(
                f"patch_role_logits must be [B, N, R], got {tuple(patch_role_logits.shape)}"
            )

        B, N, R = patch_role_logits.shape
        gaps = []

        for r in range(R):
            cur = patch_role_logits[..., r]
            if R == 1:
                other_max = torch.zeros_like(cur)
            else:
                other_ids = [i for i in range(R) if i != r]
                other_max = patch_role_logits[..., other_ids].max(dim=-1).values
            gaps.append(cur - other_max)

        return torch.stack(gaps, dim=-1)

    @staticmethod
    def compute_top1_gap(patch_role_logits: torch.Tensor) -> torch.Tensor:
        R = patch_role_logits.shape[-1]
        top2 = torch.topk(patch_role_logits, k=min(2, R), dim=-1).values
        if R >= 2:
            return (top2[..., 0] - top2[..., 1]).unsqueeze(-1)
        return torch.ones_like(top2[..., 0:1])

    def forward(self, token_role_logits: torch.Tensor) -> dict:
        if token_role_logits.ndim != 4:
            raise ValueError(
                f"token_role_logits must be [B, N, T, R], got {tuple(token_role_logits.shape)}"
            )

        patch_role_logits = token_role_logits.mean(dim=2)
        patch_role_probs = torch.softmax(patch_role_logits / self.tau, dim=-1)
        patch_role_gaps = self.compute_one_vs_rest_gaps(patch_role_logits)
        patch_top1_gap = self.compute_top1_gap(patch_role_logits)

        return {
            "patch_role_logits": patch_role_logits,
            "patch_role_probs": patch_role_probs,
            "patch_role_gaps": patch_role_gaps,
            "patch_top1_gap": patch_top1_gap,
        }


class RoleAwareTailPlugin(nn.Module):
    """
    Encoder-tail role-aware plugin (patch-level)

    输入:
        patch_feat:        [B, N, D]
        patch_role_probs:  [B, N, R]
        patch_role_gaps:   [B, N, R]
        patch_role_logits: [B, N, R] or None
        patch_top1_gap:    [B, N, 1] or None
    """

    def __init__(
        self,
        feat_dim: int,
        num_roles: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        use_role_logits: bool = False,
        use_top1_gap: bool = True,
        use_beta: bool = True,
        init_scale: float = 0.05,
    ):
        super().__init__()

        self.feat_dim = int(feat_dim)
        self.num_roles = int(num_roles)
        self.hidden_dim = int(hidden_dim)
        self.use_role_logits = bool(use_role_logits)
        self.use_top1_gap = bool(use_top1_gap)
        self.use_beta = bool(use_beta)

        side_dim = 0
        if self.use_role_logits:
            side_dim += self.num_roles
        side_dim += self.num_roles  # probs
        side_dim += self.num_roles  # gaps
        if self.use_top1_gap:
            side_dim += 1

        self.feat_norm = nn.LayerNorm(self.feat_dim)

        self.cond_mlp = nn.Sequential(
            nn.Linear(side_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.to_gamma = nn.Linear(self.hidden_dim, self.feat_dim)
        self.to_beta = nn.Linear(self.hidden_dim, self.feat_dim) if self.use_beta else None

        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.to_gamma.bias)
        if self.to_beta is not None:
            nn.init.zeros_(self.to_beta.bias)

    def forward(
        self,
        patch_feat: torch.Tensor,
        patch_role_probs: torch.Tensor,
        patch_role_gaps: torch.Tensor,
        patch_role_logits: torch.Tensor | None = None,
        patch_top1_gap: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        if patch_feat.ndim != 3:
            raise ValueError(f"patch_feat must be [B, N, D], got {tuple(patch_feat.shape)}")
        if patch_role_probs.ndim != 3:
            raise ValueError(f"patch_role_probs must be [B, N, R], got {tuple(patch_role_probs.shape)}")
        if patch_role_gaps.ndim != 3:
            raise ValueError(f"patch_role_gaps must be [B, N, R], got {tuple(patch_role_gaps.shape)}")

        B, N, D = patch_feat.shape
        if D != self.feat_dim:
            raise ValueError(f"feat dim mismatch: got {D}, expected {self.feat_dim}")

        side_inputs = []
        if self.use_role_logits:
            if patch_role_logits is None:
                raise ValueError("patch_role_logits is required when use_role_logits=True")
            side_inputs.append(patch_role_logits)

        side_inputs.append(patch_role_probs)
        side_inputs.append(patch_role_gaps)

        if self.use_top1_gap:
            if patch_top1_gap is None:
                raise ValueError("patch_top1_gap is required when use_top1_gap=True")
            side_inputs.append(patch_top1_gap)

        side = torch.cat(side_inputs, dim=-1)
        feat_ln = self.feat_norm(patch_feat)
        cond_h = self.cond_mlp(side)

        gamma = torch.tanh(self.to_gamma(cond_h))
        if self.use_beta:
            beta = self.to_beta(cond_h)
        else:
            beta = torch.zeros_like(gamma)

        delta = gamma * feat_ln + beta
        out = patch_feat + self.res_scale * delta

        if return_aux:
            aux = {
                "gamma_mean": float(gamma.mean().detach().cpu()),
                "gamma_abs_mean": float(gamma.abs().mean().detach().cpu()),
                "beta_abs_mean": float(beta.abs().mean().detach().cpu()),
                "delta_abs_mean": float(delta.abs().mean().detach().cpu()),
                "res_scale": float(self.res_scale.detach().cpu()),
            }
            return out, aux

        return out


class RoleAwareTailWithSharedSummary(nn.Module):
    """
    主线封装：
    plugin 只复用外部共享主 prototype 生成的 patch-level role summary
    """

    def __init__(
        self,
        feat_dim: int,
        num_roles: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        use_role_logits: bool = False,
        use_top1_gap: bool = True,
        use_beta: bool = True,
        init_scale: float = 0.05,
    ):
        super().__init__()
        self.plugin = RoleAwareTailPlugin(
            feat_dim=feat_dim,
            num_roles=num_roles,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_role_logits=use_role_logits,
            use_top1_gap=use_top1_gap,
            use_beta=use_beta,
            init_scale=init_scale,
        )

    def forward(
        self,
        patch_feat: torch.Tensor,
        patch_role_probs: torch.Tensor,
        patch_role_gaps: torch.Tensor,
        patch_role_logits: torch.Tensor | None = None,
        patch_top1_gap: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        return self.plugin(
            patch_feat=patch_feat,
            patch_role_probs=patch_role_probs,
            patch_role_gaps=patch_role_gaps,
            patch_role_logits=patch_role_logits,
            patch_top1_gap=patch_top1_gap,
            return_aux=return_aux,
        )
