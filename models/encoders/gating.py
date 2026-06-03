import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """
    Prototype routing with routing projection head:
    - token feature x 先经过 routing_proj 得到 routing subspace 表示 z
    - 每个 expert 一个 prototype (gate_vectors) in routing subspace
    - token 与 prototype 的 similarity 作为 routing similarity
    - score = sim - threshold

    routing_metric:
        - "cosine":   scaled cosine similarity (default, backward-compatible)
        - "dot":      scaled dot-product similarity
        - "semi_dot": cosine with controlled token-norm scaling
    """

    def __init__(
        self,
        dim,
        num_experts,
        input_dim=None,
        routing_strategy="proto_topany",
        top_k=2,
        min_experts=1,
        max_experts=None,
        init_scale=2.0,
        learnable_scale=True,
        init_threshold=0.0,
        threshold_min=-2.0,
        threshold_max=2.0,
        noise_std=0.02,
        proj_dim=None,
        proj_hidden_dim=None,
        use_routing_proj=True,
        proj_dropout=0.0,
        routing_metric="cosine",
        semi_dot_min_scale=0.5,      # 新增
        semi_dot_max_scale=2.0,      # 新增
        attn_use_q_proj=True,
    ):
        super().__init__()
        self.routing_strategy = routing_strategy
        self.top_k = top_k
        self.min_experts = min_experts
        self.max_experts = max_experts
        self.num_experts = num_experts
        self.dim = dim
        self.input_dim = dim if input_dim is None else input_dim
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.noise_std = noise_std
        self.use_routing_proj = use_routing_proj
        self.routing_metric = routing_metric
        self.semi_dot_min_scale = semi_dot_min_scale
        self.semi_dot_max_scale = semi_dot_max_scale

        if proj_dim is None:
            proj_dim = dim
        if proj_hidden_dim is None:
            proj_hidden_dim = dim

        self.proj_dim = proj_dim
        self.proj_hidden_dim = proj_hidden_dim

        if self.use_routing_proj:
            self.routing_proj = nn.Sequential(
                nn.Linear(self.input_dim, proj_hidden_dim),
                nn.GELU(),
                nn.Dropout(proj_dropout),
                nn.Linear(proj_hidden_dim, proj_dim),
            )
        else:
            assert proj_dim == dim, "If use_routing_proj=False, proj_dim must equal dim."
            self.routing_proj = nn.Identity()

        self.attn_use_q_proj = attn_use_q_proj
        if self.routing_metric == "attn":
            if self.attn_use_q_proj:
                self.gate_query = nn.Linear(proj_dim, proj_dim)
            else:
                self.gate_query = nn.Identity()

            self.expert_keys = nn.Parameter(torch.empty(num_experts, proj_dim))
            nn.init.normal_(self.expert_keys, mean=0.0, std=0.02)
        # expert prototypes
        self.gate_vectors = nn.Parameter(torch.empty(num_experts, proj_dim))
        nn.init.normal_(self.gate_vectors, mean=0.0, std=0.02)

        self.expert_threshold = nn.Parameter(torch.ones(num_experts) * init_threshold)

        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        else:
            self.register_buffer("logit_scale", torch.tensor(math.log(init_scale)))

        # debug cache
        self.last_input = None
        self.last_proj = None
        self.last_sim = None
        self.last_score = None

    def get_threshold(self):
        return self.expert_threshold.clamp(self.threshold_min, self.threshold_max)

    def compute_similarity(self, z, gate_vectors):
        """
        z: [T, D]
        gate_vectors: [E, D]
        return:
            sim: [T, E]
        """
        scale = self.logit_scale.exp().clamp(min=0.1, max=100.0)

        if self.routing_metric == "cosine":
            z_norm = F.normalize(z, dim=-1)
            gate_norm = F.normalize(gate_vectors, dim=-1)
            sim = (z_norm @ gate_norm.t()) * scale

        elif self.routing_metric == "dot":
            sim = (z @ gate_vectors.t()) * scale

        elif self.routing_metric == "semi_dot":
            # 1) cosine 主体：保留当前方向几何
            z_unit = F.normalize(z, dim=-1)
            gate_unit = F.normalize(gate_vectors, dim=-1)
            cos_sim = z_unit @ gate_unit.t()   # [T, E]

            # 2) 受控 token norm 缩放：只恢复一部分 norm 信息
            #    这里按 batch 内平均 norm 归一，再做 clip，避免像 pure dot 那样爆炸
            z_norm = z.norm(dim=-1, keepdim=True).clamp(min=1e-6)   # [T, 1]
            mean_norm = z_norm.detach().mean().clamp(min=1e-6)

            token_scale = (z_norm / mean_norm).clamp(
                min=self.semi_dot_min_scale,
                max=self.semi_dot_max_scale
            )   # [T, 1]

            sim = cos_sim * token_scale * scale
        elif self.routing_metric == "attn":
            q = self.gate_query(z)   # [T, D]
            sim = (q @ self.expert_keys.t()) / math.sqrt(q.shape[-1])
            sim = sim * scale
        else:
            raise ValueError(f"Unsupported routing_metric: {self.routing_metric}")

        return sim

    def forward(self, x, cluster_bias=None, bias_scale=1.0):
        """
        x: [T, D]
        return:
            sim:   [T, E]
            score: [T, E] sim - threshold
        """
        z = self.routing_proj(x)   # [T, proj_dim]

        sim = self.compute_similarity(z, self.gate_vectors)

        if self.training and torch.rand(1).item() < 0.001:
            print(f"[Gating] routing_metric = {self.routing_metric}")

        if self.training and self.noise_std > 0:
            sim = sim + torch.randn_like(sim) * self.noise_std

        threshold = self.get_threshold().unsqueeze(0)  # [1, E]
        score = sim - threshold

        if cluster_bias is not None:
            score = score + bias_scale * cluster_bias

        self.last_input = x
        self.last_proj = z
        self.last_sim = sim
        self.last_score = score
        return sim, score