import torch
import torch.nn as nn
import torch.nn.functional as F
from .masking import generate_block_mask


class MoEDistiller(nn.Module):
    def __init__(self, student_model, teacher_model, stu_dim=384, tea_dim=1280, grid_size=16, moe_train_cfg=None):
        super().__init__()

        self.student = student_model
        self.teacher = teacher_model
        self.grid_size = grid_size

        print("⚡ 正在 GPU 上预生成 Mask Bank...")
        self.mask_pool_size = 8192
        self.register_buffer("mask_bank", self._build_mask_bank(mask_ratio=0.3))

        self.moe_train_cfg = moe_train_cfg or {}

        self.diversity_weight = self.moe_train_cfg.get("diversity_weight", 0.2)
        self.load_balance_weight = self.moe_train_cfg.get("load_balance_weight", 0.1)
        self.coverage_weight = self.moe_train_cfg.get("coverage_weight", 0.05)
        self.sparsity_weight = self.moe_train_cfg.get("sparsity_weight", 0.05)
        self.routing_sim_weight = self.moe_train_cfg.get("routing_sim_weight", 0.0)
        self.cluster_weight = self.moe_train_cfg.get("cluster_weight", 0.0)
        self.target_active_experts = self.moe_train_cfg.get("target_active_experts", 1.4)

        self.use_group_guided_routing = self.moe_train_cfg.get("use_group_guided_routing", False)
        self.group_guided_weight = self.moe_train_cfg.get("group_guided_weight", 0.05)

        self.num_offline_clusters = self.moe_train_cfg.get("num_offline_clusters", 6)
        self.valid_cluster_ids = self.moe_train_cfg.get("valid_cluster_ids", [0, 1, 2, 3, 4])
        self.background_cluster_ids = self.moe_train_cfg.get("background_cluster_ids", [5])

        # 只针对 normal experts，不含 shared expert
        self.group_guided_num_experts = self.moe_train_cfg.get("group_guided_num_experts", 4)

        self.cluster_expert_logits = nn.Parameter(
            0.01 * torch.randn(self.num_offline_clusters, self.group_guided_num_experts)
        )
        self.use_cluster_specialization = self.moe_train_cfg.get("use_cluster_specialization", False)
        self.spec_weight = self.moe_train_cfg.get("spec_weight", 0.1)
        self.spec_align_weight = self.moe_train_cfg.get("spec_align_weight", 1.0)
        self.spec_margin_weight = self.moe_train_cfg.get("spec_margin_weight", 1.0)

        self.use_cluster_anchor = self.moe_train_cfg.get("use_cluster_anchor", False)
        self.anchor_weight = self.moe_train_cfg.get("anchor_weight", 0.1)

        # 允许从配置里显式指定 cluster -> anchor expert
        self.cluster_anchor_map = self.moe_train_cfg.get("cluster_anchor_map", None)

        # 如果不显式给，就先用一个默认映射
        if self.cluster_anchor_map is None:
            self.cluster_anchor_map = {
                0: 0,
                1: 1,
                2: 2,
                3: 3,
                4: 1,   # 你也可以改成 2，后面再试
            }

        self.use_prototype_specialization = self.moe_train_cfg.get("use_prototype_specialization", False)
        self.proto_spec_weight = self.moe_train_cfg.get("proto_spec_weight", 0.02)

        self.proto_layer_weights = self.moe_train_cfg.get("proto_layer_weights", [1.0, 0.4])

        self.proto_main_margin = self.moe_train_cfg.get("proto_main_margin", 0.05)
        self.proto_main_weight = self.moe_train_cfg.get("proto_main_weight", 1.0)
        
        self.use_cluster_pull = self.moe_train_cfg.get("use_cluster_pull", False)
        self.cluster_pull_weight = self.moe_train_cfg.get("cluster_pull_weight", 0.02)
        self.cluster_pull_layers = self.moe_train_cfg.get("cluster_pull_layers", [0, 1])
        self.cluster_pull_use_cosine = self.moe_train_cfg.get("cluster_pull_use_cosine", True)
        self.cluster_pull_only_valid = self.moe_train_cfg.get("cluster_pull_only_valid", True)

        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        self._apply_student_freeze_strategy()

        self.proj_l9 = nn.Linear(stu_dim, tea_dim)
        self.proj_l12 = nn.Linear(stu_dim, tea_dim)

        self.stu_features = {}
        self.tea_features = {}
        self._register_hooks()

        nn.init.xavier_uniform_(self.proj_l9.weight)
        nn.init.xavier_uniform_(self.proj_l12.weight)

    def _build_mask_bank(self, mask_ratio):
        masks = generate_block_mask(
            batch_size=self.mask_pool_size,
            grid_size=self.grid_size,
            mask_ratio=mask_ratio
        )
        return masks

    def _apply_student_freeze_strategy(self):
        for name, param in self.student.named_parameters():
            param.requires_grad = False

            if "norm" in name or "layernorm" in name:
                param.requires_grad = True

            # 你原先这段未必能匹配到真实 DINOv2 名称，先保留
            if any(
                name.startswith(f"base_encoder.model.encoder.layer.{i}")
                for i in [9, 10, 11]
            ):
                param.requires_grad = True

            # MoE 相关参数全部解冻
            if ".mlp.experts." in name or ".mlp.shared_expert." in name or ".mlp.gate." in name:
                param.requires_grad = True

    def _register_hooks(self):
        def get_tea_hook(name):
            def hook(module, input, output):
                self.tea_features[name] = output[0] if isinstance(output, tuple) else output
            return hook

        self.teacher.blocks[23].register_forward_hook(get_tea_hook('layer_24'))
        self.teacher.blocks[31].register_forward_hook(get_tea_hook('layer_32'))

    def forward(self, images, mask_ratio=0.3, epoch=0, moe_warmup_epochs=3, routing_sim_start_epoch=5, is_eval=False,offline_cluster_ids=None,):
        self.tea_features.clear()

        B = images.shape[0]

        rand_indices = torch.randint(0, self.mask_pool_size, (B,), device=images.device)
        mask = self.mask_bank[rand_indices]

        current_ratio = mask.float().mean()
        if mask_ratio < current_ratio:
            keep_prob = mask_ratio / current_ratio
            drop_mask = torch.rand_like(mask.float()) < keep_prob
            mask = mask & drop_mask

        with torch.no_grad():
            _ = self.teacher(images)
            for k, v in self.tea_features.items():
                self.tea_features[k] = v.detach()

        _, gate_info_list, stu_feature_dict,_ = self.student(
            images,
            mask=mask,
            return_gates=True,
            is_eval=is_eval,
            return_features=True,
            offline_cluster_ids=offline_cluster_ids,
        )

        s_feat_9 = stu_feature_dict['layer_9']
        s_feat_12 = stu_feature_dict['layer_12']
        t_feat_24 = self.tea_features['layer_24']
        t_feat_32 = self.tea_features['layer_32']

        s_proj_9 = self.proj_l9(s_feat_9)
        s_proj_12 = self.proj_l12(s_feat_12)

        loss, loss_dict = self.compute_distill_loss(
            s_proj_9, s_proj_12, t_feat_24, t_feat_32, mask,
            gate_info_list, epoch, moe_warmup_epochs, routing_sim_start_epoch,
            offline_cluster_ids=offline_cluster_ids,
        )

        if self.training and torch.rand(1).item() < 0.01:
            if gate_info_list is not None and len(gate_info_list) > 0:
                B = images.shape[0]
                N = s_proj_12.shape[1] - 1   # 去掉CLS后的patch token数

                # hard usage
                last_mask = gate_info_list[-1]["dispatch_mask"].float()
                E = last_mask.shape[-1]
                tokens_per_batch = last_mask.shape[0] // B
                last_mask = last_mask.view(B, tokens_per_batch, E)
                if tokens_per_batch == N + 1:
                    last_mask = last_mask[:, 1:, :]
                hard_counts = last_mask.sum(dim=(0, 1))
                hard_frac = hard_counts / hard_counts.sum().clamp_min(1e-8)

                # soft usage
                last_weight = self.get_last_dispatch_weight(gate_info_list, B, N).float()  # [B, N, E]
                soft_mass = last_weight.sum(dim=(0, 1))
                soft_frac = soft_mass / soft_mass.sum().clamp_min(1e-8)

                print("[Debug] hard expert fraction:", hard_frac.detach().cpu().numpy())
                print("[Debug] soft expert fraction:", soft_frac.detach().cpu().numpy())

        return loss, loss_dict, gate_info_list

    def compute_token_distill_error(self, pred, target):
        """
        pred, target: [B, N, D]
        return: [B, N]
        """
        return F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)
    @torch.no_grad()
    def cluster_tokens_torch(self, features, n_clusters=8, iters=5):
        """
        features: [B, N, D]
        return:
            assign: [B, N] long
            cluster_weight: [B, N] float
        """
        B, N, D = features.shape
        feat_flat = features.reshape(B * N, D)  # [BN, D]

        # 初始化中心
        perm = torch.randperm(feat_flat.shape[0], device=feat_flat.device)[:n_clusters]
        centers = feat_flat[perm].clone()  # [C, D]

        for _ in range(iters):
            dists = torch.cdist(feat_flat, centers, p=2)   # [BN, C]
            assign_flat = dists.argmin(dim=1)              # [BN]

            new_centers = []
            for c in range(n_clusters):
                mask = assign_flat == c
                if mask.sum() == 0:
                    new_centers.append(centers[c:c+1])
                else:
                    new_centers.append(feat_flat[mask].mean(dim=0, keepdim=True))
            centers = torch.cat(new_centers, dim=0)

        assign = assign_flat.view(B, N)  # [B, N]

        cluster_weight = torch.zeros(B, N, device=features.device, dtype=features.dtype)

        for b in range(B):
            ids = assign[b]  # [N]
            counts = torch.bincount(ids, minlength=n_clusters).float()  # [C]
            freqs = counts / counts.sum().clamp_min(1.0)                # [C]

            token_freq = freqs[ids]                                     # [N]
            rarity = (1.0 / (token_freq + 1e-6)) ** self.rare_gamma

            # normalize to [0,1]
            r_min = rarity.min()
            r_max = rarity.max()
            rarity = (rarity - r_min) / (r_max - r_min + 1e-6)

            # map to [rare_min_weight, rare_max_weight]
            rarity = self.rare_min_weight + (self.rare_max_weight - self.rare_min_weight) * rarity
            cluster_weight[b] = rarity

        return assign, cluster_weight
    
    
    #取 cluster→expert 目标分布
    def get_cluster_expert_prior(self):
        return F.softmax(self.cluster_expert_logits, dim=-1)  # [K, E]
    #有效 cluster mask，用于计算 group-guided routing loss 时只关注有效 cluster
    def build_valid_cluster_mask(self, cluster_ids):
        cluster_ids = cluster_ids.long()
        valid_ids = torch.tensor(self.valid_cluster_ids, device=cluster_ids.device)
        valid_mask = (cluster_ids.unsqueeze(-1) == valid_ids.view(1, 1, -1)).any(dim=-1)
        return valid_mask

    def get_last_dispatch_weight(self, gate_info_list, B, N):
        """
        return:
            dispatch_weight: [B, N, E]
        """
        last_dispatch = gate_info_list[-1]["dispatch_weight"]   # [B*(N+1) or B*N, E]
        tokens_per_batch = last_dispatch.shape[0] // B
        E = last_dispatch.shape[-1]

        last_dispatch = last_dispatch.view(B, tokens_per_batch, E)

        # 去掉 CLS
        if tokens_per_batch == N + 1:
            last_dispatch = last_dispatch[:, 1:, :]

        return last_dispatch
    
    def get_layer_sim(self, gate_info_list, layer_idx, B, N):
        """
        return:
            sim: [B, N, E]   patch-only
        """
        sim = gate_info_list[layer_idx]["sim"]   # [B*(N+1) or B*N, E]
        tokens_per_batch = sim.shape[0] // B
        E = sim.shape[-1]

        sim = sim.view(B, tokens_per_batch, E)

        # 去掉 CLS
        if tokens_per_batch == N + 1:
            sim = sim[:, 1:, :]

        return sim
    
    def get_layer_routing_z(self, gate_info_list, layer_idx, B, N):
        """
        return:
            z: [B, N, D]   patch-only routing projection output
        """
        z = gate_info_list[layer_idx]["routing_z"]   # [B*(N+1) or B*N, D]
        tokens_per_batch = z.shape[0] // B
        D = z.shape[-1]

        z = z.view(B, tokens_per_batch, D)

        # 去掉 CLS
        if tokens_per_batch == N + 1:
            z = z[:, 1:, :]

        return z

    def compute_single_layer_prototype_specialization_loss(self, sim, cluster_ids):
        """
        sim:        [B, N, E]  patch-level scaled cosine similarity
        cluster_ids:[B, N]
        """
        device = sim.device
        B, N, E = sim.shape

        cluster_ids = cluster_ids.long()
        valid_mask = self.build_valid_cluster_mask(cluster_ids)   # [B, N]

        sim_flat = sim.reshape(B * N, E)
        cluster_flat = cluster_ids.reshape(B * N)
        valid_flat = valid_mask.reshape(B * N)

        sim_valid = sim_flat[valid_flat]          # [M, E]
        cluster_valid = cluster_flat[valid_flat]  # [M]

        if sim_valid.shape[0] == 0:
            zero = torch.tensor(0.0, device=device)
            stats = {
                "proto_main": 0.0,
                
                "proto_tokens": 0,
            }
            return zero, stats

        main_expert_map = {0: 0, 1: 1, 4: 2}
        #shared_expert_idx = 3

        main_losses = []
        #shared_losses = []
        used_tokens = 0

        for cid in self.valid_cluster_ids:
            cid = int(cid)
            if cid not in main_expert_map:
                continue

            mask_c = (cluster_valid == cid)
            if mask_c.sum() == 0:
                continue

            sim_c = sim_valid[mask_c]   # [Nc, E]
            used_tokens += int(mask_c.sum().item())

            main_e = main_expert_map[cid]

            # 只和另外两个 cluster-specific experts 比
            neg_experts = [e for e in main_expert_map.values() if e != main_e]
            if len(neg_experts) > 0:
                main_sim = sim_c[:, main_e:main_e+1]      # [Nc, 1]
                neg_sim = sim_c[:, neg_experts]           # [Nc, 2]
                loss_main = F.relu(self.proto_main_margin - (main_sim - neg_sim)).mean()
                main_losses.append(loss_main)

            # 对 shared expert 只做弱约束
            # if shared_expert_idx < E:
            #     shared_sim = sim_c[:, shared_expert_idx]
            #     loss_shared = F.relu(
            #         self.proto_shared_margin - (sim_c[:, main_e] - shared_sim)
            #     ).mean()
            #     shared_losses.append(loss_shared)

        if len(main_losses) == 0:
            zero = torch.tensor(0.0, device=device)
            stats = {
                "proto_main": 0.0,
                "proto_shared": 0.0,
                "proto_tokens": used_tokens,
            }
            return zero, stats

        proto_main = torch.stack(main_losses).mean()
        # proto_shared = (
        #     torch.stack(shared_losses).mean()
        #     if len(shared_losses) > 0
        #     else torch.tensor(0.0, device=device)
        # )

        total_loss = (
            self.proto_main_weight * proto_main 
        )

        stats = {
            "proto_main": float(proto_main.detach().cpu()),
            #"proto_shared": float(proto_shared.detach().cpu()),
            "proto_tokens": used_tokens,
        }
        return total_loss, stats
    
    def compute_two_layer_prototype_specialization_loss(self, gate_info_list, B, N, cluster_ids):
        """
        对两层 MoE 分别计算 prototype specialization:
        - 第一层权重大
        - 第二层权重小
        """
        device = cluster_ids.device

        if gate_info_list is None or len(gate_info_list) < 2:
            zero = torch.tensor(0.0, device=device)
            stats = {
                "proto_layer0": 0.0,
                "proto_layer1": 0.0,
                "proto_l0_main": 0.0,
                "proto_l1_main": 0.0,
            }
            return zero, stats

        # 假设当前主线就是两层 MoE，对应 gate_info_list[0], gate_info_list[1]
        sim_l0 = self.get_layer_sim(gate_info_list, 0, B, N)
        sim_l1 = self.get_layer_sim(gate_info_list, 1, B, N)

        loss_l0, stats_l0 = self.compute_single_layer_prototype_specialization_loss(sim_l0, cluster_ids)
        loss_l1, stats_l1 = self.compute_single_layer_prototype_specialization_loss(sim_l1, cluster_ids)

        w0 = float(self.proto_layer_weights[0])
        w1 = float(self.proto_layer_weights[1])

        total_loss = w0 * loss_l0 + w1 * loss_l1

        stats = {
            "proto_layer0": float(loss_l0.detach().cpu()),
            "proto_layer1": float(loss_l1.detach().cpu()),
            "proto_l0_main": stats_l0["proto_main"],
            #"proto_l0_shared": stats_l0["proto_shared"],
            "proto_l1_main": stats_l1["proto_main"],
            #"proto_l1_shared": stats_l1["proto_shared"],
        }
        return total_loss, stats

    def compute_group_guided_routing_loss(self, router_probs, cluster_ids):
        if router_probs.shape[-1] != self.cluster_expert_logits.shape[-1]:
            raise ValueError(
                f"Expert dim mismatch: router_probs {router_probs.shape[-1]} "
                f"vs cluster prior {self.cluster_expert_logits.shape[-1]}"
            )
        if cluster_ids.min() < 0 or cluster_ids.max() >= self.num_offline_clusters:
            raise ValueError(
                f"cluster_ids out of range: min={cluster_ids.min().item()}, "
                f"max={cluster_ids.max().item()}, num_clusters={self.num_offline_clusters}"
            )
        cluster_ids = cluster_ids.long()
        eps = 1e-8
        prior = self.get_cluster_expert_prior()          # [K, E]
        target = prior[cluster_ids]                      # [B, N, E]
        valid_mask = self.build_valid_cluster_mask(cluster_ids)  # [B, N]

        p = router_probs.clamp_min(eps)
        q = target.clamp_min(eps)

        kl = (p * (p.log() - q.log())).sum(dim=-1)       # [B, N]
        valid = valid_mask.float()

        loss = (kl * valid).sum() / (valid.sum() + 1e-6)

        stats = {
            "group_valid_token_ratio": float(valid.mean().detach().cpu()),
            "group_bg_token_ratio": float((1.0 - valid).mean().detach().cpu()),
        }
        return loss, stats
    
    def compute_cluster_specialization_loss(self, router_probs, cluster_ids):
        """
        router_probs: [B, N, E]   最后一层 normal experts 的 dispatch_weight
        cluster_ids:  [B, N]      offline token cluster ids

        思路：
        1) 对每个有效 cluster，计算其平均 expert responsibility 分布 mean_c
        2) 用软目标分布 target_c 做 KL 对齐
        3) 再加一个小的主专家 vs shared expert(E3) margin
        """
        device = router_probs.device
        B, N, E = router_probs.shape
        eps = 1e-8

        cluster_ids = cluster_ids.long()
        valid_mask = self.build_valid_cluster_mask(cluster_ids)   # [B, N]

        probs_flat = router_probs.reshape(B * N, E)               # [BN, E]
        cluster_flat = cluster_ids.reshape(B * N)                 # [BN]
        valid_flat = valid_mask.reshape(B * N)                    # [BN]

        probs_valid = probs_flat[valid_flat]                      # [M, E]
        cluster_valid = cluster_flat[valid_flat]                  # [M]

        if probs_valid.shape[0] < 2:
            zero = torch.tensor(0.0, device=device)
            stats = {
                "spec_align": 0.0,
                "spec_margin": 0.0,
                "spec_num_valid_tokens": int(probs_valid.shape[0]),
                "spec_num_valid_clusters": 0,
            }
            return zero, stats

        # 你当前主线下的“逻辑 group”目标分布
        # E0 / E1 / E2 是各自 cluster 的主专家，E3 是共享 / bridge expert
        target_map = {
            0: torch.tensor([0.45, 0.10, 0.10, 0.35], device=device),
            1: torch.tensor([0.10, 0.45, 0.10, 0.35], device=device),
            4: torch.tensor([0.10, 0.10, 0.45, 0.35], device=device),
        }

        # 每个 cluster 的主专家 index
        main_expert_map = {
            0: 0,
            1: 1,
            4: 2,
        }
        shared_expert_idx = 3

        align_losses = []
        margin_losses = []
        used_clusters = []

        margin = 0.05  # 很小，防止 E3 抢走入口，但不做硬控制

        for cid in self.valid_cluster_ids:
            cid = int(cid)
            mask_c = (cluster_valid == cid)
            if mask_c.sum() < 2:
                continue
            if cid not in target_map:
                continue

            p_c = probs_valid[mask_c]                  # [Nc, E]
            mean_c = p_c.mean(dim=0)                   # [E]
            mean_c = mean_c / mean_c.sum().clamp_min(eps)

            target_c = target_map[cid]
            target_c = target_c / target_c.sum().clamp_min(eps)

            # 让 cluster 平均分布靠近目标分布
            # 用 KL(target || mean) 更稳一点
            align_c = F.kl_div(
                mean_c.clamp_min(eps).log(),
                target_c,
                reduction="sum"
            )
            align_losses.append(align_c)

            # 防止共享专家 E3 压过主专家
            main_idx = main_expert_map[cid]
            margin_c = F.relu(margin - (mean_c[main_idx] - mean_c[shared_expert_idx]))
            margin_losses.append(margin_c)

            used_clusters.append(cid)

        if len(align_losses) == 0:
            zero = torch.tensor(0.0, device=device)
            stats = {
                "spec_align": 0.0,
                "spec_margin": 0.0,
                "spec_num_valid_tokens": int(probs_valid.shape[0]),
                "spec_num_valid_clusters": 0,
            }
            return zero, stats

        align_loss = torch.stack(align_losses).mean()
        margin_loss = (
            torch.stack(margin_losses).mean()
            if len(margin_losses) > 0
            else torch.tensor(0.0, device=device)
        )

        total_loss = (
            self.spec_align_weight * align_loss +
            self.spec_margin_weight * margin_loss
        )

        stats = {
            "spec_align": float(align_loss.detach().cpu()),
            "spec_margin": float(margin_loss.detach().cpu()),
            "spec_num_valid_tokens": int(probs_valid.shape[0]),
            "spec_num_valid_clusters": len(used_clusters),
        }
        return total_loss, stats
    
    def compute_expert_given_cluster_stats(self, router_probs, cluster_ids):
        """
        router_probs: [B, N, E]   最后一层 normal experts 的 dispatch_weight
        cluster_ids:  [B, N]

        return:
            stats: dict
                包含：
                - p_e0_given_c0, p_e1_given_c0, ...
                - cluster_token_count_0, ...
                - main_minus_shared_c0, main_minus_shared_c1, main_minus_shared_c4
        """
        B, N, E = router_probs.shape
        eps = 1e-8

        cluster_ids = cluster_ids.long()
        valid_mask = self.build_valid_cluster_mask(cluster_ids)   # [B, N]

        probs_flat = router_probs.reshape(B * N, E)               # [BN, E]
        cluster_flat = cluster_ids.reshape(B * N)                 # [BN]
        valid_flat = valid_mask.reshape(B * N)                    # [BN]

        probs_valid = probs_flat[valid_flat]                      # [M, E]
        cluster_valid = cluster_flat[valid_flat]                  # [M]

        stats = {}
        if probs_valid.shape[0] == 0:
            return stats

        main_expert_map = {0: 0, 1: 1, 4: 2}
        shared_expert_idx = 3

        for cid in self.valid_cluster_ids:
            cid = int(cid)
            mask_c = (cluster_valid == cid)
            n_c = int(mask_c.sum().item())

            stats[f"cluster_token_count_{cid}"] = n_c

            if n_c == 0:
                for e in range(E):
                    stats[f"p_e{e}_given_c{cid}"] = 0.0
                if cid in main_expert_map and shared_expert_idx < E:
                    stats[f"main_minus_shared_c{cid}"] = 0.0
                continue

            mean_c = probs_valid[mask_c].mean(dim=0)              # [E]
            mean_c = mean_c / mean_c.sum().clamp_min(eps)

            for e in range(E):
                stats[f"p_e{e}_given_c{cid}"] = float(mean_c[e].detach().cpu())

            if cid in main_expert_map and shared_expert_idx < E:
                main_e = main_expert_map[cid]
                stats[f"main_minus_shared_c{cid}"] = float(
                    (mean_c[main_e] - mean_c[shared_expert_idx]).detach().cpu()
                )

        return stats
    
    def compute_cluster_given_expert_stats(self, router_probs, cluster_ids):
        """
        router_probs: [B, N, E]   最后一层 normal experts 的 dispatch_weight
        cluster_ids:  [B, N]

        return:
            stats: dict
                包含：
                - p_c0_given_e0, p_c1_given_e0, ...
                - expert_token_mass_0, ...
                - expert_purity_0, expert_main_cluster_0, ...
        """
        B, N, E = router_probs.shape
        eps = 1e-8

        cluster_ids = cluster_ids.long()
        valid_mask = self.build_valid_cluster_mask(cluster_ids)   # [B, N]

        probs_flat = router_probs.reshape(B * N, E)               # [BN, E]
        cluster_flat = cluster_ids.reshape(B * N)                 # [BN]
        valid_flat = valid_mask.reshape(B * N)                    # [BN]

        probs_valid = probs_flat[valid_flat]                      # [M, E]
        cluster_valid = cluster_flat[valid_flat]                  # [M]

        stats = {}
        if probs_valid.shape[0] == 0:
            return stats

        # 只看有效 cluster
        valid_cluster_ids = [int(cid) for cid in self.valid_cluster_ids]

        # 对每个 expert，统计它“负责”的 token mass 里，各 cluster 占比
        # mass_c_e = sum_i p(e | token_i) over tokens in cluster c
        # p(c | e) = mass_c_e / sum_c mass_c_e
        for e in range(E):
            expert_mass_e = probs_valid[:, e].sum()   # 标量
            stats[f"expert_token_mass_{e}"] = float(expert_mass_e.detach().cpu())

            if expert_mass_e.item() <= 0:
                for cid in valid_cluster_ids:
                    stats[f"p_c{cid}_given_e{e}"] = 0.0
                stats[f"expert_purity_{e}"] = 0.0
                stats[f"expert_main_cluster_{e}"] = -1
                continue

            cluster_probs = []
            for cid in valid_cluster_ids:
                mask_c = (cluster_valid == cid)
                if mask_c.sum() == 0:
                    p_c_given_e = torch.tensor(0.0, device=probs_valid.device)
                else:
                    mass_c_e = probs_valid[mask_c, e].sum()
                    p_c_given_e = mass_c_e / expert_mass_e.clamp_min(eps)

                stats[f"p_c{cid}_given_e{e}"] = float(p_c_given_e.detach().cpu())
                cluster_probs.append(p_c_given_e)

            cluster_probs = torch.stack(cluster_probs)   # [C]
            purity, argmax_idx = cluster_probs.max(dim=0)

            stats[f"expert_purity_{e}"] = float(purity.detach().cpu())
            stats[f"expert_main_cluster_{e}"] = int(valid_cluster_ids[int(argmax_idx.item())])

        return stats

    def compute_cluster_anchor_loss(self, router_probs, cluster_ids):
        """
        router_probs: [B, N, E]
        cluster_ids:  [B, N]
        """
        device = router_probs.device
        B, N, E = router_probs.shape

        cluster_ids = cluster_ids.long()
        valid_mask = self.build_valid_cluster_mask(cluster_ids)   # [B, N]

        probs_flat = router_probs.reshape(B * N, E)
        cluster_flat = cluster_ids.reshape(B * N)
        valid_flat = valid_mask.reshape(B * N)

        probs_valid = probs_flat[valid_flat]
        cluster_valid = cluster_flat[valid_flat]

        if probs_valid.shape[0] < 2:
            zero = torch.tensor(0.0, device=device)
            stats = {
                "anchor_num_valid_tokens": int(probs_valid.shape[0]),
                "anchor_num_valid_clusters": 0,
                "anchor_avg_conf": 0.0,
            }
            return zero, stats

        cluster_mean_list = []
        anchor_target_list = []

        for cid in self.valid_cluster_ids:
            cid = int(cid)
            mask_c = (cluster_valid == cid)
            if mask_c.sum() < 2:
                continue

            if cid not in self.cluster_anchor_map:
                continue

            mean_c = probs_valid[mask_c].mean(dim=0)  # [E]
            cluster_mean_list.append(mean_c)
            anchor_target_list.append(int(self.cluster_anchor_map[cid]))

        if len(cluster_mean_list) == 0:
            zero = torch.tensor(0.0, device=device)
            stats = {
                "anchor_num_valid_tokens": int(probs_valid.shape[0]),
                "anchor_num_valid_clusters": 0,
                "anchor_avg_conf": 0.0,
            }
            return zero, stats

        cluster_means = torch.stack(cluster_mean_list, dim=0)   # [C, E]
        anchor_targets = torch.tensor(anchor_target_list, device=device, dtype=torch.long)

        # router_probs 本身是概率，先取 log 再做 NLL / CE
        log_cluster_means = torch.log(cluster_means.clamp_min(1e-8))
        anchor_loss = F.nll_loss(log_cluster_means, anchor_targets)

        anchor_conf = cluster_means.gather(1, anchor_targets.unsqueeze(1)).mean()

        stats = {
            "anchor_num_valid_tokens": int(probs_valid.shape[0]),
            "anchor_num_valid_clusters": int(cluster_means.shape[0]),
            "anchor_avg_conf": float(anchor_conf.detach().cpu()),
        }
        return anchor_loss, stats
    def compute_cluster_pull_loss(self, z, cluster_ids):
        """
        z: [B, N, D]
        cluster_ids: [B, N]
        """
        device = z.device
        cluster_ids = cluster_ids.long()

        if self.cluster_pull_use_cosine:
            z = F.normalize(z, dim=-1)

        valid_mask = self.build_valid_cluster_mask(cluster_ids)  # [B, N]

        losses = []
        used_clusters = 0
        used_tokens = 0

        if self.cluster_pull_only_valid:
            cluster_iter = self.valid_cluster_ids
        else:
            cluster_iter = torch.unique(cluster_ids).tolist()

        for cid in cluster_iter:
            cid = int(cid)
            mask_c = (cluster_ids == cid)

            if self.cluster_pull_only_valid:
                mask_c = mask_c & valid_mask

            if mask_c.sum() < 2:
                continue

            z_c = z[mask_c]   # [Nc, D]
            center = z_c.mean(dim=0, keepdim=True)

            if self.cluster_pull_use_cosine:
                center = F.normalize(center, dim=-1)
                sim = (z_c * center).sum(dim=-1)
                loss_c = (1.0 - sim).mean()
            else:
                loss_c = ((z_c - center) ** 2).sum(dim=-1).mean()

            losses.append(loss_c)
            used_clusters += 1
            used_tokens += int(mask_c.sum().item())

        if len(losses) == 0:
            zero = torch.tensor(0.0, device=device)
            stats = {
                "cluster_pull": 0.0,
                "cluster_pull_used_clusters": 0,
                "cluster_pull_used_tokens": 0,
            }
            return zero, stats

        loss = torch.stack(losses).mean()
        stats = {
            "cluster_pull": float(loss.detach().cpu()),
            "cluster_pull_used_clusters": used_clusters,
            "cluster_pull_used_tokens": used_tokens,
        }
        return loss, stats
    # ---------------- router losses ----------------

    def prototype_diversity_loss(self):
        losses = []
        for blk in self.student.blocks:
            if hasattr(blk.mlp, "gate"):
                proto = blk.mlp.gate.gate_vectors              # [E, D]
                proto = F.normalize(proto, dim=-1)
                sim_mat = proto @ proto.t()
                eye = torch.eye(sim_mat.size(0), device=sim_mat.device, dtype=sim_mat.dtype)
                losses.append(((sim_mat - eye) ** 2).mean())

        if len(losses) == 0:
            return torch.tensor(0.0, device=next(self.student.parameters()).device)
        return torch.stack(losses).mean()

    def load_balance_loss(self, dispatch_mask, dispatch_weight):
        """
        dispatch_mask: [B, N, E]
        dispatch_weight: [B, N, E]
        """
        E = dispatch_mask.shape[-1]

        load = dispatch_mask.float().mean(dim=(0, 1))          # [E]
        importance = dispatch_weight.mean(dim=(0, 1))          # [E]
        target = torch.full_like(load, 1.0 / E)

        loss_load = F.mse_loss(load, target)
        loss_imp = F.mse_loss(importance, target)
        return loss_load + loss_imp

    def coverage_loss(self, score):
        """
        score: [B, N, E]
        希望每个 token 至少有一个 expert 的 score 不要太低
        """
        best_score = score.max(dim=-1).values
        return F.relu(0.1 - best_score).mean()

    def sparsity_loss(self, active_counts):
        """
        active_counts: [B, N]
        希望平均激活 expert 数接近 target_active_experts
        """
        return ((active_counts.float().mean() - self.target_active_experts) ** 2)

    def routing_similarity_loss(self, features, gates):
        """
        features: (B, N, D)
        gates:    (B, N, E)
        """
        B, N, D = features.shape
        sample_k = min(64, N)

        idx = torch.randint(0, N, (B, sample_k), device=features.device)

        f = torch.gather(features, 1, idx.unsqueeze(-1).expand(-1, -1, D))
        g = torch.gather(gates, 1, idx.unsqueeze(-1).expand(-1, -1, gates.shape[-1]))

        f_norm = F.normalize(f, dim=-1)
        sim_feat = torch.matmul(f_norm, f_norm.transpose(1, 2))

        g_norm = F.normalize(g, dim=-1)
        sim_gate = torch.matmul(g_norm, g_norm.transpose(1, 2))

        return F.mse_loss(sim_feat, sim_gate)

    def cluster_routing_loss(self, features, gates, n_clusters=4, sample_k=128, iters=5):
        B, N, E = gates.shape
        k = min(sample_k, N)
        idx = torch.randint(0, N, (B, k), device=gates.device)
        g_sample = torch.gather(gates, 1, idx.unsqueeze(-1).expand(-1, -1, E))
        g_flat = g_sample.reshape(-1, E)

        perm = torch.randperm(g_flat.shape[0], device=gates.device)[:n_clusters]
        centers = g_flat[perm].clone()

        for _ in range(iters):
            dists = torch.cdist(g_flat, centers, p=2)
            assignments = dists.argmin(dim=1)

            new_centers = []
            for i in range(n_clusters):
                mask = assignments == i
                if mask.sum() == 0:
                    new_centers.append(centers[i].unsqueeze(0))
                else:
                    new_centers.append(g_flat[mask].mean(dim=0, keepdim=True))
            centers = torch.cat(new_centers, dim=0)

        assigned_centers = centers[assignments]
        return F.mse_loss(g_flat, assigned_centers)

    def compute_router_z_loss(self, student_model):
        z_loss = 0.0
        n = 0

        for module in student_model.modules():
            if hasattr(module, "gate"):
                sim = module.gate.last_sim
                if sim is None:
                    continue
                z_loss = z_loss + (sim ** 2).mean()
                n += 1

        if n > 0:
            z_loss = z_loss / n
        return z_loss

    def compute_moe_loss(self, gate_info_list, features, epoch=0, moe_warmup_epochs=3, routing_sim_start_epoch=5,offline_cluster_ids=None,):
        """
        gate_info_list: list of gate_info from MoE layers
        features: [B, N, D]  (通常传 s_patch_12)
        offline_cluster_ids: [B, N] or None
        """
        device = features.device
        B, N, _ = features.shape

        moe_weight_scale = min(1.0, epoch / max(moe_warmup_epochs, 1))
        

        diversity_loss = self.prototype_diversity_loss()

        load_losses = []
        coverage_losses = []
        sparsity_losses = []


        for gate_info in gate_info_list:
            dispatch_mask = gate_info["dispatch_mask"]
            dispatch_weight = gate_info["dispatch_weight"]
            score = gate_info["score"]
            active_counts = gate_info["active_counts"]

            tokens_per_batch = dispatch_mask.shape[0] // B
            E = dispatch_mask.shape[-1]

            dispatch_mask = dispatch_mask.view(B, tokens_per_batch, E)
            dispatch_weight = dispatch_weight.view(B, tokens_per_batch, E)
            score = score.view(B, tokens_per_batch, E)
            active_counts = active_counts.view(B, tokens_per_batch)
            
            # 去掉 CLS
            if tokens_per_batch == N + 1:
                dispatch_mask = dispatch_mask[:, 1:, :]
                dispatch_weight = dispatch_weight[:, 1:, :]
                score = score[:, 1:, :]
                active_counts = active_counts[:, 1:]

            load_losses.append(self.load_balance_loss(dispatch_mask, dispatch_weight))
            coverage_losses.append(self.coverage_loss(score))
            sparsity_losses.append(self.sparsity_loss(active_counts))

        if len(load_losses) > 0:
            load_loss = torch.stack(load_losses).mean()
            coverage_loss = torch.stack(coverage_losses).mean()
            sparsity_loss = torch.stack(sparsity_losses).mean()
        else:
            device = features.device
            load_loss = torch.tensor(0.0, device=device)
            coverage_loss = torch.tensor(0.0, device=device)
            sparsity_loss = torch.tensor(0.0, device=device)

        # ---------------- group-guided routing loss ----------------
        # 只在最后一层 normal expert dispatch 上做
        last_dispatch = self.get_last_dispatch_weight(gate_info_list, B, N)  # [B, N, E]

        if self.use_group_guided_routing and (offline_cluster_ids is not None):
            if offline_cluster_ids.shape[:2] != last_dispatch.shape[:2]:
                raise ValueError(
                    f"offline_cluster_ids shape mismatch: "
                    f"{offline_cluster_ids.shape} vs last_dispatch {last_dispatch.shape}"
                )

            group_guided_loss, group_stats = self.compute_group_guided_routing_loss(
                last_dispatch, offline_cluster_ids
            )
        else:
            group_guided_loss = torch.tensor(0.0, device=device)
            group_stats = {
                "group_valid_token_ratio": 0.0,
                "group_bg_token_ratio": 0.0,
            }
        if offline_cluster_ids is not None:
            if offline_cluster_ids.shape[:2] != last_dispatch.shape[:2]:
                raise ValueError(
                    f"offline_cluster_ids shape mismatch for prototype specialization: "
                    f"{offline_cluster_ids.shape} vs last_dispatch {last_dispatch.shape}"
                )

            proto_loss, proto_stats = self.compute_two_layer_prototype_specialization_loss(
                gate_info_list, B, N, offline_cluster_ids
            )
        else:
            proto_loss = torch.tensor(0.0, device=device)
            proto_stats = {
                "proto_layer0": 0.0,
                "proto_layer1": 0.0,
                "proto_l0_main": 0.0,
                "proto_l0_shared": 0.0,
                "proto_l1_main": 0.0,
                "proto_l1_shared": 0.0,
            }

        # ---------------- cluster pull loss on routing_z ----------------
        if self.use_cluster_pull and (offline_cluster_ids is not None):
            if offline_cluster_ids.shape[:2] != last_dispatch.shape[:2]:
                raise ValueError(
                    f"offline_cluster_ids shape mismatch for cluster pull: "
                    f"{offline_cluster_ids.shape} vs last_dispatch {last_dispatch.shape}"
                )

            pull_losses = []
            pull_stats_list = []

            for layer_idx in self.cluster_pull_layers:
                if layer_idx >= len(gate_info_list):
                    continue
                if "routing_z" not in gate_info_list[layer_idx]:
                    raise KeyError(
                        f"gate_info_list[{layer_idx}] does not contain 'routing_z'. "
                        f"Please add routing_z to gate_info in student forward."
                    )

                z_l = self.get_layer_routing_z(gate_info_list, layer_idx, B, N)  # [B, N, D]
                pull_l, stats_l = self.compute_cluster_pull_loss(z_l, offline_cluster_ids)
                pull_losses.append(pull_l)
                pull_stats_list.append(stats_l)

            if len(pull_losses) > 0:
                cluster_pull_loss = torch.stack(pull_losses).mean()
                cluster_pull_stats = {
                    "cluster_pull": float(cluster_pull_loss.detach().cpu()),
                    "cluster_pull_used_clusters": max(s["cluster_pull_used_clusters"] for s in pull_stats_list),
                    "cluster_pull_used_tokens": max(s["cluster_pull_used_tokens"] for s in pull_stats_list),
                }
            else:
                cluster_pull_loss = torch.tensor(0.0, device=device)
                cluster_pull_stats = {
                    "cluster_pull": 0.0,
                    "cluster_pull_used_clusters": 0,
                    "cluster_pull_used_tokens": 0,
                }
        else:
            cluster_pull_loss = torch.tensor(0.0, device=device)
            cluster_pull_stats = {
                "cluster_pull": 0.0,
                "cluster_pull_used_clusters": 0,
                "cluster_pull_used_tokens": 0,
            }

        # ---------------- cluster specialization loss ----------------
        expert_cluster_stats = {}
        cluster_expert_stats = {}
        if offline_cluster_ids is not None:
            if offline_cluster_ids.shape[:2] != last_dispatch.shape[:2]:
                raise ValueError(
                    f"offline_cluster_ids shape mismatch for specialization: "
                    f"{offline_cluster_ids.shape} vs last_dispatch {last_dispatch.shape}"
                )

            
            expert_cluster_stats = self.compute_expert_given_cluster_stats(
                last_dispatch, offline_cluster_ids
            )
            cluster_expert_stats = self.compute_cluster_given_expert_stats(
                last_dispatch, offline_cluster_ids
            )
        if self.use_cluster_specialization and (offline_cluster_ids is not None):
            spec_loss, spec_stats = self.compute_cluster_specialization_loss(
                last_dispatch, offline_cluster_ids
            )
        else:
            spec_loss = torch.tensor(0.0, device=device)
            spec_stats = {
                "spec_align": 0.0,
                "spec_margin": 0.0,
                "spec_num_valid_tokens": 0,
                "spec_num_valid_clusters": 0,
            }
        

        # ---------------- cluster anchor loss ----------------
        if self.use_cluster_anchor and (offline_cluster_ids is not None):
            if offline_cluster_ids.shape[:2] != last_dispatch.shape[:2]:
                raise ValueError(
                    f"offline_cluster_ids shape mismatch for anchor loss: "
                    f"{offline_cluster_ids.shape} vs last_dispatch {last_dispatch.shape}"
                )

            anchor_loss, anchor_stats = self.compute_cluster_anchor_loss(
                last_dispatch, offline_cluster_ids
            )
        else:
            anchor_loss = torch.tensor(0.0, device=device)
            anchor_stats = {
                "anchor_num_valid_tokens": 0,
                "anchor_num_valid_clusters": 0,
                "anchor_avg_conf": 0.0,
            }
        moe_loss = moe_weight_scale * (
            self.diversity_weight * diversity_loss
            + self.load_balance_weight * load_loss
            + self.coverage_weight * coverage_loss
            + self.sparsity_weight * sparsity_loss
        )
        #moe_loss = moe_loss + routing_weight * routing_sim + self.cluster_weight * cluster_loss
        if self.use_group_guided_routing:
            moe_loss = moe_loss + self.group_guided_weight * group_guided_loss
        if self.use_cluster_pull:
            moe_loss = moe_loss + self.cluster_pull_weight * cluster_pull_loss
        if self.use_cluster_specialization:
            moe_loss = moe_loss + self.spec_weight * spec_loss
        if self.use_cluster_anchor:
            moe_loss = moe_loss + self.anchor_weight * anchor_loss
        if self.use_prototype_specialization:
            moe_loss = moe_loss + self.proto_spec_weight * proto_loss

        moe_loss_dict = {
            "moe_total": float(moe_loss.detach().cpu()),
            "moe_diversity": float(diversity_loss.detach().cpu()),
            "moe_load": float(load_loss.detach().cpu()),
            "moe_coverage": float(coverage_loss.detach().cpu()),
            "moe_sparsity": float(sparsity_loss.detach().cpu()),
            # "moe_routing_sim": float(routing_sim.detach().cpu()),
            # "moe_cluster": float(cluster_loss.detach().cpu()),
            "moe_weight_scale": float(moe_weight_scale),

        }
        if self.use_group_guided_routing:
            moe_loss_dict.update({
                "moe_group_guided": float(group_guided_loss.detach().cpu()),
                "group_valid_token_ratio": group_stats["group_valid_token_ratio"],
                "group_bg_token_ratio": group_stats["group_bg_token_ratio"],
            })
        if self.use_cluster_pull:
            moe_loss_dict.update({
                "moe_cluster_pull": float(cluster_pull_loss.detach().cpu()),
                "cluster_pull_used_clusters": cluster_pull_stats["cluster_pull_used_clusters"],
                "cluster_pull_used_tokens": cluster_pull_stats["cluster_pull_used_tokens"],
            })
        if self.use_cluster_specialization:
            moe_loss_dict.update({
                "moe_spec": float(spec_loss.detach().cpu()),
                "spec_align": spec_stats["spec_align"],
                "spec_margin": spec_stats["spec_margin"],
                "spec_num_valid_tokens": spec_stats["spec_num_valid_tokens"],
                "spec_num_valid_clusters": spec_stats["spec_num_valid_clusters"],
            })
            
        if self.use_cluster_anchor:
            moe_loss_dict.update({
                "moe_anchor": float(anchor_loss.detach().cpu()),
                "anchor_num_valid_tokens": anchor_stats["anchor_num_valid_tokens"],
                "anchor_num_valid_clusters": anchor_stats["anchor_num_valid_clusters"],
                "anchor_avg_conf": anchor_stats["anchor_avg_conf"],
            })
        if self.use_prototype_specialization:
            moe_loss_dict.update({
                "moe_proto": float(proto_loss.detach().cpu()),
                **proto_stats
            })
        moe_loss_dict.update(cluster_expert_stats)
        moe_loss_dict.update(expert_cluster_stats)
        return moe_loss, moe_loss_dict

    # ---------------- distillation losses ----------------
    def patch_relation_consistency_loss(self, student_patch, teacher_patch):
        B, N, D = student_patch.shape
        sample_k = min(64, N)

        idx1 = torch.randint(0, N, (B, sample_k), device=student_patch.device)
        idx2 = torch.randint(0, N, (B, sample_k), device=student_patch.device)

        s1 = torch.gather(student_patch, 1, idx1.unsqueeze(-1).expand(-1, -1, D))
        s2 = torch.gather(student_patch, 1, idx2.unsqueeze(-1).expand(-1, -1, D))
        t1 = torch.gather(teacher_patch, 1, idx1.unsqueeze(-1).expand(-1, -1, D))
        t2 = torch.gather(teacher_patch, 1, idx2.unsqueeze(-1).expand(-1, -1, D))

        return F.mse_loss(s1 - s2, t1 - t2)

    def weighted_smooth_l1(self, pred, target, token_mask):
        """
        pred, target: [B, N, D]
        token_mask:   [B, N] float or bool
        """
        diff = F.smooth_l1_loss(pred, target, reduction="none").mean(dim=-1)  # [B, N]
        token_mask = token_mask.float()
        return (diff * token_mask).sum() / (token_mask.sum() + 1e-6)
    def compute_distill_loss(self, s_9, s_12, t_24, t_32, mask, gate_info_list, epoch=0, moe_warmup_epochs=3, routing_sim_start_epoch=5,offline_cluster_ids=None,):
        def split_tokens(feat, is_teacher=False):
            cls_token = feat[:, 0:1, :]
            patch_tokens = feat[:, 5:, :] if is_teacher else feat[:, 1:, :]
            return cls_token, patch_tokens

        s_cls_9, s_patch_9 = split_tokens(s_9, False)
        t_cls_24, t_patch_24 = split_tokens(t_24, True)
        s_cls_12, s_patch_12 = split_tokens(s_12, False)
        t_cls_32, t_patch_32 = split_tokens(t_32, True)

        # ---- safety check: token count must match ----
        if s_patch_9.shape[:2] != t_patch_24.shape[:2]:
            raise ValueError(
                f"Shape mismatch: s_patch_9 {s_patch_9.shape} vs t_patch_24 {t_patch_24.shape}"
            )
        if s_patch_12.shape[:2] != t_patch_32.shape[:2]:
            raise ValueError(
                f"Shape mismatch: s_patch_12 {s_patch_12.shape} vs t_patch_32 {t_patch_32.shape}"
            )

        target_ones = torch.ones(s_cls_9.shape[0]).to(s_cls_9.device)
        loss_cls_9 = F.cosine_embedding_loss(s_cls_9.squeeze(1), t_cls_24.squeeze(1), target_ones)
        loss_cls_12 = F.cosine_embedding_loss(s_cls_12.squeeze(1), t_cls_32.squeeze(1), target_ones)

       # ---------------- patch distill ----------------
        mask_float = mask.float()
        unmask_float = (~mask).float()

        loss_p9_masked = self.weighted_smooth_l1(
            s_patch_9, t_patch_24, mask_float
        )
        loss_p12_masked = self.weighted_smooth_l1(
            s_patch_12, t_patch_32, mask_float
        )

        loss_p9_unmasked = self.weighted_smooth_l1(
            s_patch_9, t_patch_24, unmask_float
        )
        loss_p12_unmasked = self.weighted_smooth_l1(
            s_patch_12, t_patch_32, unmask_float
        )

        loss_moe, moe_loss_dict = self.compute_moe_loss(
            gate_info_list, s_patch_12, epoch=epoch,
            moe_warmup_epochs=moe_warmup_epochs,
            routing_sim_start_epoch=routing_sim_start_epoch,
            offline_cluster_ids=offline_cluster_ids,
        )

        prc_loss = self.patch_relation_consistency_loss(s_patch_12, t_patch_32)
        z_loss = self.compute_router_z_loss(self.student)

        
        weight_cls = 1.0
        weight_masked = 1.5
        weight_unmasked = 0.5
        weight_moe = 0.6
        weight_prc = 0.01
        weight_z = 1e-4

        total_loss = (
            weight_cls * (loss_cls_9 * 0.5 + loss_cls_12 * 1.0) +
            weight_masked * (loss_p9_masked * 0.5 + loss_p12_masked * 1.0) +
            weight_unmasked * (loss_p9_unmasked * 0.5 + loss_p12_unmasked * 1.0) +
            weight_moe * loss_moe +
            weight_prc * prc_loss +
            weight_z * z_loss 
        )

        loss_dict = {
            "cls_l9": loss_cls_9.item(),
            "cls_l12": loss_cls_12.item(),
            "patch_masked_l9": loss_p9_masked.item(),
            "patch_masked_l12": loss_p12_masked.item(),
            "patch_unmasked_l9": loss_p9_unmasked.item(),
            "patch_unmasked_l12": loss_p12_unmasked.item(),
            "moe_loss": loss_moe.item(),
            "prc_loss": prc_loss.item(),
            "z_loss": z_loss.item(),
            "total_loss": total_loss.item(),

        }
        loss_dict.update(moe_loss_dict)
        return total_loss, loss_dict