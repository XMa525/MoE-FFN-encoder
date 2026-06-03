import torch
import torch.nn as nn
from .dinov2_encoder import DINOv2Encoder
from .moe_FFN import MoEFFN


class MoEEncoder(nn.Module):

    def __init__(self, base_encoder_cfg, moe_cfg):
        super().__init__()

        self.base_encoder = DINOv2Encoder(**base_encoder_cfg)

        self.blocks = self.base_encoder.blocks
        self.norm = self.base_encoder.norm
        self.embed_dim = self.base_encoder.embed_dim
        self.device = self.base_encoder.device

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.moe_layers_idx = moe_cfg.get("moe_layers", [-3, -2, -1])
        self.num_layers = len(self.moe_layers_idx)
        self.num_experts = moe_cfg.get("num_experts", 4)
        self.shared_expert = moe_cfg.get("shared_expert", True)
        self.routing_strategy = moe_cfg.get("routing_strategy", "proto_topany")
        self.top_k = moe_cfg.get("top_k", 2)
        self.init_threshold = moe_cfg.get("init_threshold", 0.0)
        self.min_experts = moe_cfg.get("min_experts", 1)
        self.max_experts = moe_cfg.get("max_experts", 2)
        self.gate_init_scale = moe_cfg.get("gate_init_scale", 2.0)
        self.gate_noise_std = moe_cfg.get("gate_noise_std", 0.02)
        self.shared_alpha = moe_cfg.get("shared_alpha", 0.05)
        self.use_routing_proj = moe_cfg.get("use_routing_proj", True)
        self.routing_proj_dim = moe_cfg.get("routing_proj_dim", None)
        self.routing_proj_hidden_dim = moe_cfg.get("routing_proj_hidden_dim", None)
        self.routing_proj_dropout = moe_cfg.get("routing_proj_dropout", 0.0)

        self.use_cluster_bias = moe_cfg.get("use_cluster_bias", False)
        self.num_clusters = moe_cfg.get("num_clusters", None)
        self.cluster_bias_scale = moe_cfg.get("cluster_bias_scale", 1.0)
        self.background_cluster_ids = moe_cfg.get("background_cluster_ids", [5])
        self.routing_metric=moe_cfg.get("routing_metric", "cosine")

        self.use_patch_guided_routing = moe_cfg.get("use_patch_guided_routing", False)
        self.patch_guided_mode = moe_cfg.get("patch_guided_mode", "none")
        self.patch_context_alpha = moe_cfg.get("patch_context_alpha", 0.5)
        self.patch_summary_type = moe_cfg.get("patch_summary_type", "global")
        self.patch_summary_kernel = moe_cfg.get("patch_summary_kernel", 3)


        depth = len(self.blocks)
        self.moe_layer_map = {}

        for moe_layer_idx, idx in enumerate(self.moe_layers_idx):
            real_idx = idx if idx >= 0 else depth + idx
            self.moe_layer_map[real_idx] = moe_layer_idx

            block = self.blocks[real_idx]

            dim = block.mlp.fc1.in_features
            hidden_dim = block.mlp.fc1.out_features

            print(f"[INFO] Replacing block {real_idx} MLP with MoE")

            
            block.mlp = MoEFFN(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=self.num_experts,
                num_layers=self.num_layers,   # 为兼容保留
                shared_expert=self.shared_expert,
                routing_strategy=self.routing_strategy,
                top_k=self.top_k,
                init_threshold=self.init_threshold,
                min_experts=self.min_experts,
                max_experts=self.max_experts,
                gate_init_scale=self.gate_init_scale,
                gate_noise_std=self.gate_noise_std,
                shared_alpha=self.shared_alpha,
                use_cluster_bias=self.use_cluster_bias,
                num_clusters=self.num_clusters,
                cluster_bias_scale=self.cluster_bias_scale,
                background_cluster_ids=self.background_cluster_ids,
                use_routing_proj=self.use_routing_proj,
                routing_proj_dim=self.routing_proj_dim,
                routing_proj_hidden_dim=self.routing_proj_hidden_dim,
                routing_proj_dropout=self.routing_proj_dropout,
                routing_metric=self.routing_metric, 
                use_patch_guided_routing=self.use_patch_guided_routing,
                patch_guided_mode=self.patch_guided_mode,
                patch_context_alpha=self.patch_context_alpha,
                patch_summary_type=self.patch_summary_type,
                patch_summary_kernel=self.patch_summary_kernel,
            )

    def _forward_moe_block(self, blk, x, moe_layer_idx, return_gates=False, is_eval=False, offline_cluster_ids=None):
        residual = x
        x_norm = blk.norm1(x)

        attn_out = blk.attention(x_norm)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]

        attn_out = blk.layer_scale1(attn_out)
        x = residual + blk.drop_path(attn_out)

        residual = x
        mlp_input = blk.norm2(x)

        if isinstance(blk.mlp, MoEFFN):
            mlp_out = blk.mlp(
                mlp_input,
                layer_idx=moe_layer_idx,
                return_gates=return_gates,
                is_eval=is_eval,
                offline_cluster_ids=offline_cluster_ids,
            )
            if return_gates:
                mlp_out, gate_info = mlp_out
            else:
                gate_info = None
        else:
            mlp_out = blk.mlp(mlp_input)
            gate_info = None

        mlp_out = blk.layer_scale2(mlp_out)
        x = residual + blk.drop_path(mlp_out)

        return x, gate_info

    def forward(self, x, return_gates=False, mask=None, is_eval=False, return_features=False,offline_cluster_ids=None,):
        x = self.base_encoder.patch_embed_forward(x)

        gate_info_list = []
        feature_dict = {}
        moe_feature_list = []

        cls_token = x[:, 0:1, :]
        patch_tokens = x[:, 1:, :]

        if mask is not None:
            B, N, D = patch_tokens.shape
            mask_expanded = mask.unsqueeze(-1).to(patch_tokens.device)
            mask_token = self.mask_token.expand(B, N, -1)
            patch_tokens = torch.where(mask_expanded, mask_token, patch_tokens)

        x = torch.cat([cls_token, patch_tokens], dim=1)

        for i, blk in enumerate(self.blocks):
            if i in self.moe_layer_map:
                moe_layer_idx = self.moe_layer_map[i]

                x, gate_info = self._forward_moe_block(
                    blk,
                    x,
                    moe_layer_idx=moe_layer_idx,
                    return_gates=return_gates,
                    is_eval=is_eval,
                    offline_cluster_ids=offline_cluster_ids,
                )

                if return_gates:
                    gate_info_list.append(gate_info)
                # 记录该 MoE block 输出特征
                moe_feature_list.append(x)
            else:
                x = blk(x)

            if i == 8:
                feature_dict["layer_9"] = x
            if i == 11:
                feature_dict["layer_12"] = x

        x_out = self.norm(x)

        if return_gates and return_features:
            return x_out, gate_info_list, feature_dict,moe_feature_list
        elif return_gates:
            return x_out, gate_info_list
        elif return_features:
            return x_out, feature_dict,moe_feature_list
        else:
            return x_out