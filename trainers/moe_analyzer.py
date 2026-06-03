import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openslide

class MoEAnalyzer:
    def __init__(self, routing_strategy="top1", top_k=1, threshold=0.3, min_experts=1, shared_expert_idx=None):
        self.routing_strategy = routing_strategy
        self.top_k = top_k
        self.threshold = threshold
        self.min_experts = min_experts
        self.shared_expert_idx = shared_expert_idx

    # ------------------------------
    # 计算每个专家使用比例 & shared expert 激活率
    # ------------------------------
    def compute_expert_usage(self, gate_probs):
        T, E_total = gate_probs.shape
        # shared expert 激活率
        shared_active = gate_probs[:, self.shared_expert_idx].mean().item() if self.shared_expert_idx is not None else 0.0

        return gate_probs.mean(dim=0).detach().cpu().numpy(), shared_active

    # 平均激活专家数量（普通专家 & shared 分开）
    def compute_avg_active(self, gate_probs):
        T, E_total = gate_probs.shape

        if self.routing_strategy == "top1":
            topk = 1
        elif self.routing_strategy == "topk":
            topk = self.top_k
        elif self.routing_strategy == "topany":
            mask = gate_probs > self.threshold
            active_count = mask.sum(dim=-1)
            active_count = torch.max(active_count, torch.tensor(self.min_experts))
            avg_active_normal = active_count.float().mean().item()
            shared_extra = 0.0
            if self.shared_expert_idx is not None:
                shared_extra = mask[:, self.shared_expert_idx].float().mean().item()
            return avg_active_normal, shared_extra
        else:
            raise ValueError(f"Unknown routing_strategy: {self.routing_strategy}")

        avg_active_normal = float(topk)
        shared_extra = 1.0 if self.shared_expert_idx is not None else 0.0
        return avg_active_normal, shared_extra

    # ------------------------------
    # 计算 gating 熵
    # ------------------------------
    def compute_entropy(self, gate_probs):
        T, E_total = gate_probs.shape
        E_normal = E_total if self.shared_expert_idx is None else E_total - 1
        gate_probs_no_shared = gate_probs[:, :E_normal]
        entropy = - (gate_probs_no_shared * torch.log(gate_probs_no_shared + 1e-9)).sum(dim=-1)
        return entropy.mean().item()

    # ------------------------------
    # 画 layer × expert heatmap（普通专家 & shared 分开显示）
    # ------------------------------
    def plot_layer_expert_heatmap(self, gating_probs_list, save_path=None):
        layer_usages_normal = []
        layer_usages_shared = []

        for gate_probs in gating_probs_list:
            usage, shared_rate = self.compute_expert_usage(gate_probs)
            if self.shared_expert_idx is not None:
                usage_normal = np.delete(usage, self.shared_expert_idx)
                layer_usages_normal.append(usage_normal)
                layer_usages_shared.append([usage[self.shared_expert_idx]])
            else:
                layer_usages_normal.append(usage)
                layer_usages_shared.append([0.0])

        heatmap_normal = np.stack(layer_usages_normal, axis=0)
        heatmap_shared = np.stack(layer_usages_shared, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [heatmap_normal.shape[1], 1]})
        sns.heatmap(heatmap_normal, annot=True, fmt=".2f", cmap="viridis", ax=axes[0])
        axes[0].set_xlabel("Normal Expert ID")
        axes[0].set_ylabel("MoE Layer")

        sns.heatmap(heatmap_shared, annot=True, fmt=".2f", cmap="magma", ax=axes[1])
        axes[1].set_xlabel("Shared Expert")
        axes[1].set_ylabel("")

        plt.suptitle(f"Layer-wise Expert Usage (average probability) [{self.routing_strategy}]")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def compute_patch_dominant_expert(self, gate_probs):
        """
        gate_probs: [N_patch, E]
        """
        dominant = gate_probs.argmax(dim=1)  # [N_patch]
        return dominant.detach().cpu().numpy()

   
    def compute_patch_entropy(self, gate_probs):
        entropy = - (gate_probs * torch.log(gate_probs + 1e-9)).sum(dim=1)
        return entropy.detach().cpu().numpy()

    def compute_patch_specialization(self, gate_probs):
        """
        衡量专家是否在不同 patch 上呈现分工
        """
        mean_prob = gate_probs.mean(dim=0, keepdim=True)
        kl = gate_probs * (torch.log(gate_probs + 1e-9) - torch.log(mean_prob + 1e-9))
        kl = kl.sum(dim=1)
        return kl.mean().item()

    def compute_patch_level_gates(self, gate_probs, coords, tokens_per_patch=None):
        """
        将 token-level gate 转为 patch-level gate。
        如果使用 ViT，每个 patch 有 tokens_per_patch 个 token（包括 CLS token）。
        如果 tokens_per_patch=None，则默认每个 patch 用第一个 token (CLS token)。

        参数:
            gate_probs: [T, E]  token-level gate probs
            coords: [N_patch, 2]  WSI patch 坐标
            tokens_per_patch: 每个 patch 的 token 数（可选）

        返回:
            gate_patch: [N_patch, E] 每个 patch 的平均 gate
        """
        T, E = gate_probs.shape
        N_patch = len(coords)

        if tokens_per_patch is None:
            # 直接用 CLS token（假设每个 patch 的第一个 token）
            gate_patch = gate_probs[:N_patch, :]
        else:
            # 如果 token 数匹配 coords × tokens_per_patch
            if T != N_patch * tokens_per_patch:
                print(f"[Warning] token数({T}) != coords数({N_patch})*tokens_per_patch({tokens_per_patch}), fallback to first token per patch")
                gate_patch = gate_probs[:N_patch, :]
            else:
                gate_patch = gate_probs.view(N_patch, tokens_per_patch, E).mean(dim=1)

        return gate_patch

    
    def plot_wsi_spatial_heatmap(self, coords,slide_path,  dominant_expert, save_path=None,patch_size=256):
        """
        coords: [N_patch, 2]  (x,y)
        dominant_expert: [N_patch]
        """
        # 读取 WSI 的缩略图，用作背景
        slide = openslide.OpenSlide(slide_path)
        thumb = slide.get_thumbnail(slide.level_dimensions[-1])  # 获取最小层的缩略图
        thumb_np = np.array(thumb)

        # 坐标映射到缩略图尺寸
        scale_x = thumb_np.shape[1] / slide.dimensions[0]
        scale_y = thumb_np.shape[0] / slide.dimensions[1]

        coords = np.array(coords)
        x = coords[:, 0] * scale_x + patch_size*scale_x/2  # 用 patch 中心
        y = coords[:, 1] * scale_y + patch_size*scale_y/2

        plt.figure(figsize=(10,10))
        plt.imshow(thumb_np)  # 背景图
        scatter = plt.scatter(x, y, c=dominant_expert, cmap="tab20", s=20, alpha=0.6)
        plt.gca().invert_yaxis()
        plt.colorbar(scatter, label="Dominant Expert ID")
        plt.title("WSI Patch-Level Expert Routing Map")
        plt.xlabel("X")
        plt.ylabel("Y")

        if save_path:
            plt.savefig(save_path, dpi=300)

    plt.show()

    def plot_moe_debug(self,gating_probs_list, shared_expert_idx=-1, save_path=None):
        
        #可视化每层 MoE gating 前期分布，突出普通专家（不包含 shared expert）。
        num_layers = len(gating_probs_list)
        plt.figure(figsize=(12, num_layers * 2.5))

        for i, gate_probs in enumerate(gating_probs_list):
            gate_probs = gate_probs.detach().cpu()
            if gate_probs.dim() == 1:
                # 只有一个 expert 或 batch concat 出现1D
                gate_probs_normal = gate_probs.unsqueeze(1)  # 变成 [N_patch, 1]
            else:
                # 多个专家，排除 shared expert
                if shared_expert_idx is not None and shared_expert_idx < gate_probs.shape[1]:
                    mask = torch.arange(gate_probs.shape[1]) != shared_expert_idx
                    gate_probs_normal = gate_probs[:, mask]
                else:
                    gate_probs_normal = gate_probs

            # 平均每个专家激活概率
            mean_probs = gate_probs_normal.mean(dim=0).numpy()

            plt.subplot(num_layers, 1, i + 1)
            sns.barplot(x=np.arange(len(mean_probs)), y=mean_probs)
            plt.ylim(0, 1)
            plt.title(f"Layer {i} - Mean Activation Probabilities (Normal Experts)")
            plt.xlabel("Normal Expert ID")
            plt.ylabel("Mean Prob")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


    def summarize(self, gating_probs_list, save_path=None, coords=None, slide_path=None):
        print("\n[MoE Patch-Level Diagnostics]")
        
        for i, gate_probs in enumerate(gating_probs_list):

            entropy = self.compute_entropy(gate_probs)
            usage, shared_rate = self.compute_expert_usage(gate_probs)
            specialization_score = self.compute_patch_specialization(gate_probs)

            print(f"Layer {i}:")
            print(f"  Mean Entropy: {entropy:.4f}")
            print(f"  Shared Expert Rate: {shared_rate:.4f}")
            print(f"  Patch Specialization Score (KL): {specialization_score:.6f}")

        self.plot_layer_expert_heatmap(gating_probs_list, save_path)


        # 只对第3层做空间可视化
        if coords is not None and slide_path is not None:
            gate_probs = gating_probs_list[2]
            gate_patch = self.compute_patch_level_gates(gating_probs_list[2], coords, tokens_per_patch=None)
            gate_patch_no_shared = gate_patch[:, :-1]  # 去掉 shared expert
            
            dominant = self.compute_patch_dominant_expert(gate_patch_no_shared)
            self.plot_wsi_spatial_heatmap(
                    slide_path=slide_path,
                    coords=coords,
                    dominant_expert=dominant,
                    save_path=None,       # 可选保存路径
                    patch_size=256        # patch 尺寸
                )

       
        # # 收集每层 gating probs
        # layer_all_probs = []
        # for l in range(len(gating_probs_list[0])):  # MoE 层数
        #     layer_probs = [batch_gates[l] for batch_gates in gating_probs_list]
        #     layer_all_probs.append(torch.cat(layer_probs, dim=0))

        # # 调用 debug 可视化
        # self.plot_moe_debug(layer_all_probs)
           