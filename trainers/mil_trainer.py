import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import h5py
import openslide
import numpy as np
from torchvision import transforms
from trainers.moe_analyzer import MoEAnalyzer


class MILTrainer:
    def __init__(self, encoder, mil_model, classifier, dataloader, feature_extractor, device="cuda", moe_train_cfg=None,max_patches=None,use_chunk=True):
        self.device = device
        self.encoder = encoder.to(device)
        self.mil_model = mil_model.to(device)
        self.classifier = classifier.to(device)
        self.dataloader = dataloader
        self.feature_extractor = feature_extractor
        self.criterion = nn.BCEWithLogitsLoss()

        # 参数优化控制（适配 MoE）
        moe_train_cfg = moe_train_cfg or {}
        self.moe_train_cfg = moe_train_cfg
        self.max_patches = max_patches
     
        freeze_encoder = moe_train_cfg.get("freeze_encoder", True)
        train_gate = moe_train_cfg.get("train_gate", True)
        train_shared_expert = moe_train_cfg.get("train_shared_expert", True)

        params_to_optimize = []

        # encoder 参数
        for name, param in self.encoder.named_parameters():
            param.requires_grad = True  # 默认可训练

            # 冻结非 MoE block 参数
            if freeze_encoder and "blocks" in name and "moe" not in name:
                param.requires_grad = False

            # 冻结 gating 网络
            if not train_gate and "gate" in name:
                param.requires_grad = False

            # 冻结 shared-expert
            if not train_shared_expert and "shared_expert" in name:
                param.requires_grad = False

            if param.requires_grad:
                params_to_optimize.append(param)

        # MIL model & classifier 参数全部优化
        #params_to_optimize += list(self.mil_model.parameters()) + list(self.classifier.parameters())
        params_to_optimize += list(self.mil_model.parameters())

        self.optimizer = optim.Adam(params_to_optimize, lr=1e-4)
        # 转换 patch 图像
        self.transform = transforms.Compose([transforms.ToTensor()])

    def train_epoch(self, batch_size=4, max_patches=None,use_chunk=True):
        self.encoder.train()
        self.mil_model.train()
        self.classifier.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # MoE loss配置
        
        moe_loss_enable = self.moe_train_cfg.get("moe_loss_enable", True)
        load_balance_weight = self.moe_train_cfg.get("load_balance_weight", 1.0)
        diversity_weight = self.moe_train_cfg.get("diversity_weight", 0.1)
        eps = 1e-9

        for batch in tqdm(self.dataloader):
            slide_path = batch["slide_path"][0]
            h5_path = batch["h5_path"][0]
            label = torch.tensor([batch["label"]]).float().to(self.device)

            # 读 WSI
            slide = openslide.OpenSlide(slide_path)
            with h5py.File(h5_path, "r") as f:
                coords = f["coords"][:]
            # 随机选择 patch（可选，避免OOM）
            
            if self.max_patches is not None and len(coords) > self.max_patches:
                idxs = np.random.choice(len(coords), size=self.max_patches, replace=False)
                coords = coords[idxs]

            print(f"[INFO] Number of patches used: {len(coords)}")

            # ===== 使用 FeatureExtractor 提取 patch 特征 =====
            patch_feats_list, gating_probs_list = self.feature_extractor.extract_features(
                slide, coords, use_chunk=use_chunk
            )
            slide.close()
           
             # 合并所有 patch 特征 -> 一个 bag
            bag_feat = 0
            for feats_cls in patch_feats_list:
                feats_cls = feats_cls.unsqueeze(0).to(self.device)  # [1, chunk_size, D]
                bag_feat += self.mil_model(feats_cls)

            #out = self.classifier(bag_feat)  # [1, num_classes]
            out = bag_feat.view(-1)           # [1]
            label = label.float().view(-1)    # [1]
            cl_loss = self.criterion(out, label)


            # MoE loss
            moe_loss = 0.0
            if moe_loss_enable:
                num_layers = len(gating_probs_list[0])  # MoE 层数
                for l in range(num_layers):
                    # 收集所有 batch 该层的 gate probs
                    layer_probs = [batch_gates[l] for batch_gates in gating_probs_list]
                    layer_probs = torch.cat(layer_probs, dim=0)  # [总token数, num_experts]
                    # 切除最后一列的 Shared Expert 概率
                    layer_probs = layer_probs[:, :-1]  # 恢复为 [总token数, num_experts]
                    # 保证 shape 至少 2D
                    if layer_probs.dim() == 1:
                        layer_probs = layer_probs.unsqueeze(0)  # [1, E]

                    mean_prob = layer_probs.mean(dim=0)
                    load_loss = torch.sum(mean_prob ** 2)
                    diversity_loss = -torch.sum(mean_prob * torch.log(mean_prob + 1e-9))
                    # 计算当前层的 loss
                    layer_loss = (load_balance_weight * load_loss + diversity_weight * diversity_loss)
                    
                    moe_loss += layer_loss 
                moe_loss = moe_loss.to(self.device) 
            
            # backward & step
            self.optimizer.zero_grad()
            total_loss_step = cl_loss + (moe_loss if moe_loss_enable else 0.0)
            total_loss_step.backward()
            self.optimizer.step()


            total_loss += cl_loss.item() + (moe_loss.item() if moe_loss_enable else 0.0)
            # accuracy
            #preds = torch.argmax(out, dim=1)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).long()
            total_correct += (preds == label.long()).sum().item()
            total_samples += 1

        # ===== MoE 可视化统计 =====
        if moe_loss_enable:
            # 合并每层 gating_probs list -> tensor
            for l in range(len(gating_probs_list)):
                if isinstance(gating_probs_list[l], list):
                    gating_probs_list[l] = torch.cat(gating_probs_list[l], dim=0)
            # 创建 analyzer 并打印 + 画热力图
            analyzer = MoEAnalyzer(
                routing_strategy=self.encoder.routing_strategy,
                top_k=getattr(self.encoder, "top_k", 1),
                threshold=getattr(self.encoder, "threshold", 0.3),
                min_experts=getattr(self.encoder, "min_experts", 1),
                shared_expert_idx=-1  # 最后一列是 shared expert
            )
            analyzer.summarize(gating_probs_list, coords=coords,slide_path=slide_path)

        avg_loss = total_loss / len(self.dataloader)
        acc = total_correct / total_samples
        print(f"[INFO] Average loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        return avg_loss, acc

    def fit(self, num_epochs=10):
        history = {"loss": [], "acc": []}
        for epoch in range(num_epochs):
            print(f"=== Epoch {epoch+1}/{num_epochs} ===")
            loss, acc = self.train_epoch()
            history["loss"].append(loss)
            history["acc"].append(acc)
        return history
