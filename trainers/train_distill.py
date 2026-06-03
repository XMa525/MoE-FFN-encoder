#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml
import os
import sys
from collections import Counter
import torchvision.transforms.v2 as T
import numpy as np
from itertools import islice
from datetime import datetime
import random

# 导入你的模块 (根据你的实际路径调整)
from models.encoders.moe_encoder import MoEEncoder
from models.encoders.moe_FFN import MoEFFN
from models.distill_teacher.virchow2 import Virchow2FeatureExtractor
from distillation.distiller import MoEDistiller
from .moe_analyzer import MoEAnalyzer
from visualization.distill_visualizer import DistillVisualizer
from scripts.check_freeze import full_freeze_report
from distillation.dataset.spider_dataset import SpiderPatchDataset
from distillation.dataset.organ_sampler import OrganBalancedBatchSampler
from utils.earlystopping import EarlyStopping


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage1 MoE-DINOv2 distillation training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase2.yaml",
        help="Path to yaml config.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        required=True,
        help="Experiment name. Used for log/checkpoint directory.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Optional override for cfg['training']['num_epochs'].",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from distill_checkpoints/{exp_name}/latest.pth if available.",
    )
    return parser.parse_args()


def setup_logger(exp_name: str):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{exp_name}_{timestamp}.log")

    sys.stdout = sys.stderr = Logger(log_file)
    print(f"Logging to {log_file}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证 DataLoader / CUDA 更稳定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_gradients(model):
    """
    打印哪些参数在参与训练，以及梯度大小
    """
    print("\n===== Gradient Check =====")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.grad is None:
            print(f"[NO GRAD] {name}")
        else:
            grad_norm = param.grad.norm().item()
            print(f"[GRAD] {name:60s} | grad_norm={grad_norm:.6e}")

    print("==========================\n")


def check_param_update(model, snapshot):
    print("\n===== Parameter Update Check =====")

    for name, param in model.named_parameters():
        if name not in snapshot:
            continue

        diff = (param.detach() - snapshot[name]).abs().mean().item()

        if diff > 0:
            print(f"[UPDATED] {name:60s} | mean_change={diff:.6e}")
        else:
            print(f"[NOT UPDATED] {name}")

    print("==========================\n")


def set_routing_schedule(model, min_experts, max_experts):
    for blk in model.blocks:
        if hasattr(blk, "mlp") and hasattr(blk.mlp, "gate"):
            blk.mlp.gate.min_experts = min_experts
            blk.mlp.gate.max_experts = max_experts


def main():
    args = parse_args()
    setup_logger(args.exp_name)

    seed = 42
    set_seed(seed)

    print(f"Using random seed: {seed}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    print(f"[Config] {args.config}")
    print(f"[Exp] {args.exp_name}")
    print(f"[moe_encoder.num_experts] {cfg['moe_encoder'].get('num_experts')}")
    print(f"[moe_loss.group_guided_num_experts] {cfg['moe_loss'].get('group_guided_num_experts', None)}")

    max_steps = 5000
    val_max_batches = 100

    # ================= 1. 初始化模型 =================
    print("Loading Teacher Model (Virchow2)...")
    teacher_wrapper = Virchow2FeatureExtractor(device=device)
    teacher_model = teacher_wrapper.model
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    try:
        teacher_model = torch.compile(teacher_model, mode="reduce-overhead")
    except Exception as e:
        print(f"⚠️ torch.compile 失败，降级回普通模式: {e}")

    print("Loading Student Model (MoE-DINOv2)...")
    student = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"]).to(device)

    # ================= 2. 初始化蒸馏器 =================
    distiller = MoEDistiller(
        student_model=student,
        teacher_model=teacher_model,
        stu_dim=384,
        tea_dim=1280,
        grid_size=16,  # 224//14 = 16
        moe_train_cfg=cfg["moe_loss"],
    ).to(device)

    # 检查 student 的解冻情况
    # full_freeze_report(distiller.student)

    # ================= 3. 优化器设置 =================
    trainable_params = [p for p in distiller.parameters() if p.requires_grad]

    gate_proto_params = [
        p for n, p in distiller.student.named_parameters()
        if ".mlp.gate.gate_vectors" in n or ".mlp.gate.logit_scale" in n
    ]
    gate_proto_set = set(id(p) for p in gate_proto_params)

    threshold_params = [
        p for n, p in distiller.student.named_parameters()
        if ".mlp.gate.expert_threshold" in n
    ]
    threshold_set = set(id(p) for p in threshold_params)

    other_params = [
        p for p in trainable_params
        if id(p) not in gate_proto_set and id(p) not in threshold_set
    ]

    base_lr = float(cfg.get("training", {}).get("lr", 1e-4))

    optimizer = optim.AdamW(
        [
            {"params": other_params, "lr": base_lr, "name": "other"},
            {"params": gate_proto_params, "lr": base_lr, "name": "gate_proto"},
            {"params": threshold_params, "lr": base_lr, "name": "gate_threshold"},
        ],
        weight_decay=0.05,
    )

    # Gradient clipping threshold
    clip_grad_norm = 1.0

    # 打印一下真正参与训练的参数量，确保冻结策略生效
    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {total_trainable / 1e6:.2f} M")

    # ================= 4. 数据加载 =================
    train_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        # T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        # T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        # T.ToTensor()
    ])

    val_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        # T.Resize((224, 224)),
        # T.ToTensor()
    ])

    print("Step 1: Dataset initialized")

    train_full_dataset = SpiderPatchDataset(
        root="../data/raw",
        transform=train_transform,
        cluster_cache_path="outputs/token_clustering_layer24_fullassign/path_to_cluster_ids.pkl",
        num_patch_tokens=256,
        missing_cluster_mode="error",
        enable_tissue_filter=True,
        white_threshold=0.85,
        tissue_threshold=0.15,
        samples_cache_path="outputs/dataset_cache/samples_t015.pkl",
        rebuild_samples_cache=False,
    )

    val_full_dataset = SpiderPatchDataset(
        root="../data/raw",
        transform=val_transform,
        cluster_cache_path="outputs/token_clustering_layer24_fullassign/path_to_cluster_ids.pkl",
        num_patch_tokens=256,
        missing_cluster_mode="error",
        enable_tissue_filter=True,
        white_threshold=0.85,
        tissue_threshold=0.15,
        samples_cache_path="outputs/dataset_cache/samples_t015.pkl",
        rebuild_samples_cache=False,
    )

    # 2. 先确认两份 dataset 长度一致
    assert len(train_full_dataset) == len(val_full_dataset)

    # 更稳：确认样本路径顺序一致
    for i in range(10):
        assert train_full_dataset.samples[i][0] == val_full_dataset.samples[i][0]

    # 3. 再切 index
    N = len(train_full_dataset)
    train_ratio = 0.8
    train_size = int(train_ratio * N)
    val_size = N - train_size

    indices = np.arange(N)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 4. 用同一组 index 去切两份 dataset
    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    print("Step 2: Creating batch sampler")
    batch_sampler = OrganBalancedBatchSampler(
        train_dataset,
        batch_size=128,
        cache_path="organ_indices_t015_aligned.pkl",
    )
    print("Batch sampler ready")

    print("Step 3: Creating DataLoader")
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    print("DataLoader ready, starting iteration")

    # 验证集 Dataloader (普通的顺序采样)
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    print(f"Batches per epoch: {len(train_loader)}")

    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # 初始化可视化工具
    analyzer = MoEAnalyzer(shared_expert_idx=-1)
    visualizer = DistillVisualizer()

    # 训练开始前保留参数
    param_snapshot = {
        name: p.clone().detach()
        for name, p in student.named_parameters()
        if p.requires_grad
    }

    # ====== checkpoint 配置 ======
    ckpt_dir = os.path.join("distill_checkpoints", args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[Checkpoint dir] {ckpt_dir}")

    routing_history = []

    # 初始化早停类
    early_stopping = EarlyStopping(
        patience=5,
        min_delta=1e-4,
        save_path=os.path.join(ckpt_dir, "moe_encoder_best.pth"),
        # plateau_patience=3,
        # eps=1e-2,
    )

    # ================= 6. 恢复训练 =================
    resume_ckpt = os.path.join(ckpt_dir, "latest.pth")
    start_epoch = 0

    if args.resume and os.path.exists(resume_ckpt):
        print(f"⚡ Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        distiller.student.load_state_dict(ckpt["student_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1

    # ================= 5. 训练循环 =================
    epochs = int(cfg.get("training", {}).get("num_epochs", 10))
    if args.num_epochs is not None:
        epochs = int(args.num_epochs)

    print(f"[Training epochs] {epochs}")
    print(f"[Base LR] {base_lr}")

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )

    warmup_epochs = 3
    routing_sim_start_epoch = 3
    threshold_freeze_epochs = 6
    routing_warm_epochs = 2

    target_mask_ratio = 0.3
    init_mask_ratio = 0.15
    min_distill_weight = 0.3
    max_distill_weight = 1.0

    print("\n🔥 Start Distillation Training")

    for epoch in range(start_epoch, epochs):
        distiller.train()
        epoch_loss = 0.0
        step_count = 0
        running_loss = {}

        for name, param in distiller.student.named_parameters():
            if ".mlp.gate.expert_threshold" in name:
                param.requires_grad = epoch >= threshold_freeze_epochs

        # ===== Warm-up 学习率 & mask ratio & 蒸馏权重 =====
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = base_lr * lr_scale

            mask_ratio = init_mask_ratio + (target_mask_ratio - init_mask_ratio) * lr_scale
            distill_weight = min_distill_weight + (max_distill_weight - min_distill_weight) * lr_scale
        else:
            mask_ratio = target_mask_ratio
            distill_weight = max_distill_weight

        # ===== Routing schedule =====
        # if epoch < routing_warm_epochs:
        #     cur_min_experts = 2
        #     cur_max_experts = 2
        # else:
        #     cur_min_experts = 1
        #     cur_max_experts = 2
        # set_routing_schedule(distiller.student, cur_min_experts, cur_max_experts)
        # print(
        #     f"[RoutingSchedule] epoch={epoch+1}, "
        #     f"min_experts={cur_min_experts}, max_experts={cur_max_experts}"
        # )

        for images, organs, offline_cluster_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device, non_blocking=True)
            offline_cluster_ids = offline_cluster_ids.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                # Forward distillation
                loss, loss_dict, gate_info_list = distiller(
                    images,
                    mask_ratio=mask_ratio,
                    epoch=epoch,
                    routing_sim_start_epoch=routing_sim_start_epoch,
                    is_eval=False,
                    offline_cluster_ids=offline_cluster_ids,
                )

            total_loss = distill_weight * loss
            scaler.scale(total_loss).backward()

            # 🔍 调试：检查梯度
            # check_gradients(student)
            # check_param_update(student, param_snapshot)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(distiller.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.detach().float().cpu().item()
            step_count += 1

            for k, v in loss_dict.items():
                running_loss[k] = running_loss.get(k, 0) + float(v)

            # MoE routing 统计
            if step_count % 10 == 0:
                last_info = gate_info_list[-1]
                dispatch_weight = last_info["dispatch_weight"].detach()
                dispatch_mask = last_info["dispatch_mask"].detach()
                active_counts = last_info["active_counts"].detach()

                entropy = -(dispatch_weight * torch.log(dispatch_weight + 1e-8)).sum(dim=-1).mean().item()
                usage = dispatch_mask.float().mean(dim=0).detach().cpu().numpy()
                avg_k = active_counts.float().mean().item()

                visualizer.update(
                    loss_dict,
                    entropy=entropy,
                    expert_usage=usage,
                    mode="train",
                )
                print(f"[train] avg_active_experts={avg_k:.4f}")

        avg_loss = epoch_loss / max(step_count, 1)
        avg_loss_dict = {k: v / step_count for k, v in running_loss.items()}

        print(f"Loss detail: {avg_loss_dict}")

        # ===== 验证阶段 (用于早停监控) =====
        distiller.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch_idx, (images, organs, offline_cluster_ids) in enumerate(
                tqdm(islice(val_loader, val_max_batches), desc=f"Epoch {epoch+1} [Val]")
            ):
                if batch_idx >= val_max_batches:
                    break

                images = images.to(device, non_blocking=True)
                offline_cluster_ids = offline_cluster_ids.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    loss, loss_dict, gate_info_list = distiller(
                        images,
                        mask_ratio=target_mask_ratio,
                        is_eval=True,
                        epoch=epoch,
                        routing_sim_start_epoch=routing_sim_start_epoch,
                        offline_cluster_ids=offline_cluster_ids,
                    )

                val_loss += loss.item()
                val_steps += 1

                last_info = gate_info_list[-1]
                dispatch_weight = last_info["dispatch_weight"].detach()
                dispatch_mask = last_info["dispatch_mask"].detach()
                active_counts = last_info["active_counts"].detach()

                entropy = -(dispatch_weight * torch.log(dispatch_weight + 1e-8)).sum(dim=-1).mean().item()
                usage = dispatch_mask.float().mean(dim=0).detach().cpu().numpy()
                avg_k = active_counts.float().mean().item()

                visualizer.update(
                    loss_dict,
                    entropy=entropy,
                    expert_usage=usage,
                    mode="val",
                )

        avg_val_loss = val_loss / max(val_steps, 1)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        scheduler.step()

        # ===== 保存 epoch checkpoint =====
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
        torch.save(
            {
                "epoch": epoch,
                "student_state_dict": distiller.student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "routing_history": routing_history,
            },
            ckpt_path,
        )

        # 最新 checkpoint
        torch.save(
            {
                "epoch": epoch,
                "student_state_dict": distiller.student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(ckpt_dir, "latest.pth"),
        )

        print(f"Checkpoint saved: {ckpt_path}")

        # ===== 触发早停机制 =====
        early_stopping(avg_val_loss, distiller.student)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}! Training stopped.")
            break

        # ================= Epoch-level Routing/Usage summary =================
        if step_count > 0:
            last_info = gate_info_list[-1]
            dispatch_weight = last_info["dispatch_weight"].detach()
            dispatch_mask = last_info["dispatch_mask"].detach()
            active_counts = last_info["active_counts"].detach()

            entropy = -(dispatch_weight * torch.log(dispatch_weight + 1e-8)).sum(dim=-1).mean().item()
            usage = dispatch_mask.float().mean(dim=0).detach().cpu().numpy()
            avg_k = active_counts.float().mean().item()

            print(f"Routing entropy: {entropy:.4f}, Expert usage: {usage}, Avg active experts: {avg_k:.4f}")

    # ================= 6. 保存权重 =================
    print("Training Complete. Saving Student weights...")

    visualizer.summarize()

    # eval时
    # analyzer.summarize(gating_probs_list)


if __name__ == "__main__":
    main()