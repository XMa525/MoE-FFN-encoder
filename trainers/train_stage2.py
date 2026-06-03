import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml
import os
import sys
import torchvision.transforms.v2 as T
import numpy as np
from itertools import islice
from datetime import datetime
import random

from models.encoders.moe_encoder import MoEEncoder
from models.distill_teacher.virchow2 import Virchow2FeatureExtractor
from distillation.distiller_stage2 import MoEDistillerStage2
from visualization.distill_visualizer import DistillVisualizer
from distillation.dataset.spider_dataset import SpiderPatchDataset
from distillation.dataset.organ_sampler import OrganBalancedBatchSampler
from distillation.dataset.build_camelyon_stage2_dataset import CamelyonWSIBagDataset
from utils.earlystopping import EarlyStopping
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

# ===================== 日志记录 =====================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"stage2_roleproto_static_train_{timestamp}.log")


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


sys.stdout = sys.stderr = Logger(log_file)
print(f"Logging to {log_file}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(config_path, ckpt_path, device="cuda"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    print("Best model loaded")
    print(f"Current moe_layers_idx = {model.moe_layers_idx}")
    return model, cfg


def print_trainable_params(model):
    total = 0
    trainable = 0
    print("\n===== Trainable Parameters =====")
    for n, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            print(f"[Trainable] {n:80s} {tuple(p.shape)}")
    print(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)")
    print("================================\n")


def build_stage2_optimizer(distiller, base_lr=5e-5, weight_decay=0.05):
    expert_params = []
    routing_proj_params = []
    gate_proto_params = []
    threshold_params = []
    norm_params = []
    other_params = []

    for n, p in distiller.named_parameters():
        if not p.requires_grad:
            continue

        if ".student." in n:
            if ".mlp.experts." in n or ".mlp.shared_expert." in n:
                expert_params.append(p)
            elif ".mlp.gate.routing_proj." in n:
                routing_proj_params.append(p)
            elif ".mlp.gate.gate_vectors" in n or ".mlp.gate.logit_scale" in n:
                gate_proto_params.append(p)
            elif ".mlp.gate.expert_threshold" in n:
                threshold_params.append(p)
            elif ".norm1." in n or ".norm2." in n:
                norm_params.append(p)
            else:
                other_params.append(p)
        else:
            # 比如 proj_l12
            other_params.append(p)

    param_groups = []
    if expert_params:
        param_groups.append({"params": expert_params, "lr": base_lr, "name": "expert"})
    if routing_proj_params:
        param_groups.append({"params": routing_proj_params, "lr": base_lr, "name": "routing_proj"})
    if gate_proto_params:
        param_groups.append({"params": gate_proto_params, "lr": base_lr, "name": "gate_proto"})
    if threshold_params:
        param_groups.append({"params": threshold_params, "lr": base_lr * 0.5, "name": "gate_threshold"})
    if norm_params:
        param_groups.append({"params": norm_params, "lr": base_lr * 0.5, "name": "norm"})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "name": "other"})

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    return optimizer

def resume_stage2_checkpoint(distiller, optimizer, scheduler, ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)

    # ===== MOD A: 优先恢复整个 distiller =====
    if "distiller_state_dict" in ckpt:
        distiller.load_state_dict(ckpt["distiller_state_dict"], strict=False)
        print("Loaded full distiller_state_dict")
    elif "student_state_dict" in ckpt:
        distiller.student.load_state_dict(ckpt["student_state_dict"], strict=True)
        print("Loaded student_state_dict only")
    else:
        raise KeyError("No distiller_state_dict or student_state_dict found in checkpoint")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    start_epoch = ckpt.get("epoch", -1) + 1

    print(f"Resumed from: {ckpt_path}")
    print(f"Start epoch: {start_epoch}")
    if "train_loss" in ckpt:
        print(f"Previous train_loss: {ckpt['train_loss']}")
    if "val_loss" in ckpt:
        print(f"Previous val_loss: {ckpt['val_loss']}")
    if "wsi_val_loss" in ckpt:
        print(f"Previous wsi_val_loss: {ckpt['wsi_val_loss']}")
    if "combined_val_loss" in ckpt:
        print(f"Previous combined_val_loss: {ckpt['combined_val_loss']}")

    return start_epoch

def main():
    seed = 42
    set_seed(seed)
    print(f"Using random seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_path = "configs/stage2.yaml"
    stage1_ckpt_path = "results/distilled_best_model/moe_encoder_best.pth"   

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    val_max_batches = cfg["stage2_train"].get("val_max_batches", 200)

    # ================= 1. Teacher =================
    print("Loading Teacher Model (Virchow2)...")
    teacher_wrapper = Virchow2FeatureExtractor(device=device)
    teacher_model = teacher_wrapper.model
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    try:
        teacher_model = torch.compile(teacher_model, mode="reduce-overhead")
    except Exception as e:
        print(f"⚠️ torch.compile failed, fallback to eager: {e}")

    # ================= 2. Student from Stage1 =================
    print("Loading Student Model from Stage1 checkpoint...")
    student, cfg_loaded = load_model(config_path, stage1_ckpt_path, device=device)

    # ================= 3. Stage2 Distiller =================
    distiller = MoEDistillerStage2(
        student_model=student,
        teacher_model=teacher_model,
        stu_dim=384,
        tea_dim=1280,
        grid_size=16,
        stage2_cfg=cfg["stage2_train"]
    ).to(device)

    print_trainable_params(distiller)

    print("[RoleProto] weight =", cfg["stage2_train"].get("role_proto_weight", 0.0))
    print("[RoleProto] dir    =", cfg["stage2_train"].get("role_proto_dir", None))
    print("[RoleProto] free expert id =", cfg["stage2_train"].get("free_expert_id", 3))
    # ================= 4. Optimizer =================
    base_lr = float(cfg["stage2_train"].get("lr", 5e-5))
    weight_decay = cfg["stage2_train"].get("weight_decay", 0.05)
    optimizer = build_stage2_optimizer(distiller, base_lr=base_lr, weight_decay=weight_decay)


    epochs = int(cfg["stage2_train"].get("epochs", 15))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    clip_grad_norm = cfg["stage2_train"].get("clip_grad_norm", 1.0)

    # ================= 5. Dataset =================
    train_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ])
    val_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])

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

    assert len(train_full_dataset) == len(val_full_dataset)
    N = len(train_full_dataset)

    train_ratio = 0.8
    train_size = int(train_ratio * N)
    indices = np.arange(N)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    batch_sampler = OrganBalancedBatchSampler(
        train_dataset,
        batch_size=128,
        cache_path="organ_indices_t015_aligned.pkl"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    print(f"Batches per epoch: {len(train_loader)}")

    # ================= 5.5 WSI bag-level dataset =================
    wsi_train_dataset = CamelyonWSIBagDataset(
        csv_path="../data/CAMELYON17/stage2_wsi_train.csv",
        raw_dir="../data/CAMELYON17/images",
        h5_dir="../data/CAMELYON17/patches/patches",
        patch_size=256,
        read_level=0,
        resize_to=224,
        max_patches=64,            # 第一版保守一点
        sample_mode="random",
        seed=42,
        return_pil=False,
        transform=T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ]),
        check_files=True,
    )

    wsi_val_dataset = CamelyonWSIBagDataset(
        csv_path="../data/CAMELYON17/stage2_wsi_val.csv",
        raw_dir="../data/CAMELYON17/images",
        h5_dir="../data/CAMELYON17/patches/patches",
        patch_size=256,
        read_level=0,
        resize_to=224,
        max_patches=64,
        sample_mode="random",
        seed=42,
        return_pil=False,
        transform=T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ]),
        check_files=True,
    )
    def wsi_bag_collate_fn(batch):
        return batch[0]
    wsi_train_loader = DataLoader(
        wsi_train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=wsi_bag_collate_fn,
    )

    wsi_val_loader = DataLoader(
        wsi_val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=wsi_bag_collate_fn,
    )

    # ================= 6. Train utils =================
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    visualizer = DistillVisualizer()

    ckpt_dir = "distill_checkpoints_stage2_roleproto_static"
    os.makedirs(ckpt_dir, exist_ok=True)

    early_stopping = EarlyStopping(
        patience=5,
        min_delta=1e-4,
        save_path=os.path.join(ckpt_dir, "moe_encoder_stage2_best.pth"),
    )
    best_full_path = os.path.join(ckpt_dir, "best_full.pth")
    best_metric = float("inf")

    #恢复训练
    resume_ckpt = os.path.join(ckpt_dir, "latest.pth")
    start_epoch = 0

    if args.resume and os.path.exists(resume_ckpt):
        start_epoch = resume_stage2_checkpoint(
            distiller=distiller,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=resume_ckpt,
            device=device
        )
    else:
        print("start from scratch.")
    # ================= 7. Training loop =================
    #epochs = cfg["stage2_train"].get("epochs", 15)

    print("\n🔥 Start Stage2 Specialization Training")

    for epoch in range(start_epoch, epochs):
        # ===== MOD 1: epoch 开头初始化 WSI 相关 iterator 和配置 =====
        wsi_train_iter = iter(wsi_train_loader)
        bag_interval = 5
        wsi_patch_batch_size = 8
        wsi_bag_loss_weight = float(cfg["stage2_train"].get("wsi_bag_loss_weight", 0.1))
        use_wsi_bag_loss = bool(cfg["stage2_train"].get("use_wsi_bag_loss", False))

        # ---------------- train ----------------
        distiller.train()

        # ===== MOD 2: patch / wsi loss 分开统计 =====
        epoch_loss_patch = 0.0
        epoch_loss_wsi = 0.0

        step_count = 0
        patch_metric_steps = 0
        wsi_metric_steps = 0

        running_patch_loss = {}
        running_wsi_loss = {}

        for images, organs, offline_cluster_ids in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images = images.to(device, non_blocking=True)
            offline_cluster_ids = offline_cluster_ids.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                loss, loss_dict, gate_info_list = distiller(
                    images,
                    offline_cluster_ids=offline_cluster_ids,
                    is_eval=False,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(distiller.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            # ===== MOD 3: patch loss 单独统计 =====
            epoch_loss_patch += loss.detach().float().cpu().item()
            patch_metric_steps += 1

            for k, v in loss_dict.items():
                running_patch_loss[k] = running_patch_loss.get(k, 0.0) + float(v)

            # ===== MOD 4: bag_interval 判断改成 (step_count + 1) % bag_interval == 0 =====
            if use_wsi_bag_loss and ((step_count + 1) % bag_interval == 0):
                try:
                    wsi_batch = next(wsi_train_iter)
                except StopIteration:
                    wsi_train_iter = iter(wsi_train_loader)
                    wsi_batch = next(wsi_train_iter)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda"):
                    loss_wsi, wsi_stats = distiller.compute_wsi_bag_loss(
                        images=wsi_batch["images"],
                        slide_label=wsi_batch["label"],
                        patch_batch_size=wsi_patch_batch_size,
                        is_eval=False,
                    )
                    loss_wsi_total = wsi_bag_loss_weight * loss_wsi

                scaler.scale(loss_wsi_total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(distiller.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                # ===== MOD 5: wsi loss / stats 单独统计 =====
                epoch_loss_wsi += loss_wsi_total.detach().float().cpu().item()
                wsi_metric_steps += 1

                for k, v in wsi_stats.items():
                    running_wsi_loss[k] = running_wsi_loss.get(k, 0.0) + float(v)
                running_wsi_loss["wsi_bag_loss_weighted"] = (
                    running_wsi_loss.get("wsi_bag_loss_weighted", 0.0)
                    + float(loss_wsi_total.detach().cpu())
                )

            step_count += 1

            if step_count % 10 == 0:
                last_info = gate_info_list[-1]
                dispatch_weight = last_info["dispatch_weight"].detach()
                dispatch_mask = last_info["dispatch_mask"].detach()

                entropy = -(dispatch_weight * torch.log(dispatch_weight + 1e-8)).sum(dim=-1).mean().item()
                usage = dispatch_mask.float().mean(dim=0).detach().cpu().numpy()

                visualizer.update(
                    loss_dict,
                    entropy=entropy,
                    expert_usage=usage,
                    mode="train"
                )

        # ===== MOD 6: patch / wsi 分开平均，不要混在一起 =====
        avg_patch_train_loss = epoch_loss_patch / max(step_count, 1)
        avg_wsi_train_loss = epoch_loss_wsi / max(wsi_metric_steps, 1) if wsi_metric_steps > 0 else 0.0

        avg_patch_loss_dict = {
            k: v / max(patch_metric_steps, 1)
            for k, v in running_patch_loss.items()
        }
        avg_wsi_loss_dict = {
            k: v / max(wsi_metric_steps, 1)
            for k, v in running_wsi_loss.items()
        } if wsi_metric_steps > 0 else {}

        print(f"[Train][Patch] Epoch {epoch+1} Loss detail: {avg_patch_loss_dict}")
        if wsi_metric_steps > 0:
            print(f"[Train][WSI]   Epoch {epoch+1} Loss detail: {avg_wsi_loss_dict}")

        # ---------------- val ----------------
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
                        offline_cluster_ids=offline_cluster_ids,
                        is_eval=True,
                    )

                val_loss += loss.item()
                val_steps += 1

                last_info = gate_info_list[-1]
                dispatch_weight = last_info["dispatch_weight"].detach()
                dispatch_mask = last_info["dispatch_mask"].detach()

                entropy = -(dispatch_weight * torch.log(dispatch_weight + 1e-8)).sum(dim=-1).mean().item()
                usage = dispatch_mask.float().mean(dim=0).detach().cpu().numpy()

                visualizer.update(
                    loss_dict,
                    entropy=entropy,
                    expert_usage=usage,
                    mode="val"
                )

        avg_val_loss = val_loss / max(val_steps, 1)

        # ---------------- WSI bag val ----------------
        if use_wsi_bag_loss:
            wsi_val_loss = 0.0
            wsi_val_steps = 0

            with torch.no_grad():
                for wsi_batch in tqdm(wsi_val_loader, desc=f"Epoch {epoch+1} [WSI Val]"):
                    with torch.amp.autocast("cuda"):
                        loss_wsi, wsi_stats = distiller.compute_wsi_bag_loss(
                            images=wsi_batch["images"],
                            slide_label=wsi_batch["label"],
                            patch_batch_size=8,
                            is_eval=True,
                        )

                    wsi_val_loss += loss_wsi.item()
                    wsi_val_steps += 1

            avg_wsi_val_loss = wsi_val_loss / max(wsi_val_steps, 1)
            print(f"[WSI Val] Epoch {epoch+1} Loss: {avg_wsi_val_loss:.6f}")
        else:
            avg_wsi_val_loss = 0.0

        # ===== MOD 7: combined val loss，用于 early stopping / checkpoint 观察 =====
        combined_val_loss = avg_val_loss + wsi_bag_loss_weight * avg_wsi_val_loss

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Patch Train Loss: {avg_patch_train_loss:.6f} | "
            f"WSI Train Loss: {avg_wsi_train_loss:.6f} | "
            f"Patch Val Loss: {avg_val_loss:.6f} | "
            f"WSI Val Loss: {avg_wsi_val_loss:.6f} | "
            f"Combined Val: {combined_val_loss:.6f}"
        )

        scheduler.step()

        # ---------------- save ----------------
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "student_state_dict": distiller.student.state_dict(),
            "distiller_state_dict": distiller.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "cfg": cfg,
            "train_patch_loss": avg_patch_train_loss,
            "train_wsi_loss": avg_wsi_train_loss,
            "val_loss": avg_val_loss,
            "wsi_val_loss": avg_wsi_val_loss,
            "combined_val_loss": combined_val_loss,
        }, ckpt_path)

        # ===== MOD 8: latest.pth 也保存 distiller_state_dict =====
        torch.save({
            "epoch": epoch,
            "student_state_dict": distiller.student.state_dict(),
            "distiller_state_dict": distiller.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_patch_loss": avg_patch_train_loss,
            "train_wsi_loss": avg_wsi_train_loss,
            "val_loss": avg_val_loss,
            "wsi_val_loss": avg_wsi_val_loss,
            "combined_val_loss": combined_val_loss,
        }, os.path.join(ckpt_dir, "latest.pth"))

        print(f"Checkpoint saved: {ckpt_path}")
        # ---------------- save best full checkpoint ----------------
        if combined_val_loss < best_metric:
            best_metric = combined_val_loss
            torch.save({
                "epoch": epoch,
                "student_state_dict": distiller.student.state_dict(),
                "distiller_state_dict": distiller.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cfg": cfg,
                "train_patch_loss": avg_patch_train_loss,
                "train_wsi_loss": avg_wsi_train_loss,
                "val_loss": avg_val_loss,
                "wsi_val_loss": avg_wsi_val_loss,
                "combined_val_loss": combined_val_loss,
            }, best_full_path)
            print(f"✅ Best full checkpoint updated: {best_full_path} (combined_val_loss={combined_val_loss:.6f})")

        # ---------------- early stop ----------------
        # ===== MOD 9: early stopping 改盯 combined_val_loss =====
        early_stopping(combined_val_loss, distiller.student)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

    print("Stage2 Training Complete.")
    visualizer.summarize()


if __name__ == "__main__":
    main()