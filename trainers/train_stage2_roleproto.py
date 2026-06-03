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
import pandas as pd
import argparse
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.encoders.moe_encoder import MoEEncoder
from models.distill_teacher.virchow2 import Virchow2FeatureExtractor
from distillation.distiller_stage2 import MoEDistillerStage2
from visualization.distill_visualizer import DistillVisualizer
from distillation.dataset.tcga_stage2_dataset import (
    TCGARolePatchDataset,
    tcga_stage2_collate_fn,
    build_tcga_stage2_collate_fn,
    canonicalize_path
)
from distillation.dataset.slide_label_balanced_sampler import SlideLabelBalancedBatchSampler
from distillation.dataset.tcga_stage2_sampler import ProjectBalancedBatchSampler
from utils.earlystopping import EarlyStopping


parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
parser.add_argument("--config", type=str, default="configs/stage2_roleproto.yaml")
parser.add_argument("--pool-csv", type=str,
                    default="analysis_outputs/tcga_conch_semantic_outputs/tcga_stage2_train_pool_with_slide_label.csv")
parser.add_argument("--stage1-ckpt", type=str,
                    default="results/distilled_best_model/moe_encoder_best.pth")
args = parser.parse_args()


# ===================== logging =====================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"stage2_tcga_roleproto_{timestamp}.log")


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
    #model.load_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)

    allowed_missing = [
        k for k in missing
        if ".mlp.patch_context_proj." in k
    ]

    real_missing = [k for k in missing if k not in allowed_missing]
    if len(real_missing) > 0:
        raise RuntimeError(f"Unexpected missing keys when loading ckpt: {real_missing}")
    if len(unexpected) > 0:
        raise RuntimeError(f"Unexpected keys in ckpt: {unexpected}")
    print("[load_model] allowed missing keys:", allowed_missing)
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
    patch_context_params = []
    gate_proto_params = []
    threshold_params = []
    norm_params = []
    other_params = []

    for n, p in distiller.named_parameters():
        if not p.requires_grad:
            continue

        if n.startswith("student."):
            if ".mlp.experts." in n or ".mlp.shared_expert." in n:
                expert_params.append(p)
            elif ".mlp.gate.routing_proj." in n:
                routing_proj_params.append(p)
            elif ".mlp.patch_context_proj." in n:
                patch_context_params.append(p)
            elif ".mlp.gate.gate_vectors" in n or ".mlp.gate.logit_scale" in n:
                gate_proto_params.append(p)
            elif ".mlp.gate.expert_threshold" in n:
                threshold_params.append(p)
            elif ".norm1." in n or ".norm2." in n:
                norm_params.append(p)
            else:
                other_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if expert_params:
        param_groups.append({"params": expert_params, "lr": base_lr, "name": "expert"})
    if routing_proj_params:
        param_groups.append({"params": routing_proj_params, "lr": base_lr, "name": "routing_proj"})
    if patch_context_params:
        param_groups.append({"params": patch_context_params, "lr": base_lr, "name": "patch_context"})
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
    return start_epoch


def build_train_val_indices_by_slide(df: pd.DataFrame, val_ratio: float, seed: int):
    if "slide_id" not in df.columns:
        raise ValueError("slide_id column is required for slide-level split")

    rng = np.random.default_rng(seed)

    slide_ids = df["slide_id"].astype(str).unique()
    slide_ids = np.array(slide_ids)
    rng.shuffle(slide_ids)

    val_num_slides = int(round(len(slide_ids) * val_ratio))
    val_slide_ids = set(slide_ids[:val_num_slides].tolist())
    train_slide_ids = set(slide_ids[val_num_slides:].tolist())

    train_indices = df.index[df["slide_id"].astype(str).isin(train_slide_ids)].to_numpy()
    val_indices = df.index[df["slide_id"].astype(str).isin(val_slide_ids)].to_numpy()

    return train_indices, val_indices, train_slide_ids, val_slide_ids

def update_running_dict(running_dict, cur_dict):
    for k, v in cur_dict.items():
        running_dict[k] = running_dict.get(k, 0.0) + float(v)


def average_running_dict(running_dict, denom):
    return {k: v / max(denom, 1) for k, v in running_dict.items()}

def main():
    seed = 42
    set_seed(seed)
    print(f"Using random seed: {seed}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    config_path = args.config
    stage1_ckpt_path = args.stage1_ckpt

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    val_max_batches = int(cfg["stage2_train"].get("val_max_batches", 200))

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

    print("[WSI Bag] use_wsi_bag_loss       =", cfg["stage2_train"].get("use_wsi_bag_loss", False))
    print("[WSI Bag] wsi_bag_loss_weight    =", cfg["stage2_train"].get("wsi_bag_loss_weight", 0.0))
    print("[WSI Bag] use_wsi_bag_margin_loss=", cfg["stage2_train"].get("use_wsi_bag_margin_loss", False))
    print("[WSI Bag] wsi_bag_margin_weight  =", cfg["stage2_train"].get("wsi_bag_margin_weight", 0.0))
    print("[WSI Bag] wsi_bag_margin         =", cfg["stage2_train"].get("wsi_bag_margin", 0.10))
    print("[WSI Bag] wsi_topk_ratio         =", cfg["stage2_train"].get("wsi_topk_ratio", 0.1))
    print("[WSI Bag] wsi_patch_batch_size   =", cfg["stage2_train"].get("wsi_patch_batch_size", 8))

    # ================= 4. Optimizer =================
    base_lr = float(cfg["stage2_train"].get("lr", 5e-5))
    weight_decay = float(cfg["stage2_train"].get("weight_decay", 0.05))
    optimizer = build_stage2_optimizer(distiller, base_lr=base_lr, weight_decay=weight_decay)

    epochs = int(cfg["stage2_train"].get("epochs", 15))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    clip_grad_norm = float(cfg["stage2_train"].get("clip_grad_norm", 1.0))

    # ================= 5. Dataset =================
    train_transform = T.Compose([
        T.ToImage(),
        T.Resize((224, 224), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ])
    val_transform = T.Compose([
        T.ToImage(),
        T.Resize((224, 224), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])

    print(f"[Dataset] Loading pool csv: {args.pool_csv}")
    full_df = pd.read_csv(args.pool_csv)
    full_df["svs_path"] = full_df["svs_path"].map(canonicalize_path)

    if "prefilter_white" in full_df.columns:
        full_df = full_df[full_df["prefilter_white"].fillna(0).astype(int) == 0].copy()

    full_df = full_df.reset_index(drop=True)

    print(f"[Dataset] total rows after prefilter = {len(full_df)}")
    if "project" in full_df.columns:
        print("[Dataset] project counts:")
        print(full_df["project"].value_counts())
    if "pred_label" in full_df.columns:
        print("[Dataset] label counts:")
        print(full_df["pred_label"].value_counts())

    if "slide_id" not in full_df.columns:
        full_df["slide_id"] = full_df["svs_path"].astype(str)

    train_indices, val_indices, train_slide_ids, val_slide_ids = build_train_val_indices_by_slide(
        df=full_df,
        val_ratio=float(cfg["stage2_train"].get("val_ratio", 0.2)),
        seed=seed,
    )

    print(f"[Split] #train slides = {len(train_slide_ids)}")
    print(f"[Split] #val slides   = {len(val_slide_ids)}")
    print(f"[Split] #train rows   = {len(train_indices)}")
    print(f"[Split] #val rows     = {len(val_indices)}")
    print(f"[Split] slide overlap = {len(train_slide_ids & val_slide_ids)}")

    train_dataset = TCGARolePatchDataset(
        csv_path=args.pool_csv,
        transform=train_transform,
        indices=train_indices,
        filter_prefilter_white=True,
        verbose=True,
        use_wsi_bag_sampling=bool(
            cfg["stage2_train"].get("use_wsi_bag_sampling", False)
            or cfg["stage2_train"].get("use_wsi_bag_loss", False)
            or cfg["stage2_train"].get("use_wsi_bag_margin_loss", False)
            or cfg["stage2_train"].get("use_neg_global_topk_suppression", False)
        ),
        wsi_bag_size=int(cfg["stage2_train"].get("wsi_bag_size", 64)),
        wsi_min_bag_size=int(cfg["stage2_train"].get("wsi_min_bag_size", 8)),
        slide_label_col=cfg["stage2_train"].get("slide_label_col", "slide_label"),
        random_seed=seed,
        use_spatial_neighbor_sampling=bool(
            cfg["stage2_train"].get("use_spatial_neighbor_sampling", False)
        ),
        spatial_neighbor_csv=cfg["stage2_train"].get("spatial_neighbor_csv", None),
        spatial_neighbor_max_k=int(
            cfg["stage2_train"].get("spatial_neighbor_max_k", 8)
        ),
    )

    val_dataset = TCGARolePatchDataset(
        csv_path=args.pool_csv,
        transform=val_transform,
        indices=val_indices,
        filter_prefilter_white=True,
        verbose=True,
        use_wsi_bag_sampling=bool(
            cfg["stage2_train"].get("use_wsi_bag_sampling", False)
            or cfg["stage2_train"].get("use_wsi_bag_loss", False)
            or cfg["stage2_train"].get("use_wsi_bag_margin_loss", False)
            or cfg["stage2_train"].get("use_neg_global_topk_suppression", False)
        ),
        wsi_bag_size=int(cfg["stage2_train"].get("wsi_bag_size", 64)),
        wsi_min_bag_size=int(cfg["stage2_train"].get("wsi_min_bag_size", 8)),
        slide_label_col=cfg["stage2_train"].get("slide_label_col", "slide_label"),
        random_seed=seed + 999,
        use_spatial_neighbor_sampling=bool(
            cfg["stage2_train"].get("use_spatial_neighbor_sampling", False)
        ),
        spatial_neighbor_csv=cfg["stage2_train"].get("spatial_neighbor_csv", None),
        spatial_neighbor_max_k=int(
            cfg["stage2_train"].get("spatial_neighbor_max_k", 8)
        ),
    )
    
    batch_size = int(cfg["stage2_train"].get("batch_size", 128))
    num_workers = int(cfg["stage2_train"].get("num_workers", 8))
    num_batches_per_epoch = cfg["stage2_train"].get("num_batches_per_epoch", None)

    batch_sampler = ProjectBalancedBatchSampler(
        train_dataset,
        batch_size=batch_size,
        num_batches_per_epoch=num_batches_per_epoch,
        drop_last=True,
        seed=seed,
        cache_path=cfg["stage2_train"].get(
            "project_sampler_cache",
            "outputs/dataset_cache/tcga_project_indices_splitbyslide.pkl"
        ),
        verbose=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=build_tcga_stage2_collate_fn(train_dataset),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_batch_sampler = SlideLabelBalancedBatchSampler(
        val_dataset,
        batch_size=batch_size,
        num_batches=val_max_batches,
        drop_last=True,
        seed=seed + 2024,
        verbose=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=build_tcga_stage2_collate_fn(val_dataset),
    )

    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # ================= 6. Train utils =================
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    visualizer = DistillVisualizer()

    ckpt_dir = cfg["stage2_train"].get("ckpt_dir", "distill_checkpoints_stage2_tcga_roleproto_v1")
    os.makedirs(ckpt_dir, exist_ok=True)

    early_stopping = EarlyStopping(
        patience=int(cfg["stage2_train"].get("early_stop_patience", 5)),
        min_delta=float(cfg["stage2_train"].get("early_stop_min_delta", 1e-4)),
        save_path=os.path.join(ckpt_dir, "moe_encoder_stage2_best.pth"),
    )

    best_full_path = os.path.join(ckpt_dir, "best_full.pth")
    best_metric = float("inf")

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
    print("\n🔥 Start Stage2 TCGA RoleProto Training")

    for epoch in range(start_epoch, epochs):
        # ---------------- train ----------------
        distiller.train()
        epoch_loss = 0.0
        step_count = 0
        running_loss_dict = {}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            images = batch["image"].to(device, non_blocking=True)
            offline_cluster_ids = None

            wsi_images = batch.get("wsi_images", None)
            if wsi_images is not None:
                wsi_images = wsi_images.to(device, non_blocking=True)

            wsi_slide_label = batch.get("wsi_slide_label", None)
            if wsi_slide_label is not None:
                wsi_slide_label = wsi_slide_label.to(device, non_blocking=True)

            slide_label_batch = batch.get("slide_label_batch", None)
            if slide_label_batch is not None:
                slide_label_batch = slide_label_batch.to(device, non_blocking=True)

            neighbor_images_list = batch.get("neighbor_images_list", None)

            optimizer.zero_grad(set_to_none=True)

            slide_id_batch = batch.get("slide_id", None)
            with torch.amp.autocast("cuda"):
                loss, loss_dict, gate_info_list = distiller(
                    images,
                    offline_cluster_ids=offline_cluster_ids,
                    is_eval=False,
                    wsi_images=wsi_images,
                    wsi_slide_label=wsi_slide_label,
                    slide_label_batch=slide_label_batch,
                    neighbor_images_list=neighbor_images_list,
                    slide_id_batch=slide_id_batch,
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(distiller.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.detach().float().cpu().item()
            step_count += 1

            update_running_dict(running_loss_dict, loss_dict)

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

        avg_train_loss = epoch_loss / max(step_count, 1)
        avg_train_loss_dict = average_running_dict(running_loss_dict, step_count)
        print(f"[Train] Epoch {epoch+1} Loss detail: {avg_train_loss_dict}")

        if "wsi_bag_total" in avg_train_loss_dict:
            print(
                f"[Train][WSI] "
                f"bag_total={avg_train_loss_dict.get('wsi_bag_total', 0.0):.6f} | "
                f"bag_bce={avg_train_loss_dict.get('wsi_bag_loss_raw', 0.0):.6f} | "
                f"bag_margin={avg_train_loss_dict.get('wsi_bag_margin_raw', 0.0):.6f} | "
                f"topk_mean={avg_train_loss_dict.get('wsi_topk_mean_score', 0.0):.6f} | "
                f"prob={avg_train_loss_dict.get('wsi_prob', 0.0):.6f}"
            )

        # ---------------- val ----------------
        distiller.eval()
        val_loss = 0.0
        val_steps = 0
        val_running_loss_dict = {}

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            ):
                if batch_idx >= val_max_batches:
                    break

                images = batch["image"].to(device, non_blocking=True)
                offline_cluster_ids = None

                wsi_images = batch.get("wsi_images", None)
                if wsi_images is not None:
                    wsi_images = wsi_images.to(device, non_blocking=True)

                wsi_slide_label = batch.get("wsi_slide_label", None)
                if wsi_slide_label is not None:
                    wsi_slide_label = wsi_slide_label.to(device, non_blocking=True)

                neighbor_images_list = batch.get("neighbor_images_list", None)
                slide_label_batch = batch.get("slide_label_batch", None)
                if slide_label_batch is not None:
                    slide_label_batch = slide_label_batch.to(device, non_blocking=True)

                slide_id_batch = batch.get("slide_id", None)
                with torch.amp.autocast("cuda"):
                    loss, loss_dict, gate_info_list = distiller(
                        images,
                        offline_cluster_ids=offline_cluster_ids,
                        is_eval=True,
                        wsi_images=wsi_images,
                        wsi_slide_label=wsi_slide_label,
                        slide_label_batch=slide_label_batch,
                        neighbor_images_list=neighbor_images_list,
                        slide_id_batch=slide_id_batch,
                    )

                val_loss += loss.item()
                val_steps += 1

                last_info = gate_info_list[-1]
                dispatch_weight = last_info["dispatch_weight"].detach()
                dispatch_mask = last_info["dispatch_mask"].detach()

                entropy = -(dispatch_weight * torch.log(dispatch_weight + 1e-8)).sum(dim=-1).mean().item()
                usage = dispatch_mask.float().mean(dim=0).detach().cpu().numpy()

                if batch_idx == 0:
                    slb = batch["slide_label_batch"]
                    num_pos = int((slb == 1).sum().item())
                    num_neg = int((slb == 0).sum().item())
                    print(f"[Val batch0] pos={num_pos}, neg={num_neg}")

                visualizer.update(
                    loss_dict,
                    entropy=entropy,
                    expert_usage=usage,
                    mode="val"
                )
                update_running_dict(val_running_loss_dict, loss_dict)

        avg_val_loss = val_loss / max(val_steps, 1)
        avg_val_loss_dict = average_running_dict(val_running_loss_dict, val_steps)
                
        print(f"[Val] Epoch {epoch+1} Loss detail: {avg_val_loss_dict}")

        if "wsi_bag_total" in avg_val_loss_dict:
            print(
                f"[Val][WSI] "
                f"bag_total={avg_val_loss_dict.get('wsi_bag_total', 0.0):.6f} | "
                f"bag_bce={avg_val_loss_dict.get('wsi_bag_loss_raw', 0.0):.6f} | "
                f"bag_margin={avg_val_loss_dict.get('wsi_bag_margin_raw', 0.0):.6f} | "
                f"topk_mean={avg_val_loss_dict.get('wsi_topk_mean_score', 0.0):.6f} | "
                f"prob={avg_val_loss_dict.get('wsi_prob', 0.0):.6f}"
            )

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
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
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }, ckpt_path)

        torch.save({
            "epoch": epoch,
            "student_state_dict": distiller.student.state_dict(),
            "distiller_state_dict": distiller.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }, os.path.join(ckpt_dir, "latest.pth"))

        print(f"Checkpoint saved: {ckpt_path}")

        if avg_val_loss < best_metric:
            best_metric = avg_val_loss
            torch.save({
                "epoch": epoch,
                "student_state_dict": distiller.student.state_dict(),
                "distiller_state_dict": distiller.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "cfg": cfg,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }, best_full_path)
            print(f"✅ Best full checkpoint updated: {best_full_path} (val_loss={avg_val_loss:.6f})")

        early_stopping(avg_val_loss, distiller.student)
        if early_stopping.early_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break

    print("Stage2 Training Complete.")
    visualizer.summarize()


if __name__ == "__main__":
    main()