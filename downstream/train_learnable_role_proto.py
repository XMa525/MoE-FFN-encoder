# downstream/train_learnable_role_proto.py
from __future__ import annotations

import os
import sys
import json
import math
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.encoders.moe_encoder import MoEEncoder
from downstream.downstream_learnable_role_proto import build_downstream_proto_model
from distillation.dataset.build_camelyon_stage2_dataset import CamelyonWSIBagDataset
from utils.earlystopping import EarlyStopping


# =========================================================
# Utils
# =========================================================
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def move_batch_to_device(batch: Dict, device: str) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# =========================================================
# Model / checkpoint loading
# =========================================================
def load_stage2_bundle(config_path: str, full_ckpt_path: str, device: str = "cuda"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(full_ckpt_path, map_location="cpu")
    if "student_state_dict" not in ckpt:
        raise KeyError(f"student_state_dict not found in {full_ckpt_path}")
    if "distiller_state_dict" not in ckpt:
        raise KeyError(f"distiller_state_dict not found in {full_ckpt_path}")

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    missing, unexpected = model.load_state_dict(ckpt["student_state_dict"], strict=False)
    print(f"[load_stage2_bundle:model] missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device)
    model.eval()

    distiller_sd = ckpt["distiller_state_dict"]
    if "proj_l12.weight" not in distiller_sd or "proj_l12.bias" not in distiller_sd:
        raise RuntimeError(f"proj_l12 not found in distiller_state_dict of {full_ckpt_path}")

    proj_out_dim, proj_in_dim = distiller_sd["proj_l12.weight"].shape
    proj_head = nn.Linear(proj_in_dim, proj_out_dim)
    proj_head.load_state_dict({
        "weight": distiller_sd["proj_l12.weight"],
        "bias": distiller_sd["proj_l12.bias"],
    })
    proj_head = proj_head.to(device)
    proj_head.eval()

    print(f"[load_stage2_bundle] loaded matched model + proj_l12 from {full_ckpt_path}")
    print(f"[load_stage2_bundle] moe_layers_idx = {model.moe_layers_idx}")
    print(f"[load_stage2_bundle] proj_l12: {proj_in_dim} -> {proj_out_dim}")
    return model, proj_head, cfg


# =========================================================
# Dataset builders
# =========================================================
def build_camelyon_downstream_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    ds_cfg = cfg["downstream_proto_train"]

    train_dataset = CamelyonWSIBagDataset(
        csv_path=ds_cfg["train_csv_path"],
        raw_dir=ds_cfg["raw_dir"],
        h5_dir=ds_cfg["h5_dir"],
        patch_size=ds_cfg.get("patch_size", 256),
        read_level=ds_cfg.get("read_level", 0),
        resize_to=ds_cfg.get("resize_to", 224),
        max_patches=ds_cfg.get("max_patches", 64),
        sample_mode=ds_cfg.get("sample_mode", "random"),
        seed=ds_cfg.get("seed", 42),
        return_pil=False,
        transform=None,
        check_files=ds_cfg.get("check_files", True),
    )

    val_dataset = CamelyonWSIBagDataset(
        csv_path=ds_cfg["val_csv_path"],
        raw_dir=ds_cfg["raw_dir"],
        h5_dir=ds_cfg["h5_dir"],
        patch_size=ds_cfg.get("patch_size", 256),
        read_level=ds_cfg.get("read_level", 0),
        resize_to=ds_cfg.get("resize_to", 224),
        max_patches=ds_cfg.get("max_patches", 64),
        sample_mode=ds_cfg.get("sample_mode", "random"),
        seed=ds_cfg.get("seed", 42),
        return_pil=False,
        transform=None,
        check_files=ds_cfg.get("check_files", True),
    )

    def bag_collate_fn(batch):
        # dataset already returns one bag dict; keep batch_size=1
        return batch[0]

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=bag_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=ds_cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=bag_collate_fn,
    )
    return train_loader, val_loader


# =========================================================
# Optimizer
# =========================================================
def build_optimizer(model: nn.Module, lr_proto: float, lr_proj: float, lr_bag: float, weight_decay: float):
    proto_params = [model.role_head.role_prototypes]
    proj_params = [p for p in model.role_head.proj_head.parameters() if p.requires_grad]
    bag_params = [p for p in model.slide_classifier.parameters() if p.requires_grad]

    param_groups = [
        {"params": proto_params, "lr": lr_proto, "name": "role_proto"},
    ]
    if len(proj_params) > 0:
        param_groups.append({"params": proj_params, "lr": lr_proj, "name": "proj_head"})
    if len(bag_params) > 0:
        param_groups.append({"params": bag_params, "lr": lr_bag, "name": "bag_head"})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# =========================================================
# Epoch runners
# =========================================================
def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    patch_batch_size: int,
    train: bool,
) -> Tuple[float, Dict[str, float]]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    count = 0
    running: Dict[str, float] = {}

    for batch in loader:
        images = batch["images"]  # [N,3,H,W]
        label = batch["label"]

        if not torch.is_tensor(images):
            images = torch.stack(images, dim=0)
        images = images.to(device, non_blocking=True)
        label_t = torch.tensor([float(label)], device=device)

        # Optional micro-batching over patches within a bag to reduce memory.
        # Wrapper expects [B_img, 3, H, W]; we keep batch dimension = num_patches.
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        output = model(images=images, slide_label=label_t, is_eval=not train)
        loss = output.total_loss

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.detach().cpu())
        count += 1
        for k, v in output.loss_dict.items():
            running[k] = running.get(k, 0.0) + float(v)

    mean_loss = total_loss / max(count, 1)
    mean_stats = {k: v / max(count, 1) for k, v in running.items()}
    return mean_loss, mean_stats


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train downstream learnable role prototypes")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage2-full-ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["downstream_proto_train"]
    seed_everything(ds_cfg.get("seed", 42))
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    ckpt_dir = ds_cfg["ckpt_dir"]
    ensure_dir(ckpt_dir)

    student, proj_head, _ = load_stage2_bundle(
        config_path=cfg["stage2_base_config"],
        full_ckpt_path=args.stage2_full_ckpt,
        device=device,
    )

    model = build_downstream_proto_model(
        student_model=student,
        proj_head=proj_head,
        role_proto_dir=ds_cfg["role_proto_dir"],
        learn_proj=ds_cfg.get("learn_proj", False),
        use_last_moe_output=ds_cfg.get("use_last_moe_output", True),
        tumor_role_name=ds_cfg.get("tumor_role_name", "tumor"),
        role_temperature=ds_cfg.get("role_temperature", 0.07),
        bag_topk_ratio=ds_cfg.get("bag_topk_ratio", 0.1),
        bag_topk_min=ds_cfg.get("bag_topk_min", 4),
        bag_topk_max=ds_cfg.get("bag_topk_max", 32),
        tumor_margin_loss_weight=ds_cfg.get("tumor_margin_loss_weight", 0.0),
    ).to(device)

    train_loader, val_loader = build_camelyon_downstream_loaders(cfg)

    optimizer = build_optimizer(
        model=model,
        lr_proto=ds_cfg.get("lr_proto", 1e-3),
        lr_proj=ds_cfg.get("lr_proj", 1e-4),
        lr_bag=ds_cfg.get("lr_bag", 1e-3),
        weight_decay=ds_cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=ds_cfg.get("epochs", 10),
    )

    early_stopping = EarlyStopping(
        patience=ds_cfg.get("early_stop_patience", 5),
        min_delta=ds_cfg.get("early_stop_min_delta", 1e-4),
        save_path=os.path.join(ckpt_dir, "best_role_proto_only.pth"),
    )

    start_epoch = 0
    latest_ckpt = os.path.join(ckpt_dir, "latest.pth")
    if args.resume and os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"[Resume] start_epoch={start_epoch}")

    best_metric = float("inf")
    epochs = ds_cfg.get("epochs", 10)

    for epoch in range(start_epoch, epochs):
        train_loss, train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            patch_batch_size=ds_cfg.get("patch_batch_size", 16),
            train=True,
        )
        val_loss, val_stats = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            patch_batch_size=ds_cfg.get("patch_batch_size", 16),
            train=False,
        )
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] train={train_loss:.6f} val={val_loss:.6f}")
        print("[Train Stats]", train_stats)
        print("[Val Stats]", val_stats)

        save_obj = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_stats": train_stats,
            "val_stats": val_stats,
            "config": cfg,
        }
        torch.save(save_obj, os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"))
        torch.save(save_obj, latest_ckpt)

        if val_loss < best_metric:
            best_metric = val_loss
            torch.save(save_obj, os.path.join(ckpt_dir, "best_full.pth"))
            print(f"[Best] updated: val_loss={val_loss:.6f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"[Early Stop] at epoch {epoch+1}")
            break

    print("Done.")


if __name__ == "__main__":
    main()
