#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
from collections import Counter
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
import h5py
import openslide
from PIL import Image

# ===== 改成你真实路径 =====
from models.encoders.virchow2_moe_encoder import (
    Virchow2MoEEncoder,
    BridgeMoEFFN,
)
# =========================


# =========================================================
# Image / WSI helpers
# =========================================================
def collect_test_images(img_dir: str, max_images: int = 4) -> List[Image.Image]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    img_paths = []
    for p in sorted(Path(img_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            img_paths.append(p)
        if len(img_paths) >= max_images:
            break

    if len(img_paths) == 0:
        raise FileNotFoundError(f"No test images found in: {img_dir}")

    images = [Image.open(p).convert("RGB") for p in img_paths]
    print("[Test images]")
    for p in img_paths:
        print(f"  - {p}")
    return images


def read_coords_from_h5(h5_path: str) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
    return torch.from_numpy(coords).long()


def read_patch_from_wsi(
    slide: openslide.OpenSlide,
    coord_xy,
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    patch = slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")
    return patch


def collect_test_images_from_wsi_h5(
    wsi_path: str,
    h5_path: str,
    max_patches: int = 4,
    patch_size: int = 256,
    random_sample: bool = True,
) -> List[Image.Image]:
    coords = read_coords_from_h5(h5_path)
    n_total = coords.shape[0]

    if n_total == 0:
        raise RuntimeError(f"No coords found in h5: {h5_path}")

    if random_sample:
        k = min(max_patches, n_total)
        idx = torch.randperm(n_total)[:k]
        coords = coords[idx]
    else:
        coords = coords[:max_patches]

    slide = openslide.OpenSlide(wsi_path)

    images = []
    sampled_coords = []
    for xy in coords.tolist():
        img = read_patch_from_wsi(
            slide=slide,
            coord_xy=xy,
            patch_size=patch_size,
            read_level=0,
        )
        img = img.resize((224, 224), resample=Image.BICUBIC)
        images.append(img)
        sampled_coords.append(xy)

    slide.close()

    print("[Sampled test patches from WSI]")
    print(f"  wsi_path: {wsi_path}")
    print(f"  h5_path : {h5_path}")
    print(f"  sampled : {len(images)} / {n_total}")
    for i, xy in enumerate(sampled_coords):
        print(f"  - patch[{i}] coord={xy}")

    return images


# =========================================================
# Model sanity helpers
# =========================================================
def print_block_replacement_status(model: Virchow2MoEEncoder, target_blocks: List[int]):
    print("\n[1] Block replacement check")
    for idx in target_blocks:
        blk = model.blocks[idx]
        ok = isinstance(blk.mlp, BridgeMoEFFN)
        print(f"  block {idx}: mlp={blk.mlp.__class__.__name__}, replaced={ok}")
        if ok:
            print(f"    proj_in : {blk.mlp.proj_in.in_features} -> {blk.mlp.proj_in.out_features}")
            print(f"    proj_out: {blk.mlp.proj_out.in_features} -> {blk.mlp.proj_out.out_features}")

            # 更稳的打印方式：从模块结构推断
            if hasattr(blk.mlp.moe, "num_experts"):
                print(f"    experts : {blk.mlp.moe.num_experts}")
            elif hasattr(blk.mlp.moe, "experts"):
                print(f"    experts : {len(blk.mlp.moe.experts)}")
            else:
                print("    experts : <unknown>")

            if hasattr(blk.mlp.moe, "experts") and len(blk.mlp.moe.experts) > 0:
                e0 = blk.mlp.moe.experts[0]
                if hasattr(e0, "ffn") and len(e0.ffn) > 0 and hasattr(e0.ffn[0], "in_features"):
                    print(f"    moe dim : {e0.ffn[0].in_features}")


def print_trainable_param_summary(model: Virchow2MoEEncoder):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n[2] Parameter summary")
    print(f"  total params    : {total:,}")
    print(f"  trainable params: {trainable:,}")

    print("\n[Trainable parameter names: first 80]")
    count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  - {name}: {tuple(p.shape)}")
            count += 1
            if count >= 80:
                break


def try_load_stage2_weights(
    model: Virchow2MoEEncoder,
    stage2_ckpt: str,
    target_to_source_layer_map: dict,
    strict: bool = False,
):
    print("\n[3] Loading stage2 MoE weights")
    try:
        model.load_stage2_moe_from_ckpt(
            stage2_ckpt_path=stage2_ckpt,
            target_to_source_layer_map=target_to_source_layer_map,
            strict=strict,
        )
        print("[OK] stage2 moe weights loaded.")
    except Exception as e:
        print(f"[ERROR] load_stage2_moe_from_ckpt failed: {e}")
        raise


def summarize_gate_info(gate_info, idx: int):
    print(f"\n  [gate_info #{idx}] type={type(gate_info).__name__}")
    if gate_info is None:
        print("    gate_info is None")
        return

    if isinstance(gate_info, dict):
        print(f"    keys: {list(gate_info.keys())}")
        for k, v in gate_info.items():
            if torch.is_tensor(v):
                print(f"    - {k}: tensor shape={tuple(v.shape)}, dtype={v.dtype}")
            elif isinstance(v, (list, tuple)):
                print(f"    - {k}: {type(v).__name__}, len={len(v)}")
            else:
                print(f"    - {k}: {type(v).__name__} = {v}")
    elif torch.is_tensor(gate_info):
        print(f"    tensor shape={tuple(gate_info.shape)}, dtype={gate_info.dtype}")
    else:
        print(f"    repr={repr(gate_info)}")


@torch.no_grad()
def run_forward_check(
    model: Virchow2MoEEncoder,
    images: List[Image.Image],
):
    print("\n[4] Forward sanity check")

    out = model(
        images,
        return_gates=True,
        is_eval=True,
        return_features=True,
        offline_cluster_ids=None,
    )

    if not isinstance(out, tuple) or len(out) != 4:
        raise RuntimeError(
            f"Unexpected forward output type/len: {type(out)}, "
            f"len={len(out) if isinstance(out, tuple) else 'N/A'}"
        )

    x_out, gate_info_list, feature_dict, moe_feature_list = out

    print(f"  x_out shape         : {tuple(x_out.shape)}")
    print(f"  num gate_info       : {len(gate_info_list)}")
    print(f"  feature_dict keys   : {list(feature_dict.keys())}")
    for k, v in feature_dict.items():
        if torch.is_tensor(v):
            print(f"    - {k}: {tuple(v.shape)}")

    print(f"  num moe_feature_list: {len(moe_feature_list)}")
    for i, feat in enumerate(moe_feature_list):
        if torch.is_tensor(feat):
            print(f"    - moe_feature[{i}]: {tuple(feat.shape)}")

    for i, g in enumerate(gate_info_list):
        summarize_gate_info(g, i)

    cls = x_out[:, 0, :]
    print(f"\n  cls shape           : {tuple(cls.shape)}")

    reg_tokens = getattr(model, "reg_tokens", 0)
    patch_start = 1 + reg_tokens
    patches = x_out[:, patch_start:, :]
    print(f"  patch tokens shape  : {tuple(patches.shape)}")
    print(f"  patch_start         : {patch_start}")

    if torch.isnan(x_out).any():
        print("[WARN] NaN detected in x_out")
    else:
        print("[OK] No NaN in x_out")


def freeze_backbone_except_moe(model: Virchow2MoEEncoder):
    for p in model.parameters():
        p.requires_grad = False

    for idx in model.moe_layer_map.keys():
        blk = model.blocks[idx]
        for p in blk.mlp.parameters():
            p.requires_grad = True

    print("[Info] backbone frozen, only moe bridge params are trainable.")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--virchow2_weight", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    # 两种测试输入方式：patch目录 或 WSI+h5
    parser.add_argument("--test_img_dir", type=str, default="")
    parser.add_argument("--test_wsi", type=str, default="")
    parser.add_argument("--test_h5", type=str, default="")
    parser.add_argument("--max_test_images", type=int, default=4)
    parser.add_argument("--max_test_patches", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--sequential_sample", action="store_true")

    # Virchow2 插层位置
    parser.add_argument("--target_block_1", type=int, default=29)
    parser.add_argument("--target_block_2", type=int, default=30)

    # stage2 来源层
    parser.add_argument("--source_stage2_layer_1", type=int, default=9)
    parser.add_argument("--source_stage2_layer_2", type=int, default=10)

    # adapter / moe 超参
    parser.add_argument("--adapter_dim", type=int, default=384)
    parser.add_argument("--adapter_hidden_dim", type=int, default=1536)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--shared_expert", action="store_true")
    parser.add_argument("--routing_strategy", type=str, default="proto_topany")
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--init_threshold", type=float, default=0.0)
    parser.add_argument("--min_experts", type=int, default=1)
    parser.add_argument("--max_experts", type=int, default=2)
    parser.add_argument("--gate_init_scale", type=float, default=2.0)
    parser.add_argument("--gate_noise_std", type=float, default=0.02)
    parser.add_argument("--shared_alpha", type=float, default=0.05)
    parser.add_argument("--use_routing_proj", action="store_true")
    parser.add_argument("--routing_metric", type=str, default="cosine")

    parser.add_argument("--strict_load_stage2", action="store_true")
    parser.add_argument("--freeze_backbone_except_moe", action="store_true")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    virchow2_cfg = {
        "weight_path": args.virchow2_weight,
        "device": device,
    }

    moe_cfg = {
        "moe_layers": [args.target_block_1, args.target_block_2],
        "adapter_dim": args.adapter_dim,
        "adapter_hidden_dim": args.adapter_hidden_dim,

        "num_experts": args.num_experts,
        "shared_expert": args.shared_expert,
        "routing_strategy": args.routing_strategy,
        "top_k": args.top_k,
        "init_threshold": args.init_threshold,
        "min_experts": args.min_experts,
        "max_experts": args.max_experts,
        "gate_init_scale": args.gate_init_scale,
        "gate_noise_std": args.gate_noise_std,
        "shared_alpha": args.shared_alpha,
        "use_routing_proj": args.use_routing_proj,
        "routing_metric": args.routing_metric,
    }

    print("[Build] Virchow2MoEEncoder")
    model = Virchow2MoEEncoder(virchow2_cfg, moe_cfg).to(device)
    model.eval()

    if args.freeze_backbone_except_moe:
        freeze_backbone_except_moe(model)

    target_blocks = [args.target_block_1, args.target_block_2]
    print_block_replacement_status(model, target_blocks)
    print_trainable_param_summary(model)

    target_to_source_layer_map = {
        args.target_block_1: args.source_stage2_layer_1,
        args.target_block_2: args.source_stage2_layer_2,
    }
    try_load_stage2_weights(
        model,
        stage2_ckpt=args.stage2_ckpt,
        target_to_source_layer_map=target_to_source_layer_map,
        strict=args.strict_load_stage2,
    )

    if args.test_wsi and args.test_h5:
        images = collect_test_images_from_wsi_h5(
            wsi_path=args.test_wsi,
            h5_path=args.test_h5,
            max_patches=args.max_test_patches,
            patch_size=args.patch_size,
            random_sample=not args.sequential_sample,
        )
    elif args.test_img_dir:
        images = collect_test_images(
            args.test_img_dir,
            max_images=args.max_test_images,
        )
    else:
        raise ValueError("Please provide either --test_img_dir OR (--test_wsi and --test_h5).")

    run_forward_check(model, images)

    print("\n[Done] Sanity check passed.")


if __name__ == "__main__":
    main()