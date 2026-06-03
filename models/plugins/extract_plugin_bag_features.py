#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models.plugins.shared_role_prototype import (
    SharedRolePrototype,
    PatchRoleSummaryFromSharedProto,
)
from models.plugins.role_aware_tail_plugin import RoleAwareTailWithSharedSummary
from models.encoders.moe_encoder import MoEEncoder


# =========================================================
# Utils
# =========================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_device(device_str: str) -> str:
    if device_str == "cpu":
        return "cpu"
    return device_str if torch.cuda.is_available() else "cpu"


def load_stage2_proj_l12(
    config_path: str,
    full_ckpt_path: str,
    device: str = "cuda",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not os.path.exists(full_ckpt_path):
        raise FileNotFoundError(f"Full checkpoint not found: {full_ckpt_path}")

    ckpt = torch.load(full_ckpt_path, map_location="cpu")

    if "distiller_state_dict" not in ckpt:
        raise KeyError("distiller_state_dict not found in full checkpoint")

    distiller_sd = ckpt["distiller_state_dict"]
    if "proj_l12.weight" not in distiller_sd or "proj_l12.bias" not in distiller_sd:
        raise KeyError("proj_l12 not found in distiller_state_dict")

    proj_out_dim, proj_in_dim = distiller_sd["proj_l12.weight"].shape
    proj_l12 = nn.Linear(proj_in_dim, proj_out_dim)
    proj_l12.load_state_dict(
        {
            "weight": distiller_sd["proj_l12.weight"],
            "bias": distiller_sd["proj_l12.bias"],
        }
    )
    proj_l12 = proj_l12.to(device)
    proj_l12.eval()

    for p in proj_l12.parameters():
        p.requires_grad = False

    print(f"[INFO] loaded proj_l12 from stage2 ckpt: {proj_in_dim} -> {proj_out_dim}")
    return proj_l12, proj_in_dim, proj_out_dim


def build_teacher_like_feats(
    x_384: torch.Tensor,
    proj_l12: nn.Module,
) -> torch.Tensor:
    """
    x_384: [1, N, 384] or [B, N, 384]
    return: normalized teacher-like feature [1, N, 1280]
    """
    x_teacher = proj_l12(x_384)
    x_teacher = F.normalize(x_teacher, dim=-1)
    return x_teacher


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Apply trained plugin to cached bag features")

    parser.add_argument("--in_feature_dir", type=str, required=True,
                        help="Input bag feature dir, each file is slide_id.pt")
    parser.add_argument("--plugin_ckpt", type=str, required=True,
                        help="Checkpoint from train_plugin_on_cached_features.py")
    parser.add_argument("--role_proto_dir", type=str, required=True,
                        help="Directory containing role_prototypes_init.npy and role_names.json")
    parser.add_argument("--out_feature_dir", type=str, required=True,
                        help="Output dir for plugin-enhanced bag features")

    # new: load stage2 proj_l12
    parser.add_argument("--stage2_config", type=str, required=True,
                        help="stage2 yaml config used to train the student/distiller")
    parser.add_argument("--stage2_full_ckpt", type=str, required=True,
                        help="stage2 best_full checkpoint containing distiller_state_dict/proj_l12")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_token_size", type=int, default=8192,
                        help="Chunk size along bag instances to avoid OOM")

    args = parser.parse_args()

    ensure_dir(args.out_feature_dir)
    device = resolve_device(args.device)

    # -----------------------------------------------------
    # load plugin ckpt
    # -----------------------------------------------------
    ckpt = torch.load(args.plugin_ckpt, map_location="cpu")
    plugin_args = ckpt["args"]
    feat_dim = int(ckpt["feat_dim"])

    # -----------------------------------------------------
    # load stage2 proj_l12
    # -----------------------------------------------------
    proj_l12, proj_in_dim, proj_out_dim = load_stage2_proj_l12(
        config_path=args.stage2_config,
        full_ckpt_path=args.stage2_full_ckpt,
        device=device,
    )

    if feat_dim != proj_in_dim:
        raise ValueError(
            f"Plugin feat_dim ({feat_dim}) != proj_l12 input dim ({proj_in_dim}). "
            f"Current cached bag features may not match the stage2 projector."
        )

    # -----------------------------------------------------
    # rebuild shared role proto
    # -----------------------------------------------------
    shared_role_proto = SharedRolePrototype.from_files(
        role_proto_dir=args.role_proto_dir,
        normalize=True,
        learnable=False,
        device=device,
    )

    if "shared_role_proto_state_dict" in ckpt:
        shared_role_proto.load_state_dict(ckpt["shared_role_proto_state_dict"], strict=True)
    elif "shared_proto_state_dict" in ckpt:
        shared_role_proto.load_state_dict(ckpt["shared_proto_state_dict"], strict=True)
    else:
        raise KeyError("Neither 'shared_role_proto_state_dict' nor 'shared_proto_state_dict' found in plugin ckpt")

    shared_role_proto.eval()

    summary_builder = PatchRoleSummaryFromSharedProto(
        shared_role_proto=shared_role_proto,
        tau=float(plugin_args.get("role_tau", 1.0)),
        use_softmax=True,
    ).to(device)

    # -----------------------------------------------------
    # rebuild plugin
    # -----------------------------------------------------
    plugin = RoleAwareTailWithSharedSummary(
        feat_dim=feat_dim,
        num_roles=shared_role_proto.num_roles,
        hidden_dim=int(plugin_args.get("plugin_hidden_dim", 128)),
        dropout=float(plugin_args.get("plugin_dropout", 0.0)),
        use_role_logits=bool(plugin_args.get("use_role_logits", False)),
        use_top1_gap=bool(plugin_args.get("use_top1_gap", False)),
        use_beta=bool(plugin_args.get("use_beta", False)),
        init_scale=float(plugin_args.get("plugin_init_scale", 0.1)),
    ).to(device)

    plugin.load_state_dict(ckpt["plugin_state_dict"], strict=True)
    plugin.eval()

    print(f"[INFO] device = {device}")
    print(f"[INFO] feat_dim = {feat_dim}")
    print(f"[INFO] proj_l12: {proj_in_dim} -> {proj_out_dim}")
    print(f"[INFO] num_roles = {shared_role_proto.num_roles}")
    print(f"[INFO] role_names = {shared_role_proto.role_names}")

    # -----------------------------------------------------
    # collect input pt files
    # -----------------------------------------------------
    pt_files = sorted([
        x for x in os.listdir(args.in_feature_dir)
        if x.endswith(".pt")
    ])
    if len(pt_files) == 0:
        raise ValueError(f"No .pt files found in {args.in_feature_dir}")

    print(f"[INFO] num_bags = {len(pt_files)}")

    # -----------------------------------------------------
    # apply plugin to each bag
    # -----------------------------------------------------
    with torch.no_grad():
        for fname in tqdm(pt_files, desc="Extract plugin bags"):
            in_path = os.path.join(args.in_feature_dir, fname)
            out_path = os.path.join(args.out_feature_dir, fname)

            obj = torch.load(in_path, map_location="cpu")
            if "features" not in obj:
                raise KeyError(f"{in_path} missing 'features'")

            feats = obj["features"].float()  # [N, D=384]
            if feats.ndim != 2:
                raise ValueError(f"{in_path} features must be [N, D], got {tuple(feats.shape)}")
            if feats.shape[1] != feat_dim:
                raise ValueError(
                    f"{in_path} feature dim mismatch: got {feats.shape[1]}, expected {feat_dim}"
                )

            N = feats.shape[0]

            out_feats_chunks = []
            out_logits_chunks = []
            out_probs_chunks = []
            out_gaps_chunks = []
            out_top1_chunks = []

            for start in range(0, N, args.batch_token_size):
                end = min(start + args.batch_token_size, N)
                x = feats[start:end].to(device).unsqueeze(0)  # [1, n, 384]

                # -------------------------------------------------
                # build role summary from projected teacher-like space
                # -------------------------------------------------
                x_teacher_like = build_teacher_like_feats(x, proj_l12)  # [1, n, 1280]
                role_dict_in = summary_builder(x_teacher_like)

                # -------------------------------------------------
                # plugin works in raw 384d feature space
                # -------------------------------------------------
                x_out = plugin(
                    patch_feat=x,
                    patch_role_probs=role_dict_in["patch_role_probs"],
                    patch_role_gaps=role_dict_in["patch_role_gaps"],
                    patch_role_logits=role_dict_in["patch_role_logits"],
                    patch_top1_gap=role_dict_in["patch_top1_gap"],
                    return_aux=False,
                )  # [1, n, 384]

                # -------------------------------------------------
                # recompute role summary AFTER plugin
                # still need proj_l12 -> teacher-like space first
                # -------------------------------------------------
                x_out_teacher_like = build_teacher_like_feats(x_out, proj_l12)  # [1, n, 1280]
                role_dict_out = summary_builder(x_out_teacher_like)

                out_feats_chunks.append(x_out.squeeze(0).cpu())
                out_logits_chunks.append(role_dict_out["patch_role_logits"].squeeze(0).cpu())
                out_probs_chunks.append(role_dict_out["patch_role_probs"].squeeze(0).cpu())
                out_gaps_chunks.append(role_dict_out["patch_role_gaps"].squeeze(0).cpu())
                out_top1_chunks.append(role_dict_out["patch_top1_gap"].squeeze(0).cpu())

            obj["features"] = torch.cat(out_feats_chunks, dim=0)          # [N, 384]
            obj["role_logits"] = torch.cat(out_logits_chunks, dim=0)      # [N, R]
            obj["role_probs"] = torch.cat(out_probs_chunks, dim=0)        # [N, R]
            obj["role_gaps"] = torch.cat(out_gaps_chunks, dim=0)          # [N, R]
            obj["role_top1_gap"] = torch.cat(out_top1_chunks, dim=0)      # [N, 1]

            obj["plugin_applied"] = True
            obj["plugin_ckpt"] = args.plugin_ckpt
            obj["role_names"] = shared_role_proto.role_names
            obj["role_summary_space"] = "projected_teacher_space_via_proj_l12"
            obj["proj_l12_input_dim"] = int(proj_in_dim)
            obj["proj_l12_output_dim"] = int(proj_out_dim)

            torch.save(obj, out_path)

    print(f"[Done] saved plugin-enhanced bags to: {args.out_feature_dir}")


if __name__ == "__main__":
    main()