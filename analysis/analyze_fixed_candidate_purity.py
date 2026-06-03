#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import math
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import yaml
import numpy as np
import pandas as pd
import openslide

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as T

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoders.moe_encoder import MoEEncoder
from models.plugins.shared_role_prototype import SharedRolePrototype, PatchRoleSummaryFromSharedProto


# =========================================================
# utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def build_transform(img_size: int = 224):
    return T.Compose([
        T.ToImage(),
        T.Resize((img_size, img_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


def load_coords_attrs(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs.items())
    patch_size = int(attrs.get("patch_size", 256))
    patch_level = int(attrs.get("patch_level", 0))
    return coords, patch_size, patch_level


def read_patch_from_wsi(
    slide: openslide.OpenSlide,
    coord_xy: Tuple[int, int],
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    patch = slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")
    return patch


def make_label_column_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("Need 'label' or 'slide_binary_label' in input csv.")
    return df


# =========================================================
# model loading
# =========================================================
def load_encoder_from_ckpt(
    config_path: str,
    student_ckpt_path: str,
    device: str = "cuda",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(student_ckpt_path, map_location="cpu")
    if "student_state_dict" not in ckpt:
        raise KeyError(f"student_state_dict not found in {student_ckpt_path}")

    encoder = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    encoder.load_state_dict(ckpt["student_state_dict"], strict=True)
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print(f"[Encoder] loaded from {student_ckpt_path}")
    return encoder, cfg


def load_proj_l12_from_stage2(
    stage2_full_ckpt: str,
    device: str,
):
    ckpt = torch.load(stage2_full_ckpt, map_location="cpu")
    if "distiller_state_dict" not in ckpt:
        raise KeyError("distiller_state_dict not found in stage2 full checkpoint")

    distiller_sd = ckpt["distiller_state_dict"]
    if "proj_l12.weight" not in distiller_sd or "proj_l12.bias" not in distiller_sd:
        raise KeyError("proj_l12 not found in distiller_state_dict")

    proj_out_dim, proj_in_dim = distiller_sd["proj_l12.weight"].shape
    proj_l12 = nn.Linear(proj_in_dim, proj_out_dim)
    proj_l12.load_state_dict({
        "weight": distiller_sd["proj_l12.weight"],
        "bias": distiller_sd["proj_l12.bias"],
    })
    proj_l12 = proj_l12.to(device)
    proj_l12.eval()
    for p in proj_l12.parameters():
        p.requires_grad = False

    print(f"[proj_l12] loaded: {proj_in_dim} -> {proj_out_dim}")
    return proj_l12


# =========================================================
# feature extraction
# =========================================================
@torch.no_grad()
def extract_patch_features_stage2_style(
    encoder: nn.Module,
    patch_imgs: torch.Tensor,
    use_last_moe_output: bool = True,
):
    out = encoder(
        patch_imgs,
        return_gates=True,
        mask=None,
        is_eval=True,
        return_features=True,
        offline_cluster_ids=None,
    )

    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise RuntimeError(f"Unexpected encoder output type/len: {type(out)}")

    _, _, feature_dict, moe_feature_list = out

    if use_last_moe_output and len(moe_feature_list) > 0:
        feat_tokens = moe_feature_list[-1]
    else:
        if "layer_12" not in feature_dict:
            raise KeyError(f"'layer_12' not found in feature_dict keys={list(feature_dict.keys())}")
        feat_tokens = feature_dict["layer_12"]

    patch_tokens = feat_tokens[:, 1:, :]
    if patch_tokens.shape[1] == 0:
        raise RuntimeError(f"No patch tokens found, shape={tuple(patch_tokens.shape)}")

    patch_feat = patch_tokens.mean(dim=1)
    return patch_feat


@torch.no_grad()
def extract_selected_patch_features(
    encoder: nn.Module,
    svs_path: str,
    h5_path: str,
    selected_patch_indices: List[int],
    device: str,
    img_size: int = 224,
    batch_size: int = 64,
    use_last_moe_output: bool = True,
):
    coords_all, patch_size, patch_level = load_coords_attrs(h5_path)
    transform = build_transform(img_size)

    selected_patch_indices = [int(x) for x in selected_patch_indices]
    selected_coords = coords_all[selected_patch_indices]

    slide = openslide.OpenSlide(svs_path)
    feats = []
    try:
        for start in range(0, len(selected_coords), batch_size):
            batch_coords = selected_coords[start:start + batch_size]
            imgs = []
            for xy in batch_coords:
                img = read_patch_from_wsi(
                    slide=slide,
                    coord_xy=(int(xy[0]), int(xy[1])),
                    patch_size=patch_size,
                    read_level=patch_level,
                )
                imgs.append(transform(img))
            x = torch.stack(imgs, dim=0).to(device, non_blocking=True)
            feat = extract_patch_features_stage2_style(
                encoder=encoder,
                patch_imgs=x,
                use_last_moe_output=use_last_moe_output,
            )
            feats.append(feat.cpu())
    finally:
        slide.close()

    return torch.cat(feats, dim=0), selected_coords, patch_size, patch_level


@torch.no_grad()
def score_patch_features(
    patch_feat_raw: torch.Tensor,
    proj_l12: nn.Module,
    summary_builder: PatchRoleSummaryFromSharedProto,
    role_names: List[str],
    tumor_name: str,
    negative_role_names: List[str],
    device: str,
):
    x = patch_feat_raw.to(device, non_blocking=True)
    x_teacher = proj_l12(x)
    x_teacher = F.normalize(x_teacher, dim=-1)

    role_dict = summary_builder(x_teacher.unsqueeze(0))
    role_logits = role_dict["patch_role_logits"][0].detach().cpu()
    role_probs = role_dict["patch_role_probs"][0].detach().cpu()
    top1_gap = role_dict["patch_top1_gap"][0].detach().cpu().squeeze(-1)

    pred_role_idx = role_probs.argmax(dim=-1)
    pred_role_name = [role_names[int(i)] for i in pred_role_idx.tolist()]

    # second-best role
    top2 = torch.topk(role_probs, k=min(2, role_probs.shape[1]), dim=-1)
    second_role_idx = []
    second_role_name = []
    for i in range(role_probs.shape[0]):
        if role_probs.shape[1] >= 2:
            idx2 = int(top2.indices[i, 1].item())
        else:
            idx2 = int(top2.indices[i, 0].item())
        second_role_idx.append(idx2)
        second_role_name.append(role_names[idx2])

    role_to_idx = {n: i for i, n in enumerate(role_names)}
    if tumor_name not in role_to_idx:
        raise KeyError(f"tumor role '{tumor_name}' not found in role_names={role_names}")

    tumor_idx = role_to_idx[tumor_name]
    neg_ids = [role_to_idx[n] for n in negative_role_names if n in role_to_idx]
    if len(neg_ids) == 0:
        raise ValueError(f"No valid negative role names found. got={negative_role_names}")

    tumor_logit = role_logits[:, tumor_idx]
    tumor_prob = role_probs[:, tumor_idx]
    neg_logit = role_logits[:, neg_ids].max(dim=-1).values
    tumor_gap = tumor_logit - neg_logit

    return {
        "role_logits": role_logits,
        "role_probs": role_probs,
        "top1_gap": top1_gap,
        "pred_role_idx": pred_role_idx,
        "pred_role_name": pred_role_name,
        "second_role_idx": second_role_idx,
        "second_role_name": second_role_name,
        "tumor_logit": tumor_logit,
        "tumor_prob": tumor_prob,
        "tumor_gap": tumor_gap,
    }


# =========================================================
# csv loading
# =========================================================
def load_candidate_csv(candidate_csv: str) -> pd.DataFrame:
    if not os.path.exists(candidate_csv):
        raise FileNotFoundError(candidate_csv)

    df = pd.read_csv(candidate_csv)
    df = make_label_column_if_needed(df)

    required_cols = {
        "slide_id", "label", "patch_idx", "coord_x", "coord_y", "svs_path", "h5_path"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"candidate csv missing columns: {missing}")

    if "candidate_type" not in df.columns:
        df["candidate_type"] = "unknown"

    return df.reset_index(drop=True)


# =========================================================
# scoring candidates
# =========================================================
def score_candidate_csv(
    candidate_df: pd.DataFrame,
    config_path: str,
    student_ckpt: str,
    stage2_full_ckpt: str,
    role_proto_dir: str,
    out_dir: str,
    device: str,
    img_size: int,
    batch_size: int,
    use_last_moe_output: bool,
    tumor_name: str,
    negative_role_names: List[str],
):
    ensure_dir(out_dir)

    encoder, cfg = load_encoder_from_ckpt(config_path, student_ckpt, device=device)
    proj_l12 = load_proj_l12_from_stage2(stage2_full_ckpt, device=device)

    shared_role_proto = SharedRolePrototype.from_files(
        role_proto_dir=role_proto_dir,
        normalize=True,
        learnable=False,
        device=device,
    )
    role_names = list(shared_role_proto.role_names)

    summary_builder = PatchRoleSummaryFromSharedProto(
        shared_role_proto=shared_role_proto,
        tau=1.0,
        use_softmax=True,
    ).to(device)
    summary_builder.eval()

    all_rows = []
    grouped = candidate_df.groupby("slide_id")
    pbar = tqdm(grouped, total=candidate_df["slide_id"].nunique(), desc="Scoring fixed candidates")

    for slide_id, sub_df in pbar:
        sub_df = sub_df.sort_values("patch_idx").reset_index(drop=True)
        svs_path = str(sub_df["svs_path"].iloc[0])
        h5_path = str(sub_df["h5_path"].iloc[0])

        patch_indices = sub_df["patch_idx"].astype(int).tolist()

        patch_feat, coords_sel, patch_size, patch_level = extract_selected_patch_features(
            encoder=encoder,
            svs_path=svs_path,
            h5_path=h5_path,
            selected_patch_indices=patch_indices,
            device=device,
            img_size=img_size,
            batch_size=batch_size,
            use_last_moe_output=use_last_moe_output,
        )

        score_out = score_patch_features(
            patch_feat_raw=patch_feat,
            proj_l12=proj_l12,
            summary_builder=summary_builder,
            role_names=role_names,
            tumor_name=tumor_name,
            negative_role_names=negative_role_names,
            device=device,
        )

        for i in range(len(sub_df)):
            row = sub_df.iloc[i].to_dict()
            row["pred_role"] = score_out["pred_role_name"][i]
            row["second_role"] = score_out["second_role_name"][i]
            row["top1_gap"] = safe_float(score_out["top1_gap"][i])
            row["tumor_prob"] = safe_float(score_out["tumor_prob"][i])
            row["tumor_gap"] = safe_float(score_out["tumor_gap"][i])

            for ridx, rname in enumerate(role_names):
                row[f"prob_{rname}"] = safe_float(score_out["role_probs"][i, ridx])
                row[f"logit_{rname}"] = safe_float(score_out["role_logits"][i, ridx])

            # infer target role from candidate_type if possible
            ctype = str(row.get("candidate_type", ""))
            target_role = "unknown"
            if "tumor" in ctype:
                target_role = "tumor"
            elif "stroma" in ctype:
                target_role = "stroma"
            elif "normal" in ctype or "epithelium" in ctype:
                target_role = "normal_epithelium"
            row["target_role_inferred"] = target_role

            all_rows.append(row)

    scored_df = pd.DataFrame(all_rows)
    scored_csv = os.path.join(out_dir, "scored_candidates.csv")
    scored_df.to_csv(scored_csv, index=False)
    print(f"[Saved] {scored_csv}")

    with open(os.path.join(out_dir, "role_names.json"), "w", encoding="utf-8") as f:
        json.dump(role_names, f, indent=2, ensure_ascii=False)

    return scored_df, role_names


# =========================================================
# statistics
# =========================================================
def summarize_purity(scored_df: pd.DataFrame, role_names: List[str], out_dir: str):
    ensure_dir(out_dir)

    summary_rows = []

    group_keys = ["candidate_type"]
    if "target_role_inferred" in scored_df.columns:
        group_keys = ["candidate_type", "target_role_inferred"]

    grouped = scored_df.groupby(group_keys, dropna=False)

    for key, sub in grouped:
        if len(sub) == 0:
            continue

        if isinstance(key, tuple):
            candidate_type, target_role = key
        else:
            candidate_type, target_role = key, "unknown"

        slide_counts = sub["slide_id"].value_counts()
        num_unique_slides = int(slide_counts.shape[0])
        max_slide_fraction = float(slide_counts.max() / max(len(sub), 1))

        pred_counts = sub["pred_role"].value_counts(normalize=True).to_dict()
        second_counts = sub["second_role"].value_counts(normalize=True).to_dict()

        row = {
            "candidate_type": candidate_type,
            "target_role_inferred": target_role,
            "num_candidates": int(len(sub)),
            "num_unique_slides": num_unique_slides,
            "max_slide_fraction": max_slide_fraction,

            "mean_top1_gap": float(sub["top1_gap"].mean()),
            "median_top1_gap": float(sub["top1_gap"].median()),
            "std_top1_gap": float(sub["top1_gap"].std(ddof=0)),

            "mean_tumor_prob": float(sub["tumor_prob"].mean()),
            "median_tumor_prob": float(sub["tumor_prob"].median()),
            "mean_tumor_gap": float(sub["tumor_gap"].mean()),
            "median_tumor_gap": float(sub["tumor_gap"].median()),
        }

        for r in role_names:
            row[f"frac_pred_{r}"] = float(pred_counts.get(r, 0.0))
            row[f"frac_second_{r}"] = float(second_counts.get(r, 0.0))

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, "purity_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")

    # overall slide concentration
    slide_conc_rows = []
    for (candidate_type, slide_id), sub in scored_df.groupby(["candidate_type", "slide_id"]):
        slide_conc_rows.append({
            "candidate_type": candidate_type,
            "slide_id": slide_id,
            "count": int(len(sub)),
        })
    slide_conc_df = pd.DataFrame(slide_conc_rows)
    slide_conc_csv = os.path.join(out_dir, "slide_concentration.csv")
    slide_conc_df.to_csv(slide_conc_csv, index=False)
    print(f"[Saved] {slide_conc_csv}")

    return summary_df, slide_conc_df


# =========================================================
# plots
# =========================================================
def plot_score_distributions(scored_df: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)

    for candidate_type, sub in scored_df.groupby("candidate_type"):
        if len(sub) == 0:
            continue

        # top1 gap hist
        plt.figure(figsize=(7, 5))
        plt.hist(sub["top1_gap"], bins=20, alpha=0.8)
        plt.xlabel("top1_gap")
        plt.ylabel("count")
        plt.title(f"{candidate_type}: top1_gap histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{candidate_type}_top1_gap_hist.png"), dpi=180)
        plt.close()

        # tumor prob hist
        plt.figure(figsize=(7, 5))
        plt.hist(sub["tumor_prob"], bins=20, alpha=0.8)
        plt.xlabel("tumor_prob")
        plt.ylabel("count")
        plt.title(f"{candidate_type}: tumor_prob histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{candidate_type}_tumor_prob_hist.png"), dpi=180)
        plt.close()

        # tumor gap hist
        plt.figure(figsize=(7, 5))
        plt.hist(sub["tumor_gap"], bins=20, alpha=0.8)
        plt.xlabel("tumor_gap")
        plt.ylabel("count")
        plt.title(f"{candidate_type}: tumor_gap histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{candidate_type}_tumor_gap_hist.png"), dpi=180)
        plt.close()

        # per-role prob boxplot
        prob_cols = [c for c in sub.columns if c.startswith("prob_")]
        if len(prob_cols) > 0:
            plt.figure(figsize=(max(6, 1.2 * len(prob_cols)), 5))
            data = [sub[c].values for c in prob_cols]
            plt.boxplot(data, labels=prob_cols, vert=True)
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("prob")
            plt.title(f"{candidate_type}: per-role probability boxplot")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{candidate_type}_role_prob_box.png"), dpi=180)
            plt.close()


def plot_slide_concentration(slide_conc_df: pd.DataFrame, out_dir: str, topn: int = 20):
    ensure_dir(out_dir)

    for candidate_type, sub in slide_conc_df.groupby("candidate_type"):
        sub = sub.sort_values("count", ascending=False).head(topn)
        if len(sub) == 0:
            continue

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(sub)), sub["count"].values)
        plt.xticks(range(len(sub)), sub["slide_id"].values, rotation=60, ha="right")
        plt.ylabel("candidate count")
        plt.title(f"{candidate_type}: top-{topn} slide concentration")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{candidate_type}_slide_concentration_top{topn}.png"), dpi=180)
        plt.close()


def plot_confusion(scored_df: pd.DataFrame, role_names: List[str], out_dir: str):
    ensure_dir(out_dir)

    sub = scored_df[scored_df["target_role_inferred"].isin(role_names)].copy()
    if len(sub) == 0:
        print("[Warn] No rows with valid target_role_inferred, skip confusion heatmap.")
        return

    mat = np.zeros((len(role_names), len(role_names)), dtype=np.float32)
    role_to_idx = {r: i for i, r in enumerate(role_names)}

    for _, row in sub.iterrows():
        tr = row["target_role_inferred"]
        pr = row["pred_role"]
        if tr in role_to_idx and pr in role_to_idx:
            mat[role_to_idx[tr], role_to_idx[pr]] += 1.0

    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    mat_norm = mat / row_sums

    plt.figure(figsize=(6, 5))
    plt.imshow(mat_norm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(role_names)), role_names, rotation=30, ha="right")
    plt.yticks(range(len(role_names)), role_names)
    plt.xlabel("predicted role")
    plt.ylabel("target role")
    plt.title("Candidate purity confusion (row-normalized)")
    for i in range(len(role_names)):
        for j in range(len(role_names)):
            plt.text(j, i, f"{mat_norm[i, j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "candidate_confusion_heatmap.png"), dpi=180)
    plt.close()


# =========================================================
# montage
# =========================================================
def sample_balanced_by_slide(df: pd.DataFrame, total_k: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= total_k:
        return df.copy()

    rng = np.random.default_rng(seed)
    slide_groups = list(df.groupby("slide_id"))
    if len(slide_groups) == 0:
        return df.head(total_k).copy()

    per_slide = max(1, total_k // len(slide_groups))
    parts = []
    remain = total_k

    for _, sub in slide_groups:
        take = min(per_slide, len(sub), remain)
        if take <= 0:
            continue
        idx = rng.choice(len(sub), size=take, replace=False)
        parts.append(sub.iloc[idx])
        remain -= take
        if remain <= 0:
            break

    used_idx = set(pd.concat(parts).index.tolist()) if len(parts) > 0 else set()
    if remain > 0:
        leftover = df[~df.index.isin(used_idx)]
        if len(leftover) > 0:
            take = min(remain, len(leftover))
            idx = rng.choice(len(leftover), size=take, replace=False)
            parts.append(leftover.iloc[idx])

    return pd.concat(parts, axis=0).reset_index(drop=True)


def save_patch_montage(
    patch_df: pd.DataFrame,
    out_path: str,
    title: str,
    tile_size: int = 224,
    cols: int = 4,
):
    if len(patch_df) == 0:
        return

    rows = math.ceil(len(patch_df) / cols)
    title_h = 45
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size + title_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(0, 0, 0))

    for idx, row in enumerate(patch_df.itertuples()):
        svs_path = str(row.svs_path)
        x = int(row.coord_x)
        y = int(row.coord_y)
        h5_path = str(row.h5_path)

        _, patch_size, patch_level = load_coords_attrs(h5_path)

        slide = openslide.OpenSlide(svs_path)
        try:
            patch = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
        finally:
            slide.close()

        patch = patch.resize((tile_size, tile_size), resample=Image.BICUBIC)

        rr = idx // cols
        cc = idx % cols
        ox = cc * tile_size
        oy = rr * tile_size + title_h

        canvas.paste(patch, (ox, oy))

        overlay = ImageDraw.Draw(canvas)
        txt = (
            f"{row.slide_id}\n"
            f"pred={row.pred_role}\n"
            f"p={row.top1_gap:.3f} gap\n"
            f"tumor={row.tumor_prob:.3f}"
        )
        overlay.rectangle([ox, oy, ox + tile_size, oy + 48], fill=(255, 255, 255))
        overlay.text((ox + 4, oy + 2), txt, fill=(0, 0, 0))

    canvas.save(out_path)


def build_montages(scored_df: pd.DataFrame, out_dir: str, topk: int = 16, random_k: int = 16, balanced_k: int = 16):
    ensure_dir(out_dir)

    for candidate_type, sub in scored_df.groupby("candidate_type"):
        if len(sub) == 0:
            continue

        sub = sub.reset_index(drop=True)

        # top-k by top1 gap then max prob
        top_df = sub.sort_values(["top1_gap"], ascending=[False]).head(topk).copy()
        save_patch_montage(
            top_df,
            os.path.join(out_dir, f"{candidate_type}_topk_montage.png"),
            title=f"Top-{len(top_df)} fixed candidates: {candidate_type}",
        )

        # random
        rand_df = sub.sample(n=min(random_k, len(sub)), random_state=42).copy()
        save_patch_montage(
            rand_df,
            os.path.join(out_dir, f"{candidate_type}_random_montage.png"),
            title=f"Random-{len(rand_df)} fixed candidates: {candidate_type}",
        )

        # balanced by slide
        bal_df = sample_balanced_by_slide(sub, total_k=min(balanced_k, len(sub)), seed=42)
        save_patch_montage(
            bal_df,
            os.path.join(out_dir, f"{candidate_type}_balanced_by_slide_montage.png"),
            title=f"Balanced-by-slide-{len(bal_df)} candidates: {candidate_type}",
        )


# =========================================================
# main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Analyze fixed candidate purity")

    parser.add_argument("--candidate_csv", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--student_ckpt", type=str, required=True,
                        help="encoder to analyze; can be frozen or adapted student ckpt")
    parser.add_argument("--stage2_full_ckpt", type=str, required=True,
                        help="used only to load proj_l12")
    parser.add_argument("--role_proto_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_last_moe_output", action="store_true")

    parser.add_argument("--tumor_name", type=str, default="tumor")
    parser.add_argument("--negative_role_names", type=str, nargs="+", default=["stroma", "normal_epithelium"])

    parser.add_argument("--topk_montage", type=int, default=16)
    parser.add_argument("--random_montage", type=int, default=16)
    parser.add_argument("--balanced_montage", type=int, default=16)

    args = parser.parse_args()
    set_seed(42)
    ensure_dir(args.out_dir)

    print("=" * 80)
    print(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print("=" * 80)

    candidate_df = load_candidate_csv(args.candidate_csv)
    scored_df, role_names = score_candidate_csv(
        candidate_df=candidate_df,
        config_path=args.config,
        student_ckpt=args.student_ckpt,
        stage2_full_ckpt=args.stage2_full_ckpt,
        role_proto_dir=args.role_proto_dir,
        out_dir=args.out_dir,
        device=args.device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        use_last_moe_output=args.use_last_moe_output,
        tumor_name=args.tumor_name,
        negative_role_names=args.negative_role_names,
    )

    summary_df, slide_conc_df = summarize_purity(
        scored_df=scored_df,
        role_names=role_names,
        out_dir=args.out_dir,
    )

    plot_score_distributions(scored_df, out_dir=args.out_dir)
    plot_slide_concentration(slide_conc_df, out_dir=args.out_dir, topn=20)
    plot_confusion(scored_df, role_names=role_names, out_dir=args.out_dir)
    build_montages(
        scored_df=scored_df,
        out_dir=args.out_dir,
        topk=args.topk_montage,
        random_k=args.random_montage,
        balanced_k=args.balanced_montage,
    )

    with open(os.path.join(args.out_dir, "final_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_candidates": int(len(scored_df)),
                "role_names": role_names,
                "candidate_types": sorted(scored_df["candidate_type"].astype(str).unique().tolist()),
                "outputs": {
                    "scored_candidates": os.path.join(args.out_dir, "scored_candidates.csv"),
                    "purity_summary": os.path.join(args.out_dir, "purity_summary.csv"),
                    "slide_concentration": os.path.join(args.out_dir, "slide_concentration.csv"),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("[Done]")


if __name__ == "__main__":
    main()