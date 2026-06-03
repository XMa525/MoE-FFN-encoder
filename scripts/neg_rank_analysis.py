import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import yaml
import openslide
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from models.encoders.moe_encoder import MoEEncoder
from distillation.role_prototype_losses import RolePrototypeBank

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================
# 说明
# =========================================================
# 这个脚本聚焦你当前这条路线最值得分析的几件事：
# 1) negative ranking selected token 主要进入了哪些 expert
# 2) 按 expert 拆分后，selected negative token 的 gap 分布怎样
# 3) positive selected token 数量是否在萎缩，以及主要进入哪些 expert
# 4) E1 / free expert 各自接的 token 在 gap / sim_tumor / sim_other_max 上是什么特征
# 5) 可选：做 2D embedding（PCA，若安装 umap-learn 也可改成 UMAP）
#
# 这个脚本默认尽量少入侵你项目，但由于每个人项目的数据装载方式不同，
# 你只需要改下面两个适配点：
#
# ADAPTER POINT A:
#   load_project_objects(...)
#   - 按你项目方式返回 distiller / student / device
#
# ADAPTER POINT B:
#   build_slide_dataset(...)
#   - 按你当前用于 val / analysis 的 slide 清单返回 Dataset
#
# 其余分析逻辑尽量已经写完整。
# =========================================================


# =========================================================
# 数据结构
# =========================================================
@dataclass
class SlideItem:
    slide_id: str
    slide_label: int
    image_paths: List[str]


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


class WSICSVSlideDataset(Dataset):
    """
    CSV 需要至少包含：
      slide_id, slide_label, svs_path, coord_x, coord_y, patch_level, patch_size
    可选：coord_idx
    """
    def __init__(self, slide_items: List[SlideItem], image_size: int = 224):
        self.slide_items = slide_items
        self.image_size = image_size
        self.transform = T.Compose([
            T.ToImage(),
            T.Resize((image_size, image_size), antialias=True),
            T.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.slide_items)

    def _load_patch(self, svs_path: str, x: int, y: int, patch_level: int, patch_size: int) -> torch.Tensor:
        slide = openslide.OpenSlide(svs_path)
        try:
            image = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
        finally:
            slide.close()
        return self.transform(image)

    def __getitem__(self, idx: int):
        item = self.slide_items[idx]
        images = []
        coord_rows = []
        for row in item.image_paths:
            images.append(
                self._load_patch(
                    svs_path=row["svs_path"],
                    x=int(row["coord_x"]),
                    y=int(row["coord_y"]),
                    patch_level=int(row["patch_level"]),
                    patch_size=int(row["patch_size"]),
                )
            )
            coord_rows.append(row)

        return {
            "slide_id": item.slide_id,
            "slide_label": int(item.slide_label),
            "images": torch.stack(images, dim=0),
            "coord_rows": coord_rows,
        }


def collate_one_slide(batch):
    assert len(batch) == 1, "Use batch_size=1 for slide-level analysis"
    return batch[0]


# =========================================================
# 适配点 A：按你项目方式加载模型 / distiller / checkpoint
# =========================================================
class AnalysisDistillerAdapter(nn.Module):
    def __init__(
        self,
        student: nn.Module,
        proj_l12: nn.Module,
        role_bank: RolePrototypeBank,
        cfg: Dict,
        device: torch.device,
    ):
        super().__init__()
        self.student = student
        self.proj_l12 = proj_l12
        self.role_bank = role_bank
        self.device = device

        stage2_cfg = cfg.get("stage2_train", cfg)
        self.use_last_moe_output = bool(stage2_cfg.get("use_last_moe_output", True))
        self.free_expert_id = int(stage2_cfg.get("free_expert_id", 3))

        self.cond_rank_neg_topk = int(stage2_cfg.get("cond_rank_neg_topk", 8))
        self.cond_rank_neg_topk_ratio = float(stage2_cfg.get("cond_rank_neg_topk_ratio", 0.0))
        self.cond_rank_pos_topk = int(stage2_cfg.get("cond_rank_pos_topk", 4))
        self.cond_rank_pos_topk_ratio = float(stage2_cfg.get("cond_rank_pos_topk_ratio", 0.0))
        self.cond_rank_pos_select_mode = str(stage2_cfg.get("cond_rank_pos_select_mode", "tumor_minus_other"))
        self.cond_rank_pos_min_tumor_score = float(stage2_cfg.get("cond_rank_pos_min_tumor_score", -1e6))
        self.cond_rank_pos_min_gap = float(stage2_cfg.get("cond_rank_pos_min_gap", -1e6))
        self.cond_rank_allow_empty_pos = bool(stage2_cfg.get("cond_rank_allow_empty_pos", True))

        self.role_tau = float(stage2_cfg.get("role_tau", 0.07))
        self.role_names = list(role_bank.role_names)
        if "tumor" not in self.role_names:
            raise ValueError(f"'tumor' not found in role_names: {self.role_names}")
        self.tumor_role_id = self.role_names.index("tumor")
        self.role_protos = F.normalize(role_bank.prototypes.to(device), dim=-1)

        self.student.eval()
        self.proj_l12.eval()

    def get_last_dispatch_weight(self, gate_info_list, B, N):
        dispatch_weight = gate_info_list[-1]["dispatch_weight"]
        E = dispatch_weight.shape[-1]
        dispatch_weight = dispatch_weight.view(B, N + 1, E)
        return dispatch_weight[:, 1:, :]

    def compute_role_scores_for_patch_repr(self, patch_repr):
        logits = self.compute_role_affinity_logits(patch_repr)
        sim_tumor = logits[:, self.tumor_role_id]
        other_ids = [i for i in range(logits.shape[1]) if i != self.tumor_role_id]
        sim_other_max = logits[:, other_ids].max(dim=1).values
        gap = sim_tumor - sim_other_max
        return sim_tumor, sim_other_max, gap, logits

    def compute_role_affinity_logits(self, features_teacher_space):
        feats = F.normalize(features_teacher_space, dim=-1)
        return feats @ self.role_protos.t()

    def _resolve_cond_rank_topk(self, num_tokens, topk_fixed, topk_ratio=0.0):
        if num_tokens <= 0:
            return 0
        k = int(topk_fixed)
        if topk_ratio is not None and topk_ratio > 0:
            k_ratio = int(round(num_tokens * topk_ratio))
            k = max(k, k_ratio)
        return max(1, min(k, num_tokens))


def load_project_objects(config_path: str, ckpt_path: str, device: torch.device):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    student = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"]).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if "student_state_dict" in ckpt:
        student.load_state_dict(ckpt["student_state_dict"], strict=True)
    elif "distiller_state_dict" in ckpt:
        dist_state = ckpt["distiller_state_dict"]
        student_state = {}
        for k, v in dist_state.items():
            if k.startswith("student."):
                student_state[k[len("student."):]] = v
        student.load_state_dict(student_state, strict=True)
    else:
        raise KeyError("checkpoint must contain student_state_dict or distiller_state_dict")

    proj_l12 = nn.Linear(384, 1280).to(device)
    loaded_proj = False
    if "distiller_state_dict" in ckpt:
        dist_state = ckpt["distiller_state_dict"]
        if "proj_l12.weight" in dist_state and "proj_l12.bias" in dist_state:
            proj_l12.load_state_dict({
                "weight": dist_state["proj_l12.weight"],
                "bias": dist_state["proj_l12.bias"],
            })
            loaded_proj = True
    if not loaded_proj:
        raise KeyError(
            "proj_l12 weights not found in checkpoint['distiller_state_dict']. "
            "Please use a stage2 full checkpoint that contains distiller_state_dict."
        )

    role_proto_dir = cfg.get("stage2_train", cfg).get("role_proto_dir", None)
    if role_proto_dir is None:
        raise ValueError("role_proto_dir not found in config['stage2_train']")

    role_bank = RolePrototypeBank(
        prototype_path=os.path.join(role_proto_dir, "role_prototypes_init.npy"),
        role_names_path=os.path.join(role_proto_dir, "role_names.json"),
        normalize=True,
    )

    return AnalysisDistillerAdapter(
        student=student,
        proj_l12=proj_l12,
        role_bank=role_bank,
        cfg=cfg,
        device=device,
    ).to(device)


# =========================================================
# 适配点 B：按你项目方式构造 slide dataset
# =========================================================
def build_slide_dataset(manifest_csv: str, image_size: int = 224) -> Dataset:
    df = pd.read_csv(manifest_csv)
    df["svs_path"] = df["svs_path"].map(canonicalize_path)

    required_cols = {
        "slide_id", "slide_label", "svs_path",
        "coord_x", "coord_y", "patch_level", "patch_size"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest CSV missing columns: {missing}")

    if "coord_idx" not in df.columns:
        df["coord_idx"] = -1

    slide_items = []
    for slide_id, subdf in df.groupby("slide_id"):
        slide_label = int(subdf["slide_label"].iloc[0])
        rows = []
        for _, row in subdf.iterrows():
            rows.append({
                "svs_path": row["svs_path"],
                "coord_x": int(row["coord_x"]),
                "coord_y": int(row["coord_y"]),
                "coord_idx": int(row["coord_idx"]) if pd.notna(row["coord_idx"]) else -1,
                "patch_level": int(row["patch_level"]),
                "patch_size": int(row["patch_size"]),
            })
        slide_items.append(SlideItem(slide_id=slide_id, slide_label=slide_label, image_paths=rows))

    return WSICSVSlideDataset(slide_items=slide_items, image_size=image_size)


# =========================================================
# 通用 helper
# =========================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def tensor_to_float(x: torch.Tensor) -> float:
    return float(x.detach().cpu().item())


# =========================================================
# 关键前向：单张 slide 提取 patch-level 表征 / dispatch / role score
# =========================================================
@torch.no_grad()
def forward_one_slide(distiller, images: torch.Tensor, patch_batch_size: int, device: torch.device):
    """
    images: [N, 3, H, W]

    return dict:
      patch_repr:        [N, D_teacher]
      dispatch_weight:   [N, E]  image-level dispatch weight（token 平均）
      hard_expert:       [N]
      sim_tumor:         [N]
      sim_other_max:     [N]
      gap:               [N]
    """
    all_repr = []
    all_dispatch = []

    student = distiller.student
    student.eval()
    distiller.eval()

    for start in range(0, images.shape[0], patch_batch_size):
        end = min(start + patch_batch_size, images.shape[0])
        batch = images[start:end].to(device, non_blocking=True)

        _, gate_info_list, feature_dict, moe_feature_list = student(
            batch,
            return_gates=True,
            mask=None,
            is_eval=True,
            return_features=True,
            offline_cluster_ids=None,
        )

        if distiller.use_last_moe_output and len(moe_feature_list) > 0:
            feat = moe_feature_list[-1]
        else:
            feat = feature_dict["layer_12"]

        patch_tokens = feat[:, 1:, :]
        patch_tokens_proj = distiller.proj_l12(patch_tokens)
        patch_repr = patch_tokens_proj.mean(dim=1)
        patch_repr = F.normalize(patch_repr, dim=-1)

        B_img = batch.shape[0]
        N_tok = patch_tokens.shape[1]
        dispatch_weight_tok = distiller.get_last_dispatch_weight(gate_info_list, B_img, N_tok)
        dispatch_weight_img = dispatch_weight_tok.mean(dim=1)

        all_repr.append(patch_repr)
        all_dispatch.append(dispatch_weight_img)

    patch_repr = torch.cat(all_repr, dim=0)
    dispatch_weight = torch.cat(all_dispatch, dim=0)
    hard_expert = dispatch_weight.argmax(dim=-1)

    sim_tumor, sim_other_max, gap, _ = distiller.compute_role_scores_for_patch_repr(patch_repr)

    return {
        "patch_repr": patch_repr,
        "dispatch_weight": dispatch_weight,
        "hard_expert": hard_expert,
        "sim_tumor": sim_tumor,
        "sim_other_max": sim_other_max,
        "gap": gap,
    }


# =========================================================
# 当前路线的核心 token 选择逻辑
# =========================================================
def select_negative_rank_tokens(distiller, gap: torch.Tensor, sim_tumor: torch.Tensor) -> torch.Tensor:
    N = gap.shape[0]
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=gap.device)

    k = distiller._resolve_cond_rank_topk(
        num_tokens=N,
        topk_fixed=distiller.cond_rank_neg_topk,
        topk_ratio=distiller.cond_rank_neg_topk_ratio,
    )
    _, topk_idx = torch.topk(sim_tumor, k=k, largest=True)
    return topk_idx


def select_positive_rank_tokens(distiller, gap: torch.Tensor, sim_tumor: torch.Tensor) -> torch.Tensor:
    N = gap.shape[0]
    if N == 0:
        return torch.empty(0, dtype=torch.long, device=gap.device)

    if distiller.cond_rank_pos_select_mode == "tumor":
        ranking_signal = sim_tumor
    else:
        ranking_signal = gap

    valid_mask = torch.ones_like(ranking_signal, dtype=torch.bool)

    if distiller.cond_rank_pos_min_tumor_score > -1e5:
        valid_mask = valid_mask & (sim_tumor >= distiller.cond_rank_pos_min_tumor_score)
    if distiller.cond_rank_pos_min_gap > -1e5:
        valid_mask = valid_mask & (gap >= distiller.cond_rank_pos_min_gap)

    valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        if distiller.cond_rank_allow_empty_pos:
            return torch.empty(0, dtype=torch.long, device=gap.device)
        valid_idx = torch.arange(N, device=gap.device)

    ranking_signal_valid = ranking_signal[valid_idx]
    k = distiller._resolve_cond_rank_topk(
        num_tokens=int(valid_idx.numel()),
        topk_fixed=distiller.cond_rank_pos_topk,
        topk_ratio=distiller.cond_rank_pos_topk_ratio,
    )
    topk_local = torch.topk(ranking_signal_valid, k=k, largest=True).indices
    topk_idx = valid_idx[topk_local]
    return topk_idx


# =========================================================
# 分析主逻辑
# =========================================================
def analyze_route(
    distiller,
    dataset: Dataset,
    out_dir: str,
    patch_batch_size: int,
    max_slides: int,
    device: torch.device,
):
    ensure_dir(out_dir)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_one_slide)

    # per-token records
    selected_neg_rows = []
    selected_pos_rows = []
    expert_all_rows = []

    # for embedding visualization
    embed_feats = []
    embed_meta = []

    for slide_idx, batch in enumerate(loader):
        if max_slides > 0 and slide_idx >= max_slides:
            break

        slide_id = batch["slide_id"]
        slide_label = int(batch["slide_label"])
        images = batch["images"]

        out = forward_one_slide(
            distiller=distiller,
            images=images,
            patch_batch_size=patch_batch_size,
            device=device,
        )

        patch_repr = out["patch_repr"]
        dispatch_weight = out["dispatch_weight"]
        hard_expert = out["hard_expert"]
        sim_tumor = out["sim_tumor"]
        sim_other_max = out["sim_other_max"]
        gap = out["gap"]

        # 全 token expert 统计
        for i in range(patch_repr.shape[0]):
            expert_all_rows.append({
                "slide_id": slide_id,
                "slide_label": slide_label,
                "token_idx": int(i),
                "hard_expert": int(hard_expert[i].item()),
                "free_mass": float(dispatch_weight[i, distiller.free_expert_id].detach().cpu()),
                "sim_tumor": float(sim_tumor[i].detach().cpu()),
                "sim_other_max": float(sim_other_max[i].detach().cpu()),
                "gap": float(gap[i].detach().cpu()),
            })

        # selected negative tokens
        if slide_label == 0:
            neg_idx = select_negative_rank_tokens(distiller, gap, sim_tumor)
            for i in neg_idx.tolist():
                selected_neg_rows.append({
                    "slide_id": slide_id,
                    "slide_label": slide_label,
                    "token_idx": int(i),
                    "hard_expert": int(hard_expert[i].item()),
                    "free_mass": float(dispatch_weight[i, distiller.free_expert_id].detach().cpu()),
                    "sim_tumor": float(sim_tumor[i].detach().cpu()),
                    "sim_other_max": float(sim_other_max[i].detach().cpu()),
                    "gap": float(gap[i].detach().cpu()),
                })

        # selected positive tokens
        if slide_label == 1:
            pos_idx = select_positive_rank_tokens(distiller, gap, sim_tumor)
            for i in pos_idx.tolist():
                selected_pos_rows.append({
                    "slide_id": slide_id,
                    "slide_label": slide_label,
                    "token_idx": int(i),
                    "hard_expert": int(hard_expert[i].item()),
                    "free_mass": float(dispatch_weight[i, distiller.free_expert_id].detach().cpu()),
                    "sim_tumor": float(sim_tumor[i].detach().cpu()),
                    "sim_other_max": float(sim_other_max[i].detach().cpu()),
                    "gap": float(gap[i].detach().cpu()),
                })

        # embedding sample（为避免太大，每 slide 最多取 64 个）
        keep = min(64, patch_repr.shape[0])
        choose = torch.randperm(patch_repr.shape[0], device=patch_repr.device)[:keep]
        feat_sel = patch_repr[choose].detach().cpu().numpy()
        for local_j, global_i in enumerate(choose.tolist()):
            embed_feats.append(feat_sel[local_j])
            embed_meta.append({
                "slide_id": slide_id,
                "slide_label": slide_label,
                "hard_expert": int(hard_expert[global_i].item()),
                "gap": float(gap[global_i].detach().cpu()),
                "is_neg_selected": int(slide_label == 0 and global_i in set(select_negative_rank_tokens(distiller, gap, sim_tumor).tolist())),
                "is_pos_selected": int(slide_label == 1 and global_i in set(select_positive_rank_tokens(distiller, gap, sim_tumor).tolist())),
            })

        print(f"[Analyze] {slide_idx + 1}/{min(len(dataset), max_slides if max_slides > 0 else len(dataset))} | slide={slide_id} | label={slide_label}")

    df_neg = pd.DataFrame(selected_neg_rows)
    df_pos = pd.DataFrame(selected_pos_rows)
    df_all = pd.DataFrame(expert_all_rows)
    df_embed = pd.DataFrame(embed_meta)

    df_neg.to_csv(os.path.join(out_dir, "negative_selected_tokens.csv"), index=False)
    df_pos.to_csv(os.path.join(out_dir, "positive_selected_tokens.csv"), index=False)
    df_all.to_csv(os.path.join(out_dir, "all_tokens_expert_stats.csv"), index=False)
    df_embed.to_csv(os.path.join(out_dir, "embedding_meta.csv"), index=False)

    save_summary_tables(df_neg, df_pos, df_all, out_dir)
    make_plots(df_neg, df_pos, df_all, out_dir)
    make_embedding_plot(np.asarray(embed_feats), df_embed, out_dir)


# =========================================================
# 汇总表
# =========================================================
def save_summary_tables(df_neg: pd.DataFrame, df_pos: pd.DataFrame, df_all: pd.DataFrame, out_dir: str):
    summary = {}

    if len(df_neg) > 0:
        neg_exp = df_neg["hard_expert"].value_counts(normalize=True).sort_index()
        neg_exp.to_csv(os.path.join(out_dir, "negative_selected_expert_ratio.csv"), header=["ratio"])
        summary["negative_selected_expert_ratio"] = neg_exp.to_dict()

        neg_gap = df_neg.groupby("hard_expert")["gap"].agg(["count", "mean", "min", "max", "std"])
        neg_gap.to_csv(os.path.join(out_dir, "negative_selected_gap_by_expert.csv"))

    if len(df_pos) > 0:
        pos_exp = df_pos["hard_expert"].value_counts(normalize=True).sort_index()
        pos_exp.to_csv(os.path.join(out_dir, "positive_selected_expert_ratio.csv"), header=["ratio"])
        summary["positive_selected_expert_ratio"] = pos_exp.to_dict()

        pos_gap = df_pos.groupby("hard_expert")["gap"].agg(["count", "mean", "min", "max", "std"])
        pos_gap.to_csv(os.path.join(out_dir, "positive_selected_gap_by_expert.csv"))

    if len(df_all) > 0:
        all_exp = df_all["hard_expert"].value_counts(normalize=True).sort_index()
        all_exp.to_csv(os.path.join(out_dir, "all_token_expert_ratio.csv"), header=["ratio"])
        summary["all_token_expert_ratio"] = all_exp.to_dict()

        expert_profile = df_all.groupby("hard_expert")[["gap", "sim_tumor", "sim_other_max", "free_mass"]].agg(["mean", "std", "min", "max"])
        expert_profile.to_csv(os.path.join(out_dir, "expert_profile_summary.csv"))

    save_json(summary, os.path.join(out_dir, "summary.json"))


# =========================================================
# 图
# =========================================================
def make_plots(df_neg: pd.DataFrame, df_pos: pd.DataFrame, df_all: pd.DataFrame, out_dir: str):
    # 1) negative selected expert distribution
    if len(df_neg) > 0:
        vc = df_neg["hard_expert"].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        plt.bar(vc.index.astype(str), vc.values)
        plt.xlabel("Expert")
        plt.ylabel("# negative selected tokens")
        plt.title("Negative ranking-selected token expert distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "negative_selected_expert_distribution.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(7, 4))
        data = [df_neg[df_neg["hard_expert"] == e]["gap"].values for e in sorted(df_neg["hard_expert"].unique())]
        labels = [str(e) for e in sorted(df_neg["hard_expert"].unique())]
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.axhline(0.0)
        plt.xlabel("Expert")
        plt.ylabel("gap = sim_tumor - sim_other_max")
        plt.title("Negative selected gap by expert")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "negative_selected_gap_by_expert_box.png"), dpi=200)
        plt.close()

    # 2) positive selected expert distribution
    if len(df_pos) > 0:
        vc = df_pos["hard_expert"].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        plt.bar(vc.index.astype(str), vc.values)
        plt.xlabel("Expert")
        plt.ylabel("# positive selected tokens")
        plt.title("Positive ranking-selected token expert distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "positive_selected_expert_distribution.png"), dpi=200)
        plt.close()

        pos_counts = df_pos.groupby("slide_id").size().values
        plt.figure(figsize=(6, 4))
        plt.hist(pos_counts, bins=min(20, max(5, len(pos_counts))))
        plt.xlabel("# selected positive tokens / slide")
        plt.ylabel("# slides")
        plt.title("Positive selected token count per slide")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "positive_selected_count_per_slide.png"), dpi=200)
        plt.close()

    # 3) all-token expert ratio
    if len(df_all) > 0:
        vc = df_all["hard_expert"].value_counts().sort_index()
        plt.figure(figsize=(6, 4))
        plt.bar(vc.index.astype(str), vc.values)
        plt.xlabel("Expert")
        plt.ylabel("# all tokens")
        plt.title("All-token expert distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "all_token_expert_distribution.png"), dpi=200)
        plt.close()

        # E1 vs free gap hist
        for expert_id, name in [(1, "E1"), (3, "Free")]:
            sub = df_all[df_all["hard_expert"] == expert_id]
            if len(sub) == 0:
                continue
            plt.figure(figsize=(6, 4))
            plt.hist(sub["gap"].values, bins=40)
            plt.axvline(0.0)
            plt.xlabel("gap")
            plt.ylabel("# tokens")
            plt.title(f"Gap distribution for {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"gap_distribution_{name}.png"), dpi=200)
            plt.close()


# =========================================================
# 2D 可视化
# =========================================================
def make_embedding_plot(feats: np.ndarray, df_meta: pd.DataFrame, out_dir: str):
    if feats.size == 0 or len(df_meta) == 0:
        return

    # 为了少依赖，默认用 PCA 2D
    feats = feats.astype(np.float32)
    feats = feats - feats.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(feats, full_matrices=False)
    z = feats @ vt[:2].T

    # by expert
    plt.figure(figsize=(6, 5))
    for e in sorted(df_meta["hard_expert"].unique()):
        mask = df_meta["hard_expert"].values == e
        plt.scatter(z[mask, 0], z[mask, 1], s=8, label=f"E{e}", alpha=0.7)
    plt.legend(markerscale=2)
    plt.title("PCA of patch_repr by expert")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "embedding_pca_by_expert.png"), dpi=200)
    plt.close()

    # negative selected highlight
    plt.figure(figsize=(6, 5))
    base = np.ones(len(df_meta), dtype=bool)
    neg_sel = df_meta["is_neg_selected"].values.astype(bool)
    plt.scatter(z[base, 0], z[base, 1], s=6, alpha=0.15)
    if neg_sel.sum() > 0:
        plt.scatter(z[neg_sel, 0], z[neg_sel, 1], s=12, alpha=0.8)
    plt.title("PCA with negative selected tokens highlighted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "embedding_pca_negative_selected.png"), dpi=200)
    plt.close()


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_batch_size", type=int, default=8)
    parser.add_argument("--max_slides", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    distiller = load_project_objects(args.config, args.ckpt, device)
    dataset = build_slide_dataset(args.manifest_csv, image_size=args.image_size)

    analyze_route(
        distiller=distiller,
        dataset=dataset,
        out_dir=args.out_dir,
        patch_batch_size=args.patch_batch_size,
        max_slides=args.max_slides,
        device=device,
    )

    print(f"[Done] saved analysis to: {args.out_dir}")


if __name__ == "__main__":
    main()
