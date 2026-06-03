import os
import sys
import math
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import yaml
import openslide

from PIL import Image, ImageDraw, ImageFile
from tqdm import tqdm

from models.encoders.moe_encoder import MoEEncoder
from distillation.role_prototype_losses import RolePrototypeBank

ImageFile.LOAD_TRUNCATED_IMAGES = True


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Stage2PatchEvaluator:
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        role_proto_dir: str,
        device: str = "cuda",
        use_last_moe_output: bool = True,
    ):
        self.device = device
        self.use_last_moe_output = use_last_moe_output

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg

        self.student = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"]).to(device)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "student_state_dict" in ckpt:
            self.student.load_state_dict(ckpt["student_state_dict"], strict=True)
        elif "distiller_state_dict" in ckpt:
            dist_state = ckpt["distiller_state_dict"]
            student_state = {}
            for k, v in dist_state.items():
                if k.startswith("student."):
                    student_state[k[len("student."):]] = v
            self.student.load_state_dict(student_state, strict=True)
        else:
            raise KeyError("checkpoint must contain student_state_dict or distiller_state_dict")

        self.student.eval()

        self.proj_l12 = nn.Linear(384, 1280).to(device)
        loaded_proj = False
        if "distiller_state_dict" in ckpt:
            dist_state = ckpt["distiller_state_dict"]
            if "proj_l12.weight" in dist_state and "proj_l12.bias" in dist_state:
                self.proj_l12.load_state_dict({
                    "weight": dist_state["proj_l12.weight"],
                    "bias": dist_state["proj_l12.bias"],
                })
                loaded_proj = True

        if not loaded_proj:
            raise KeyError("proj_l12 not found in distiller_state_dict")

        self.proj_l12.eval()

        self.role_bank = RolePrototypeBank(
            prototype_path=os.path.join(role_proto_dir, "role_prototypes_init.npy"),
            role_names_path=os.path.join(role_proto_dir, "role_names.json"),
            normalize=True,
        )
        self.role_names = list(self.role_bank.role_names)
        self.role_protos = F.normalize(self.role_bank.prototypes.to(device), dim=-1)

        if "tumor" not in self.role_names:
            raise ValueError(f"'tumor' not found in role names: {self.role_names}")
        self.tumor_role_id = self.role_names.index("tumor")

    @torch.no_grad()
    def encode_patch_batch(self, images: torch.Tensor):
        student_out, gate_info_list, feature_dict, moe_feature_list = self.student(
            images,
            return_gates=True,
            mask=None,
            is_eval=True,
            return_features=True,
            offline_cluster_ids=None,
        )

        if self.use_last_moe_output and len(moe_feature_list) > 0:
            feat = moe_feature_list[-1]
        else:
            feat = feature_dict["layer_12"]

        patch_tokens = feat[:, 1:, :]
        patch_tokens_proj = self.proj_l12(patch_tokens)
        patch_repr = patch_tokens_proj.mean(dim=1)
        patch_repr = F.normalize(patch_repr, dim=-1)
        return patch_repr

    @torch.no_grad()
    def compute_role_logits(self, patch_repr):
        feat = F.normalize(patch_repr, dim=-1)
        logits = feat @ self.role_protos.t()
        return logits

    @torch.no_grad()
    def compute_tumor_evidence(self, patch_repr):
        logits = self.compute_role_logits(patch_repr)
        sim_tumor = logits[:, self.tumor_role_id]
        other_ids = [i for i in range(logits.shape[1]) if i != self.tumor_role_id]
        sim_other_max = logits[:, other_ids].max(dim=1).values
        score = sim_tumor - sim_other_max
        pred_role = torch.argmax(logits, dim=1)
        return score, sim_tumor, sim_other_max, pred_role


def load_patch_image_tensor(svs_path, x, y, patch_level, patch_size, transform):
    slide = openslide.OpenSlide(svs_path)
    try:
        image = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
    finally:
        slide.close()

    if transform is not None:
        image = transform(image)
    return image


def load_patch_image_pil(svs_path, x, y, patch_level, patch_size, out_size=224):
    slide = openslide.OpenSlide(svs_path)
    try:
        image = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
    finally:
        slide.close()

    if out_size is not None:
        image = image.resize((out_size, out_size))
    return image


def evaluate_two_models_on_df(
    baseline_eval,
    current_eval,
    df,
    batch_size=16,
    max_patches_per_slide=0,
):
    transform = T.Compose([
        T.ToImage(),
        T.Resize((224, 224), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])

    rows = []

    grouped = df.groupby("slide_id")
    for slide_id, sdf in tqdm(grouped, total=len(grouped), desc="Evaluating slides"):
        sdf = sdf.reset_index(drop=True)

        slide_label_vals = sdf["slide_label"].dropna().unique().tolist()
        if len(slide_label_vals) == 0:
            continue
        if len(slide_label_vals) > 1:
            raise ValueError(f"slide_id={slide_id} has inconsistent slide_label: {slide_label_vals}")
        slide_label = int(slide_label_vals[0])

        if max_patches_per_slide > 0 and len(sdf) > max_patches_per_slide:
            sdf = sdf.sample(n=max_patches_per_slide, random_state=42).reset_index(drop=True)

        meta_buf = []
        image_buf = []

        for i, row in sdf.iterrows():
            img = load_patch_image_tensor(
                svs_path=row["svs_path"],
                x=int(row["coord_x"]),
                y=int(row["coord_y"]),
                patch_level=int(row["patch_level"]),
                patch_size=int(row["patch_size"]),
                transform=transform,
            )
            image_buf.append(img)
            meta_buf.append(row.to_dict())

            if len(image_buf) == batch_size or i == len(sdf) - 1:
                images = torch.stack(image_buf, dim=0).to(baseline_eval.device, non_blocking=True)

                b_repr = baseline_eval.encode_patch_batch(images)
                b_score, b_tumor, b_other, b_pred = baseline_eval.compute_tumor_evidence(b_repr)

                c_repr = current_eval.encode_patch_batch(images)
                c_score, c_tumor, c_other, c_pred = current_eval.compute_tumor_evidence(c_repr)

                for j in range(len(meta_buf)):
                    meta = meta_buf[j]
                    rows.append({
                        "slide_id": str(meta["slide_id"]),
                        "slide_label": int(meta["slide_label"]),
                        "svs_path": meta["svs_path"],
                        "coord_x": int(meta["coord_x"]),
                        "coord_y": int(meta["coord_y"]),
                        "coord_idx": int(meta["coord_idx"]) if pd.notna(meta.get("coord_idx", np.nan)) else -1,
                        "patch_level": int(meta["patch_level"]),
                        "patch_size": int(meta["patch_size"]),
                        "baseline_score": float(b_score[j].detach().cpu()),
                        "baseline_sim_tumor": float(b_tumor[j].detach().cpu()),
                        "baseline_sim_other": float(b_other[j].detach().cpu()),
                        "baseline_pred_role": int(b_pred[j].detach().cpu()),
                        "current_score": float(c_score[j].detach().cpu()),
                        "current_sim_tumor": float(c_tumor[j].detach().cpu()),
                        "current_sim_other": float(c_other[j].detach().cpu()),
                        "current_pred_role": int(c_pred[j].detach().cpu()),
                    })

                image_buf = []
                meta_buf = []

    out_df = pd.DataFrame(rows)
    out_df["delta_score"] = out_df["current_score"] - out_df["baseline_score"]
    return out_df


def select_negative_suppressed_patches(
    df,
    baseline_score_min=0.05,
    baseline_sim_tumor_min=0.80,
    delta_max=-0.10,
    current_score_max=0.05,
    max_keep=64,
):
    neg = df[df["slide_label"] == 0].copy()

    sel = neg[
        (neg["baseline_score"] >= baseline_score_min) &
        (neg["baseline_sim_tumor"] >= baseline_sim_tumor_min) &
        (neg["delta_score"] <= delta_max) &
        (neg["current_score"] <= current_score_max)
    ].copy()

    if len(sel) == 0:
        return sel

    sel["rank_metric"] = (
        1.3 * sel["baseline_score"]
        + 0.8 * sel["baseline_sim_tumor"]
        - 1.0 * sel["current_score"]
        - 2.0 * sel["delta_score"]
    )
    sel = sel.sort_values("rank_metric", ascending=False).head(max_keep).reset_index(drop=True)
    return sel


def select_positive_preserved_highscore_patches(
    df,
    baseline_score_min=0.20,
    current_score_min=0.15,
    delta_min=-0.10,
    max_keep=64,
):
    pos = df[df["slide_label"] == 1].copy()

    sel = pos[
        (pos["baseline_score"] >= baseline_score_min) &
        (pos["current_score"] >= current_score_min) &
        (pos["delta_score"] >= delta_min)
    ].copy()

    if len(sel) == 0:
        return sel

    sel["rank_metric"] = (
        0.8 * sel["baseline_score"]
        + 1.2 * sel["current_score"]
        + 0.5 * sel["current_sim_tumor"]
        - 0.2 * sel["current_sim_other"]
    )
    sel = sel.sort_values("rank_metric", ascending=False).head(max_keep).reset_index(drop=True)
    return sel


def select_positive_tumor_anchor_patches(
    df,
    baseline_score_min=0.22,
    current_score_min=0.18,
    delta_min=-0.08,
    current_sim_tumor_min=0.88,
    current_sim_other_max=0.90,
    require_current_pred_tumor=True,
    tumor_role_id=0,
    max_keep=64,
):
    pos = df[df["slide_label"] == 1].copy()

    sel = pos[
        (pos["baseline_score"] >= baseline_score_min) &
        (pos["current_score"] >= current_score_min) &
        (pos["delta_score"] >= delta_min) &
        (pos["current_sim_tumor"] >= current_sim_tumor_min) &
        (pos["current_sim_other"] <= current_sim_other_max)
    ].copy()

    if require_current_pred_tumor:
        sel = sel[sel["current_pred_role"] == tumor_role_id].copy()

    if len(sel) == 0:
        return sel

    sel["rank_metric"] = (
        1.5 * sel["current_score"]
        + 1.2 * sel["current_sim_tumor"]
        - 0.8 * sel["current_sim_other"]
        + 0.3 * sel["delta_score"]
    )
    sel = sel.sort_values("rank_metric", ascending=False).head(max_keep).reset_index(drop=True)
    return sel


def select_positive_suppressed_tumor_like_patches(
    df,
    baseline_score_min=0.22,
    baseline_sim_tumor_min=0.88,
    delta_max=-0.10,
    current_score_max=0.10,
    require_baseline_pred_tumor=True,
    tumor_role_id=0,
    max_keep=64,
):
    pos = df[df["slide_label"] == 1].copy()

    sel = pos[
        (pos["baseline_score"] >= baseline_score_min) &
        (pos["baseline_sim_tumor"] >= baseline_sim_tumor_min) &
        (pos["delta_score"] <= delta_max) &
        (pos["current_score"] <= current_score_max)
    ].copy()

    if require_baseline_pred_tumor:
        sel = sel[sel["baseline_pred_role"] == tumor_role_id].copy()

    if len(sel) == 0:
        return sel

    sel["rank_metric"] = (
        1.4 * sel["baseline_score"]
        + 1.0 * sel["baseline_sim_tumor"]
        - 1.2 * sel["current_score"]
        - 2.0 * sel["delta_score"]
    )
    sel = sel.sort_values("rank_metric", ascending=False).head(max_keep).reset_index(drop=True)
    return sel


# =========================
# 新增1：positive slides top tumor-evidence patches
# =========================
def select_positive_top_patches_by_model(
    df,
    model_prefix="baseline",
    topk_per_slide=3,
    max_slides=16,
):
    """
    返回 positive slide 中每张 slide 按某模型 score 排序后的 top-k patch。
    model_prefix: "baseline" or "current"
    """
    pos = df[df["slide_label"] == 1].copy()
    if len(pos) == 0:
        return pos.iloc[:0].copy()

    keep_rows = []
    grouped = pos.groupby("slide_id")
    for slide_id, sdf in grouped:
        sdf = sdf.sort_values(f"{model_prefix}_score", ascending=False).head(topk_per_slide)
        sdf = sdf.copy()
        sdf["top_source_model"] = model_prefix
        sdf["top_rank_in_slide"] = np.arange(1, len(sdf) + 1)
        keep_rows.append(sdf)

    out = pd.concat(keep_rows, axis=0).reset_index(drop=True)

    # 优先保留当前 top1 更高的 slide
    slide_order = (
        out.groupby("slide_id")[f"{model_prefix}_score"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if max_slides > 0:
        slide_order = slide_order[:max_slides]

    out = out[out["slide_id"].isin(slide_order)].copy()
    out["slide_order_rank"] = out["slide_id"].map({sid: i for i, sid in enumerate(slide_order)})
    out = out.sort_values(["slide_order_rank", "top_rank_in_slide"]).reset_index(drop=True)
    return out


# =========================
# 新增2：baseline vs current same-slide paired comparison
# =========================
def build_positive_slide_top_pair_df(
    df,
    baseline_topk=2,
    current_topk=2,
    max_slides=12,
):
    """
    每张 positive slide 取 baseline top-k 和 current top-k，
    后面画成同一行对照图。
    """
    pos = df[df["slide_label"] == 1].copy()
    if len(pos) == 0:
        return pos.iloc[:0].copy()

    slide_priority = (
        pos.groupby("slide_id")[["baseline_score", "current_score"]]
        .max()
        .max(axis=1)
        .sort_values(ascending=False)
        .index.tolist()
    )
    if max_slides > 0:
        slide_priority = slide_priority[:max_slides]

    rows = []
    for slide_id in slide_priority:
        sdf = pos[pos["slide_id"] == slide_id].copy()

        btop = sdf.sort_values("baseline_score", ascending=False).head(baseline_topk).copy()
        btop["pair_panel"] = "baseline"
        btop["pair_rank"] = np.arange(1, len(btop) + 1)

        ctop = sdf.sort_values("current_score", ascending=False).head(current_topk).copy()
        ctop["pair_panel"] = "current"
        ctop["pair_rank"] = np.arange(1, len(ctop) + 1)

        rows.extend([btop, ctop])

    out = pd.concat(rows, axis=0).reset_index(drop=True)
    out["slide_order_rank"] = out["slide_id"].map({sid: i for i, sid in enumerate(slide_priority)})
    out = out.sort_values(["slide_order_rank", "pair_panel", "pair_rank"]).reset_index(drop=True)
    return out


def draw_patch_grid(
    df_sel,
    out_png,
    title,
    patch_size_vis=224,
    ncols=4,
    delta_eps=1e-6,
):
    if len(df_sel) == 0:
        dummy = Image.new("RGB", (1200, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(dummy)
        draw.text((20, 20), f"{title}\nNo patches selected.", fill=(0, 0, 0))
        dummy.save(out_png)
        print(f"[Saved] {out_png}")
        return

    n = len(df_sel)
    nrows = math.ceil(n / ncols)

    pad = 20
    text_h = 90
    title_h = 80
    legend_h = 30

    canvas_w = ncols * (patch_size_vis + pad) + pad
    canvas_h = title_h + legend_h + nrows * (patch_size_vis + text_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # 标题
    draw.text((20, 12), title, fill=(0, 0, 0))

    # 图例
    legend_y = 42
    # blue: delta > 0
    draw.rectangle([20, legend_y, 40, legend_y + 20], outline=(0, 102, 204), width=4)
    draw.text((48, legend_y - 1), "score up (delta > 0)", fill=(0, 0, 0))

    # red: delta < 0
    draw.rectangle([250, legend_y, 270, legend_y + 20], outline=(220, 20, 60), width=4)
    draw.text((278, legend_y - 1), "suppressed (delta < 0)", fill=(0, 0, 0))

    # black: ~ unchanged
    draw.rectangle([520, legend_y, 540, legend_y + 20], outline=(0, 0, 0), width=3)
    draw.text((548, legend_y - 1), "unchanged", fill=(0, 0, 0))

    for idx, row in df_sel.reset_index(drop=True).iterrows():
        r = idx // ncols
        c = idx % ncols

        x0 = pad + c * (patch_size_vis + pad)
        y0 = title_h + legend_h + pad + r * (patch_size_vis + text_h + pad)

        patch = load_patch_image_pil(
            svs_path=row["svs_path"],
            x=int(row["coord_x"]),
            y=int(row["coord_y"]),
            patch_level=int(row["patch_level"]),
            patch_size=int(row["patch_size"]),
            out_size=patch_size_vis,
        )
        canvas.paste(patch, (x0, y0))

        delta = float(row["delta_score"])

        if delta > delta_eps:
            border_color = (0, 102, 204)      # 蓝
            border_width = 5
        elif delta < -delta_eps:
            border_color = (220, 20, 60)      # 红
            border_width = 5
        else:
            border_color = (0, 0, 0)          # 黑
            border_width = 3

        txt = (
            f"slide={row['slide_id']}\n"
            f"coord=({int(row['coord_x'])},{int(row['coord_y'])})\n"
            f"base={row['baseline_score']:.3f}  cur={row['current_score']:.3f}\n"
            f"delta={row['delta_score']:.3f}"
        )
        draw.text((x0, y0 + patch_size_vis + 6), txt, fill=(0, 0, 0))

        draw.rectangle(
            [x0, y0, x0 + patch_size_vis, y0 + patch_size_vis],
            outline=border_color,
            width=border_width,
        )

    canvas.save(out_png)
    print(f"[Saved] {out_png}")


def draw_positive_slide_top_pair_grid(
    df_pair,
    out_png,
    title,
    patch_size_vis=196,
    baseline_topk=2,
    current_topk=2,
):
    """
    每张 positive slide 一行：
    [baseline top1][baseline top2] | [current top1][current top2]
    """
    if len(df_pair) == 0:
        dummy = Image.new("RGB", (1400, 300), color=(255, 255, 255))
        draw = ImageDraw.Draw(dummy)
        draw.text((20, 20), f"{title}\nNo slides selected.", fill=(0, 0, 0))
        dummy.save(out_png)
        print(f"[Saved] {out_png}")
        return

    slide_ids = df_pair["slide_id"].drop_duplicates().tolist()
    nrows = len(slide_ids)
    ncols = baseline_topk + current_topk

    pad = 16
    gap_mid = 36
    title_h = 56
    text_h = 96
    row_label_w = 180

    canvas_w = row_label_w + baseline_topk * (patch_size_vis + pad) + gap_mid + current_topk * (patch_size_vis + pad) + pad
    canvas_h = title_h + nrows * (patch_size_vis + text_h + pad) + pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), title, fill=(0, 0, 0))
    draw.text((row_label_w + 20, 15), "Baseline top patches", fill=(0, 0, 0))
    draw.text((row_label_w + baseline_topk * (patch_size_vis + pad) + gap_mid, 15), "Current top patches", fill=(0, 0, 0))

    slide_to_row = {sid: i for i, sid in enumerate(slide_ids)}

    for slide_id in slide_ids:
        sdf = df_pair[df_pair["slide_id"] == slide_id].copy()
        row_idx = slide_to_row[slide_id]
        y0_base = title_h + row_idx * (patch_size_vis + text_h + pad) + pad

        draw.text((20, y0_base + 12), f"slide={slide_id}", fill=(0, 0, 0))

        # baseline panel
        sb = sdf[sdf["pair_panel"] == "baseline"].sort_values("pair_rank")
        for j, (_, row) in enumerate(sb.iterrows()):
            x0 = row_label_w + j * (patch_size_vis + pad)
            patch = load_patch_image_pil(
                svs_path=row["svs_path"],
                x=int(row["coord_x"]),
                y=int(row["coord_y"]),
                patch_level=int(row["patch_level"]),
                patch_size=int(row["patch_size"]),
                out_size=patch_size_vis,
            )
            canvas.paste(patch, (x0, y0_base))
            txt = (
                f"B{int(row['pair_rank'])}\n"
                f"score={row['baseline_score']:.3f}\n"
                f"tumor={row['baseline_sim_tumor']:.3f}\n"
                f"other={row['baseline_sim_other']:.3f}"
            )
            draw.text((x0, y0_base + patch_size_vis + 6), txt, fill=(0, 0, 0))
            draw.rectangle([x0, y0_base, x0 + patch_size_vis, y0_base + patch_size_vis], outline=(0, 0, 0), width=2)

        # current panel
        sc = sdf[sdf["pair_panel"] == "current"].sort_values("pair_rank")
        for j, (_, row) in enumerate(sc.iterrows()):
            x0 = row_label_w + baseline_topk * (patch_size_vis + pad) + gap_mid + j * (patch_size_vis + pad)
            patch = load_patch_image_pil(
                svs_path=row["svs_path"],
                x=int(row["coord_x"]),
                y=int(row["coord_y"]),
                patch_level=int(row["patch_level"]),
                patch_size=int(row["patch_size"]),
                out_size=patch_size_vis,
            )
            canvas.paste(patch, (x0, y0_base))
            txt = (
                f"C{int(row['pair_rank'])}\n"
                f"score={row['current_score']:.3f}\n"
                f"tumor={row['current_sim_tumor']:.3f}\n"
                f"other={row['current_sim_other']:.3f}"
            )
            draw.text((x0, y0_base + patch_size_vis + 6), txt, fill=(0, 0, 0))
            draw.rectangle([x0, y0_base, x0 + patch_size_vis, y0_base + patch_size_vis], outline=(0, 0, 0), width=2)

    canvas.save(out_png)
    print(f"[Saved] {out_png}")


def write_summary(
    all_df,
    neg_sel,
    pos_preserved_sel,
    pos_anchor_sel,
    pos_suppressed_sel,
    pos_top_baseline_sel,
    pos_top_current_sel,
    out_txt,
):
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Rank/Residual patch suppression summary\n")
        f.write("=====================================\n\n")

        f.write(f"Total patches: {len(all_df)}\n")
        f.write(f"Negative patches: {int((all_df['slide_label'] == 0).sum())}\n")
        f.write(f"Positive patches: {int((all_df['slide_label'] == 1).sum())}\n\n")

        neg_df = all_df[all_df["slide_label"] == 0]
        pos_df = all_df[all_df["slide_label"] == 1]

        if len(neg_df) > 0:
            f.write("[Negative]\n")
            f.write(f"baseline_score mean = {neg_df['baseline_score'].mean():.6f}\n")
            f.write(f"current_score  mean = {neg_df['current_score'].mean():.6f}\n")
            f.write(f"delta_score    mean = {neg_df['delta_score'].mean():.6f}\n\n")

        if len(pos_df) > 0:
            f.write("[Positive]\n")
            f.write(f"baseline_score mean = {pos_df['baseline_score'].mean():.6f}\n")
            f.write(f"current_score  mean = {pos_df['current_score'].mean():.6f}\n")
            f.write(f"delta_score    mean = {pos_df['delta_score'].mean():.6f}\n\n")

        f.write("[Selected suppressed HN patches]\n")
        f.write(f"num_selected = {len(neg_sel)}\n")
        if len(neg_sel) > 0:
            f.write(f"baseline_score mean = {neg_sel['baseline_score'].mean():.6f}\n")
            f.write(f"current_score  mean = {neg_sel['current_score'].mean():.6f}\n")
            f.write(f"delta_score    mean = {neg_sel['delta_score'].mean():.6f}\n")
            f.write(f"baseline_sim_tumor mean = {neg_sel['baseline_sim_tumor'].mean():.6f}\n")
            f.write(f"current_sim_tumor  mean = {neg_sel['current_sim_tumor'].mean():.6f}\n")
        f.write("\n")

        f.write("[Selected preserved positive high-score patches]\n")
        f.write(f"num_selected = {len(pos_preserved_sel)}\n")
        if len(pos_preserved_sel) > 0:
            f.write(f"baseline_score mean = {pos_preserved_sel['baseline_score'].mean():.6f}\n")
            f.write(f"current_score  mean = {pos_preserved_sel['current_score'].mean():.6f}\n")
            f.write(f"delta_score    mean = {pos_preserved_sel['delta_score'].mean():.6f}\n")
            f.write(f"current_sim_tumor mean = {pos_preserved_sel['current_sim_tumor'].mean():.6f}\n")
            f.write(f"current_sim_other mean = {pos_preserved_sel['current_sim_other'].mean():.6f}\n")
        f.write("\n")

        f.write("[Selected positive tumor-anchor patches]\n")
        f.write(f"num_selected = {len(pos_anchor_sel)}\n")
        if len(pos_anchor_sel) > 0:
            f.write(f"baseline_score mean = {pos_anchor_sel['baseline_score'].mean():.6f}\n")
            f.write(f"current_score  mean = {pos_anchor_sel['current_score'].mean():.6f}\n")
            f.write(f"delta_score    mean = {pos_anchor_sel['delta_score'].mean():.6f}\n")
            f.write(f"current_sim_tumor mean = {pos_anchor_sel['current_sim_tumor'].mean():.6f}\n")
            f.write(f"current_sim_other mean = {pos_anchor_sel['current_sim_other'].mean():.6f}\n")
        f.write("\n")

        f.write("[Selected positive suppressed tumor-like patches]\n")
        f.write(f"num_selected = {len(pos_suppressed_sel)}\n")
        if len(pos_suppressed_sel) > 0:
            f.write(f"baseline_score mean = {pos_suppressed_sel['baseline_score'].mean():.6f}\n")
            f.write(f"current_score  mean = {pos_suppressed_sel['current_score'].mean():.6f}\n")
            f.write(f"delta_score    mean = {pos_suppressed_sel['delta_score'].mean():.6f}\n")
            f.write(f"baseline_sim_tumor mean = {pos_suppressed_sel['baseline_sim_tumor'].mean():.6f}\n")
            f.write(f"current_sim_tumor  mean = {pos_suppressed_sel['current_sim_tumor'].mean():.6f}\n")
            f.write(f"current_sim_other  mean = {pos_suppressed_sel['current_sim_other'].mean():.6f}\n")
        f.write("\n")

        f.write("[Positive top tumor-evidence patches by baseline]\n")
        f.write(f"num_selected = {len(pos_top_baseline_sel)}\n")
        if len(pos_top_baseline_sel) > 0:
            f.write(f"baseline_score mean = {pos_top_baseline_sel['baseline_score'].mean():.6f}\n")
            f.write(f"baseline_sim_tumor mean = {pos_top_baseline_sel['baseline_sim_tumor'].mean():.6f}\n")
            f.write(f"baseline_sim_other mean = {pos_top_baseline_sel['baseline_sim_other'].mean():.6f}\n")
        f.write("\n")

        f.write("[Positive top tumor-evidence patches by current]\n")
        f.write(f"num_selected = {len(pos_top_current_sel)}\n")
        if len(pos_top_current_sel) > 0:
            f.write(f"current_score mean = {pos_top_current_sel['current_score'].mean():.6f}\n")
            f.write(f"current_sim_tumor mean = {pos_top_current_sel['current_sim_tumor'].mean():.6f}\n")
            f.write(f"current_sim_other mean = {pos_top_current_sel['current_sim_other'].mean():.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--baseline-ckpt", type=str, required=True)
    parser.add_argument("--current-ckpt", type=str, required=True)
    parser.add_argument("--role-proto-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--max-patches-per-slide", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--neg-baseline-score-min", type=float, default=0.0)
    parser.add_argument("--neg-delta-max", type=float, default=-0.10)
    parser.add_argument("--neg-current-score-max", type=float, default=0.05)
    parser.add_argument("--neg-max-keep", type=int, default=32)

    parser.add_argument("--pos-baseline-score-min", type=float, default=0.20)
    parser.add_argument("--pos-current-score-min", type=float, default=0.15)
    parser.add_argument("--pos-delta-min", type=float, default=-0.10)
    parser.add_argument("--pos-max-keep", type=int, default=32)

    parser.add_argument("--anchor-baseline-score-min", type=float, default=0.22)
    parser.add_argument("--anchor-current-score-min", type=float, default=0.18)
    parser.add_argument("--anchor-delta-min", type=float, default=-0.08)
    parser.add_argument("--anchor-current-sim-tumor-min", type=float, default=0.88)
    parser.add_argument("--anchor-current-sim-other-max", type=float, default=0.90)
    parser.add_argument("--anchor-max-keep", type=int, default=32)

    parser.add_argument("--pos-supp-baseline-score-min", type=float, default=0.22)
    parser.add_argument("--pos-supp-baseline-sim-tumor-min", type=float, default=0.88)
    parser.add_argument("--pos-supp-delta-max", type=float, default=-0.10)
    parser.add_argument("--pos-supp-current-score-max", type=float, default=0.10)
    parser.add_argument("--pos-supp-max-keep", type=int, default=32)

    # 新增：positive top tumor evidence
    parser.add_argument("--pos-topk-per-slide", type=int, default=3)
    parser.add_argument("--pos-top-max-slides", type=int, default=16)

    # 新增：paired slide comparison
    parser.add_argument("--pair-baseline-topk", type=int, default=2)
    parser.add_argument("--pair-current-topk", type=int, default=2)
    parser.add_argument("--pair-max-slides", type=int, default=12)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.csv)
    df["svs_path"] = df["svs_path"].map(canonicalize_path)

    required_cols = [
        "slide_id", "slide_label", "svs_path",
        "coord_x", "coord_y", "coord_idx",
        "patch_level", "patch_size"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in csv: {missing}")

    if args.max_slides > 0:
        keep_slides = df["slide_id"].drop_duplicates().sample(
            n=min(args.max_slides, df["slide_id"].nunique()),
            random_state=args.seed,
        ).tolist()
        df = df[df["slide_id"].isin(keep_slides)].reset_index(drop=True)

    print("[CSV] num rows   =", len(df))
    print("[CSV] num slides =", df["slide_id"].nunique())
    print(df.groupby("slide_label")["slide_id"].nunique())

    baseline_eval = Stage2PatchEvaluator(
        config_path=args.config,
        ckpt_path=args.baseline_ckpt,
        role_proto_dir=args.role_proto_dir,
        device=args.device,
        use_last_moe_output=True,
    )

    current_eval = Stage2PatchEvaluator(
        config_path=args.config,
        ckpt_path=args.current_ckpt,
        role_proto_dir=args.role_proto_dir,
        device=args.device,
        use_last_moe_output=True,
    )

    compare_df = evaluate_two_models_on_df(
        baseline_eval=baseline_eval,
        current_eval=current_eval,
        df=df,
        batch_size=args.batch_size,
        max_patches_per_slide=args.max_patches_per_slide,
    )

    compare_csv = os.path.join(args.out_dir, "patch_compare_all.csv")
    compare_df.to_csv(compare_csv, index=False)
    print(f"[Saved] {compare_csv}")

    neg_sel = select_negative_suppressed_patches(
        compare_df,
        baseline_score_min=args.neg_baseline_score_min,
        delta_max=args.neg_delta_max,
        current_score_max=args.neg_current_score_max,
        max_keep=args.neg_max_keep,
    )
    neg_csv = os.path.join(args.out_dir, "negative_suppressed_hn_patches.csv")
    neg_sel.to_csv(neg_csv, index=False)
    print(f"[Saved] {neg_csv}")

    pos_preserved_sel = select_positive_preserved_highscore_patches(
        compare_df,
        baseline_score_min=args.pos_baseline_score_min,
        current_score_min=args.pos_current_score_min,
        delta_min=args.pos_delta_min,
        max_keep=args.pos_max_keep,
    )
    pos_preserved_csv = os.path.join(args.out_dir, "positive_preserved_highscore_patches.csv")
    pos_preserved_sel.to_csv(pos_preserved_csv, index=False)
    print(f"[Saved] {pos_preserved_csv}")

    pos_anchor_sel = select_positive_tumor_anchor_patches(
        compare_df,
        baseline_score_min=args.anchor_baseline_score_min,
        current_score_min=args.anchor_current_score_min,
        delta_min=args.anchor_delta_min,
        current_sim_tumor_min=args.anchor_current_sim_tumor_min,
        current_sim_other_max=args.anchor_current_sim_other_max,
        require_current_pred_tumor=True,
        tumor_role_id=baseline_eval.tumor_role_id,
        max_keep=args.anchor_max_keep,
    )
    pos_anchor_csv = os.path.join(args.out_dir, "positive_tumor_anchor_patches.csv")
    pos_anchor_sel.to_csv(pos_anchor_csv, index=False)
    print(f"[Saved] {pos_anchor_csv}")

    pos_suppressed_sel = select_positive_suppressed_tumor_like_patches(
        compare_df,
        baseline_score_min=args.pos_supp_baseline_score_min,
        baseline_sim_tumor_min=args.pos_supp_baseline_sim_tumor_min,
        delta_max=args.pos_supp_delta_max,
        current_score_max=args.pos_supp_current_score_max,
        require_baseline_pred_tumor=True,
        tumor_role_id=baseline_eval.tumor_role_id,
        max_keep=args.pos_supp_max_keep,
    )
    pos_suppressed_csv = os.path.join(args.out_dir, "positive_suppressed_tumor_like_patches.csv")
    pos_suppressed_sel.to_csv(pos_suppressed_csv, index=False)
    print(f"[Saved] {pos_suppressed_csv}")

    # 新增1：正片中 baseline / current 各自 top tumor-evidence
    pos_top_baseline_sel = select_positive_top_patches_by_model(
        compare_df,
        model_prefix="baseline",
        topk_per_slide=args.pos_topk_per_slide,
        max_slides=args.pos_top_max_slides,
    )
    pos_top_baseline_csv = os.path.join(args.out_dir, "positive_top_tumor_by_baseline.csv")
    pos_top_baseline_sel.to_csv(pos_top_baseline_csv, index=False)
    print(f"[Saved] {pos_top_baseline_csv}")

    pos_top_current_sel = select_positive_top_patches_by_model(
        compare_df,
        model_prefix="current",
        topk_per_slide=args.pos_topk_per_slide,
        max_slides=args.pos_top_max_slides,
    )
    pos_top_current_csv = os.path.join(args.out_dir, "positive_top_tumor_by_current.csv")
    pos_top_current_sel.to_csv(pos_top_current_csv, index=False)
    print(f"[Saved] {pos_top_current_csv}")

    # 新增2：同slide baseline vs current 对照
    pos_pair_df = build_positive_slide_top_pair_df(
        compare_df,
        baseline_topk=args.pair_baseline_topk,
        current_topk=args.pair_current_topk,
        max_slides=args.pair_max_slides,
    )
    pos_pair_csv = os.path.join(args.out_dir, "positive_slide_top_tumor_pair.csv")
    pos_pair_df.to_csv(pos_pair_csv, index=False)
    print(f"[Saved] {pos_pair_csv}")

    draw_patch_grid(
        neg_sel,
        out_png=os.path.join(args.out_dir, "negative_suppressed_hn_grid.png"),
        title="Negative slides: suppressed HN-like patches",
        patch_size_vis=224,
        ncols=4,
    )

    draw_patch_grid(
        pos_preserved_sel,
        out_png=os.path.join(args.out_dir, "positive_preserved_highscore_grid.png"),
        title="Positive slides: preserved high-score patches",
        patch_size_vis=224,
        ncols=4,
    )

    draw_patch_grid(
        pos_anchor_sel,
        out_png=os.path.join(args.out_dir, "positive_tumor_anchor_grid.png"),
        title="Positive slides: tumor-anchor candidate patches",
        patch_size_vis=224,
        ncols=4,
    )

    draw_patch_grid(
        pos_suppressed_sel,
        out_png=os.path.join(args.out_dir, "positive_suppressed_tumor_like_grid.png"),
        title="Positive slides: suppressed tumor-like patches",
        patch_size_vis=224,
        ncols=4,
    )

    # 新增图1
    draw_patch_grid(
        pos_top_baseline_sel,
        out_png=os.path.join(args.out_dir, "positive_baseline_top_tumor_grid.png"),
        title="Positive slides: baseline top tumor-evidence patches",
        patch_size_vis=224,
        ncols=args.pos_topk_per_slide,
    )

    draw_patch_grid(
        pos_top_current_sel,
        out_png=os.path.join(args.out_dir, "positive_current_top_tumor_grid.png"),
        title="Positive slides: current top tumor-evidence patches",
        patch_size_vis=224,
        ncols=args.pos_topk_per_slide,
    )

    # 新增图2
    draw_positive_slide_top_pair_grid(
        pos_pair_df,
        out_png=os.path.join(args.out_dir, "positive_slide_top_tumor_pair_grid.png"),
        title="Positive slides: baseline vs current top tumor-evidence patches",
        patch_size_vis=196,
        baseline_topk=args.pair_baseline_topk,
        current_topk=args.pair_current_topk,
    )

    write_summary(
        all_df=compare_df,
        neg_sel=neg_sel,
        pos_preserved_sel=pos_preserved_sel,
        pos_anchor_sel=pos_anchor_sel,
        pos_suppressed_sel=pos_suppressed_sel,
        pos_top_baseline_sel=pos_top_baseline_sel,
        pos_top_current_sel=pos_top_current_sel,
        out_txt=os.path.join(args.out_dir, "summary.txt"),
    )


if __name__ == "__main__":
    main()