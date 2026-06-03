import os
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import yaml
import openslide

from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================
# project imports
# =========================
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.encoders.moe_encoder import MoEEncoder
from distillation.role_prototype_losses import RolePrototypeBank


# =========================
# utils
# =========================
def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_patch_image(svs_path, x, y, patch_level, patch_size, transform):
    slide = openslide.OpenSlide(svs_path)
    try:
        image = slide.read_region(
            (int(x), int(y)),
            int(patch_level),
            (int(patch_size), int(patch_size)),
        ).convert("RGB")
    finally:
        slide.close()

    if transform is not None:
        image = transform(image)
    return image


def batched(iterable, batch_size):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


# =========================
# evaluator
# =========================
class PatchHNEvaluator:
    """
    用 stage2 checkpoint 的 student + proj_l12:
    - 提取 patch teacher-space repr
    - 算 role logits
    - 算 tumor evidence
    - 算到 HN center 的相似度
    """
    def __init__(
        self,
        config_path: str,
        ckpt_path: str,
        role_proto_dir: str,
        hn_bank_dir: str,
        hn_classes,
        device="cuda",
        use_last_moe_output=True,
    ):
        self.device = device
        self.use_last_moe_output = use_last_moe_output
        self.hn_classes = list(hn_classes)

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg

        # student
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

        # proj_l12
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

        # role bank
        self.role_bank = RolePrototypeBank(
            prototype_path=os.path.join(role_proto_dir, "role_prototypes_init.npy"),
            role_names_path=os.path.join(role_proto_dir, "role_names.json"),
            normalize=True,
        )
        self.role_names = list(self.role_bank.role_names)
        self.role_protos = F.normalize(self.role_bank.prototypes.to(device), dim=-1)

        if "tumor" not in self.role_names:
            raise ValueError(f"'tumor' not in role names: {self.role_names}")
        self.tumor_role_id = self.role_names.index("tumor")

        # HN centers
        self.hn_centers = {}
        for cls_name in self.hn_classes:
            feat_path = os.path.join(hn_bank_dir, f"{cls_name}_features.npy")
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"HN feature bank not found: {feat_path}")

            arr = np.load(feat_path)
            if arr.ndim != 2:
                raise ValueError(f"HN feature bank must be 2D, got {arr.shape} for {cls_name}")

            feat = torch.from_numpy(arr).float().to(device)
            feat = F.normalize(feat, dim=-1)
            center = F.normalize(feat.mean(dim=0), dim=-1)
            self.hn_centers[cls_name] = center

        print("[HN centers loaded]")
        for k, v in self.hn_centers.items():
            print(f"  - {k}: {tuple(v.shape)}")

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
            feat = moe_feature_list[-1]   # [B, T+1, 384]
        else:
            feat = feature_dict["layer_12"]

        patch_tokens = feat[:, 1:, :]                # [B, T, 384]
        patch_tokens_proj = self.proj_l12(patch_tokens)  # [B, T, 1280]
        patch_repr = patch_tokens_proj.mean(dim=1)       # [B, 1280]
        patch_repr = F.normalize(patch_repr, dim=-1)
        return patch_repr

    @torch.no_grad()
    def compute_role_logits(self, patch_repr: torch.Tensor):
        feat = F.normalize(patch_repr, dim=-1)
        logits = feat @ self.role_protos.t()
        return logits

    @torch.no_grad()
    def compute_tumor_evidence(self, patch_repr: torch.Tensor):
        logits = self.compute_role_logits(patch_repr)  # [B, R]
        sim_tumor = logits[:, self.tumor_role_id]
        other_ids = [i for i in range(logits.shape[1]) if i != self.tumor_role_id]
        sim_other_max = logits[:, other_ids].max(dim=1).values
        score = sim_tumor - sim_other_max

        nearest_role_id = torch.argmax(logits, dim=1)
        return {
            "tumor_minus_max_other": score,
            "sim_tumor": sim_tumor,
            "sim_other_max": sim_other_max,
            "nearest_role_id": nearest_role_id,
        }

    @torch.no_grad()
    def compute_hn_similarity(self, patch_repr: torch.Tensor):
        """
        return dict:
            sim_to_<cls>, nearest_hn_class, nearest_hn_sim
        """
        patch_repr = F.normalize(patch_repr, dim=-1)

        sim_dict = {}
        sim_stack = []
        names = []

        for cls_name, center in self.hn_centers.items():
            sim = patch_repr @ center  # [B]
            sim_dict[f"sim_to_{cls_name}"] = sim
            sim_stack.append(sim.unsqueeze(1))
            names.append(cls_name)

        sim_mat = torch.cat(sim_stack, dim=1)   # [B, C]
        best_idx = torch.argmax(sim_mat, dim=1)
        best_sim = sim_mat.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        nearest_names = [names[i] for i in best_idx.detach().cpu().tolist()]

        sim_dict["nearest_hn_sim"] = best_sim
        sim_dict["nearest_hn_class"] = nearest_names
        return sim_dict


# =========================
# core analysis
# =========================
def run_patch_analysis(
    evaluator: PatchHNEvaluator,
    df: pd.DataFrame,
    out_dir: str,
    batch_size: int = 16,
    max_rows: int = 0,
    per_slide_topk: int = 16,
):
    os.makedirs(out_dir, exist_ok=True)

    transform = T.Compose([
        T.ToImage(),
        T.Resize((224, 224), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])

    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)

    patch_rows = []
    iterator = range(len(df))

    batch_items = []
    batch_meta = []

    for i in tqdm(iterator, total=len(df), desc="Patch analysis"):
        row = df.iloc[i]

        img = load_patch_image(
            svs_path=row["svs_path"],
            x=row["coord_x"],
            y=row["coord_y"],
            patch_level=row["patch_level"],
            patch_size=row["patch_size"],
            transform=transform,
        )

        batch_items.append(img)
        batch_meta.append(row.to_dict())

        flush = (len(batch_items) == batch_size) or (i == len(df) - 1)
        if not flush:
            continue

        images = torch.stack(batch_items, dim=0).to(evaluator.device, non_blocking=True)
        patch_repr = evaluator.encode_patch_batch(images)

        ev = evaluator.compute_tumor_evidence(patch_repr)
        hn = evaluator.compute_hn_similarity(patch_repr)

        bs = images.shape[0]
        for j in range(bs):
            meta = batch_meta[j]

            out_row = dict(meta)

            out_row["tumor_minus_max_other"] = float(ev["tumor_minus_max_other"][j].detach().cpu())
            out_row["sim_tumor"] = float(ev["sim_tumor"][j].detach().cpu())
            out_row["sim_other_max"] = float(ev["sim_other_max"][j].detach().cpu())

            rid = int(ev["nearest_role_id"][j].detach().cpu())
            out_row["nearest_role_id"] = rid
            out_row["nearest_role_name"] = evaluator.role_names[rid]

            for cls_name in evaluator.hn_classes:
                out_row[f"sim_to_{cls_name}"] = float(hn[f"sim_to_{cls_name}"][j].detach().cpu())

            out_row["nearest_hn_sim"] = float(hn["nearest_hn_sim"][j].detach().cpu())
            out_row["nearest_hn_class"] = hn["nearest_hn_class"][j]

            patch_rows.append(out_row)

        batch_items = []
        batch_meta = []

    patch_df = pd.DataFrame(patch_rows)
    patch_csv = os.path.join(out_dir, "patch_hn_overlap.csv")
    patch_df.to_csv(patch_csv, index=False)
    print(f"[Saved] {patch_csv}")

    # per-slide rank by tumor evidence
    patch_df["rank_within_slide_by_tumor_evidence"] = (
        patch_df.groupby("slide_id")["tumor_minus_max_other"]
        .rank(method="first", ascending=False)
    )
    patch_df["is_topk_patch_within_slide"] = (
        patch_df["rank_within_slide_by_tumor_evidence"] <= per_slide_topk
    ).astype(int)

    patch_df.to_csv(patch_csv, index=False)

    return patch_df


def summarize_topk_overlap(
    patch_df: pd.DataFrame,
    hn_classes,
    out_dir: str,
    per_slide_topk: int = 16,
):
    os.makedirs(out_dir, exist_ok=True)

    top_df = patch_df[patch_df["is_topk_patch_within_slide"] == 1].copy()

    # 1) 正负 top-k patch 的 HN 相似度统计
    rows = []
    for label_val, label_name in [(0, "negative"), (1, "positive")]:
        sdf = top_df[top_df["slide_label"] == label_val]
        if len(sdf) == 0:
            continue

        for cls_name in hn_classes:
            col = f"sim_to_{cls_name}"
            rows.append({
                "group": label_name,
                "metric": col,
                "mean": float(sdf[col].mean()),
                "std": float(sdf[col].std()),
                "median": float(sdf[col].median()),
                "p90": float(sdf[col].quantile(0.90)),
                "p95": float(sdf[col].quantile(0.95)),
                "count": int(len(sdf)),
            })

        rows.append({
            "group": label_name,
            "metric": "nearest_hn_sim",
            "mean": float(sdf["nearest_hn_sim"].mean()),
            "std": float(sdf["nearest_hn_sim"].std()),
            "median": float(sdf["nearest_hn_sim"].median()),
            "p90": float(sdf["nearest_hn_sim"].quantile(0.90)),
            "p95": float(sdf["nearest_hn_sim"].quantile(0.95)),
            "count": int(len(sdf)),
        })

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(out_dir, "topk_patch_hn_similarity_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")

    # 2) nearest_hn_class 占比
    nearest_rows = []
    for label_val, label_name in [(0, "negative"), (1, "positive")]:
        sdf = top_df[top_df["slide_label"] == label_val]
        if len(sdf) == 0:
            continue

        vc = sdf["nearest_hn_class"].value_counts(normalize=True)
        for cls_name, frac in vc.items():
            nearest_rows.append({
                "group": label_name,
                "nearest_hn_class": cls_name,
                "frac": float(frac),
                "count": int((sdf["nearest_hn_class"] == cls_name).sum()),
                "total": int(len(sdf)),
            })

    nearest_df = pd.DataFrame(nearest_rows)
    nearest_csv = os.path.join(out_dir, "topk_patch_nearest_hn_class_fraction.csv")
    nearest_df.to_csv(nearest_csv, index=False)
    print(f"[Saved] {nearest_csv}")

    # 3) 每张 slide 的 top1 / topk patch summary
    slide_rows = []
    for slide_id, sdf in patch_df.groupby("slide_id"):
        sdf = sdf.sort_values("tumor_minus_max_other", ascending=False).reset_index(drop=True)
        top1 = sdf.iloc[0]
        topk = sdf[sdf["is_topk_patch_within_slide"] == 1]

        row = {
            "slide_id": slide_id,
            "slide_label": int(sdf.iloc[0]["slide_label"]),
            "num_patches": int(len(sdf)),
            "top1_tumor_minus_max_other": float(top1["tumor_minus_max_other"]),
            "top1_nearest_hn_class": top1["nearest_hn_class"],
            "top1_nearest_hn_sim": float(top1["nearest_hn_sim"]),
        }
        for cls_name in hn_classes:
            row[f"top1_sim_to_{cls_name}"] = float(top1[f"sim_to_{cls_name}"])
            row[f"topk_mean_sim_to_{cls_name}"] = float(topk[f"sim_to_{cls_name}"].mean())

        row["topk_mean_tumor_minus_max_other"] = float(topk["tumor_minus_max_other"].mean())
        slide_rows.append(row)

    slide_df = pd.DataFrame(slide_rows)
    slide_csv = os.path.join(out_dir, "per_slide_topk_hn_overlap_summary.csv")
    slide_df.to_csv(slide_csv, index=False)
    print(f"[Saved] {slide_csv}")

    # 4) 高 HN 相似度 patch 提取
    hard_rows = []
    for cls_name in hn_classes:
        col = f"sim_to_{cls_name}"
        thr = float(top_df[col].quantile(0.95)) if len(top_df) > 0 else 1.0
        sdf = top_df[top_df[col] >= thr].copy()
        sdf["hn_class"] = cls_name
        sdf["hn_sim_threshold_p95"] = thr
        hard_rows.append(sdf)

    if len(hard_rows) > 0:
        hard_df = pd.concat(hard_rows, axis=0).reset_index(drop=True)
        hard_csv = os.path.join(out_dir, "high_hn_similarity_topk_patches.csv")
        hard_df.to_csv(hard_csv, index=False)
        print(f"[Saved] {hard_csv}")

    # 5) 文本 summary
    txt = os.path.join(out_dir, "hn_overlap_summary.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"Num all patches: {len(patch_df)}\n")
        f.write(f"Num top-k patches: {len(top_df)}\n\n")

        for label_val, label_name in [(0, "negative"), (1, "positive")]:
            sdf = top_df[top_df["slide_label"] == label_val]
            if len(sdf) == 0:
                continue

            f.write(f"[{label_name} top-k patches]\n")
            f.write(f"count = {len(sdf)}\n")
            f.write(f"tumor_minus_max_other mean = {sdf['tumor_minus_max_other'].mean():.6f}\n")
            f.write(f"nearest_hn_sim mean = {sdf['nearest_hn_sim'].mean():.6f}\n")
            for cls_name in hn_classes:
                f.write(f"sim_to_{cls_name} mean = {sdf[f'sim_to_{cls_name}'].mean():.6f}\n")
            f.write("nearest_hn_class frac:\n")
            f.write(sdf["nearest_hn_class"].value_counts(normalize=True).to_string())
            f.write("\n\n")

    print(f"[Saved] {txt}")


def merge_with_context_if_available(
    patch_df: pd.DataFrame,
    context_csv: str,
    out_dir: str,
):
    """
    把 patch overlap 结果和已有邻域分析表 merge，
    看 positive/negative 中高 HN 相似 patch 是否也有高 context gap
    """
    if context_csv is None or (not os.path.exists(context_csv)):
        print("[Skip] context csv not provided or not found")
        return

    os.makedirs(out_dir, exist_ok=True)

    ctx = pd.read_csv(context_csv)

    need_cols = ["slide_id", "coord_x", "coord_y"]
    for c in need_cols:
        if c not in ctx.columns:
            raise ValueError(f"context csv missing required col: {c}")

    merged = patch_df.merge(
        ctx,
        on=["slide_id", "coord_x", "coord_y"],
        how="left",
        suffixes=("", "_ctx"),
    )

    merged_csv = os.path.join(out_dir, "patch_hn_overlap_with_context.csv")
    merged.to_csv(merged_csv, index=False)
    print(f"[Saved] {merged_csv}")

    rows = []
    top_df = merged[merged["is_topk_patch_within_slide"] == 1].copy()

    for label_val, label_name in [(0, "negative"), (1, "positive")]:
        sdf = top_df[top_df["slide_label"] == label_val]
        if len(sdf) == 0:
            continue

        row = {
            "group": label_name,
            "count": int(len(sdf)),
            "mean_tumor_minus_max_other": float(sdf["tumor_minus_max_other"].mean()),
        }

        for c in [
            "neighbor_mean_score",
            "neighbor_top5_mean_score",
            "neighbor_max_score",
            "context_gap_mean_neighbor",
            "context_gap_top5_neighbor",
            "num_neighbors",
        ]:
            if c in sdf.columns:
                row[f"mean_{c}"] = float(sdf[c].dropna().mean()) if sdf[c].notna().any() else np.nan

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(out_dir, "topk_patch_context_overlap_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")


# =========================
# main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--role-proto-dir", type=str, required=True)

    parser.add_argument("--hn-bank-dir", type=str, required=True)
    parser.add_argument(
        "--hn-classes",
        type=str,
        nargs="+",
        default=["gland_like_sub3", "fibrous_dense"],
    )

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--context-csv", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--per-slide-topk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.csv)
    df["svs_path"] = df["svs_path"].map(canonicalize_path)

    required_cols = [
        "slide_id", "slide_label", "svs_path",
        "coord_x", "coord_y", "patch_level", "patch_size"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in csv: {missing}")

    evaluator = PatchHNEvaluator(
        config_path=args.config,
        ckpt_path=args.ckpt,
        role_proto_dir=args.role_proto_dir,
        hn_bank_dir=args.hn_bank_dir,
        hn_classes=args.hn_classes,
        device=args.device,
        use_last_moe_output=True,
    )

    patch_df = run_patch_analysis(
        evaluator=evaluator,
        df=df,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        per_slide_topk=args.per_slide_topk,
    )

    summarize_topk_overlap(
        patch_df=patch_df,
        hn_classes=args.hn_classes,
        out_dir=args.out_dir,
        per_slide_topk=args.per_slide_topk,
    )

    if args.context_csv is not None:
        merge_with_context_if_available(
            patch_df=patch_df,
            context_csv=args.context_csv,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()