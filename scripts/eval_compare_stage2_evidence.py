import os
import json
import math
import argparse
from collections import defaultdict
import sys
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
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ===== 你项目里的模块 =====
from models.encoders.moe_encoder import MoEEncoder
from distillation.role_prototype_losses import RolePrototypeBank


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Stage2EvidenceEvaluator:
    """
    只做 inference/eval，不需要 teacher。
    需要：
    - student model
    - proj_l12
    - role bank
    """

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

        # ---- build student ----
        self.student = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"]).to(device)

        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "student_state_dict" in ckpt:
            self.student.load_state_dict(ckpt["student_state_dict"], strict=True)
        elif "distiller_state_dict" in ckpt:
            # 兜底：若只有 distiller_state_dict，则抽 student.*
            dist_state = ckpt["distiller_state_dict"]
            student_state = {}
            for k, v in dist_state.items():
                if k.startswith("student."):
                    student_state[k[len("student."):]] = v
            self.student.load_state_dict(student_state, strict=True)
        else:
            raise KeyError("checkpoint must contain student_state_dict or distiller_state_dict")

        self.student.eval()

        # ---- build proj_l12 ----
        self.proj_l12 = nn.Linear(384, 1280).to(device)

        loaded_proj = False
        if "distiller_state_dict" in ckpt:
            dist_state = ckpt["distiller_state_dict"]
            w_key = "proj_l12.weight"
            b_key = "proj_l12.bias"
            if w_key in dist_state and b_key in dist_state:
                self.proj_l12.load_state_dict({
                    "weight": dist_state[w_key],
                    "bias": dist_state[b_key],
                })
                loaded_proj = True

        if not loaded_proj:
            raise KeyError(
                "proj_l12 weights not found in checkpoint['distiller_state_dict']. "
                "Please use a stage2 full checkpoint that contains distiller_state_dict."
            )

        self.proj_l12.eval()

        # ---- role bank ----
        self.role_bank = RolePrototypeBank(
            prototype_path=os.path.join(role_proto_dir, "role_prototypes_init.npy"),
            role_names_path=os.path.join(role_proto_dir, "role_names.json"),
            normalize=True,
        )
        self.role_names = list(self.role_bank.role_names)
        self.role_protos = F.normalize(
            self.role_bank.prototypes.to(device),
            dim=-1,
        )

        if "tumor" not in self.role_names:
            raise ValueError(f"'tumor' not found in role_names: {self.role_names}")
        self.tumor_role_id = self.role_names.index("tumor")

    @torch.no_grad()
    def encode_patch_batch(self, images: torch.Tensor):
        """
        images: [B, 3, H, W]
        return:
            patch_repr: [B, 1280]
        """
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

        patch_tokens = feat[:, 1:, :]            # [B, T, 384]
        patch_tokens_proj = self.proj_l12(patch_tokens)   # [B, T, 1280]

        patch_repr = patch_tokens_proj.mean(dim=1)        # [B, 1280]
        patch_repr = F.normalize(patch_repr, dim=-1)
        return patch_repr

    @torch.no_grad()
    def compute_role_logits(self, patch_repr: torch.Tensor):
        """
        patch_repr: [B, 1280]
        return:
            logits: [B, R]
        """
        feat = F.normalize(patch_repr, dim=-1)
        logits = feat @ self.role_protos.t()
        return logits

    @torch.no_grad()
    def compute_tumor_evidence(self, patch_repr: torch.Tensor):
        """
        tumor evidence = sim(tumor) - max(sim(other roles))
        """
        logits = self.compute_role_logits(patch_repr)   # [B, R]
        sim_tumor = logits[:, self.tumor_role_id]

        other_ids = [i for i in range(logits.shape[1]) if i != self.tumor_role_id]
        sim_other_max = logits[:, other_ids].max(dim=1).values
        score = sim_tumor - sim_other_max
        return score, sim_tumor, sim_other_max


def load_patch_image(svs_path, x, y, patch_level, patch_size, transform):
    slide = openslide.OpenSlide(svs_path)
    try:
        image = slide.read_region((x, y), patch_level, (patch_size, patch_size)).convert("RGB")
    finally:
        slide.close()

    if transform is not None:
        image = transform(image)
    return image


def evaluate_one_model(
    evaluator: Stage2EvidenceEvaluator,
    df: pd.DataFrame,
    out_prefix: str,
    batch_size: int = 16,
    topk_ratio: float = 0.1,
    topk_min: int = 4,
    topk_max: int = 16,
    max_patches_per_slide: int = 0,
):
    device = evaluator.device

    transform = T.Compose([
        T.ToImage(),
        T.Resize((224, 224), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])

    slide_rows = []
    top_patch_rows = []

    grouped = df.groupby("slide_id")

    for slide_id, sdf in tqdm(grouped, total=len(grouped), desc=f"Eval {out_prefix}"):
        sdf = sdf.reset_index(drop=True)

        slide_label_vals = sdf["slide_label"].dropna().unique().tolist()
        if len(slide_label_vals) == 0:
            continue
        if len(slide_label_vals) > 1:
            raise ValueError(f"slide_id={slide_id} has inconsistent slide_label: {slide_label_vals}")
        slide_label = int(slide_label_vals[0])

        if max_patches_per_slide > 0 and len(sdf) > max_patches_per_slide:
            sdf = sdf.sample(n=max_patches_per_slide, random_state=42).reset_index(drop=True)

        patch_scores = []
        sim_tumor_all = []
        sim_other_max_all = []

        meta_rows = []

        images_buf = []

        for i, row in sdf.iterrows():
            img = load_patch_image(
                svs_path=row["svs_path"],
                x=int(row["coord_x"]),
                y=int(row["coord_y"]),
                patch_level=int(row["patch_level"]),
                patch_size=int(row["patch_size"]),
                transform=transform,
            )
            images_buf.append(img)
            meta_rows.append({
                "slide_id": row["slide_id"],
                "slide_label": slide_label,
                "svs_path": row["svs_path"],
                "coord_x": int(row["coord_x"]),
                "coord_y": int(row["coord_y"]),
                "coord_idx": int(row["coord_idx"]) if pd.notna(row["coord_idx"]) else -1,
                "patch_level": int(row["patch_level"]),
                "patch_size": int(row["patch_size"]),
            })

            if len(images_buf) == batch_size or i == len(sdf) - 1:
                images = torch.stack(images_buf, dim=0).to(device, non_blocking=True)
                patch_repr = evaluator.encode_patch_batch(images)
                score, sim_tumor, sim_other_max = evaluator.compute_tumor_evidence(patch_repr)

                patch_scores.extend(score.detach().cpu().tolist())
                sim_tumor_all.extend(sim_tumor.detach().cpu().tolist())
                sim_other_max_all.extend(sim_other_max.detach().cpu().tolist())

                images_buf = []

        scores_np = np.asarray(patch_scores, dtype=np.float32)
        sim_tumor_np = np.asarray(sim_tumor_all, dtype=np.float32)
        sim_other_np = np.asarray(sim_other_max_all, dtype=np.float32)

        N = len(scores_np)
        if N == 0:
            continue

        k = max(topk_min, int(round(N * topk_ratio)))
        k = min(k, topk_max, N)

        top_idx = np.argsort(-scores_np)[:k]
        top_vals = scores_np[top_idx]

        slide_rows.append({
            "slide_id": slide_id,
            "slide_label": slide_label,
            "num_patches": N,
            "topk_k": k,
            "topk_mean_score": float(top_vals.mean()),
            "topk_max_score": float(top_vals.max()),
            "topk_min_score": float(top_vals.min()),
            "score_mean_all": float(scores_np.mean()),
            "score_std_all": float(scores_np.std()),
            "sim_tumor_mean_all": float(sim_tumor_np.mean()),
            "sim_other_max_mean_all": float(sim_other_np.mean()),
        })

        # 保存 top evidence patch 元信息
        for rank, idx in enumerate(top_idx.tolist(), start=1):
            meta = meta_rows[idx]
            top_patch_rows.append({
                "slide_id": slide_id,
                "slide_label": slide_label,
                "rank": rank,
                "score": float(scores_np[idx]),
                "sim_tumor": float(sim_tumor_np[idx]),
                "sim_other_max": float(sim_other_np[idx]),
                **meta,
            })

    slide_df = pd.DataFrame(slide_rows)
    top_patch_df = pd.DataFrame(top_patch_rows)

    slide_csv = f"{out_prefix}_per_slide.csv"
    top_patch_csv = f"{out_prefix}_top_patches.csv"
    summary_txt = f"{out_prefix}_summary.txt"

    slide_df.to_csv(slide_csv, index=False)
    top_patch_df.to_csv(top_patch_csv, index=False)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Num slides: {len(slide_df)}\n\n")

        if len(slide_df) > 0:
            f.write("[Overall]\n")
            f.write(slide_df.describe().to_string())
            f.write("\n\n")

            for label_val, name in [(0, "negative"), (1, "positive")]:
                sdf = slide_df[slide_df["slide_label"] == label_val]
                if len(sdf) == 0:
                    continue
                f.write(f"[{name} slides]\n")
                f.write(sdf.describe().to_string())
                f.write("\n\n")

                f.write(f"{name} topk_mean_score mean = {sdf['topk_mean_score'].mean():.6f}\n")
                f.write(f"{name} topk_max_score  mean = {sdf['topk_max_score'].mean():.6f}\n")
                f.write(f"{name} score_mean_all  mean = {sdf['score_mean_all'].mean():.6f}\n\n")

    print(f"[Saved] {slide_csv}")
    print(f"[Saved] {top_patch_csv}")
    print(f"[Saved] {summary_txt}")

    return slide_df, top_patch_df


def make_comparison_csv(df_a, df_b, name_a, name_b, out_csv):
    merged = df_a.merge(
        df_b,
        on=["slide_id", "slide_label"],
        how="inner",
        suffixes=(f"_{name_a}", f"_{name_b}")
    )

    for metric in ["topk_mean_score", "topk_max_score", "topk_min_score", "score_mean_all"]:
        ca = f"{metric}_{name_a}"
        cb = f"{metric}_{name_b}"
        if ca in merged.columns and cb in merged.columns:
            merged[f"delta_{metric}_{name_b}_minus_{name_a}"] = merged[cb] - merged[ca]

    merged.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")
    return merged


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

    parser.add_argument("--topk-ratio", type=float, default=0.1)
    parser.add_argument("--topk-min", type=int, default=4)
    parser.add_argument("--topk-max", type=int, default=16)

    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--max-patches-per-slide", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

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
    print("[CSV] slide_label counts:")
    print(df.groupby("slide_label")["slide_id"].nunique())

    # ===== baseline =====
    baseline_eval = Stage2EvidenceEvaluator(
        config_path=args.config,
        ckpt_path=args.baseline_ckpt,
        role_proto_dir=args.role_proto_dir,
        device=args.device,
        use_last_moe_output=True,
    )

    baseline_slide_df, baseline_top_df = evaluate_one_model(
        evaluator=baseline_eval,
        df=df,
        out_prefix=os.path.join(args.out_dir, "baseline"),
        batch_size=args.batch_size,
        topk_ratio=args.topk_ratio,
        topk_min=args.topk_min,
        topk_max=args.topk_max,
        max_patches_per_slide=args.max_patches_per_slide,
    )

    # ===== current =====
    current_eval = Stage2EvidenceEvaluator(
        config_path=args.config,
        ckpt_path=args.current_ckpt,
        role_proto_dir=args.role_proto_dir,
        device=args.device,
        use_last_moe_output=True,
    )

    current_slide_df, current_top_df = evaluate_one_model(
        evaluator=current_eval,
        df=df,
        out_prefix=os.path.join(args.out_dir, "current"),
        batch_size=args.batch_size,
        topk_ratio=args.topk_ratio,
        topk_min=args.topk_min,
        topk_max=args.topk_max,
        max_patches_per_slide=args.max_patches_per_slide,
    )

    # ===== merged compare =====
    compare_csv = os.path.join(args.out_dir, "baseline_vs_current_compare.csv")
    merged = make_comparison_csv(
        baseline_slide_df,
        current_slide_df,
        name_a="baseline",
        name_b="current",
        out_csv=compare_csv,
    )

    # 再输出一个简要对比 summary
    summary_txt = os.path.join(args.out_dir, "compare_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Num merged slides: {len(merged)}\n\n")

        for label_val, name in [(0, "negative"), (1, "positive")]:
            sdf = merged[merged["slide_label"] == label_val]
            if len(sdf) == 0:
                continue

            f.write(f"[{name} slides]\n")
            for metric in [
                "delta_topk_mean_score_current_minus_baseline" 
                "delta_topk_max_score_current_minus_baseline"
                "delta_topk_min_score_current_minus_baseline"
                "delta_score_mean_all_current_minus_baseline"
            ]:
                if metric in sdf.columns:
                    f.write(f"{metric}: mean={sdf[metric].mean():.6f}, std={sdf[metric].std():.6f}\n")
            f.write("\n")

    print(f"[Saved] {summary_txt}")


if __name__ == "__main__":
    main()