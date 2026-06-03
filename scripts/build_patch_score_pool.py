import os
import argparse
import sys
from typing import List, Dict

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


def set_seed(seed: int = 42):
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
            dist_state = ckpt["distiller_state_dict"]
            student_state = {}
            for k, v in dist_state.items():
                if k.startswith("student."):
                    student_state[k[len("student."):]] = v
            self.student.load_state_dict(student_state, strict=True)
        else:
            raise KeyError(
                "checkpoint must contain student_state_dict or distiller_state_dict"
            )

        self.student.eval()

        # ---- build proj_l12 ----
        self.proj_l12 = nn.Linear(384, 1280).to(device)

        loaded_proj = False
        if "distiller_state_dict" in ckpt:
            dist_state = ckpt["distiller_state_dict"]
            w_key = "proj_l12.weight"
            b_key = "proj_l12.bias"
            if w_key in dist_state and b_key in dist_state:
                self.proj_l12.load_state_dict(
                    {
                        "weight": dist_state[w_key],
                        "bias": dist_state[b_key],
                    }
                )
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
        _, _, feature_dict, moe_feature_list = self.student(
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

        patch_tokens = feat[:, 1:, :]                    # [B, T, 384]
        patch_tokens_proj = self.proj_l12(patch_tokens)  # [B, T, 1280]

        patch_repr = patch_tokens_proj.mean(dim=1)       # [B, 1280]
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


def load_patch_image(
    svs_path: str,
    x: int,
    y: int,
    patch_level: int,
    patch_size: int,
    transform,
):
    slide = openslide.OpenSlide(svs_path)
    try:
        image = slide.read_region(
            (x, y),
            patch_level,
            (patch_size, patch_size),
        ).convert("RGB")
    finally:
        slide.close()

    if transform is not None:
        image = transform(image)
    return image


def validate_input_df(df: pd.DataFrame):
    required_cols = [
        "slide_id",
        "slide_label",
        "svs_path",
        "coord_x",
        "coord_y",
        "coord_idx",
        "patch_level",
        "patch_size",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in csv: {missing}")
def filter_df_by_candidate_slides(
    df: pd.DataFrame,
    candidate_csv: str,
    candidate_subclass_id: int | None = None,
):
    cand = pd.read_csv(candidate_csv)

    required_cols = ["slide_id"]
    missing = [c for c in required_cols if c not in cand.columns]
    if missing:
        raise ValueError(f"Missing columns in candidate csv: {missing}")

    if candidate_subclass_id is not None:
        if "subclass_id" not in cand.columns:
            raise ValueError(
                "candidate_subclass_id was given, but candidate csv has no subclass_id column"
            )
        cand = cand[cand["subclass_id"] == candidate_subclass_id].copy()

    keep_slides = cand["slide_id"].drop_duplicates().tolist()
    print(f"[Candidate] num candidate rows   = {len(cand)}")
    print(f"[Candidate] num candidate slides = {len(keep_slides)}")

    df = df[df["slide_id"].isin(keep_slides)].reset_index(drop=True)
    print(f"[Filtered by candidate slides] num rows   = {len(df)}")
    print(f"[Filtered by candidate slides] num slides = {df['slide_id'].nunique()}")

    return df

def build_patch_score_pool(
    evaluator: Stage2EvidenceEvaluator,
    df: pd.DataFrame,
    out_csv: str,
    batch_size: int = 16,
    max_patches_per_slide: int = 0,
):
    device = evaluator.device

    transform = T.Compose([
        T.ToImage(),
        T.Resize((224, 224), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])

    patch_rows: List[Dict] = []
    grouped = df.groupby("slide_id")

    for slide_id, sdf in tqdm(grouped, total=len(grouped), desc="Build patch score pool"):
        sdf = sdf.reset_index(drop=True)

        slide_label_vals = sdf["slide_label"].dropna().unique().tolist()
        if len(slide_label_vals) == 0:
            continue
        if len(slide_label_vals) > 1:
            raise ValueError(
                f"slide_id={slide_id} has inconsistent slide_label: {slide_label_vals}"
            )
        slide_label = int(slide_label_vals[0])

        if max_patches_per_slide > 0 and len(sdf) > max_patches_per_slide:
            sdf = sdf.sample(
                n=max_patches_per_slide,
                random_state=42,
            ).reset_index(drop=True)

        batch_images = []
        batch_meta = []

        for i, row in sdf.iterrows():
            try:
                img = load_patch_image(
                    svs_path=row["svs_path"],
                    x=int(row["coord_x"]),
                    y=int(row["coord_y"]),
                    patch_level=int(row["patch_level"]),
                    patch_size=int(row["patch_size"]),
                    transform=transform,
                )
            except Exception as e:
                print(
                    f"[WARN] failed to load patch: slide_id={row['slide_id']} "
                    f"coord=({row['coord_x']},{row['coord_y']}) "
                    f"svs={row['svs_path']} err={e}"
                )
                continue

            batch_images.append(img)
            batch_meta.append(
                {
                    "slide_id": row["slide_id"],
                    "slide_label": slide_label,
                    "svs_path": row["svs_path"],
                    "coord_x": int(row["coord_x"]),
                    "coord_y": int(row["coord_y"]),
                    "coord_idx": int(row["coord_idx"]) if pd.notna(row["coord_idx"]) else -1,
                    "patch_level": int(row["patch_level"]),
                    "patch_size": int(row["patch_size"]),
                }
            )

            flush = (len(batch_images) == batch_size) or (i == len(sdf) - 1)
            if not flush:
                continue

            images = torch.stack(batch_images, dim=0).to(device, non_blocking=True)

            with torch.no_grad():
                patch_repr = evaluator.encode_patch_batch(images)
                score, sim_tumor, sim_other_max = evaluator.compute_tumor_evidence(patch_repr)

            score_cpu = score.detach().cpu().tolist()
            sim_tumor_cpu = sim_tumor.detach().cpu().tolist()
            sim_other_cpu = sim_other_max.detach().cpu().tolist()

            for meta, s, st, so in zip(batch_meta, score_cpu, sim_tumor_cpu, sim_other_cpu):
                patch_rows.append(
                    {
                        **meta,
                        "sim_tumor": float(st),
                        "sim_other_max": float(so),
                        "score": float(s),
                    }
                )

            batch_images = []
            batch_meta = []

    patch_df = pd.DataFrame(patch_rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    patch_df.to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

    # 额外输出一个简短 summary
    summary_txt = out_csv.replace(".csv", "_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Num rows: {len(patch_df)}\n")
        if len(patch_df) > 0:
            f.write(f"Num slides: {patch_df['slide_id'].nunique()}\n\n")
            f.write("[Overall]\n")
            f.write(
                patch_df[["sim_tumor", "sim_other_max", "score"]].describe().to_string()
            )
            f.write("\n\n")

            for label_val, name in [(0, "negative"), (1, "positive")]:
                sdf = patch_df[patch_df["slide_label"] == label_val]
                if len(sdf) == 0:
                    continue
                f.write(f"[{name} patches]\n")
                f.write(
                    sdf[["sim_tumor", "sim_other_max", "score"]].describe().to_string()
                )
                f.write("\n\n")

    print(f"[Saved] {summary_txt}")
    return patch_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--role-proto-dir", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--max-patches-per-slide", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-last-moe-output", action="store_true")
    parser.add_argument(
        "--candidate-csv",
        type=str,
        default=None,
        help="Optional candidate csv. If given, only keep slides appearing in this csv.",
    )
    parser.add_argument(
        "--candidate-subclass-id",
        type=int,
        default=None,
        help="Optional subclass_id filter for candidate csv.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    df = pd.read_csv(args.csv)
    df["svs_path"] = df["svs_path"].map(canonicalize_path)
    validate_input_df(df)

    # 先按 candidate slide 过滤，确保 pool 和 candidate 能对上
    if args.candidate_csv is not None:
        df = filter_df_by_candidate_slides(
            df=df,
            candidate_csv=args.candidate_csv,
            candidate_subclass_id=args.candidate_subclass_id,
        )

    # 再决定是否继续抽样 slide（一般 candidate-driven 之后通常就不需要了）
    if args.max_slides > 0 and df["slide_id"].nunique() > args.max_slides:
        keep_slides = df["slide_id"].drop_duplicates().sample(
            n=min(args.max_slides, df["slide_id"].nunique()),
            random_state=args.seed,
        ).tolist()
        df = df[df["slide_id"].isin(keep_slides)].reset_index(drop=True)

    print("[CSV] num rows   =", len(df))
    print("[CSV] num slides =", df["slide_id"].nunique())
    print("[CSV] slide_label counts:")
    print(df.groupby("slide_label")["slide_id"].nunique())

    evaluator = Stage2EvidenceEvaluator(
        config_path=args.config,
        ckpt_path=args.ckpt,
        role_proto_dir=args.role_proto_dir,
        device=args.device,
        use_last_moe_output=args.use_last_moe_output,
    )

    build_patch_score_pool(
        evaluator=evaluator,
        df=df,
        out_csv=args.out_csv,
        batch_size=args.batch_size,
        max_patches_per_slide=args.max_patches_per_slide,
    )


if __name__ == "__main__":
    main()