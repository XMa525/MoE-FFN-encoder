import os
import json
import argparse
from typing import List, Dict, Any
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

from models.encoders.moe_encoder import MoEEncoder
from distillation.role_prototype_losses import RolePrototypeBank


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


class Stage2RoleEvaluator:
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
            raise ValueError(f"'tumor' not found in role_names: {self.role_names}")
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
    def compute_role_logits(self, patch_repr: torch.Tensor):
        feat = F.normalize(patch_repr, dim=-1)
        logits = feat @ self.role_protos.t()
        return logits

    @torch.no_grad()
    def compute_patch_scores(self, patch_repr: torch.Tensor):
        logits = self.compute_role_logits(patch_repr)
        sim_tumor = logits[:, self.tumor_role_id]

        other_ids = [i for i in range(logits.shape[1]) if i != self.tumor_role_id]
        sim_other_max, other_argmax_local = logits[:, other_ids].max(dim=1)
        score = sim_tumor - sim_other_max

        nearest_role = logits.argmax(dim=1)
        return {
            "logits": logits,
            "sim_tumor": sim_tumor,
            "sim_other_max": sim_other_max,
            "tumor_minus_max_other": score,
            "nearest_role": nearest_role,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--role-proto-dir", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-slides", type=int, default=0)
    parser.add_argument("--max-patches-per-slide", type=int, default=0)

    parser.add_argument("--mine-mode", type=str, default="top_ratio",
                        choices=["top_ratio", "score_threshold", "hybrid"])
    parser.add_argument("--top-ratio", type=float, default=0.05)
    parser.add_argument("--min-top-k", type=int, default=8)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--require-nearest-tumor", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    df = pd.read_csv(args.csv)
    df["svs_path"] = df["svs_path"].map(canonicalize_path)

    required_cols = [
        "slide_id", "slide_label", "svs_path", "coord_x", "coord_y",
        "coord_idx", "patch_level", "patch_size"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    neg_df = df[df["slide_label"] == 0].copy().reset_index(drop=True)
    if args.max_slides > 0:
        keep_slides = (
            neg_df["slide_id"].drop_duplicates()
            .sample(n=min(args.max_slides, neg_df["slide_id"].nunique()), random_state=args.seed)
            .tolist()
        )
        neg_df = neg_df[neg_df["slide_id"].isin(keep_slides)].reset_index(drop=True)

    print("[NEG] rows   =", len(neg_df))
    print("[NEG] slides =", neg_df["slide_id"].nunique())

    evaluator = Stage2RoleEvaluator(
        config_path=args.config,
        ckpt_path=args.ckpt,
        role_proto_dir=args.role_proto_dir,
        device=args.device,
        use_last_moe_output=True,
    )

    transform = T.Compose([
        T.ToImage(),
        T.Resize((224, 224), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])

    grouped = neg_df.groupby("slide_id")

    mined_rows: List[Dict[str, Any]] = []
    mined_features: List[np.ndarray] = []

    for slide_id, sdf in tqdm(grouped, total=len(grouped), desc="Mining hard negatives"):
        sdf = sdf.reset_index(drop=True)

        if args.max_patches_per_slide > 0 and len(sdf) > args.max_patches_per_slide:
            sdf = sdf.sample(n=args.max_patches_per_slide, random_state=args.seed).reset_index(drop=True)

        all_meta = []
        all_feat = []
        all_scores = []
        all_sim_tumor = []
        all_sim_other_max = []
        all_nearest_role = []

        buf_imgs = []
        buf_meta = []

        for _, row in sdf.iterrows():
            img = load_patch_image(
                row["svs_path"],
                row["coord_x"],
                row["coord_y"],
                row["patch_level"],
                row["patch_size"],
                transform,
            )
            buf_imgs.append(img)
            buf_meta.append({
                "slide_id": row["slide_id"],
                "slide_label": int(row["slide_label"]),
                "svs_path": row["svs_path"],
                "coord_x": int(row["coord_x"]),
                "coord_y": int(row["coord_y"]),
                "coord_idx": int(row["coord_idx"]) if pd.notna(row["coord_idx"]) else -1,
                "patch_level": int(row["patch_level"]),
                "patch_size": int(row["patch_size"]),
            })

            flush = (len(buf_imgs) == args.batch_size)
            if flush:
                images = torch.stack(buf_imgs, dim=0).to(args.device, non_blocking=True)
                patch_repr = evaluator.encode_patch_batch(images)
                out = evaluator.compute_patch_scores(patch_repr)

                all_feat.append(patch_repr.detach().cpu().numpy())
                all_scores.append(out["tumor_minus_max_other"].detach().cpu().numpy())
                all_sim_tumor.append(out["sim_tumor"].detach().cpu().numpy())
                all_sim_other_max.append(out["sim_other_max"].detach().cpu().numpy())
                all_nearest_role.append(out["nearest_role"].detach().cpu().numpy())
                all_meta.extend(buf_meta)

                buf_imgs = []
                buf_meta = []

        if len(buf_imgs) > 0:
            images = torch.stack(buf_imgs, dim=0).to(args.device, non_blocking=True)
            patch_repr = evaluator.encode_patch_batch(images)
            out = evaluator.compute_patch_scores(patch_repr)

            all_feat.append(patch_repr.detach().cpu().numpy())
            all_scores.append(out["tumor_minus_max_other"].detach().cpu().numpy())
            all_sim_tumor.append(out["sim_tumor"].detach().cpu().numpy())
            all_sim_other_max.append(out["sim_other_max"].detach().cpu().numpy())
            all_nearest_role.append(out["nearest_role"].detach().cpu().numpy())
            all_meta.extend(buf_meta)

        feat_np = np.concatenate(all_feat, axis=0)
        score_np = np.concatenate(all_scores, axis=0)
        sim_tumor_np = np.concatenate(all_sim_tumor, axis=0)
        sim_other_np = np.concatenate(all_sim_other_max, axis=0)
        nearest_np = np.concatenate(all_nearest_role, axis=0)

        idxs = np.arange(len(score_np))

        if args.mine_mode == "top_ratio":
            k = max(args.min_top_k, int(round(len(score_np) * args.top_ratio)))
            k = min(k, len(score_np))
            selected = np.argsort(-score_np)[:k]

        elif args.mine_mode == "score_threshold":
            selected = idxs[score_np >= args.score_threshold]

        else:  # hybrid
            k = max(args.min_top_k, int(round(len(score_np) * args.top_ratio)))
            k = min(k, len(score_np))
            topk = np.argsort(-score_np)[:k]
            thresh = idxs[score_np >= args.score_threshold]
            selected = np.unique(np.concatenate([topk, thresh], axis=0))

        if args.require_nearest_tumor:
            selected = selected[nearest_np[selected] == evaluator.tumor_role_id]

        for i in selected.tolist():
            meta = all_meta[i].copy()
            meta["tumor_minus_max_other"] = float(score_np[i])
            meta["sim_tumor"] = float(sim_tumor_np[i])
            meta["sim_other_max"] = float(sim_other_np[i])
            meta["nearest_role_id"] = int(nearest_np[i])
            meta["nearest_role_name"] = evaluator.role_names[int(nearest_np[i])]
            mined_rows.append(meta)
            mined_features.append(feat_np[i])

    mined_df = pd.DataFrame(mined_rows)
    feat_arr = np.stack(mined_features, axis=0) if len(mined_features) > 0 else np.zeros((0, 1280), dtype=np.float32)

    csv_path = os.path.join(args.out_dir, "hard_negative_candidates.csv")
    feat_path = os.path.join(args.out_dir, "hard_negative_features.npy")
    meta_path = os.path.join(args.out_dir, "mining_config.json")

    mined_df.to_csv(csv_path, index=False)
    np.save(feat_path, feat_arr)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("[Saved]", csv_path)
    print("[Saved]", feat_path)
    print("[Saved]", meta_path)
    print("[Mined] num candidates =", len(mined_df))
    if len(mined_df) > 0:
        print(mined_df["nearest_role_name"].value_counts())
        print(mined_df["tumor_minus_max_other"].describe())
        print(mined_df.groupby("slide_id").size().describe())


if __name__ == "__main__":
    main()