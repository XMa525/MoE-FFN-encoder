#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import yaml
import h5py
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import openslide
import pandas as pd

from PIL import Image
from tqdm import tqdm
import torchvision.transforms.v2 as T

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.encoders.moe_encoder import MoEEncoder


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def build_transform(image_size: int = 224):
    return T.Compose([
        T.ToImage(),
        T.Resize((image_size, image_size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
    ])


def make_label_column_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("Need 'label' or 'slide_binary_label' in slides_csv.")
    return df


def load_slides_csv(
    slides_csv: str,
    split: Optional[str] = None,
    benchmark_split: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_csv(slides_csv)
    df = make_label_column_if_needed(df)

    required_cols = {"slide_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"slides_csv missing columns: {missing}")

    if benchmark_split is not None:
        if "benchmark_split" not in df.columns:
            raise ValueError("benchmark_split was specified but csv has no 'benchmark_split' column")
        df = df[df["benchmark_split"] == benchmark_split].copy()
    elif split is not None:
        if "split" not in df.columns:
            raise ValueError("split was specified but csv has no 'split' column")
        df = df[df["split"] == split].copy()

    df = df.reset_index(drop=True)
    return df


def read_coords_from_h5_with_attrs(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs.items())

    patch_size = int(attrs.get("patch_size", 256))
    patch_level = int(attrs.get("patch_level", 0))
    return torch.from_numpy(coords).long(), patch_size, patch_level


def read_patch_from_wsi(
    slide: openslide.OpenSlide,
    coord_xy: Tuple[int, int],
    patch_size: int = 256,
    read_level: int = 0,
) -> Image.Image:
    x, y = int(coord_xy[0]), int(coord_xy[1])
    patch = slide.read_region((x, y), read_level, (patch_size, patch_size)).convert("RGB")
    return patch


# =========================================================
# Load transfer student + original proj_l12
# =========================================================
def load_transfer_bundle(
    config_path: str,
    student_ckpt_path: str,
    proj_source_ckpt_path: str,
    device: str = "cuda",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not os.path.exists(student_ckpt_path):
        raise FileNotFoundError(f"Student checkpoint not found: {student_ckpt_path}")
    if not os.path.exists(proj_source_ckpt_path):
        raise FileNotFoundError(f"Projection source checkpoint not found: {proj_source_ckpt_path}")

    student_ckpt = torch.load(student_ckpt_path, map_location="cpu")
    proj_ckpt = torch.load(proj_source_ckpt_path, map_location="cpu")

    if "student_state_dict" not in student_ckpt:
        raise KeyError("student_state_dict not found in student checkpoint")

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    model.load_state_dict(student_ckpt["student_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    if "distiller_state_dict" not in proj_ckpt:
        raise KeyError("distiller_state_dict not found in projection source checkpoint")

    distiller_sd = proj_ckpt["distiller_state_dict"]
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

    for p in model.parameters():
        p.requires_grad = False
    for p in proj_l12.parameters():
        p.requires_grad = False

    print("Loaded transfer student + original proj_l12")
    print(f"student_ckpt: {student_ckpt_path}")
    print(f"proj_source_ckpt: {proj_source_ckpt_path}")
    print(f"Current moe_layers_idx = {model.moe_layers_idx}")
    print(f"proj_l12 shape: {proj_in_dim} -> {proj_out_dim}")

    return model, proj_l12, cfg


# =========================================================
# Transfer raw feature extractor
# =========================================================
class TransferPatchFeatureExtractor(nn.Module):
    """
    For each input patch image:
      - run transfer-trained student encoder
      - take last MoE block patch token feature (preferred) or layer_12
      - output raw patch feature [B, 384]
      - optionally project to teacher space [B, 1280]
    """

    def __init__(
        self,
        config_path: str,
        student_ckpt_path: str,
        proj_source_ckpt_path: str,
        device: str = "cuda",
        use_last_moe_output: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, self.proj_l12, self.cfg = load_transfer_bundle(
            config_path=config_path,
            student_ckpt_path=student_ckpt_path,
            proj_source_ckpt_path=proj_source_ckpt_path,
            device=str(self.device),
        )
        self.use_last_moe_output = bool(use_last_moe_output)
        self.transform = build_transform(image_size=224)

    @torch.no_grad()
    def _forward_model(self, batch_tensor: torch.Tensor):
        student_out, gate_info_list, feature_dict, moe_feature_list = self.model(
            batch_tensor,
            return_gates=True,
            mask=None,
            is_eval=True,
            return_features=True,
            offline_cluster_ids=None,
        )
        return student_out, gate_info_list, feature_dict, moe_feature_list

    @torch.no_grad()
    def extract_features(
        self,
        images: List[Image.Image],
        return_teacher_space: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x = torch.stack([self.transform(img) for img in images]).to(self.device)

        _, _, feature_dict, moe_feature_list = self._forward_model(x)

        if self.use_last_moe_output and len(moe_feature_list) > 0:
            feat = moe_feature_list[-1]      # [B, T+1, 384]
            feat_source = "last_moe"
        else:
            if "layer_12" not in feature_dict:
                raise KeyError(f"'layer_12' not found in feature_dict keys={list(feature_dict.keys())}")
            feat = feature_dict["layer_12"]  # [B, T+1, 384]
            feat_source = "layer12"

        patch_tokens = feat[:, 1:, :]        # [B, T, 384]
        if patch_tokens.shape[1] == 0:
            raise RuntimeError(f"No patch tokens found, got shape={tuple(patch_tokens.shape)}")

        patch_feat_raw = patch_tokens.mean(dim=1)  # [B, 384]

        out = {
            "patch_feat_raw": patch_feat_raw,
            "feat_source": feat_source,
        }

        if return_teacher_space:
            patch_feat_teacher_space = self.proj_l12(patch_feat_raw)
            patch_feat_teacher_space = F.normalize(patch_feat_teacher_space, dim=-1)
            out["patch_feat_teacher_space"] = patch_feat_teacher_space

        return out


# =========================================================
# Slide extraction
# =========================================================
@torch.no_grad()
def extract_one_slide(
    extractor: TransferPatchFeatureExtractor,
    slide_path: str,
    h5_path: str,
    label: int,
    slide_id: str,
    out_path: str,
    batch_size: int = 64,
    max_patches: Optional[int] = None,
    save_teacher_space: bool = False,
    seed: int = 42,
):
    coords, patch_size, patch_level = read_coords_from_h5_with_attrs(h5_path)

    if max_patches is not None and coords.shape[0] > max_patches:
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(coords.shape[0], generator=g)[:max_patches]
        coords = coords[perm]

    slide = openslide.OpenSlide(slide_path)

    all_raw = []
    all_teacher = []
    all_coords = []

    num_coords = coords.shape[0]

    for start in tqdm(
        range(0, num_coords, batch_size),
        desc=f"Extracting {slide_id}",
        leave=False
    ):
        end = min(start + batch_size, num_coords)
        batch_coords = coords[start:end]

        batch_images = []
        for xy in batch_coords.tolist():
            img = read_patch_from_wsi(
                slide=slide,
                coord_xy=xy,
                patch_size=patch_size,
                read_level=patch_level,
            )
            batch_images.append(img)

        out = extractor.extract_features(
            batch_images,
            return_teacher_space=save_teacher_space,
        )

        all_raw.append(out["patch_feat_raw"].cpu())
        all_coords.append(batch_coords.cpu())

        if save_teacher_space:
            all_teacher.append(out["patch_feat_teacher_space"].cpu())

    slide.close()

    patch_feat_raw = torch.cat(all_raw, dim=0) if len(all_raw) > 0 else torch.empty(0, 384)
    coords_out = torch.cat(all_coords, dim=0) if len(all_coords) > 0 else torch.empty(0, 2)

    save_obj = {
        "features": patch_feat_raw,      # for downstream MIL
        "patch_feat_raw": patch_feat_raw,
        "coords": coords_out,
        "label": int(label),
        "slide_id": slide_id,
        "num_instances": int(patch_feat_raw.shape[0]),
        "feat_dim": int(patch_feat_raw.shape[1]) if patch_feat_raw.ndim == 2 and patch_feat_raw.shape[0] > 0 else 0,
        "source": "transfer_encoder_raw_patch_feature",
        "student_ckpt": str(extractor.model.__class__.__name__),
    }

    if save_teacher_space:
        patch_feat_teacher_space = torch.cat(all_teacher, dim=0) if len(all_teacher) > 0 else torch.empty(0, 1280)
        save_obj["patch_feat_teacher_space"] = patch_feat_teacher_space

    torch.save(save_obj, out_path)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Extract slide bag features using transfer-trained encoder")

    parser.add_argument("--slides_csv", type=str, required=True,
                        help="CSV with slide_id / label / split, or benchmark csv")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for bag feature .pt files")

    parser.add_argument("--config", type=str, required=True,
                        help="stage2 yaml config used to build encoder")
    parser.add_argument("--student_ckpt", type=str, required=True,
                        help="transfer-trained checkpoint with student_state_dict")
    parser.add_argument("--proj_source_ckpt", type=str, required=True,
                        help="original stage2 full checkpoint providing distiller_state_dict/proj_l12")

    parser.add_argument("--split", type=str, default=None,
                        choices=[None, "train", "val", "test"],
                        help="Optional original split filter")
    parser.add_argument("--benchmark_split", type=str, default=None,
                        choices=[None, "benchmark_train", "benchmark_val", "benchmark_test"],
                        help="Optional benchmark split filter")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--use_last_moe_output", action="store_true",
                        help="Use last MoE block output instead of layer_12")
    parser.add_argument("--save_teacher_space", action="store_true",
                        help="Also save patch_feat_teacher_space")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    if args.split is not None and args.benchmark_split is not None:
        raise ValueError("Use either --split or --benchmark_split, not both")

    df = load_slides_csv(
        args.slides_csv,
        split=args.split,
        benchmark_split=args.benchmark_split,
    )

    need = {"slide_id", "label", "svs_path", "h5_path"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"slides_csv missing required columns for extraction: {miss}")

    extractor = TransferPatchFeatureExtractor(
        config_path=args.config,
        student_ckpt_path=args.student_ckpt,
        proj_source_ckpt_path=args.proj_source_ckpt,
        device=args.device,
        use_last_moe_output=args.use_last_moe_output,
    )

    meta_path = os.path.join(args.out_dir, "feature_meta.csv")
    meta_rows = []

    failed_rows = []
    skipped_missing = 0
    skipped_other = 0

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Extract transfer bag features"):
        slide_id = str(row["slide_id"])
        label = int(row["label"])
        slide_path = str(row["svs_path"])
        h5_path = str(row["h5_path"])

        out_path = os.path.join(args.out_dir, f"{slide_id}.pt")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        if not os.path.exists(slide_path):
            skipped_missing += 1
            msg = f"WSI not found: {slide_path}"
            print(f"[Skip][No WSI] {slide_id}: {msg}")
            failed_rows.append({
                "slide_id": slide_id,
                "label": label,
                "reason": "no_wsi",
                "message": msg,
            })
            continue

        if not os.path.exists(h5_path):
            skipped_missing += 1
            msg = f"H5 not found: {h5_path}"
            print(f"[Skip][No H5] {slide_id}: {msg}")
            failed_rows.append({
                "slide_id": slide_id,
                "label": label,
                "reason": "no_h5",
                "message": msg,
            })
            continue

        try:
            extract_one_slide(
                extractor=extractor,
                slide_path=slide_path,
                h5_path=h5_path,
                label=label,
                slide_id=slide_id,
                out_path=out_path,
                batch_size=args.batch_size,
                max_patches=args.max_patches,
                save_teacher_space=args.save_teacher_space,
                seed=args.seed + row_idx,
            )

            obj = torch.load(out_path, map_location="cpu")
            meta_rows.append({
                "slide_id": slide_id,
                "label": label,
                "num_instances": obj["num_instances"],
                "feat_dim": obj["feat_dim"],
                "out_path": out_path,
            })

        except Exception as e:
            skipped_other += 1
            print(f"[ERROR] slide_id={slide_id}: {e}")
            failed_rows.append({
                "slide_id": slide_id,
                "label": label,
                "reason": "extract_error",
                "message": str(e),
            })
            continue

    if len(meta_rows) > 0:
        pd.DataFrame(meta_rows).to_csv(meta_path, index=False)
        print(f"[Saved meta] {meta_path}")

    if len(failed_rows) > 0:
        failed_csv = os.path.join(args.out_dir, "skipped_or_failed_slides.csv")
        pd.DataFrame(failed_rows).to_csv(failed_csv, index=False)
        print(f"[Saved skipped/failed log] {failed_csv}")

    print(
        f"Done. "
        f"skip_missing={skipped_missing}, "
        f"other_errors={skipped_other}"
    )


if __name__ == "__main__":
    main()