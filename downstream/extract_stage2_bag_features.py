#!/usr/bin/env python3
from __future__ import annotations

import os
import csv
import json
import yaml
import h5py
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
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
VALID_WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


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


def load_slides_csv(slides_csv: str, split: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(slides_csv)

    required_cols = {"slide_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"slides_csv missing columns: {missing}")

    df = make_label_column_if_needed(df)

    if split is not None:
        if "split" not in df.columns:
            raise ValueError("split was specified but slides_csv has no 'split' column")
        df = df[df["split"] == split].copy()

    df = df.reset_index(drop=True)
    return df


from pathlib import Path

VALID_WSI_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}

def find_wsi_path(raw_dir: str, slide_id: str) -> str:
    raw_dir = Path(raw_dir)

    matches = []
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_WSI_EXTS:
            continue
        if slide_id in p.stem or slide_id in p.name:
            matches.append(str(p))

    if len(matches) == 0:
        raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {raw_dir}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda x: (len(x), x))
    return matches[0]


def find_h5_path(h5_dir: str, slide_id: str) -> str:
    h5_dir = Path(h5_dir)

    matches = []
    for p in h5_dir.rglob("*.h5"):
        if slide_id in p.stem or slide_id in p.name:
            matches.append(str(p))

    if len(matches) == 0:
        raise FileNotFoundError(f"H5 not found for slide_id={slide_id} in {h5_dir}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda x: (len(x), x))
    return matches[0]


def read_coords_from_h5(h5_path: str) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
    return torch.from_numpy(coords).long()


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
# Stage2 loader
# =========================================================
def load_stage2_bundle(
    config_path: str,
    full_ckpt_path: str,
    device: str = "cuda",
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not os.path.exists(full_ckpt_path):
        raise FileNotFoundError(f"Full checkpoint not found: {full_ckpt_path}")

    ckpt = torch.load(full_ckpt_path, map_location="cpu")

    if "student_state_dict" not in ckpt:
        raise KeyError("student_state_dict not found in full checkpoint")
    if "distiller_state_dict" not in ckpt:
        raise KeyError("distiller_state_dict not found in full checkpoint")

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    model.load_state_dict(ckpt["student_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

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

    for p in model.parameters():
        p.requires_grad = False
    for p in proj_l12.parameters():
        p.requires_grad = False

    print("Loaded matched stage2 student + proj_l12 from best_full checkpoint")
    print(f"Current moe_layers_idx = {model.moe_layers_idx}")
    print(f"proj_l12 shape: {proj_in_dim} -> {proj_out_dim}")

    return model, proj_l12, cfg


# =========================================================
# Stage2 raw feature extractor
# =========================================================
class Stage2RawPatchFeatureExtractor(nn.Module):
    """
    For each input patch image:
      - run stage2 student encoder
      - take last MoE block patch token feature (preferred) or layer_12
      - output raw patch feature [B, 384]
      - optionally project to teacher space [B, 1280]
    """

    def __init__(
        self,
        config_path: str,
        full_ckpt_path: str,
        device: str = "cuda",
        use_last_moe_output: bool = True,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, self.proj_l12, self.cfg = load_stage2_bundle(
            config_path=config_path,
            full_ckpt_path=full_ckpt_path,
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
        return_teacher_space: bool = True,
        return_role_summary: bool = False,
        role_proto_dir: Optional[str] = None,
        role_tau: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        x = torch.stack([self.transform(img) for img in images]).to(self.device)

        _, _, feature_dict, moe_feature_list = self._forward_model(x)

        if self.use_last_moe_output and len(moe_feature_list) > 0:
            feat = moe_feature_list[-1]      # [B, T+1, 384]
        else:
            feat = feature_dict["layer_12"]  # [B, T+1, 384]

        patch_tokens = feat[:, 1:, :]        # [B, T, 384]

        if patch_tokens.shape[1] == 0:
            raise RuntimeError(f"No patch tokens found, got shape={tuple(patch_tokens.shape)}")

        patch_feat_raw = patch_tokens.mean(dim=1)  # [B, 384]

        out = {
            "patch_feat_raw": patch_feat_raw,
        }

        if return_teacher_space or return_role_summary:
            patch_feat_teacher_space = self.proj_l12(patch_feat_raw)
            patch_feat_teacher_space = torch.nn.functional.normalize(patch_feat_teacher_space, dim=-1)
            out["patch_feat_teacher_space"] = patch_feat_teacher_space

        if return_role_summary:
            if role_proto_dir is None:
                raise ValueError("role_proto_dir must be set when return_role_summary=True")

            proto_path = os.path.join(role_proto_dir, "role_prototypes_init.npy")
            names_path = os.path.join(role_proto_dir, "role_names.json")
            if not os.path.exists(proto_path):
                raise FileNotFoundError(f"Missing prototype file: {proto_path}")
            if not os.path.exists(names_path):
                raise FileNotFoundError(f"Missing role names file: {names_path}")

            protos = torch.from_numpy(
                __import__("numpy").load(proto_path).astype("float32")
            ).to(self.device)
            with open(names_path, "r", encoding="utf-8") as f:
                role_names = json.load(f)

            protos = torch.nn.functional.normalize(protos, dim=-1)
            logits = out["patch_feat_teacher_space"] @ protos.t()
            probs = torch.softmax(logits / role_tau, dim=-1)

            gaps = []
            R = logits.shape[-1]
            for r in range(R):
                cur = logits[:, r]
                other_ids = [i for i in range(R) if i != r]
                if len(other_ids) == 0:
                    other_max = torch.zeros_like(cur)
                else:
                    other_max = logits[:, other_ids].max(dim=-1).values
                gaps.append(cur - other_max)
            role_gaps = torch.stack(gaps, dim=-1)

            top2 = torch.topk(logits, k=min(2, R), dim=-1).values
            if R >= 2:
                role_top1_gap = (top2[:, 0] - top2[:, 1]).unsqueeze(-1)
            else:
                role_top1_gap = torch.ones_like(top2[:, 0:1])

            out["role_logits"] = logits
            out["role_probs"] = probs
            out["role_gaps"] = role_gaps
            out["role_top1_gap"] = role_top1_gap
            out["role_names"] = role_names

        return out


# =========================================================
# Slide extraction
# =========================================================
@torch.no_grad()
def extract_one_slide(
    extractor: Stage2RawPatchFeatureExtractor,
    slide_path: str,
    h5_path: str,
    label: int,
    slide_id: str,
    out_path: str,
    patch_size: int = 256,
    batch_size: int = 64,
    max_patches: Optional[int] = None,
    save_teacher_space: bool = False,
    save_role_summary: bool = False,
    role_proto_dir: Optional[str] = None,
    role_tau: float = 1.0,
):
    coords = read_coords_from_h5(h5_path)
    if max_patches is not None and coords.shape[0] > max_patches:
        perm = torch.randperm(coords.shape[0])[:max_patches]
        coords = coords[perm]

    slide = openslide.OpenSlide(slide_path)

    all_raw = []
    all_teacher = []
    all_coords = []

    all_role_logits = []
    all_role_probs = []
    all_role_gaps = []
    all_role_top1_gap = []
    role_names = None

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
                read_level=0,
            )
            img = img.resize((224, 224), resample=Image.BICUBIC)
            batch_images.append(img)

        out = extractor.extract_features(
            batch_images,
            return_teacher_space=save_teacher_space or save_role_summary,
            return_role_summary=save_role_summary,
            role_proto_dir=role_proto_dir,
            role_tau=role_tau,
        )

        all_raw.append(out["patch_feat_raw"].cpu())
        all_coords.append(batch_coords.cpu())

        if save_teacher_space or save_role_summary:
            all_teacher.append(out["patch_feat_teacher_space"].cpu())

        if save_role_summary:
            all_role_logits.append(out["role_logits"].cpu())
            all_role_probs.append(out["role_probs"].cpu())
            all_role_gaps.append(out["role_gaps"].cpu())
            all_role_top1_gap.append(out["role_top1_gap"].cpu())
            role_names = out["role_names"]

    slide.close()

    patch_feat_raw = torch.cat(all_raw, dim=0) if len(all_raw) > 0 else torch.empty(0, 384)
    coords_out = torch.cat(all_coords, dim=0) if len(all_coords) > 0 else torch.empty(0, 2)

    save_obj = {
        "features": patch_feat_raw,      # downstream train_abmil.py expects this key
        "patch_feat_raw": patch_feat_raw,
        "coords": coords_out,
        "label": int(label),
        "slide_id": slide_id,
        "num_instances": int(patch_feat_raw.shape[0]),
        "feat_dim": int(patch_feat_raw.shape[1]) if patch_feat_raw.ndim == 2 and patch_feat_raw.shape[0] > 0 else 0,
        "source": "stage2_raw_patch_feature",
    }

    if save_teacher_space or save_role_summary:
        patch_feat_teacher_space = torch.cat(all_teacher, dim=0) if len(all_teacher) > 0 else torch.empty(0, 1280)
        save_obj["patch_feat_teacher_space"] = patch_feat_teacher_space

    if save_role_summary:
        save_obj["role_logits"] = torch.cat(all_role_logits, dim=0) if len(all_role_logits) > 0 else torch.empty(0)
        save_obj["role_probs"] = torch.cat(all_role_probs, dim=0) if len(all_role_probs) > 0 else torch.empty(0)
        save_obj["role_gaps"] = torch.cat(all_role_gaps, dim=0) if len(all_role_gaps) > 0 else torch.empty(0)
        save_obj["role_top1_gap"] = torch.cat(all_role_top1_gap, dim=0) if len(all_role_top1_gap) > 0 else torch.empty(0)
        save_obj["role_names"] = role_names

    torch.save(save_obj, out_path)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Extract slide bag features using stage2 raw patch features")

    parser.add_argument("--slides_csv", type=str, required=True,
                        help="CSV with slide_id / label / split")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="Directory containing raw WSI files")
    parser.add_argument("--h5_dir", type=str, required=True,
                        help="Directory containing CLAM patch h5 files")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for bag feature .pt files")

    parser.add_argument("--config", type=str, required=True,
                        help="stage2 yaml config")
    parser.add_argument("--full_ckpt", type=str, required=True,
                        help="stage2 best_full checkpoint")

    parser.add_argument("--split", type=str, default=None,
                        choices=[None, "train", "val", "test"], help="Optional split filter")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--use_last_moe_output", action="store_true",
                        help="Use last MoE block output instead of layer_12")
    parser.add_argument("--save_teacher_space", action="store_true",
                        help="Also save patch_feat_teacher_space")
    parser.add_argument("--save_role_summary", action="store_true",
                        help="Also save role logits/probs/gaps/top1_gap")
    parser.add_argument("--role_proto_dir", type=str, default=None)
    parser.add_argument("--role_tau", type=float, default=1.0)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    if args.save_role_summary and args.role_proto_dir is None:
        raise ValueError("--role_proto_dir must be set when --save_role_summary is used")

    df = load_slides_csv(args.slides_csv, split=args.split)

    extractor = Stage2RawPatchFeatureExtractor(
        config_path=args.config,
        full_ckpt_path=args.full_ckpt,
        device=args.device,
        use_last_moe_output=args.use_last_moe_output,
    )

    meta_path = os.path.join(args.out_dir, "feature_meta.csv")
    meta_rows = []

    failed_rows = []
    skipped_no_wsi = 0
    skipped_no_h5 = 0
    skipped_other = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extract stage2 bag features"):
        slide_id = str(row["slide_id"])
        label = int(row["label"])

        out_path = os.path.join(args.out_dir, f"{slide_id}.pt")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        # 1) 先找 WSI；没有就跳过
        try:
            slide_path = find_wsi_path(args.raw_dir, slide_id)
        except FileNotFoundError as e:
            skipped_no_wsi += 1
            print(f"[Skip][No WSI] {slide_id}: {e}")
            failed_rows.append({
                "slide_id": slide_id,
                "label": label,
                "reason": "no_wsi",
                "message": str(e),
            })
            continue

        # 2) 再找 h5；没有就跳过
        try:
            h5_path = find_h5_path(args.h5_dir, slide_id)
        except FileNotFoundError as e:
            skipped_no_h5 += 1
            print(f"[Skip][No H5] {slide_id}: {e}")
            failed_rows.append({
                "slide_id": slide_id,
                "label": label,
                "reason": "no_h5",
                "message": str(e),
            })
            continue

        # 3) 正常提特征；其他异常也记下来但不中断全流程
        try:
            extract_one_slide(
                extractor=extractor,
                slide_path=slide_path,
                h5_path=h5_path,
                label=label,
                slide_id=slide_id,
                out_path=out_path,
                patch_size=args.patch_size,
                batch_size=args.batch_size,
                max_patches=args.max_patches,
                save_teacher_space=args.save_teacher_space,
                save_role_summary=args.save_role_summary,
                role_proto_dir=args.role_proto_dir,
                role_tau=args.role_tau
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
        f"skip_no_wsi={skipped_no_wsi}, "
        f"skip_no_h5={skipped_no_h5}, "
        f"other_errors={skipped_other}"
    )

    print("Done.")


if __name__ == "__main__":
    main()