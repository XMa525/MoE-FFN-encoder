#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import h5py
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import openslide
from tqdm import tqdm


# =========================================================
# Utils
# =========================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_role_map(role_map_json: str) -> Dict[str, List[int]]:
    """
    Example:
    '{"stroma":[1], "benign_epithelium":[2], "cancer":[3,4,5]}'
    """
    role_map = json.loads(role_map_json)
    out = {}
    for k, v in role_map.items():
        if not isinstance(v, list):
            raise ValueError(f"role_map[{k}] must be a list, got {type(v)}")
        out[k] = [int(x) for x in v]
    return out


def get_role_specific_threshold(
    role_name: str,
    default_thresh: float,
    stroma_min_purity: Optional[float],
    benign_min_purity: Optional[float],
    cancer_min_purity: Optional[float],
) -> float:
    if role_name == "stroma" and stroma_min_purity is not None:
        return float(stroma_min_purity)
    if role_name == "benign_epithelium" and benign_min_purity is not None:
        return float(benign_min_purity)
    if role_name == "cancer" and cancer_min_purity is not None:
        return float(cancer_min_purity)
    return float(default_thresh)


def find_h5_path(h5_dir: str, slide_id: str) -> str:
    h5_dir = Path(h5_dir)

    exact = list(h5_dir.rglob(f"{slide_id}.h5"))
    if len(exact) == 1:
        return str(exact[0])
    if len(exact) > 1:
        raise RuntimeError(
            f"Found multiple exact h5 files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in exact[:10])
        )

    fuzzy = list(h5_dir.rglob(f"{slide_id}*.h5"))
    if len(fuzzy) == 1:
        return str(fuzzy[0])
    if len(fuzzy) > 1:
        exact_name = [p for p in fuzzy if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])
        raise RuntimeError(
            f"Found multiple fuzzy h5 files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in fuzzy[:10])
        )

    raise FileNotFoundError(f"H5 not found for slide_id={slide_id} in {h5_dir}")


def find_mask_path(mask_dir: str, slide_id: str) -> str:
    mask_dir = Path(mask_dir)

    candidates = []
    candidates.extend(mask_dir.rglob(f"{slide_id}_mask.tiff"))
    candidates.extend(mask_dir.rglob(f"{slide_id}_mask.tif"))

    if len(candidates) == 1:
        return str(candidates[0])
    if len(candidates) > 1:
        raise RuntimeError(
            f"Found multiple mask files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in candidates[:10])
        )

    fuzzy = []
    fuzzy.extend(mask_dir.rglob(f"{slide_id}*_mask.tiff"))
    fuzzy.extend(mask_dir.rglob(f"{slide_id}*_mask.tif"))
    if len(fuzzy) == 1:
        return str(fuzzy[0])
    if len(fuzzy) > 1:
        raise RuntimeError(
            f"Found multiple fuzzy mask files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in fuzzy[:10])
        )

    raise FileNotFoundError(f"Mask not found for slide_id={slide_id} in {mask_dir}")


def read_coords_from_h5(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
    return coords


def open_mask_slide(mask_path: str):
    return openslide.OpenSlide(mask_path)


def read_mask_patch(
    mask_slide,
    x: int,
    y: int,
    patch_size: int,
    read_level: int = 0,
) -> Optional[np.ndarray]:
    """
    按 level-0 坐标从 PANDA label mask 中读取一个 patch。
    """
    try:
        patch = mask_slide.read_region((x, y), read_level, (patch_size, patch_size))
        arr = np.array(patch)

        # OpenSlide read_region 通常返回 RGBA
        if arr.ndim == 3:
            arr = arr[..., 0]

        if arr.shape[0] != patch_size or arr.shape[1] != patch_size:
            return None

        return arr
    except Exception:
        return None


def compute_role_ratios(mask_patch: np.ndarray, role_map: Dict[str, List[int]]) -> Dict[str, float]:
    total = float(mask_patch.size)
    ratios = {}
    for role_name, vals in role_map.items():
        ratios[role_name] = float(np.isin(mask_patch, vals).sum() / total)
    return ratios


# =========================================================
# Main extraction
# =========================================================
def extract_candidates_for_slide(
    slide_id: str,
    h5_path: str,
    mask_path: str,
    role_map: Dict[str, List[int]],
    patch_size: int,
    purity_thresh: float,
    stroma_min_purity: Optional[float],
    benign_min_purity: Optional[float],
    cancer_min_purity: Optional[float],
    max_background_ratio: float,
    min_tissue_ratio: float,
    background_values: List[int],
    topk_per_role_per_slide: int,
    mask_read_level: int = 0,
) -> List[Dict]:
    coords = read_coords_from_h5(h5_path)
    mask_slide = open_mask_slide(mask_path)

    rows = []

    for xy in coords:
        x, y = int(xy[0]), int(xy[1])

        mask_patch = read_mask_patch(
            mask_slide=mask_slide,
            x=x,
            y=y,
            patch_size=patch_size,
            read_level=mask_read_level,
        )
        if mask_patch is None:
            continue

        total = float(mask_patch.size)

        background_ratio = float(np.isin(mask_patch, background_values).sum() / total)
        tissue_ratio = 1.0 - background_ratio

        if background_ratio > max_background_ratio:
            continue
        if tissue_ratio < min_tissue_ratio:
            continue

        role_ratios = compute_role_ratios(mask_patch, role_map)
        best_role = max(role_ratios, key=role_ratios.get)
        best_ratio = role_ratios[best_role]

        role_thresh = get_role_specific_threshold(
            role_name=best_role,
            default_thresh=purity_thresh,
            stroma_min_purity=stroma_min_purity,
            benign_min_purity=benign_min_purity,
            cancer_min_purity=cancer_min_purity,
        )

        if best_ratio < role_thresh:
            continue

        row = {
            "slide_id": slide_id,
            "x": x,
            "y": y,
            "best_role": best_role,
            "purity": float(best_ratio),
            "role_threshold": float(role_thresh),
            "background_ratio": float(background_ratio),
            "tissue_ratio": float(tissue_ratio),
        }

        for role_name, ratio in role_ratios.items():
            row[f"ratio_{role_name}"] = float(ratio)

        rows.append(row)

    mask_slide.close()

    if len(rows) == 0:
        return []

    df = pd.DataFrame(rows)

    # 每张 slide、每个 role 只保留 top-k purity
    kept = []
    for role_name in sorted(df["best_role"].unique()):
        sub = df[df["best_role"] == role_name].copy()
        sub = sub.sort_values(["purity", "tissue_ratio"], ascending=[False, False])
        if topk_per_role_per_slide is not None and len(sub) > topk_per_role_per_slide:
            sub = sub.head(topk_per_role_per_slide)
        kept.append(sub)

    out_df = pd.concat(kept, axis=0).reset_index(drop=True)
    return out_df.to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser("Extract PANDA mask-based role candidates")

    parser.add_argument("--slides_csv", type=str, required=True,
                        help="CSV containing slide_id and optionally split/label")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="Directory containing PANDA label masks")
    parser.add_argument("--h5_dir", type=str, required=True,
                        help="Directory containing CLAM patch h5 files")
    parser.add_argument("--out_csv", type=str, required=True)

    parser.add_argument("--split", type=str, default=None,
                        choices=[None, "train", "val", "test"],
                        help="Optional split filter; usually use train only")

    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--mask_read_level", type=int, default=0,
                        help="Usually 0 for PANDA label masks")

    parser.add_argument("--purity_thresh", type=float, default=0.75,
                        help="Default minimum dominant-role area ratio")

    parser.add_argument("--stroma_min_purity", type=float, default=None,
                        help="Optional role-specific purity threshold for stroma")
    parser.add_argument("--benign_min_purity", type=float, default=None,
                        help="Optional role-specific purity threshold for benign_epithelium")
    parser.add_argument("--cancer_min_purity", type=float, default=None,
                        help="Optional role-specific purity threshold for cancer")

    parser.add_argument("--max_background_ratio", type=float, default=0.20,
                        help="Drop patch if background ratio is larger than this")
    parser.add_argument("--min_tissue_ratio", type=float, default=0.50,
                        help="Drop patch if tissue ratio is smaller than this")
    parser.add_argument("--topk_per_role_per_slide", type=int, default=20,
                        help="Keep only top-k candidates for each role within each slide")

    parser.add_argument("--background_values", type=int, nargs="+", default=[0],
                        help="Mask values treated as background")

    parser.add_argument("--role_map_json", type=str, required=True,
                        help='Example: \'{"stroma":[1], "benign_epithelium":[2], "cancer":[3,4,5]}\'')
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    ensure_dir(str(Path(args.out_csv).parent))
    role_map = parse_role_map(args.role_map_json)

    df = pd.read_csv(args.slides_csv)
    if "slide_id" not in df.columns:
        if "image_id" in df.columns:
            df["slide_id"] = df["image_id"]
        else:
            raise ValueError("slides_csv must contain 'slide_id' or 'image_id'")

    if args.split is not None:
        if "split" not in df.columns:
            raise ValueError("split filter requested but slides_csv has no 'split' column")
        df = df[df["split"] == args.split].copy()

    df = df.reset_index(drop=True)

    all_rows = []
    fail_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extract role candidates"):
        slide_id = str(row["slide_id"])

        try:
            h5_path = find_h5_path(args.h5_dir, slide_id)
            mask_path = find_mask_path(args.mask_dir, slide_id)

            rows = extract_candidates_for_slide(
                slide_id=slide_id,
                h5_path=h5_path,
                mask_path=mask_path,
                role_map=role_map,
                patch_size=args.patch_size,
                purity_thresh=args.purity_thresh,
                stroma_min_purity=args.stroma_min_purity,
                benign_min_purity=args.benign_min_purity,
                cancer_min_purity=args.cancer_min_purity,
                max_background_ratio=args.max_background_ratio,
                min_tissue_ratio=args.min_tissue_ratio,
                background_values=args.background_values,
                topk_per_role_per_slide=args.topk_per_role_per_slide,
                mask_read_level=args.mask_read_level,
            )
            all_rows.extend(rows)

            if args.verbose:
                role_counts = {}
                for r in rows:
                    role_counts[r["best_role"]] = role_counts.get(r["best_role"], 0) + 1
                print(f"[{slide_id}] kept={len(rows)} role_counts={role_counts}")

        except Exception as e:
            fail_rows.append({
                "slide_id": slide_id,
                "error": str(e)
            })
            if args.verbose:
                print(f"[ERROR] {slide_id}: {e}")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.out_csv, index=False)
    print(f"[Saved candidates] {args.out_csv} shape={out_df.shape}")

    if len(fail_rows) > 0:
        fail_csv = str(Path(args.out_csv).with_suffix("")) + "_failures.csv"
        pd.DataFrame(fail_rows).to_csv(fail_csv, index=False)
        print(f"[Saved failures] {fail_csv} n={len(fail_rows)}")

    if len(out_df) > 0:
        print("\n[Role counts]")
        print(out_df["best_role"].value_counts())

        print("\n[Slides with candidates]")
        print(out_df["slide_id"].nunique())

        print("\n[Role-specific purity summary]")
        print(
            out_df.groupby("best_role")[["purity", "role_threshold", "background_ratio", "tissue_ratio"]]
            .agg(["count", "mean", "min", "max"])
        )


if __name__ == "__main__":
    main()