#!/usr/bin/env python3
from __future__ import annotations

import os
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =========================================================
# Utils
# =========================================================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_entropy(probs: List[float], eps: float = 1e-12) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def margin_top1_top2(probs: List[float]) -> float:
    p = sorted([float(x) for x in probs], reverse=True)
    if len(p) < 2:
        return 0.0
    return float(p[0] - p[1])


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


def find_wsi_path(wsi_dir: str, slide_id: str) -> str:
    wsi_dir = Path(wsi_dir)
    exts = [".tiff", ".tif", ".svs", ".ndpi", ".mrxs"]

    exact = []
    for ext in exts:
        exact.extend(wsi_dir.rglob(f"{slide_id}{ext}"))
    if len(exact) == 1:
        return str(exact[0])
    if len(exact) > 1:
        raise RuntimeError(
            f"Found multiple exact WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in exact[:10])
        )

    fuzzy = []
    for ext in exts:
        fuzzy.extend(wsi_dir.rglob(f"{slide_id}*{ext}"))
    if len(fuzzy) == 1:
        return str(fuzzy[0])
    if len(fuzzy) > 1:
        exact_name = [p for p in fuzzy if p.stem == slide_id]
        if len(exact_name) == 1:
            return str(exact_name[0])
        raise RuntimeError(
            f"Found multiple fuzzy WSI files for slide_id={slide_id}: "
            + ", ".join(str(x) for x in fuzzy[:10])
        )

    raise FileNotFoundError(f"WSI not found for slide_id={slide_id} in {wsi_dir}")


def maybe_build_slide_meta(slides_csv: Optional[str]) -> Optional[pd.DataFrame]:
    if slides_csv is None:
        return None
    df = pd.read_csv(slides_csv)
    if "slide_id" not in df.columns:
        if "image_id" in df.columns:
            df["slide_id"] = df["image_id"]
        else:
            raise ValueError("slides_csv must contain slide_id or image_id")
    return df


# =========================================================
# Core
# =========================================================
def filter_per_slide_role_topk(
    df: pd.DataFrame,
    topk_map: Dict[str, int],
) -> pd.DataFrame:
    kept = []

    for role_name, sub in df.groupby("best_role"):
        k = topk_map.get(role_name, None)
        for slide_id, sub_slide in sub.groupby("slide_id"):
            sub_slide = sub_slide.sort_values(
                ["purity", "tissue_ratio"],
                ascending=[False, False]
            )
            if k is not None:
                sub_slide = sub_slide.head(k)
            kept.append(sub_slide)

    if len(kept) == 0:
        return df.iloc[:0].copy()
    return pd.concat(kept, axis=0).reset_index(drop=True)


def build_one_role_csv(
    df_role: pd.DataFrame,
    role_name: str,
    slide_meta_df: Optional[pd.DataFrame],
    wsi_dir: str,
    h5_dir: str,
    project_name: str,
    organ_name: str,
    patch_level: int,
    patch_size: int,
) -> pd.DataFrame:
    rows = []

    slide_meta_map = {}
    if slide_meta_df is not None:
        for _, row in slide_meta_df.iterrows():
            slide_meta_map[str(row["slide_id"])] = row.to_dict()

    df_role = df_role.sort_values(["slide_id", "purity"], ascending=[True, False]).reset_index(drop=True)

    review_idx = 0
    for slide_id, sub in df_role.groupby("slide_id"):
        meta = slide_meta_map.get(str(slide_id), {})

        if "source_path" in meta and pd.notna(meta["source_path"]) and os.path.exists(str(meta["source_path"])):
            svs_path = str(meta["source_path"])
        else:
            svs_path = find_wsi_path(wsi_dir, str(slide_id))

        if "h5_path" in meta and pd.notna(meta["h5_path"]) and os.path.exists(str(meta["h5_path"])):
            h5_path = str(meta["h5_path"])
        else:
            h5_path = find_h5_path(h5_dir, str(slide_id))

        sub = sub.copy().sort_values(["purity", "tissue_ratio"], ascending=[False, False]).reset_index(drop=True)

        for i, row in sub.iterrows():
            probs = [
                float(row.get("ratio_stroma", 0.0)),
                float(row.get("ratio_benign_epithelium", 0.0)),
                float(row.get("ratio_cancer", 0.0)),
            ]
            ent = safe_entropy(probs)
            margin = margin_top1_top2(probs)

            out = {
                "review_idx": review_idx,
                "project": project_name,
                "slide_id": slide_id,
                "svs_path": svs_path,
                "h5_path": h5_path,
                "coord_x": int(row["x"]),
                "coord_y": int(row["y"]),
                "coord_idx": -1,  # 若后续需要可再回填
                "patch_level": int(patch_level),
                "patch_size": int(patch_size),
                "organ_name": organ_name,
                "pred_label": role_name,
                "pred_confidence": float(row["purity"]),
                "entropy": ent,
                "margin_top1_top2": margin,
                "prefilter_white": False,
                "core_rank": int(i + 1),
                "core_label": role_name,
                "score_tumor": float(row.get("ratio_cancer", 0.0)),
                "score_fibrovascular_stroma": float(row.get("ratio_stroma", 0.0)),
                "score_normal_kidney_parenchyma": float(row.get("ratio_benign_epithelium", 0.0)),
                "score_vascular_hemorrhage": 0.0,
                "score_background_artifact": float(row.get("background_ratio", 0.0)),
                "score_ambiguous_mixed": float(1.0 - row["purity"]),
                "slide_balanced_rank": int(i + 1),
                "keep": True,
                "comment": "",
                "review_sheet": "",
                "position_in_sheet": -1,
            }
            rows.append(out)
            review_idx += 1

    out_df = pd.DataFrame(rows)

    # 固定列顺序
    expected_cols = [
        "review_idx", "project", "slide_id", "svs_path", "h5_path",
        "coord_x", "coord_y", "coord_idx", "patch_level", "patch_size",
        "organ_name", "pred_label", "pred_confidence", "entropy",
        "margin_top1_top2", "prefilter_white", "core_rank", "core_label",
        "score_tumor", "score_fibrovascular_stroma", "score_normal_kidney_parenchyma",
        "score_vascular_hemorrhage", "score_background_artifact", "score_ambiguous_mixed",
        "slide_balanced_rank", "keep", "comment", "review_sheet", "position_in_sheet"
    ]
    out_df = out_df[expected_cols]
    return out_df


def main():
    parser = argparse.ArgumentParser("Build 3 PANDA proto candidate CSVs")

    parser.add_argument("--candidate_csv", type=str, required=True,
                        help="panda_role_candidates_3role_train.csv")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--slides_csv", type=str, default=None,
                        help="Optional split/meta CSV containing slide_id and maybe source_path/h5_path")
    parser.add_argument("--wsi_dir", type=str, required=True,
                        help="Directory to resolve slide_id -> WSI path if source_path unavailable")
    parser.add_argument("--h5_dir", type=str, required=True,
                        help="Directory to resolve slide_id -> h5 path if h5_path unavailable")

    parser.add_argument("--project_name", type=str, default="PANDA")
    parser.add_argument("--organ_name", type=str, default="prostate")
    parser.add_argument("--patch_level", type=int, default=0)
    parser.add_argument("--patch_size", type=int, default=256)

    parser.add_argument("--stroma_topk", type=int, default=5)
    parser.add_argument("--benign_topk", type=int, default=10)
    parser.add_argument("--cancer_topk", type=int, default=10)

    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df = pd.read_csv(args.candidate_csv)
    required = {
        "slide_id", "x", "y", "best_role", "purity",
        "background_ratio", "tissue_ratio",
        "ratio_stroma", "ratio_benign_epithelium", "ratio_cancer"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"candidate_csv missing columns: {missing}")

    slide_meta_df = maybe_build_slide_meta(args.slides_csv)

    topk_map = {
        "stroma": args.stroma_topk,
        "benign_epithelium": args.benign_topk,
        "cancer": args.cancer_topk,
    }

    df_filt = filter_per_slide_role_topk(df, topk_map=topk_map)

    role_to_filename = {
        "stroma": "stroma_candidates.csv",
        "benign_epithelium": "benign_epithelium_candidates.csv",
        "cancer": "cancer_candidates.csv",
    }

    for role_name, out_name in role_to_filename.items():
        sub = df_filt[df_filt["best_role"] == role_name].copy()
        if len(sub) == 0:
            print(f"[WARN] role={role_name} has 0 rows after filtering")
            out_df = pd.DataFrame(columns=[
                "review_idx", "project", "slide_id", "svs_path", "h5_path",
                "coord_x", "coord_y", "coord_idx", "patch_level", "patch_size",
                "organ_name", "pred_label", "pred_confidence", "entropy",
                "margin_top1_top2", "prefilter_white", "core_rank", "core_label",
                "score_tumor", "score_fibrovascular_stroma", "score_normal_kidney_parenchyma",
                "score_vascular_hemorrhage", "score_background_artifact", "score_ambiguous_mixed",
                "slide_balanced_rank", "keep", "comment", "review_sheet", "position_in_sheet"
            ])
        else:
            out_df = build_one_role_csv(
                df_role=sub,
                role_name=role_name,
                slide_meta_df=slide_meta_df,
                wsi_dir=args.wsi_dir,
                h5_dir=args.h5_dir,
                project_name=args.project_name,
                organ_name=args.organ_name,
                patch_level=args.patch_level,
                patch_size=args.patch_size,
            )

        out_path = os.path.join(args.out_dir, out_name)
        out_df.to_csv(out_path, index=False)
        print(f"[Saved] {out_path} shape={out_df.shape}")


if __name__ == "__main__":
    main()