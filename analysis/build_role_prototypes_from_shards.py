#!/usr/bin/env python3
"""
Build role prototypes directly from pre-extracted Virchow2 token-feature shards.

Purpose
-------
Use existing shard_XX.npz token feature caches to construct initial role prototypes
for the new CONCH-guided role prior branch.

Current role setup
------------------
P1: tumor core balanced
P2: stroma core balanced
P3: ambiguous_clean balanced
P4: free expert (no semantic prototype)

What this script does
---------------------
1. Read three candidate CSVs:
   - tumor core balanced
   - stroma core balanced
   - ambiguous_clean balanced
2. Traverse all shard_XX.npz files that contain token-level features.
3. Aggregate token features into patch-level features using mean pooling.
4. Match pooled patch features to the three role CSVs by canonicalized path.
5. Build three role prototypes in the existing feature space.
6. Save prototypes, metadata, pairwise cosine, and optional matched patch features.

Expected shard structure
------------------------
Each shard_XX.npz is expected to contain at least:
- features : [num_tokens, dim]
- paths    : [num_tokens] object/string, one patch path per token
Optionally:
- patch_ids, organs, token_x, token_y

The user already verified an example shard has:
- features shape (1280768, 1280)
- paths shape (1280768,)
which implies token-level feature caching per patch.

Typical usage
-------------
python build_role_prototypes_from_shards.py \
  --tumor-csv outputs/conch_analysis/candidate_core_tumor_balanced_by_organ.csv \
  --stroma-csv outputs/conch_analysis/candidate_core_stroma_balanced_by_organ.csv \
  --ambiguous-csv outputs/refined_candidates/ambiguous_clean_balanced_by_organ.csv \
  --shard-dir /data/maxinyu/WSI_WORKSPACE/path/to/layer24_shards \
  --outdir outputs/role_proto_init_layer24 \
  --save-matched-features

Outputs
-------
- role_prototypes_init.npy          shape [3, D]
- role_names.json                   ["tumor", "stroma", "ambiguous"]
- role_metadata.json
- prototype_pairwise_cosine.json
- matched_patch_features.csv        metadata for matched patches
- matched_patch_features.npz        pooled features + labels (optional)
- unmatched_paths.json              CSV paths not found in shards

Notes
-----
- This script assumes the shards come from the already selected prototype layer.
- Patch-level feature = mean over all token features belonging to the same patch.
- P4 free expert is not represented here; this file only builds the 3 semantic prototypes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / denom


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = l2_normalize(x)
    return x @ x.T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build role prototypes from token-feature shards")
    parser.add_argument("--tumor-csv", type=str, required=True)
    parser.add_argument("--stroma-csv", type=str, required=True)
    parser.add_argument("--ambiguous-csv", type=str, required=True)
    parser.add_argument("--shard-dir", type=str, required=True)
    parser.add_argument("--shard-pattern", type=str, default="shard_*.npz")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--feature-key", type=str, default="features")
    parser.add_argument("--path-key", type=str, default="paths")
    parser.add_argument("--organ-key", type=str, default="organs")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean"])
    parser.add_argument("--normalize-patch-feature", action="store_true", help="L2 normalize patch features before averaging into prototypes")
    parser.add_argument("--normalize-prototype", action="store_true", default=True, help="L2 normalize final prototypes")
    parser.add_argument("--save-matched-features", action="store_true")
    return parser.parse_args()


def load_role_csv(path: str, role_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "patch_path" not in df.columns:
        raise ValueError(f"CSV missing patch_path: {path}")
    df = df.copy()
    df["role"] = role_name
    df["canonical_path"] = df["patch_path"].map(canonicalize_path)
    return df


def build_target_map(role_dfs: List[pd.DataFrame]) -> Tuple[Dict[str, str], pd.DataFrame]:
    df_all = pd.concat(role_dfs, axis=0).reset_index(drop=True)
    # If same path appears in multiple roles, keep the first occurrence and log duplicates.
    dup_counts = df_all["canonical_path"].value_counts()
    duplicated = dup_counts[dup_counts > 1]
    if len(duplicated) > 0:
        print(f"[Warn] {len(duplicated)} canonical paths appear in multiple rows/roles; keeping first occurrence.")
        df_all = df_all.drop_duplicates(subset=["canonical_path"], keep="first").reset_index(drop=True)

    target_map = {row["canonical_path"]: row["role"] for _, row in df_all.iterrows()}
    return target_map, df_all


def aggregate_shards_for_targets(
    shard_files: List[str],
    target_map: Dict[str, str],
    feature_key: str,
    path_key: str,
    organ_key: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int], Dict[str, int]]:
    """
    Aggregate token-level features into patch-level features for targeted paths only.

    Returns
    -------
    pooled_features : canonical_path -> pooled patch feature [D]
    token_counts    : canonical_path -> number of tokens seen
    organ_ids       : canonical_path -> organ id if available else -1
    """
    sum_map: Dict[str, np.ndarray] = {}
    count_map: Dict[str, int] = defaultdict(int)
    organ_map: Dict[str, int] = {}

    target_set = set(target_map.keys())

    for shard_path in tqdm(shard_files, desc="Reading shards"):
        data = np.load(shard_path, allow_pickle=True)
        if feature_key not in data.files or path_key not in data.files:
            raise KeyError(f"Shard missing required keys in {shard_path}. Found {data.files}")

        feats = data[feature_key]   # [num_tokens, D]
        paths = data[path_key]      # [num_tokens]
        organs = data[organ_key] if organ_key in data.files else None

        # Group token rows by canonicalized patch path, but only for target patches.
        local_groups: Dict[str, List[int]] = defaultdict(list)
        for i, p in enumerate(paths):
            cp = canonicalize_path(str(p))
            if cp in target_set:
                local_groups[cp].append(i)

        if not local_groups:
            continue

        for cp, indices in local_groups.items():
            token_feats = feats[indices].astype(np.float32, copy=False)
            token_sum = token_feats.sum(axis=0)

            if cp not in sum_map:
                sum_map[cp] = token_sum
            else:
                sum_map[cp] += token_sum

            count_map[cp] += len(indices)

            if organs is not None and cp not in organ_map:
                organ_map[cp] = int(organs[indices[0]])

    pooled_map: Dict[str, np.ndarray] = {}
    for cp, feat_sum in sum_map.items():
        pooled_map[cp] = feat_sum / max(count_map[cp], 1)

    return pooled_map, dict(count_map), organ_map


def build_role_prototypes(
    pooled_map: Dict[str, np.ndarray],
    df_targets: pd.DataFrame,
    normalize_patch_feature: bool,
    normalize_prototype: bool,
) -> Tuple[np.ndarray, List[str], pd.DataFrame, Dict[str, object]]:
    role_names = ["tumor", "stroma", "ambiguous"]
    matched_rows = []
    unmatched = []

    for _, row in df_targets.iterrows():
        cp = row["canonical_path"]
        role = row["role"]
        if cp not in pooled_map:
            unmatched.append(cp)
            continue

        feat = pooled_map[cp].astype(np.float32, copy=False)
        if normalize_patch_feature:
            feat = l2_normalize(feat[None, :])[0]

        matched = row.to_dict()
        matched["feature"] = feat
        matched_rows.append(matched)

    matched_df = pd.DataFrame(matched_rows)

    prototypes = []
    role_stats: Dict[str, object] = {}
    for role in role_names:
        sub = matched_df[matched_df["role"] == role]
        if len(sub) == 0:
            raise RuntimeError(f"No matched features found for role '{role}'.")
        X = np.stack(sub["feature"].values, axis=0)
        proto = X.mean(axis=0)
        if normalize_prototype:
            proto = l2_normalize(proto[None, :])[0]
        prototypes.append(proto)

        role_stats[role] = {
            "num_matched_patches": int(len(sub)),
            "feature_dim": int(X.shape[1]),
            "mean_feature_norm": float(np.linalg.norm(X, axis=1).mean()),
        }
        if "organ_name" in sub.columns:
            role_stats[role]["organ_counts"] = sub["organ_name"].value_counts().to_dict()

    prototypes = np.stack(prototypes, axis=0).astype(np.float32)

    meta = {
        "role_names": role_names,
        "num_total_target_paths": int(len(df_targets)),
        "num_matched_paths": int(len(matched_df)),
        "num_unmatched_paths": int(len(unmatched)),
        "normalize_patch_feature": bool(normalize_patch_feature),
        "normalize_prototype": bool(normalize_prototype),
        "role_stats": role_stats,
        "unmatched_paths": unmatched,
    }
    return prototypes, role_names, matched_df, meta


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    tumor_df = load_role_csv(args.tumor_csv, "tumor")
    stroma_df = load_role_csv(args.stroma_csv, "stroma")
    ambiguous_df = load_role_csv(args.ambiguous_csv, "ambiguous")

    target_map, df_targets = build_target_map([tumor_df, stroma_df, ambiguous_df])

    shard_files = sorted(glob.glob(os.path.join(args.shard_dir, args.shard_pattern)))
    if len(shard_files) == 0:
        raise FileNotFoundError(f"No shard files found under {args.shard_dir} with pattern {args.shard_pattern}")

    print(f"Found {len(shard_files)} shard files")
    print(f"Target patches to match: {len(target_map)}")

    pooled_map, token_counts, organ_map = aggregate_shards_for_targets(
        shard_files=shard_files,
        target_map=target_map,
        feature_key=args.feature_key,
        path_key=args.path_key,
        organ_key=args.organ_key,
    )

    prototypes, role_names, matched_df, meta = build_role_prototypes(
        pooled_map=pooled_map,
        df_targets=df_targets,
        normalize_patch_feature=args.normalize_patch_feature,
        normalize_prototype=args.normalize_prototype,
    )

    # enrich matched_df with token counts / organ ids from shards
    matched_df = matched_df.copy()
    matched_df["token_count_from_shards"] = matched_df["canonical_path"].map(lambda p: token_counts.get(p, -1))
    matched_df["organ_id_from_shards"] = matched_df["canonical_path"].map(lambda p: organ_map.get(p, -1))

    pairwise = cosine_matrix(prototypes)
    pairwise_dict = {}
    for i in range(len(role_names)):
        for j in range(i + 1, len(role_names)):
            pairwise_dict[f"{role_names[i]}__vs__{role_names[j]}"] = float(pairwise[i, j])

    np.save(os.path.join(args.outdir, "role_prototypes_init.npy"), prototypes)
    with open(os.path.join(args.outdir, "role_names.json"), "w", encoding="utf-8") as f:
        json.dump(role_names, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "prototype_pairwise_cosine.json"), "w", encoding="utf-8") as f:
        json.dump(pairwise_dict, f, ensure_ascii=False, indent=2)

    meta["pairwise_cosine"] = pairwise_dict
    meta["shard_dir"] = args.shard_dir
    meta["num_shards"] = len(shard_files)
    meta["feature_key"] = args.feature_key
    meta["path_key"] = args.path_key
    meta["organ_key"] = args.organ_key
    with open(os.path.join(args.outdir, "role_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_cols = [c for c in matched_df.columns if c != "feature"]
    matched_df[save_cols].to_csv(os.path.join(args.outdir, "matched_patch_features.csv"), index=False)

    with open(os.path.join(args.outdir, "unmatched_paths.json"), "w", encoding="utf-8") as f:
        json.dump(meta["unmatched_paths"], f, ensure_ascii=False, indent=2)

    if args.save_matched_features:
        X = np.stack(matched_df["feature"].values, axis=0).astype(np.float32)
        y = matched_df["role"].values.astype(object)
        patch_paths = matched_df["patch_path"].values.astype(object)
        canonical_paths = matched_df["canonical_path"].values.astype(object)
        np.savez_compressed(
            os.path.join(args.outdir, "matched_patch_features.npz"),
            features=X,
            labels=y,
            patch_paths=patch_paths,
            canonical_paths=canonical_paths,
        )

    print("Done.")
    print(f"Saved role prototypes to: {args.outdir}")
    print("Role counts:")
    for role, stats in meta["role_stats"].items():
        print(f"  {role}: {stats['num_matched_patches']}")
    print("Pairwise cosine:")
    for k, v in pairwise_dict.items():
        print(f"  {k}: {v:.4f}")
    print(f"Unmatched target paths: {meta['num_unmatched_paths']}")


if __name__ == "__main__":
    main()
