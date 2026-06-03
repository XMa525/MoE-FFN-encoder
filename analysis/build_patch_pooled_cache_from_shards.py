#!/usr/bin/env python3
"""
Build a reusable patch-level pooled feature cache from token-level shard_XX.npz files.

Why this script exists
----------------------
Your layer24 token-feature shards store one feature vector per token. For the current
role-prototype mainline, repeated token->patch aggregation is wasteful because:
- target patches are distributed across all shards,
- every prototype / analysis step only needs patch-level pooled features.

So the best strategy is to pay the cost once and materialize a reusable patch-level cache.

What this script does
---------------------
For each token-level shard:
1. Read metadata arrays and token-level features.
2. Group contiguous token rows by patch_ids (preferred) or paths (fallback).
3. Mean-pool token features into one patch-level vector.
4. Save patch-level pooled results as one pooled shard file.

Expected token-level shard structure
------------------------------------
Each shard_XX.npz should contain at least:
- features : [num_tokens, D]
- paths    : [num_tokens]
Optionally:
- patch_ids: [num_tokens]
- organs   : [num_tokens]
- token_x, token_y

Output pooled shard structure
-----------------------------
Each pooled_shard_XX.npz will contain:
- canonical_paths  : [num_patches]
- pooled_features  : [num_patches, D]
- token_counts     : [num_patches]
- organ_ids        : [num_patches]
- source_shard_id  : [num_patches]
- source_shard_path: [num_patches]

Typical usage
-------------
python build_patch_pooled_cache_from_shards.py \
  --shard-dir /data/maxinyu/path/to/layer24_shards \
  --outdir /data/maxinyu/path/to/layer24_patch_cache

If you want to skip already built pooled shards:
python build_patch_pooled_cache_from_shards.py \
  --shard-dir /data/maxinyu/path/to/layer24_shards \
  --outdir /data/maxinyu/path/to/layer24_patch_cache \
  --skip-existing

Outputs
-------
1. pooled_shard_XXXXX.npz for each source shard
2. pooled_cache_summary.json
3. pooled_cache_manifest.csv

Notes
-----
- This is intentionally a one-time heavy preprocessing step.
- After this cache exists, downstream scripts should stop touching token-level shards.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build patch-level pooled cache from token-level shards")
    parser.add_argument("--shard-dir", type=str, required=True)
    parser.add_argument("--shard-pattern", type=str, default="shard_*.npz")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--feature-key", type=str, default="features")
    parser.add_argument("--path-key", type=str, default="paths")
    parser.add_argument("--patch-id-key", type=str, default="patch_ids")
    parser.add_argument("--organ-key", type=str, default="organs")
    parser.add_argument("--skip-existing", action="store_true", help="Skip pooled shard if output file already exists")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="Output dtype for pooled features")
    return parser.parse_args()


def contiguous_segments_from_patch_ids(patch_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if patch_ids.ndim != 1:
        raise ValueError(f"patch_ids must be 1D, got shape {patch_ids.shape}")
    if len(patch_ids) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    change = np.flatnonzero(patch_ids[1:] != patch_ids[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(patch_ids)]))
    return starts, ends


def contiguous_segments_from_paths(paths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(paths) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    change = np.flatnonzero(paths[1:] != paths[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(paths)]))
    return starts, ends


def pooled_shard_path(outdir: str, shard_id: int) -> str:
    return os.path.join(outdir, f"pooled_shard_{shard_id:05d}.npz")


def process_one_shard(
    shard_id: int,
    shard_path: str,
    outdir: str,
    feature_key: str,
    path_key: str,
    patch_id_key: str,
    organ_key: str,
    output_dtype: str,
) -> Dict[str, object]:
    data = np.load(shard_path, allow_pickle=True)

    if feature_key not in data.files or path_key not in data.files:
        raise KeyError(f"Shard {shard_path} missing required keys. Found: {data.files}")

    feats = data[feature_key]   # [num_tokens, D]
    paths = data[path_key]      # [num_tokens]
    organs = data[organ_key] if organ_key in data.files else None

    if patch_id_key in data.files:
        starts, ends = contiguous_segments_from_patch_ids(data[patch_id_key])
    else:
        starts, ends = contiguous_segments_from_paths(paths)

    num_patches = len(starts)
    dim = feats.shape[1]

    pooled_features = np.empty((num_patches, dim), dtype=np.float32)
    canonical_paths = np.empty((num_patches,), dtype=object)
    token_counts = np.empty((num_patches,), dtype=np.int32)
    organ_ids = np.empty((num_patches,), dtype=np.int16)
    source_shard_id = np.full((num_patches,), shard_id, dtype=np.int32)
    source_shard_path = np.full((num_patches,), shard_path, dtype=object)

    for i, (start, end) in enumerate(zip(starts, ends)):
        pooled_features[i] = feats[start:end].astype(np.float32, copy=False).mean(axis=0)
        canonical_paths[i] = canonicalize_path(paths[start])
        token_counts[i] = int(end - start)
        organ_ids[i] = int(organs[start]) if organs is not None else -1

    if output_dtype == "float16":
        pooled_features = pooled_features.astype(np.float16)
    else:
        pooled_features = pooled_features.astype(np.float32)

    out_path = pooled_shard_path(outdir, shard_id)
    np.savez_compressed(
        out_path,
        canonical_paths=canonical_paths,
        pooled_features=pooled_features,
        token_counts=token_counts,
        organ_ids=organ_ids,
        source_shard_id=source_shard_id,
        source_shard_path=source_shard_path,
    )

    return {
        "shard_id": shard_id,
        "source_shard_path": shard_path,
        "pooled_shard_path": out_path,
        "num_tokens": int(feats.shape[0]),
        "num_patches": int(num_patches),
        "feature_dim": int(dim),
        "output_dtype": output_dtype,
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    shard_files = sorted(glob.glob(os.path.join(args.shard_dir, args.shard_pattern)))
    if len(shard_files) == 0:
        raise FileNotFoundError(f"No shard files found in {args.shard_dir} with pattern {args.shard_pattern}")

    manifest_rows = []
    total_tokens = 0
    total_patches = 0
    feature_dim = None

    print(f"Found {len(shard_files)} token-level shard files")

    for shard_id, shard_path in enumerate(tqdm(shard_files, desc="Building pooled cache")):
        out_path = pooled_shard_path(args.outdir, shard_id)
        if args.skip_existing and os.path.exists(out_path):
            data = np.load(out_path, allow_pickle=True)
            row = {
                "shard_id": shard_id,
                "source_shard_path": shard_path,
                "pooled_shard_path": out_path,
                "num_tokens": -1,
                "num_patches": int(len(data["canonical_paths"])),
                "feature_dim": int(data["pooled_features"].shape[1]),
                "output_dtype": str(data["pooled_features"].dtype),
            }
        else:
            row = process_one_shard(
                shard_id=shard_id,
                shard_path=shard_path,
                outdir=args.outdir,
                feature_key=args.feature_key,
                path_key=args.path_key,
                patch_id_key=args.patch_id_key,
                organ_key=args.organ_key,
                output_dtype=args.dtype,
            )

        manifest_rows.append(row)
        total_patches += row["num_patches"]
        if row["num_tokens"] > 0:
            total_tokens += row["num_tokens"]
        if feature_dim is None:
            feature_dim = row["feature_dim"]

        if shard_id % 25 == 0 or shard_id == len(shard_files) - 1:
            print(f"[Info] processed shard {shard_id+1}/{len(shard_files)} | cumulative pooled patches: {total_patches}")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(os.path.join(args.outdir, "pooled_cache_manifest.csv"), index=False)

    summary = {
        "num_source_shards": int(len(shard_files)),
        "num_pooled_shards": int(len(manifest_df)),
        "total_tokens_seen": int(total_tokens),
        "total_patches_written": int(total_patches),
        "feature_dim": int(feature_dim) if feature_dim is not None else None,
        "output_dtype": args.dtype,
        "source_shard_dir": args.shard_dir,
    }
    with open(os.path.join(args.outdir, "pooled_cache_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Saved pooled patch cache to: {args.outdir}")
    print(f"Total pooled patches: {total_patches}")


if __name__ == "__main__":
    main()
