#!/usr/bin/env python3
"""
Build a lightweight shard index for target patches without reading feature arrays.

Goal
----
Before decoding huge token feature arrays from shard_XX.npz, first scan only the
lightweight metadata arrays (paths / patch_ids / organs) to determine:

1. Which target patches are present in which shards.
2. The token row ranges [start, end) for each matched patch.
3. How many shards actually need to be opened in a second-stage feature pass.

Why this matters
----------------
In the current prototype-building pipeline, reading compressed `features` arrays from
all shards is the dominant bottleneck. This script avoids touching `features` entirely
and only builds a reusable index over target patch locations.

Expected shard structure
------------------------
Each shard_XX.npz should contain at least:
- paths    : [num_tokens]
Optionally:
- patch_ids: [num_tokens]
- organs   : [num_tokens]

If patch_ids are available, contiguous runs of identical patch_id are used to define
patch segments. Otherwise, contiguous runs of identical paths are used.

Inputs
------
- Three role CSVs containing patch_path columns:
    tumor core balanced
    stroma core balanced
    ambiguous_clean balanced
- A shard directory containing shard_*.npz files.

Outputs
-------
1. target_patch_shard_index.npz
   Arrays:
   - canonical_paths  : [M] object
   - role_labels      : [M] object
   - shard_ids        : [M] int32
   - shard_paths      : [M] object
   - starts           : [M] int32
   - ends             : [M] int32
   - token_counts     : [M] int32
   - organ_ids        : [M] int16

2. shard_hit_summary.csv
   One row per shard with matched patch count.

3. index_summary.json
   Overall statistics.

Typical usage
-------------
python index_target_patches_in_shards.py \
  --tumor-csv outputs/conch_analysis/candidate_core_tumor_balanced_by_organ.csv \
  --stroma-csv outputs/conch_analysis/candidate_core_stroma_balanced_by_organ.csv \
  --ambiguous-csv outputs/refined_candidates/ambiguous_clean_balanced_by_organ.csv \
  --shard-dir /data/maxinyu/path/to/layer24_shards \
  --outdir outputs/role_proto_init_layer24_index
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index target patches in shards without reading feature arrays")
    parser.add_argument("--tumor-csv", type=str, required=True)
    parser.add_argument("--stroma-csv", type=str, required=True)
    parser.add_argument("--ambiguous-csv", type=str, required=True)
    parser.add_argument("--shard-dir", type=str, required=True)
    parser.add_argument("--shard-pattern", type=str, default="shard_*.npz")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--path-key", type=str, default="paths")
    parser.add_argument("--patch-id-key", type=str, default="patch_ids")
    parser.add_argument("--organ-key", type=str, default="organs")
    parser.add_argument("--stop-when-complete", action="store_true", help="Early stop if all target patches have been matched")
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
    dup_counts = df_all["canonical_path"].value_counts()
    duplicated = dup_counts[dup_counts > 1]
    if len(duplicated) > 0:
        print(f"[Warn] {len(duplicated)} canonical paths appear multiple times; keeping first occurrence.")
        df_all = df_all.drop_duplicates(subset=["canonical_path"], keep="first").reset_index(drop=True)

    target_map = {row["canonical_path"]: row["role"] for _, row in df_all.iterrows()}
    return target_map, df_all


def contiguous_segments_from_patch_ids(patch_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if patch_ids.ndim != 1:
        raise ValueError(f"patch_ids must be 1D, got {patch_ids.shape}")
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


def index_target_patches(
    shard_files: List[str],
    target_map: Dict[str, str],
    path_key: str,
    patch_id_key: str,
    organ_key: str,
    stop_when_complete: bool,
) -> Tuple[List[Dict[str, object]], pd.DataFrame, Dict[str, object]]:
    target_set = set(target_map.keys())
    remaining = set(target_set)

    index_rows: List[Dict[str, object]] = []
    shard_stats = []

    for shard_id, shard_path in enumerate(tqdm(shard_files, desc="Indexing shards")):
        data = np.load(shard_path, allow_pickle=True)
        if path_key not in data.files:
            raise KeyError(f"Shard missing key '{path_key}': {shard_path}. Found {data.files}")

        paths = data[path_key]
        organs = data[organ_key] if organ_key in data.files else None

        if patch_id_key in data.files:
            starts, ends = contiguous_segments_from_patch_ids(data[patch_id_key])
        else:
            starts, ends = contiguous_segments_from_paths(paths)

        shard_match_count = 0
        shard_patch_count = len(starts)

        for start, end in zip(starts, ends):
            cp = canonicalize_path(paths[start])
            if cp not in target_set:
                continue

            organ_id = int(organs[start]) if organs is not None else -1
            row = {
                "canonical_path": cp,
                "role": target_map[cp],
                "shard_id": shard_id,
                "shard_path": shard_path,
                "start": int(start),
                "end": int(end),
                "token_count": int(end - start),
                "organ_id": organ_id,
            }
            index_rows.append(row)
            shard_match_count += 1
            if cp in remaining:
                remaining.remove(cp)

        shard_stats.append({
            "shard_id": shard_id,
            "shard_path": shard_path,
            "num_patch_segments": shard_patch_count,
            "matched_target_patches": shard_match_count,
            "num_remaining_after_shard": len(remaining),
        })

        if shard_id % 25 == 0 or shard_id == len(shard_files) - 1:
            print(f"[Info] matched {len(target_set) - len(remaining)} / {len(target_set)} target patches after shard {shard_id+1}/{len(shard_files)}")

        if stop_when_complete and not remaining:
            print("[Info] All target patches indexed. Early stopping.")
            break

    shard_df = pd.DataFrame(shard_stats)
    summary = {
        "num_target_paths": int(len(target_set)),
        "num_index_rows": int(len(index_rows)),
        "num_unique_matched_paths": int(len(set(r["canonical_path"] for r in index_rows))),
        "num_unmatched_paths": int(len(remaining)),
        "num_shards_scanned": int(len(shard_df)),
        "num_hit_shards": int((shard_df["matched_target_patches"] > 0).sum()) if len(shard_df) > 0 else 0,
        "unmatched_paths": sorted(list(remaining)),
    }
    return index_rows, shard_df, summary


def save_index(index_rows: List[Dict[str, object]], out_path: str) -> None:
    if len(index_rows) == 0:
        np.savez_compressed(
            out_path,
            canonical_paths=np.array([], dtype=object),
            role_labels=np.array([], dtype=object),
            shard_ids=np.array([], dtype=np.int32),
            shard_paths=np.array([], dtype=object),
            starts=np.array([], dtype=np.int32),
            ends=np.array([], dtype=np.int32),
            token_counts=np.array([], dtype=np.int32),
            organ_ids=np.array([], dtype=np.int16),
        )
        return

    canonical_paths = np.array([r["canonical_path"] for r in index_rows], dtype=object)
    role_labels = np.array([r["role"] for r in index_rows], dtype=object)
    shard_ids = np.array([r["shard_id"] for r in index_rows], dtype=np.int32)
    shard_paths = np.array([r["shard_path"] for r in index_rows], dtype=object)
    starts = np.array([r["start"] for r in index_rows], dtype=np.int32)
    ends = np.array([r["end"] for r in index_rows], dtype=np.int32)
    token_counts = np.array([r["token_count"] for r in index_rows], dtype=np.int32)
    organ_ids = np.array([r["organ_id"] for r in index_rows], dtype=np.int16)

    np.savez_compressed(
        out_path,
        canonical_paths=canonical_paths,
        role_labels=role_labels,
        shard_ids=shard_ids,
        shard_paths=shard_paths,
        starts=starts,
        ends=ends,
        token_counts=token_counts,
        organ_ids=organ_ids,
    )


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)

    tumor_df = load_role_csv(args.tumor_csv, "tumor")
    stroma_df = load_role_csv(args.stroma_csv, "stroma")
    ambiguous_df = load_role_csv(args.ambiguous_csv, "ambiguous")
    target_map, df_targets = build_target_map([tumor_df, stroma_df, ambiguous_df])

    shard_files = sorted(glob.glob(os.path.join(args.shard_dir, args.shard_pattern)))
    if len(shard_files) == 0:
        raise FileNotFoundError(f"No shard files found in {args.shard_dir} with pattern {args.shard_pattern}")

    print(f"Found {len(shard_files)} shard files")
    print(f"Target patches to index: {len(target_map)}")

    index_rows, shard_df, summary = index_target_patches(
        shard_files=shard_files,
        target_map=target_map,
        path_key=args.path_key,
        patch_id_key=args.patch_id_key,
        organ_key=args.organ_key,
        stop_when_complete=args.stop_when_complete,
    )

    index_path = os.path.join(args.outdir, "target_patch_shard_index.npz")
    save_index(index_rows, index_path)
    shard_df.to_csv(os.path.join(args.outdir, "shard_hit_summary.csv"), index=False)

    with open(os.path.join(args.outdir, "index_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Saved index to: {index_path}")
    print(f"Unique matched target paths: {summary['num_unique_matched_paths']} / {summary['num_target_paths']}")
    print(f"Hit shards: {summary['num_hit_shards']} / {summary['num_shards_scanned']}")
    print(f"Unmatched target paths: {summary['num_unmatched_paths']}")


if __name__ == "__main__":
    main()
