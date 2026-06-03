#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


def set_seed(seed: int = 42):
    np.random.seed(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_locked_train_slide_ids(paths: List[str]) -> List[str]:
    locked = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Locked-train csv not found: {path}")
        df = pd.read_csv(path)
        if "slide_id" not in df.columns:
            raise ValueError(f"{path} missing required column: slide_id")
        locked.extend(df["slide_id"].astype(str).tolist())
    return sorted(set(locked))


def find_files_by_stem(root_dir: str, suffix: str) -> Dict[str, str]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    out = {}
    for p in root.rglob(f"*{suffix}"):
        if p.is_file():
            out[p.stem] = str(p.resolve())
    return out


def build_slide_table_from_dirs(
    svs_root: str,
    h5_root: str,
    benign_dirname: str = "Benign",
    malignant_dirname: str = "Malignant",
    image_suffix: str = ".tif",
) -> pd.DataFrame:
    svs_root = Path(svs_root)
    h5_root = Path(h5_root)

    benign_img_root = svs_root / benign_dirname
    malignant_img_root = svs_root / malignant_dirname

    if not benign_img_root.exists():
        raise FileNotFoundError(f"Benign image dir not found: {benign_img_root}")
    if not malignant_img_root.exists():
        raise FileNotFoundError(f"Malignant image dir not found: {malignant_img_root}")
    if not h5_root.exists():
        raise FileNotFoundError(f"h5 root not found: {h5_root}")

    benign_imgs = find_files_by_stem(str(benign_img_root), image_suffix)
    malignant_imgs = find_files_by_stem(str(malignant_img_root), image_suffix)
    h5_map = find_files_by_stem(str(h5_root), ".h5")

    rows = []

    def collect_rows(img_map: Dict[str, str], label: int, project: str):
        missing_h5 = 0
        for stem, img_path in sorted(img_map.items()):
            if stem not in h5_map:
                missing_h5 += 1
                continue
            rows.append({
                "slide_id": stem,
                "label": int(label),
                "project": project,
                "svs_path": img_path,
                "h5_path": h5_map[stem],
            })
        return missing_h5

    benign_missing = collect_rows(benign_imgs, label=0, project=benign_dirname)
    malignant_missing = collect_rows(malignant_imgs, label=1, project=malignant_dirname)

    print(f"[Benign] images={len(benign_imgs)}, matched={sum(r['project']==benign_dirname for r in rows)}, missing_h5={benign_missing}")
    print(f"[Malignant] images={len(malignant_imgs)}, matched={sum(r['project']==malignant_dirname for r in rows)}, missing_h5={malignant_missing}")
    print(f"[H5 total] {len(h5_map)}")
    print(f"[Matched total] {len(rows)}")

    if len(rows) == 0:
        raise RuntimeError("No matched tif-h5 pairs found.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["slide_id"]).reset_index(drop=True)
    return df


def _split_ids(
    ids: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    if len(ids) == 0:
        return [], [], []

    ids = list(ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n = len(ids)

    if n == 1:
        return ids, [], []
    elif n == 2:
        return [ids[0]], [ids[1]], []
    elif n == 3:
        return [ids[0]], [ids[1]], [ids[2]]

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    while n_train + n_val + n_test < n:
        n_train += 1
    while n_train + n_val + n_test > n:
        if n_train >= max(n_val, n_test) and n_train > 1:
            n_train -= 1
        elif n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break

    if n >= 4:
        if n_val == 0 and n_train > 1:
            n_train -= 1
            n_val = 1
        if n_test == 0 and n_train > 1:
            n_train -= 1
            n_test = 1

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def stratified_slide_split(
    df: pd.DataFrame,
    locked_train_slide_ids: set,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, str]:
    split_map = {}

    for sid in locked_train_slide_ids:
        split_map[str(sid)] = "train"

    remain = df[~df["slide_id"].astype(str).isin(locked_train_slide_ids)].copy()

    for label, sub in remain.groupby("label"):
        ids = sub["slide_id"].astype(str).tolist()
        tr, va, te = _split_ids(
            ids=ids,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed + int(label) * 1000,
        )
        for x in tr:
            split_map[x] = "train"
        for x in va:
            split_map[x] = "val"
        for x in te:
            split_map[x] = "test"

    return split_map


def print_summary(df: pd.DataFrame, title: str):
    print(f"\n===== {title} =====")
    print(f"num slides = {len(df)}")

    print("\n[Split counts]")
    print(df["split"].value_counts(dropna=False).sort_index())

    print("\n[Label x Split]")
    print(pd.crosstab(df["label"], df["split"], dropna=False))

    print("\n[Project x Split]")
    print(pd.crosstab(df["project"], df["split"], dropna=False))


def main():
    parser = argparse.ArgumentParser("Make parotid split directly from tif+h5 directories")

    parser.add_argument("--svs_root", type=str, required=True,
                        help="e.g. data/Parotid/sdpc_to_tif")
    parser.add_argument("--h5_root", type=str, required=True,
                        help="e.g. data/Parotid/patches")
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument("--benign_dirname", type=str, default="Benign")
    parser.add_argument("--malignant_dirname", type=str, default="Malignant")
    parser.add_argument("--image_suffix", type=str, default=".tif")

    parser.add_argument("--locked_train_csv", type=str, action="append", default=[],
                        help="CSV containing slide_id column. Can be repeated. These slides will be forced into train.")

    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    set_seed(args.seed)

    df = build_slide_table_from_dirs(
        svs_root=args.svs_root,
        h5_root=args.h5_root,
        benign_dirname=args.benign_dirname,
        malignant_dirname=args.malignant_dirname,
        image_suffix=args.image_suffix,
    )

    locked_train_slide_ids = set()
    if len(args.locked_train_csv) > 0:
        locked_train_slide_ids = set(read_locked_train_slide_ids(args.locked_train_csv))
        print(f"[Info] locked_train_slide_ids = {len(locked_train_slide_ids)}")

        missing_locked = sorted(list(locked_train_slide_ids - set(df["slide_id"].tolist())))
        if len(missing_locked) > 0:
            print(f"[Warn] some locked slide_ids are not found in matched dataframe, count={len(missing_locked)}")
            print("Example missing:", missing_locked[:10])

        locked_train_slide_ids = locked_train_slide_ids & set(df["slide_id"].tolist())

    split_map = stratified_slide_split(
        df=df,
        locked_train_slide_ids=locked_train_slide_ids,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    df["split"] = df["slide_id"].astype(str).map(split_map)
    if df["split"].isna().any():
        raise ValueError("Some slides were not assigned split.")

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        ensure_dir(out_dir)

    df.to_csv(args.output_csv, index=False)
    print_summary(df, "Final Split Summary")
    print(f"\n[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()