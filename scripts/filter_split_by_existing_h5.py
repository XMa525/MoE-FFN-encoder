#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd


def normalize_h5_stem(path: Path) -> str:
    """
    Match common CLAM h5 names:
      BRACS_1599.h5         -> BRACS_1599
      BRACS_1599_patches.h5 -> BRACS_1599
      BRACS_1599.ome.h5     -> BRACS_1599
    """
    stem = path.stem

    for suffix in ["_patches", ".ome"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def build_h5_index(h5_root: str):
    h5_root = Path(h5_root).resolve()

    if not h5_root.exists():
        raise FileNotFoundError(f"h5_root does not exist: {h5_root}")

    h5_map = {}
    duplicates = {}

    for p in h5_root.rglob("*.h5"):
        key = normalize_h5_stem(p)

        if key in h5_map:
            duplicates.setdefault(key, [h5_map[key]]).append(str(p))
            continue

        h5_map[key] = str(p.resolve())

    return h5_map, duplicates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", required=True, help="Original BRACS split csv")
    parser.add_argument("--h5-root", required=True, help="Root directory containing h5 files")
    parser.add_argument("--out-csv", required=True, help="Filtered split csv with only slides that have h5")
    parser.add_argument(
        "--missing-csv",
        default=None,
        help="Optional csv path to save missing h5 slides"
    )
    parser.add_argument(
        "--add-h5-path",
        action="store_true",
        help="Add matched h5_path column to output csv"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.split_csv)

    if "slide_id" not in df.columns:
        raise ValueError("split csv must contain column: slide_id")

    h5_map, duplicates = build_h5_index(args.h5_root)

    print(f"[Info] Loaded split csv: {args.split_csv}")
    print(f"[Info] Num rows in split csv: {len(df)}")
    print(f"[Info] Found h5 files: {len(h5_map)}")
    print(f"[Info] Duplicate h5 stems: {len(duplicates)}")

    if duplicates:
        print("[Warning] Duplicate h5 stems found. Keeping the first one.")
        for k, paths in list(duplicates.items())[:10]:
            print(f"  {k}:")
            for p in paths:
                print(f"    {p}")

    df["slide_id"] = df["slide_id"].astype(str)
    df["_matched_h5_path"] = df["slide_id"].map(h5_map)

    keep_df = df[df["_matched_h5_path"].notna()].copy()
    missing_df = df[df["_matched_h5_path"].isna()].copy()

    if args.add_h5_path:
        keep_df["h5_path"] = keep_df["_matched_h5_path"]

    keep_df = keep_df.drop(columns=["_matched_h5_path"])
    missing_df = missing_df.drop(columns=["_matched_h5_path"])

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keep_df.to_csv(out_csv, index=False)

    if args.missing_csv is not None:
        missing_csv = Path(args.missing_csv)
    else:
        missing_csv = out_csv.with_name(out_csv.stem + "_missing_h5.csv")

    missing_df.to_csv(missing_csv, index=False)

    print("\n========== Summary ==========")
    print(f"Kept slides:    {len(keep_df)}")
    print(f"Missing slides: {len(missing_df)}")
    print(f"Saved filtered split csv to: {out_csv}")
    print(f"Saved missing h5 list to:    {missing_csv}")

    if "split" in keep_df.columns:
        print("\n[Kept by split]")
        print(keep_df["split"].value_counts())

    if "label" in keep_df.columns:
        print("\n[Kept by label]")
        print(keep_df["label"].value_counts())

    if "split" in keep_df.columns and "label" in keep_df.columns:
        print("\n[Kept by split × label]")
        print(pd.crosstab(keep_df["split"], keep_df["label"]))

    if len(missing_df) > 0:
        print("\n[Missing examples]")
        cols = [c for c in ["slide_id", "split", "group", "label", "source_path", "svs_path"] if c in missing_df.columns]
        print(missing_df[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()