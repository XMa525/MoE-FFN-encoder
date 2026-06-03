#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


WSI_EXTS = {
    ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif"
}

# 二分类标签：按你的任务设定
LABEL_MAP = {
    "Group_AT": 1,
    "Group_BT": 0,
}


def build_bracs_slide_split_csv(
    bracs_root: str,
    output_csv: str,
    use_relative_path: bool = False,
):
    """
    bracs_root 应该是 BRACS 根目录，而不是 BRACS/PATCHES。

    预期结构：
    BRACS/
    ├── train/
    │   ├── Group_AT/
    │   └── Group_BT/
    ├── val/
    │   ├── Group_AT/
    │   └── Group_BT/
    ├── test/
    │   ├── Group_AT/
    │   └── Group_BT/
    └── PATCHES/
    """

    bracs_root = Path(bracs_root).resolve()
    rows = []

    for split in ["train", "val", "test"]:
        split_dir = bracs_root / split

        if not split_dir.exists():
            print(f"[Warning] split dir not found: {split_dir}")
            continue

        for group_name, label in LABEL_MAP.items():
            group_dir = split_dir / group_name

            if not group_dir.exists():
                print(f"[Warning] group dir not found: {group_dir}")
                continue

            wsi_files = [
                p for p in group_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in WSI_EXTS
            ]

            print(f"[Info] {split}/{group_name}: {len(wsi_files)} WSI files")

            for svs_path in tqdm(wsi_files, desc=f"{split}/{group_name}"):
                slide_id = svs_path.stem

                if use_relative_path:
                    source_path = str(svs_path.relative_to(bracs_root))
                else:
                    source_path = str(svs_path)

                rows.append({
                    "slide_id": slide_id,
                    "label": label,
                    "project": "BRACS",
                    "source_type": "file",
                    "source_path": source_path,
                    "split": split,
                    "group": group_name,
                })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            f"No WSI files found under {bracs_root}. "
            f"Please check bracs_root, folder names, and file extensions."
        )

    df = df.sort_values(["split", "group", "slide_id"]).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\n========== Summary ==========")
    print(df.groupby(["split", "group", "label"]).size())
    print(f"\nSaved to: {output_csv}")
    print(f"Total slides: {len(df)}")
    print(f"Unique slide_id: {df['slide_id'].nunique()}")

    dup = df[df["slide_id"].duplicated(keep=False)]
    if len(dup) > 0:
        print("\n[Warning] duplicated slide_id found:")
        print(dup[["slide_id", "split", "group", "source_path"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bracs_root",
        type=str,
        required=True,
        help="Path to BRACS root, not BRACS/PATCHES"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="bracs_split.csv",
        help="Output csv path"
    )
    parser.add_argument(
        "--relative_path",
        action="store_true",
        help="Save source_path as relative path to BRACS root"
    )
    args = parser.parse_args()

    build_bracs_slide_split_csv(
        bracs_root=args.bracs_root,
        output_csv=args.output_csv,
        use_relative_path=args.relative_path,
    )


if __name__ == "__main__":
    main()