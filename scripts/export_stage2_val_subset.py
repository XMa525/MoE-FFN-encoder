import os
import sys
import argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from distillation.dataset.tcga_stage2_dataset import TCGARolePatchDataset


def build_train_val_indices(n_total: int, val_ratio: float = 0.2, seed: int = 42):
    """
    与你当前训练脚本保持同风格的 patch-level 随机划分。
    注意：
    这不是 slide-level split，而是对总行数做随机切分。
    """
    if n_total <= 0:
        raise ValueError(f"n_total must be > 0, got {n_total}")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n_total, dtype=np.int64)
    rng.shuffle(all_idx)

    n_val = int(round(n_total * val_ratio))
    n_val = max(1, min(n_val, n_total - 1))

    val_indices = np.sort(all_idx[:n_val])
    train_indices = np.sort(all_idx[n_val:])
    return train_indices, val_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool-csv",
        type=str,
        required=True,
        help="原始 stage2 pool csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="导出 train/val 子集 csv 的目录",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="与训练保持一致的 val_ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="与训练保持一致的随机种子",
    )
    parser.add_argument(
        "--slide-label-col",
        type=str,
        default="slide_label",
        help="slide label 列名",
    )
    parser.add_argument(
        "--filter-prefilter-white",
        action="store_true",
        help="是否与训练一致过滤 prefilter_white=1",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pool_csv):
        raise FileNotFoundError(f"pool csv not found: {args.pool_csv}")

    os.makedirs(args.out_dir, exist_ok=True)

    full_df = pd.read_csv(args.pool_csv)
    print(f"[Pool CSV] rows = {len(full_df)}")
    print(f"[Pool CSV] slides = {full_df['slide_id'].nunique() if 'slide_id' in full_df.columns else 'N/A'}")

    train_indices, val_indices = build_train_val_indices(
        n_total=len(full_df),
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"[Split] train_indices = {len(train_indices)}")
    print(f"[Split] val_indices   = {len(val_indices)}")

    train_dataset = TCGARolePatchDataset(
        csv_path=args.pool_csv,
        transform=None,
        indices=train_indices,
        filter_prefilter_white=args.filter_prefilter_white,
        verbose=True,
        use_wsi_bag_sampling=False,
        slide_label_col=args.slide_label_col,
        random_seed=args.seed,
        use_spatial_neighbor_sampling=False,
    )

    val_dataset = TCGARolePatchDataset(
        csv_path=args.pool_csv,
        transform=None,
        indices=val_indices,
        filter_prefilter_white=args.filter_prefilter_white,
        verbose=True,
        use_wsi_bag_sampling=False,
        slide_label_col=args.slide_label_col,
        random_seed=args.seed + 999,
        use_spatial_neighbor_sampling=False,
    )

    train_csv = os.path.join(args.out_dir, "train_subset_exact.csv")
    val_csv = os.path.join(args.out_dir, "val_subset_exact.csv")
    train_slide_csv = os.path.join(args.out_dir, "train_slide_ids_exact.csv")
    val_slide_csv = os.path.join(args.out_dir, "val_slide_ids_exact.csv")
    summary_txt = os.path.join(args.out_dir, "split_summary.txt")

    train_dataset.df.to_csv(train_csv, index=False)
    val_dataset.df.to_csv(val_csv, index=False)

    if "slide_id" in train_dataset.df.columns:
        train_dataset.df[["slide_id"]].drop_duplicates().to_csv(train_slide_csv, index=False)
    else:
        pd.DataFrame(columns=["slide_id"]).to_csv(train_slide_csv, index=False)

    if "slide_id" in val_dataset.df.columns:
        val_dataset.df[["slide_id"]].drop_duplicates().to_csv(val_slide_csv, index=False)
    else:
        pd.DataFrame(columns=["slide_id"]).to_csv(val_slide_csv, index=False)

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("[Original Pool]\n")
        f.write(f"rows={len(full_df)}\n")
        if "slide_id" in full_df.columns:
            f.write(f"slides={full_df['slide_id'].nunique()}\n")
        if args.slide_label_col in full_df.columns:
            f.write("slide label counts:\n")
            f.write(full_df[args.slide_label_col].value_counts(dropna=False).to_string())
            f.write("\n")
        f.write("\n")

        f.write("[Train Subset Exported]\n")
        f.write(f"rows={len(train_dataset.df)}\n")
        if "slide_id" in train_dataset.df.columns:
            f.write(f"slides={train_dataset.df['slide_id'].nunique()}\n")
        if args.slide_label_col in train_dataset.df.columns:
            f.write("slide label counts:\n")
            f.write(train_dataset.df[args.slide_label_col].value_counts(dropna=False).to_string())
            f.write("\n")
        f.write("\n")

        f.write("[Val Subset Exported]\n")
        f.write(f"rows={len(val_dataset.df)}\n")
        if "slide_id" in val_dataset.df.columns:
            f.write(f"slides={val_dataset.df['slide_id'].nunique()}\n")
        if args.slide_label_col in val_dataset.df.columns:
            f.write("slide label counts:\n")
            f.write(val_dataset.df[args.slide_label_col].value_counts(dropna=False).to_string())
            f.write("\n")
        f.write("\n")

        if "slide_id" in train_dataset.df.columns and "slide_id" in val_dataset.df.columns:
            train_slides = set(train_dataset.df["slide_id"].astype(str).tolist())
            val_slides = set(val_dataset.df["slide_id"].astype(str).tolist())
            overlap = train_slides & val_slides
            f.write("[Slide Overlap Check]\n")
            f.write(f"train_unique_slides={len(train_slides)}\n")
            f.write(f"val_unique_slides={len(val_slides)}\n")
            f.write(f"overlap_slides={len(overlap)}\n")

    print(f"[Saved] {train_csv}")
    print(f"[Saved] {val_csv}")
    print(f"[Saved] {train_slide_csv}")
    print(f"[Saved] {val_slide_csv}")
    print(f"[Saved] {summary_txt}")


if __name__ == "__main__":
    main()