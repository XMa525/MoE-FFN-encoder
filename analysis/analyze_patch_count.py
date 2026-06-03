import os
import h5py
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def count_patches_in_h5(h5_path: str) -> int:
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        return len(f["coords"])


def main():
    parser = argparse.ArgumentParser("Analyze patch counts from CLAM h5 files")
    parser.add_argument("--h5_dir", type=str, required=True, help="Directory containing CLAM .h5 files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for csv and plots")
    args = parser.parse_args()

    h5_dir = Path(args.h5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(h5_dir.glob("*.h5"))
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No .h5 files found in {h5_dir}")

    rows = []
    for h5_path in h5_files:
        try:
            n_patches = count_patches_in_h5(str(h5_path))
            rows.append({
                "slide_id": h5_path.stem,
                "num_patches": n_patches,
            })
        except Exception as e:
            print(f"[ERROR] {h5_path.name}: {e}")

    df = pd.DataFrame(rows).sort_values("num_patches", ascending=False).reset_index(drop=True)
    csv_path = out_dir / "patch_count_per_slide.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved: {csv_path}")
    print("\n===== Basic statistics =====")
    print(df["num_patches"].describe())

    quantiles = df["num_patches"].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    print("\n===== Quantiles =====")
    print(quantiles)

    print("\n===== Threshold counts =====")
    for th in [1000, 2000, 3000, 5000, 8000, 10000]:
        cnt = (df["num_patches"] > th).sum()
        print(f"> {th}: {cnt} slides")

    # histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["num_patches"], bins=40)
    plt.xlabel("Number of patches per slide")
    plt.ylabel("Number of slides")
    plt.title("Patch count distribution")
    hist_path = out_dir / "patch_count_hist.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()

    # boxplot
    plt.figure(figsize=(6, 5))
    plt.boxplot(df["num_patches"], vert=True)
    plt.ylabel("Number of patches per slide")
    plt.title("Patch count boxplot")
    box_path = out_dir / "patch_count_boxplot.png"
    plt.tight_layout()
    plt.savefig(box_path, dpi=200)
    plt.close()

    print(f"Saved: {hist_path}")
    print(f"Saved: {box_path}")

    # top slides
    top_path = out_dir / "top20_patch_heavy_slides.csv"
    df.head(20).to_csv(top_path, index=False)
    print(f"Saved: {top_path}")


if __name__ == "__main__":
    main()