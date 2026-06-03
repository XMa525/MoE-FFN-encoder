import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)


def infer_label_column(df: pd.DataFrame) -> str:
    if "label" in df.columns:
        return "label"
    if "slide_binary_label" in df.columns:
        return "slide_binary_label"
    raise ValueError("CSV must contain 'label' or 'slide_binary_label' column.")


def balance_one_split(df_split: pd.DataFrame, label_col: str, seed: int = 42) -> pd.DataFrame:
    """
    对单个 split 做类均衡：
    - 找到正负类数量
    - 从多数类中随机下采样到和少数类一样多
    """
    if len(df_split) == 0:
        return df_split.copy()

    vc = df_split[label_col].value_counts().to_dict()
    if 0 not in vc or 1 not in vc:
        print(f"[Warn] split={df_split['split'].iloc[0]} only has one class, skip balancing.")
        return df_split.copy()

    n0 = vc[0]
    n1 = vc[1]
    n_min = min(n0, n1)

    df0 = df_split[df_split[label_col] == 0]
    df1 = df_split[df_split[label_col] == 1]

    df0_bal = df0.sample(n=n_min, random_state=seed, replace=False)
    df1_bal = df1.sample(n=n_min, random_state=seed, replace=False)

    df_bal = pd.concat([df0_bal, df1_bal], axis=0)
    df_bal = df_bal.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_bal


def print_split_stats(df: pd.DataFrame, label_col: str, prefix: str):
    print(f"\n===== {prefix} =====")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            print(f"{split}: 0")
            continue
        vc = sub[label_col].value_counts().to_dict()
        n0 = vc.get(0, 0)
        n1 = vc.get(1, 0)
        print(f"{split}: total={len(sub)}, neg={n0}, pos={n1}")


def main():
    parser = argparse.ArgumentParser("Make balanced train/val/test CSV by undersampling majority class")
    parser.add_argument("--in_csv", type=str, required=True, help="Original slides CSV")
    parser.add_argument("--out_csv", type=str, required=True, help="Balanced output CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    df = pd.read_csv(args.in_csv)
    label_col = infer_label_column(df)

    required_cols = ["slide_id", "split", label_col]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    print_split_stats(df, label_col, prefix="Original split stats")

    parts = []
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].copy()
        sub_bal = balance_one_split(sub, label_col=label_col, seed=args.seed)
        parts.append(sub_bal)

    df_bal = pd.concat(parts, axis=0).reset_index(drop=True)

    # 若原来没有 label 列但有 slide_binary_label，可额外补一个 label，方便后续统一
    if "label" not in df_bal.columns and label_col == "slide_binary_label":
        df_bal["label"] = df_bal["slide_binary_label"]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_bal.to_csv(args.out_csv, index=False)

    print_split_stats(df_bal, "label" if "label" in df_bal.columns else label_col, prefix="Balanced split stats")
    print(f"\n[OK] saved to: {args.out_csv}")


if __name__ == "__main__":
    main()