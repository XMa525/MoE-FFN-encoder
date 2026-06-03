#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd


TCGA_PATIENT_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.IGNORECASE)


def set_seed(seed: int = 42):
    np.random.seed(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "label" in df.columns:
        pass
    elif "slide_binary_label" in df.columns:
        df["label"] = df["slide_binary_label"]
    else:
        raise ValueError("Input csv must contain 'label' or 'slide_binary_label' column.")

    df["label"] = df["label"].astype(int)
    return df


def check_required_columns(df: pd.DataFrame):
    required = ["slide_id", "label", "project"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input csv missing required columns: {missing}")


def extract_tcga_patient_id(slide_id: str) -> str:
    s = str(slide_id).strip()
    m = TCGA_PATIENT_RE.search(s)
    if m is not None:
        return m.group(1).upper()

    parts = s.split("-")
    if len(parts) >= 3 and parts[0].upper() == "TCGA":
        return "-".join(parts[:3]).upper()

    raise ValueError(f"Cannot extract TCGA patient id from slide_id: {slide_id}")


def add_patient_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["patient_id"] = df["slide_id"].map(extract_tcga_patient_id)
    return df


def build_patient_group_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    patient-level grouping table for split only.
    保留 slide 原始 label，不要求 patient 单标签。

    为了近似分层，构造：
    - patient_major_project: 该 patient 最常见 project
    - patient_pos_any:      该 patient 是否含任一正样本
    """
    rows = []
    for patient_id, sub in df.groupby("patient_id", dropna=False):
        proj_counts = sub["project"].astype(str).value_counts()
        major_project = proj_counts.index[0]

        patient_pos_any = int((sub["label"].astype(int) == 1).any())
        patient_num_slides = int(len(sub))
        patient_num_pos = int((sub["label"].astype(int) == 1).sum())
        patient_num_neg = int((sub["label"].astype(int) == 0).sum())

        rows.append({
            "patient_id": patient_id,
            "patient_major_project": major_project,
            "patient_pos_any": patient_pos_any,
            "patient_num_slides": patient_num_slides,
            "patient_num_pos": patient_num_pos,
            "patient_num_neg": patient_num_neg,
        })

    return pd.DataFrame(rows)


def split_one_group(
    group_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    对单个分层子组（这里是 patient-level group）做切分。
    """
    if len(group_df) == 0:
        return (
            group_df.iloc[[]].copy(),
            group_df.iloc[[]].copy(),
            group_df.iloc[[]].copy(),
        )

    group_df = group_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(group_df)

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

    # 小组保底
    if n == 1:
        n_train, n_val, n_test = 1, 0, 0
    elif n == 2:
        n_train, n_val, n_test = 1, 1, 0
    elif n == 3:
        n_train, n_val, n_test = 1, 1, 1
    else:
        if n_val == 0:
            if n_train > 1:
                n_train -= 1
                n_val = 1
            elif n_test > 1:
                n_test -= 1
                n_val = 1

        if n_test == 0:
            if n_train > 1:
                n_train -= 1
                n_test = 1
            elif n_val > 1:
                n_val -= 1
                n_test = 1

    assert n_train + n_val + n_test == n, (n, n_train, n_val, n_test)

    train_df = group_df.iloc[:n_train].copy()
    val_df = group_df.iloc[n_train:n_train + n_val].copy()
    test_df = group_df.iloc[n_train + n_val:].copy()
    return train_df, val_df, test_df


def build_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    set_seed(seed)

    patient_df = build_patient_group_table(df)

    parts: List[pd.DataFrame] = []

    # 按 patient-level summary 分层，而不是按 slide-level label
    # 这样可确保 patient 不跨 split，同时大致维持 project 和 patient-level positive presence 的平衡
    for (project, patient_pos_any), sub_df in patient_df.groupby(
        ["patient_major_project", "patient_pos_any"], dropna=False
    ):
        tr, va, te = split_one_group(
            group_df=sub_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
        tr["split"] = "train"
        va["split"] = "val"
        te["split"] = "test"
        parts.extend([tr, va, te])

    patient_split_df = pd.concat(parts, axis=0).reset_index(drop=True)

    # 回填到 slide-level：同 patient 的所有 slide 一起进同一 split
    out_df = df.merge(
        patient_split_df[["patient_id", "split"]],
        on="patient_id",
        how="left",
        validate="many_to_one",
    )

    if out_df["split"].isna().any():
        bad = out_df[out_df["split"].isna()][["slide_id", "patient_id"]]
        raise RuntimeError(f"Some slides did not get split assigned:\n{bad.head()}")

    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out_df


def check_no_patient_leakage(df: pd.DataFrame):
    leak = (
        df.groupby("patient_id")["split"]
        .nunique()
        .reset_index(name="num_splits")
    )
    leak = leak[leak["num_splits"] > 1].copy()

    if len(leak) > 0:
        raise RuntimeError(
            f"Patient leakage detected: {len(leak)} patients appear in multiple splits.\n"
            f"Examples:\n{leak.head(10).to_string(index=False)}"
        )

    print("\n[Check] No patient leakage across splits.")


def print_summary(df: pd.DataFrame, title: str):
    print(f"\n===== {title} =====")
    print(f"num slides    = {len(df)}")
    print(f"num patients  = {df['patient_id'].nunique()}")

    if "split" in df.columns:
        print("\n[Split counts | slide-level]")
        print(df["split"].value_counts(dropna=False).sort_index())

        print("\n[Split counts | patient-level]")
        patient_split = df[["patient_id", "split"]].drop_duplicates()
        print(patient_split["split"].value_counts(dropna=False).sort_index())

    print("\n[Label x Split | slide-level]")
    print(pd.crosstab(df["label"], df["split"], dropna=False))

    print("\n[Project x Split | slide-level]")
    print(pd.crosstab(df["project"], df["split"], dropna=False))

    # 这个表更有用：patient 内是否含正样本，在不同 split 的分布
    patient_summary = (
        df.groupby(["patient_id", "split"], dropna=False)["label"]
        .agg(patient_pos_any=lambda x: int((x.astype(int) == 1).any()))
        .reset_index()
    )

    print("\n[Patient-pos-any x Split | patient-level]")
    print(pd.crosstab(patient_summary["patient_pos_any"], patient_summary["split"], dropna=False))

    print("\n[Project x Patient-pos-any x Split | patient-level]")
    patient_project = (
        df.groupby(["patient_id", "split"], dropna=False)
        .agg(
            project=("project", lambda x: x.astype(str).value_counts().index[0]),
            patient_pos_any=("label", lambda x: int((x.astype(int) == 1).any())),
        )
        .reset_index()
    )
    g = (
        patient_project.groupby(["project", "patient_pos_any", "split"])
        .size()
        .reset_index(name="count")
        .sort_values(["project", "patient_pos_any", "split"])
    )
    print(g.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        "Create patient-grouped train/val/test split csv for TCGA slide-level training"
    )
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Input slide-level csv containing slide_id, label(or slide_binary_label), project")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Output split csv path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input csv not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    df = normalize_label_column(df)
    check_required_columns(df)

    # 去重：避免同一个 slide_id 重复
    if df["slide_id"].duplicated().any():
        dup_n = int(df["slide_id"].duplicated().sum())
        print(f"[Warn] found duplicated slide_id rows: {dup_n}, keeping first occurrence")
        df = df.drop_duplicates(subset=["slide_id"], keep="first").reset_index(drop=True)

    df = add_patient_id(df)

    out_df = build_split(
        df=df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    check_no_patient_leakage(out_df)

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        ensure_dir(out_dir)

    out_df.to_csv(args.output_csv, index=False)
    print_summary(out_df, "Final Patient-Grouped Split Summary")
    print(f"\n[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()