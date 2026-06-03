import argparse
import os
import re
import pandas as pd


POSITIVE_STAGES = {"macro", "micro", "itc"}
NEGATIVE_STAGES = {"negative"}


def is_wsi_row(patient_value: str) -> bool:
    """
    只保留真正的 slide 行，比如 patient_100_node_0.tif
    跳过 patient_100.zip 这种压缩包行
    """
    if not isinstance(patient_value, str):
        return False
    return patient_value.lower().endswith((".tif", ".tiff", ".svs", ".ndpi", ".mrxs"))


def parse_slide_id(filename: str) -> str:
    """
    去掉扩展名，得到 slide_id
    例如 patient_100_node_0.tif -> patient_100_node_0
    """
    return os.path.splitext(filename)[0]


def map_stage_to_label(stage: str):
    """
    negative -> 0
    macro/micro/itc -> 1
    其他返回 None
    """
    if not isinstance(stage, str):
        return None
    s = stage.strip().lower()
    if s in NEGATIVE_STAGES:
        return 0
    if s in POSITIVE_STAGES:
        return 1
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True,
                        help="Path to submission_example.csv")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save sampled stage2 WSI csv")
    parser.add_argument("--target_total", type=int, default=100,
                        help="Target total number of sampled slides")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    # 兼容列名
    if "patient" not in df.columns or "stage" not in df.columns:
        raise ValueError("CSV 必须包含列: patient, stage")

    # 只保留真正 slide 行
    df = df[df["patient"].apply(is_wsi_row)].copy()

    # 提取 slide_id
    df["slide_id"] = df["patient"].apply(parse_slide_id)

    # 转成二分类标签
    df["label"] = df["stage"].apply(map_stage_to_label)

    # 去掉无法映射的行
    df = df[df["label"].notnull()].copy()
    df["label"] = df["label"].astype(int)

    # 去重，避免重复 slide
    df = df.drop_duplicates(subset=["slide_id"]).reset_index(drop=True)

    pos_df = df[df["label"] == 1].copy()
    neg_df = df[df["label"] == 0].copy()

    print(f"[INFO] total valid slides = {len(df)}")
    print(f"[INFO] positive slides    = {len(pos_df)}")
    print(f"[INFO] negative slides    = {len(neg_df)}")

    # 尽量正负均衡
    target_per_class = args.target_total // 2
    actual_per_class = min(target_per_class, len(pos_df), len(neg_df))

    if actual_per_class == 0:
        raise ValueError("正类或负类数量为 0，无法构建平衡数据集。")

    pos_sample = pos_df.sample(n=actual_per_class, random_state=args.seed)
    neg_sample = neg_df.sample(n=actual_per_class, random_state=args.seed)

    out_df = pd.concat([pos_sample, neg_sample], axis=0)
    out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # 只保留常用字段
    out_df = out_df[["slide_id", "patient", "stage", "label"]]

    out_df.to_csv(args.output_csv, index=False)

    print(f"[INFO] sampled positive = {(out_df['label'] == 1).sum()}")
    print(f"[INFO] sampled negative = {(out_df['label'] == 0).sum()}")
    print(f"[INFO] total sampled    = {len(out_df)}")
    print(f"[Saved] {args.output_csv}")


if __name__ == "__main__":
    main()