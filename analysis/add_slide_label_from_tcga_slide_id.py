import re
import argparse
import pandas as pd


def parse_tcga_slide_label(slide_id: str):
    """
    从 TCGA slide_id 中解析 slide-level label

    规则：
    - TCGA-B0-4693-01Z-00-DX1 里的第4段是 01Z
    - 取前两位 sample type code:
        01 -> tumor -> 1
        11 -> normal -> 0

    返回:
        1 / 0 / None
    """
    if pd.isna(slide_id):
        return None

    slide_id = str(slide_id).strip()
    parts = slide_id.split("-")

    # 标准 TCGA barcode / slide_id 一般至少有4段
    if len(parts) < 4:
        return None

    sample_field = parts[3]  # e.g. "01Z", "11A"
    if len(sample_field) < 2:
        return None

    sample_code = sample_field[:2]

    if sample_code == "01":
        return 1
    elif sample_code == "11":
        return 0
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="输入的 pool csv 路径"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="输出的新 csv 路径（带 slide_label 列）"
    )
    parser.add_argument(
        "--drop-unresolved",
        action="store_true",
        help="是否删除无法解析 slide_label 的行"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if "slide_id" not in df.columns:
        raise ValueError("CSV 中没有 slide_id 列，无法解析 TCGA slide_label")

    df = df.copy()
    df["slide_label"] = df["slide_id"].map(parse_tcga_slide_label)

    # 统计
    total_rows = len(df)
    resolved_rows = df["slide_label"].notna().sum()
    unresolved_rows = df["slide_label"].isna().sum()

    print(f"Total rows      : {total_rows}")
    print(f"Resolved rows   : {resolved_rows}")
    print(f"Unresolved rows : {unresolved_rows}")

    # 打印 label 分布
    if resolved_rows > 0:
        print("\n[slide_label value counts]")
        print(df["slide_label"].value_counts(dropna=False))

        print("\n[unique slide counts by label]")
        tmp = df.dropna(subset=["slide_label"]).copy()
        tmp["slide_label"] = tmp["slide_label"].astype(int)
        print(tmp.groupby("slide_label")["slide_id"].nunique())

    # 打印无法解析的样本示例
    if unresolved_rows > 0:
        unresolved_example = (
            df.loc[df["slide_label"].isna(), "slide_id"]
            .drop_duplicates()
            .tolist()[:20]
        )
        print("\n[Example unresolved slide_id]")
        for x in unresolved_example:
            print(x)

    if args.drop_unresolved:
        before = len(df)
        df = df[df["slide_label"].notna()].copy()
        df["slide_label"] = df["slide_label"].astype(int)
        after = len(df)
        print(f"\nDropped unresolved rows: {before - after}")
    else:
        # 保留 unresolved 时，已解析部分转 int，整体列保持 float/nullable 也没关系
        if df["slide_label"].notna().all():
            df["slide_label"] = df["slide_label"].astype(int)

    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved to: {args.output_csv}")


if __name__ == "__main__":
    main()