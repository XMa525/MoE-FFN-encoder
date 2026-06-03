import os
import json
import argparse
from typing import Optional
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import pandas as pd


def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def infer_score_col(df: pd.DataFrame) -> str:
    """
    自动推断 patch score 列名
    本项目里优先使用 tumor-minus-other 类分数
    """
    candidates = [
        "tumor_minus_max_other",   # 你当前 candidate_csv 用的是这个
        "candidate_score",
        "score",
        "tumor_score",
        "tumor_evidence",
        "wsi_tumor_score",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Cannot infer score column. Available columns: {df.columns.tolist()}"
    )


def load_candidate_df(candidate_csv: str) -> pd.DataFrame:
    df = pd.read_csv(candidate_csv)
    required = [
        "slide_id",
        "coord_idx",
        "coord_x",
        "coord_y",
        "subclass_id",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"candidate_csv missing required columns: {missing}")

    if "svs_path" in df.columns:
        df["svs_path"] = df["svs_path"].map(canonicalize_path)

    return df


def load_pool_df(pool_csv: str) -> pd.DataFrame:
    df = pd.read_csv(pool_csv)
    required = [
        "slide_id",
        "coord_idx",
        "coord_x",
        "coord_y",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"pool_csv missing required columns: {missing}")

    if "svs_path" in df.columns:
        df["svs_path"] = df["svs_path"].map(canonicalize_path)

    return df


def build_neighbor_index(
    candidate_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    radius: float = 1024.0,
    max_neighbors: int = 8,
    min_neighbors: int = 1,
    subclass_id: Optional[int] = None,
    negative_only: bool = False,
    positive_only: bool = False,
    exclude_same_coord: bool = True,
):
    pool_score_col = infer_score_col(pool_df)
    cand_score_col = infer_score_col(candidate_df)

    rows = []
    num_skipped_no_slide = 0
    num_skipped_no_neighbors = 0

    grouped_pool = {sid: sdf.reset_index(drop=True) for sid, sdf in pool_df.groupby("slide_id")}

    work_df = candidate_df.copy()

    if subclass_id is not None:
        work_df = work_df[work_df["subclass_id"] == subclass_id].copy()

    if negative_only:
        if "slide_label" not in work_df.columns:
            raise ValueError("negative_only=True but candidate_csv has no slide_label")
        work_df = work_df[work_df["slide_label"] == 0].copy()

    if positive_only:
        if "slide_label" not in work_df.columns:
            raise ValueError("positive_only=True but candidate_csv has no slide_label")
        work_df = work_df[work_df["slide_label"] == 1].copy()

    work_df = work_df.reset_index(drop=True)

    for _, row in work_df.iterrows():
        slide_id = str(row["slide_id"])
        coord_idx = int(row["coord_idx"])
        cx = float(row["coord_x"])
        cy = float(row["coord_y"])
        subclass = int(row["subclass_id"])

        if slide_id not in grouped_pool:
            num_skipped_no_slide += 1
            continue

        sdf = grouped_pool[slide_id].copy()

        if exclude_same_coord and "coord_idx" in sdf.columns:
            sdf = sdf[sdf["coord_idx"] != coord_idx].copy()

        if len(sdf) == 0:
            num_skipped_no_neighbors += 1
            continue

        dx = sdf["coord_x"].astype(float).values - cx
        dy = sdf["coord_y"].astype(float).values - cy
        dist = np.sqrt(dx * dx + dy * dy)

        sdf = sdf.copy()
        sdf["dist"] = dist
        sdf = sdf[sdf["dist"] <= radius].copy()

        if len(sdf) == 0:
            num_skipped_no_neighbors += 1
            continue

        sdf = sdf.sort_values("dist", ascending=True).reset_index(drop=True)

        if max_neighbors > 0 and len(sdf) > max_neighbors:
            sdf = sdf.iloc[:max_neighbors].copy()

        if len(sdf) < min_neighbors:
            num_skipped_no_neighbors += 1
            continue

        nb_scores = sdf[pool_score_col].astype(float).values
        nb_coord_indices = sdf["coord_idx"].astype(int).tolist()

        row_out = {
            "slide_id": slide_id,
            "coord_idx": coord_idx,
            "coord_x": int(row["coord_x"]),
            "coord_y": int(row["coord_y"]),
            "subclass_id": subclass,
            "candidate_score": float(row[cand_score_col]),
            "num_neighbors": int(len(sdf)),
            "neighbor_coord_indices": ";".join(str(x) for x in nb_coord_indices),
            "neighbor_mean_score": float(nb_scores.mean()),
            "neighbor_std_score": float(nb_scores.std()) if len(nb_scores) > 1 else 0.0,
            "neighbor_max_score": float(nb_scores.max()),
            "neighbor_min_score": float(nb_scores.min()),
            "neighbor_top5_mean_score": float(np.sort(nb_scores)[-min(5, len(nb_scores)):].mean()),
            "neighbor_frac_score_gt_0": float((nb_scores > 0).mean()),
            "neighbor_frac_score_gt_005": float((nb_scores > 0.05).mean()),
            "neighbor_frac_score_gt_010": float((nb_scores > 0.10).mean()),
            "context_gap_mean_neighbor": float(float(row[cand_score_col]) - nb_scores.mean()),
            "context_gap_top5_neighbor": float(
                float(row[cand_score_col]) - np.sort(nb_scores)[-min(5, len(nb_scores)):].mean()
            ),
        }

        for col in ["slide_label", "svs_path", "patch_level", "patch_size"]:
            if col in row.index:
                row_out[col] = row[col]

        rows.append(row_out)

    out_df = pd.DataFrame(rows)

    meta = {
        "num_input_candidates": int(len(work_df)),
        "num_output_candidates": int(len(out_df)),
        "num_skipped_no_slide_match": int(num_skipped_no_slide),
        "num_skipped_no_neighbors": int(num_skipped_no_neighbors),
        "radius": float(radius),
        "max_neighbors": int(max_neighbors),
        "min_neighbors": int(min_neighbors),
        "subclass_id_filter": None if subclass_id is None else int(subclass_id),
        "negative_only": bool(negative_only),
        "positive_only": bool(positive_only),
        "pool_score_col": pool_score_col,
        "candidate_score_col": cand_score_col,
    }
    return out_df, meta


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    rows = []

    def add_group(name: str, sdf: pd.DataFrame):
        if len(sdf) == 0:
            return
        rows.append({
            "group": name,
            "num_candidates": int(len(sdf)),
            "mean_candidate_score": float(sdf["candidate_score"].mean()),
            "mean_neighbor_score": float(sdf["neighbor_mean_score"].mean()),
            "mean_neighbor_top5_score": float(sdf["neighbor_top5_mean_score"].mean()),
            "mean_context_gap": float(sdf["context_gap_mean_neighbor"].mean()),
            "mean_context_gap_top5": float(sdf["context_gap_top5_neighbor"].mean()),
            "mean_num_neighbors": float(sdf["num_neighbors"].mean()),
        })

    add_group("overall", df)

    if "slide_label" in df.columns:
        for label_val in sorted(df["slide_label"].dropna().unique().tolist()):
            add_group(f"slide_label_{int(label_val)}", df[df["slide_label"] == label_val])

    for sid in sorted(df["subclass_id"].dropna().unique().tolist()):
        add_group(f"subclass_{int(sid)}", df[df["subclass_id"] == sid])

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=str, required=True)
    parser.add_argument("--pool-csv", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)

    parser.add_argument("--radius", type=float, default=1024.0)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--min-neighbors", type=int, default=1)

    parser.add_argument("--subclass-id", type=int, default=None)
    parser.add_argument("--negative-only", action="store_true")
    parser.add_argument("--positive-only", action="store_true")
    parser.add_argument("--exclude-same-coord", action="store_true")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    candidate_df = load_candidate_df(args.candidate_csv)
    pool_df = load_pool_df(args.pool_csv)

    out_df, meta = build_neighbor_index(
        candidate_df=candidate_df,
        pool_df=pool_df,
        radius=args.radius,
        max_neighbors=args.max_neighbors,
        min_neighbors=args.min_neighbors,
        subclass_id=args.subclass_id,
        negative_only=args.negative_only,
        positive_only=args.positive_only,
        exclude_same_coord=args.exclude_same_coord,
    )

    out_df.to_csv(args.out_csv, index=False)

    summary_df = build_summary(out_df)
    summary_csv = args.out_csv.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    meta_json = args.out_csv.replace(".csv", "_meta.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[Saved] neighbor index csv: {args.out_csv}")
    print(f"[Saved] summary csv: {summary_csv}")
    print(f"[Saved] meta json: {meta_json}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()