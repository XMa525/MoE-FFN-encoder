from __future__ import annotations

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
import argparse
from typing import Optional, List, Dict

import numpy as np
import pandas as pd


def infer_score(df: pd.DataFrame) -> pd.Series:
    if "score" in df.columns:
        return df["score"].astype(float)
    if "delta_tumor_minus_max_other" in df.columns:
        return df["delta_tumor_minus_max_other"].astype(float)
    if "sim_tumor" in df.columns and "sim_other_max" in df.columns:
        return df["sim_tumor"].astype(float) - df["sim_other_max"].astype(float)
    raise ValueError(
        "Cannot infer score. Need one of: "
        "'score', 'delta_tumor_minus_max_other', or ('sim_tumor' and 'sim_other_max')."
    )


def ensure_required_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def compute_neighbor_stats_for_one(
    cand_row: pd.Series,
    pool_slide: pd.DataFrame,
    radius: Optional[float],
    k_nearest: Optional[int],
    exclude_self: bool = True,
) -> Dict:
    """
    cand_row: candidate row
    pool_slide: all pool patches from the same slide
    """
    cx = float(cand_row["coord_x"])
    cy = float(cand_row["coord_y"])

    px = pool_slide["coord_x"].to_numpy(dtype=np.float32)
    py = pool_slide["coord_y"].to_numpy(dtype=np.float32)
    ps = pool_slide["_score"].to_numpy(dtype=np.float32)

    dist2 = (px - cx) ** 2 + (py - cy) ** 2
    dist = np.sqrt(dist2)

    mask = np.ones(len(pool_slide), dtype=bool)

    if exclude_self:
        same_xy = (px == cx) & (py == cy)
        mask &= ~same_xy

    if radius is not None:
        mask &= (dist <= radius)

    idx = np.where(mask)[0]

    if k_nearest is not None:
        if len(idx) > 0:
            local_dist = dist[idx]
            order = np.argsort(local_dist)
            idx = idx[order[:k_nearest]]

    if len(idx) == 0:
        return {
            "num_neighbors": 0,
            "neighbor_mean_score": np.nan,
            "neighbor_std_score": np.nan,
            "neighbor_top5_mean_score": np.nan,
            "neighbor_max_score": np.nan,
            "neighbor_frac_score_gt_0": np.nan,
            "neighbor_frac_score_gt_005": np.nan,
            "neighbor_frac_score_gt_010": np.nan,
        }

    ns = ps[idx]
    ns_sorted = np.sort(ns)[::-1]
    top5 = ns_sorted[: min(5, len(ns_sorted))]

    return {
        "num_neighbors": int(len(idx)),
        "neighbor_mean_score": float(np.mean(ns)),
        "neighbor_std_score": float(np.std(ns)),
        "neighbor_top5_mean_score": float(np.mean(top5)),
        "neighbor_max_score": float(np.max(ns)),
        "neighbor_frac_score_gt_0": float(np.mean(ns > 0)),
        "neighbor_frac_score_gt_005": float(np.mean(ns > 0.05)),
        "neighbor_frac_score_gt_010": float(np.mean(ns > 0.10)),
    }


def summarize_group(df: pd.DataFrame, group_name: str):
    cols = [
        "candidate_score",
        "neighbor_mean_score",
        "neighbor_top5_mean_score",
        "neighbor_max_score",
        "neighbor_frac_score_gt_0",
        "neighbor_frac_score_gt_005",
        "neighbor_frac_score_gt_010",
        "context_gap_mean_neighbor",
        "context_gap_top5_neighbor",
        "num_neighbors",
    ]
    existing = [c for c in cols if c in df.columns]
    print(f"\n[{group_name}]")
    if len(df) == 0:
        print("No rows.")
        return
    print(df[existing].describe())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=str, required=True)
    parser.add_argument("--pool-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument(
        "--radius",
        type=float,
        default=1024.0,
        help="Neighbor search radius in the same coord unit as coord_x/coord_y. "
             "Set <=0 to disable radius filter.",
    )
    parser.add_argument(
        "--k-nearest",
        type=int,
        default=32,
        help="Keep only nearest-K neighbors after radius filter. Set <=0 to disable.",
    )

    parser.add_argument(
        "--subclass-id",
        type=int,
        default=None,
        help="Optional: only analyze one subclass_id from candidate csv.",
    )
    parser.add_argument(
        "--negative-only",
        action="store_true",
        help="Optional: only analyze candidate rows with slide_label == 0.",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Optional: only analyze candidate rows with slide_label == 1.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cand = pd.read_csv(args.candidate_csv)
    pool = pd.read_csv(args.pool_csv)

    ensure_required_cols(cand, ["slide_id", "slide_label", "coord_x", "coord_y"], "candidate-csv")
    ensure_required_cols(pool, ["slide_id", "slide_label", "coord_x", "coord_y"], "pool-csv")

    cand = cand.copy()
    pool = pool.copy()

    cand["_score"] = infer_score(cand)
    pool["_score"] = infer_score(pool)

    if args.subclass_id is not None:
        if "subclass_id" not in cand.columns:
            raise ValueError("candidate-csv has no subclass_id column, but --subclass-id was given.")
        cand = cand[cand["subclass_id"] == args.subclass_id].copy()

    if args.negative_only and args.positive_only:
        raise ValueError("Cannot set both --negative-only and --positive-only")

    if args.negative_only:
        cand = cand[cand["slide_label"] == 0].copy()
    if args.positive_only:
        cand = cand[cand["slide_label"] == 1].copy()

    cand = cand.reset_index(drop=True)
    pool = pool.reset_index(drop=True)

    radius = None if args.radius <= 0 else float(args.radius)
    k_nearest = None if args.k_nearest <= 0 else int(args.k_nearest)

    pool_by_slide = {sid: g.reset_index(drop=True) for sid, g in pool.groupby("slide_id")}

    rows = []
    skipped = 0

    for _, row in cand.iterrows():
        sid = row["slide_id"]
        if sid not in pool_by_slide:
            skipped += 1
            continue

        g = pool_by_slide[sid]

        nstats = compute_neighbor_stats_for_one(
            cand_row=row,
            pool_slide=g,
            radius=radius,
            k_nearest=k_nearest,
            exclude_self=True,
        )

        out = {
            "slide_id": row["slide_id"],
            "slide_label": int(row["slide_label"]),
            "coord_x": int(row["coord_x"]),
            "coord_y": int(row["coord_y"]),
            "candidate_score": float(row["_score"]),
        }

        for c in ["subclass_id", "svs_path", "coord_idx", "sim_tumor", "sim_other_max"]:
            if c in row.index:
                out[c] = row[c]

        out.update(nstats)

        # context gap: candidate 自己比邻域高多少
        if not np.isnan(out["neighbor_mean_score"]):
            out["context_gap_mean_neighbor"] = float(out["candidate_score"] - out["neighbor_mean_score"])
            out["context_gap_top5_neighbor"] = float(out["candidate_score"] - out["neighbor_top5_mean_score"])
        else:
            out["context_gap_mean_neighbor"] = np.nan
            out["context_gap_top5_neighbor"] = np.nan

        rows.append(out)

    result = pd.DataFrame(rows)
    result_path = os.path.join(args.out_dir, "gland_like_neighbor_context_per_patch.csv")
    result.to_csv(result_path, index=False)

    summary_rows = []

    # overall
    if len(result) > 0:
        overall = {
            "group": "overall",
            "num_candidates": int(len(result)),
            "mean_candidate_score": float(result["candidate_score"].mean()),
            "mean_neighbor_score": float(result["neighbor_mean_score"].mean()),
            "mean_neighbor_top5_score": float(result["neighbor_top5_mean_score"].mean()),
            "mean_context_gap": float(result["context_gap_mean_neighbor"].mean()),
            "mean_context_gap_top5": float(result["context_gap_top5_neighbor"].mean()),
            "mean_num_neighbors": float(result["num_neighbors"].mean()),
        }
        summary_rows.append(overall)

    # by slide label
    for y, g in result.groupby("slide_label"):
        row = {
            "group": f"slide_label_{int(y)}",
            "num_candidates": int(len(g)),
            "mean_candidate_score": float(g["candidate_score"].mean()),
            "mean_neighbor_score": float(g["neighbor_mean_score"].mean()),
            "mean_neighbor_top5_score": float(g["neighbor_top5_mean_score"].mean()),
            "mean_context_gap": float(g["context_gap_mean_neighbor"].mean()),
            "mean_context_gap_top5": float(g["context_gap_top5_neighbor"].mean()),
            "mean_num_neighbors": float(g["num_neighbors"].mean()),
        }
        summary_rows.append(row)

    # by subclass
    if "subclass_id" in result.columns:
        for sid, g in result.groupby("subclass_id"):
            row = {
                "group": f"subclass_{int(sid)}",
                "num_candidates": int(len(g)),
                "mean_candidate_score": float(g["candidate_score"].mean()),
                "mean_neighbor_score": float(g["neighbor_mean_score"].mean()),
                "mean_neighbor_top5_score": float(g["neighbor_top5_mean_score"].mean()),
                "mean_context_gap": float(g["context_gap_mean_neighbor"].mean()),
                "mean_context_gap_top5": float(g["context_gap_top5_neighbor"].mean()),
                "mean_num_neighbors": float(g["num_neighbors"].mean()),
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "gland_like_neighbor_context_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # 导出两类最值得检查的 patch
    # 1) candidate 自己高，但邻域很低 => 可能是假阳性 shortcut
    if len(result) > 0:
        suspicious = result.sort_values(
            ["context_gap_mean_neighbor", "candidate_score"],
            ascending=[False, False]
        )
        suspicious.head(200).to_csv(
            os.path.join(args.out_dir, "gland_like_context_unsupported_top.csv"),
            index=False
        )

        supported = result.sort_values(
            ["context_gap_mean_neighbor", "candidate_score"],
            ascending=[True, False]
        )
        supported.head(200).to_csv(
            os.path.join(args.out_dir, "gland_like_context_supported_top.csv"),
            index=False
        )

    meta = {
        "candidate_csv": args.candidate_csv,
        "pool_csv": args.pool_csv,
        "num_input_candidates": int(len(cand)),
        "num_output_candidates": int(len(result)),
        "num_skipped_no_slide_match": int(skipped),
        "radius": radius,
        "k_nearest": k_nearest,
        "subclass_id_filter": args.subclass_id,
        "negative_only": bool(args.negative_only),
        "positive_only": bool(args.positive_only),
    }
    with open(os.path.join(args.out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[OK] saved:", result_path)
    print("[OK] saved:", summary_path)
    print("[Meta]")
    print(json.dumps(meta, indent=2))

    if len(result) > 0:
        summarize_group(result, "overall")
        for y, g in result.groupby("slide_label"):
            summarize_group(g, f"slide_label_{int(y)}")
        if "subclass_id" in result.columns:
            for sid, g in result.groupby("subclass_id"):
                summarize_group(g, f"subclass_{int(sid)}")


if __name__ == "__main__":
    main()