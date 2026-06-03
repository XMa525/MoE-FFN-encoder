import os
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--superclass-csv", type=str, required=True)
    parser.add_argument("--features-npy", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--topk-per-class", type=int, default=0,
                        help="0 means keep all; otherwise keep top-K hardest per superclass")
    parser.add_argument("--score-col", type=str, default="tumor_minus_max_other")
    args = parser.parse_args()

    safe_mkdir(args.out_dir)

    df = pd.read_csv(args.superclass_csv)
    feats = np.load(args.features_npy)

    if len(df) != len(feats):
        raise ValueError(f"csv rows ({len(df)}) != features rows ({len(feats)})")

    required_cols = ["superclass_name", args.score_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary_rows: List[Dict] = []

    for sname in sorted(df["superclass_name"].unique().tolist()):
        sdf = df[df["superclass_name"] == sname].copy()

        if args.topk_per_class > 0 and len(sdf) > args.topk_per_class:
            sdf = sdf.sort_values(args.score_col, ascending=False).head(args.topk_per_class).copy()

        idx = sdf.index.to_numpy()
        sub_feats = feats[idx]

        feat_path = os.path.join(args.out_dir, f"{sname}_features.npy")
        csv_path = os.path.join(args.out_dir, f"{sname}_metadata.csv")
        np.save(feat_path, sub_feats)
        sdf.to_csv(csv_path, index=False)

        row = {
            "superclass_name": sname,
            "count": int(len(sdf)),
            "feature_dim": int(sub_feats.shape[1]),
            "mean_score": float(sdf[args.score_col].mean()),
            "std_score": float(sdf[args.score_col].std()),
            "feature_path": os.path.basename(feat_path),
            "metadata_path": os.path.basename(csv_path),
        }
        summary_rows.append(row)

        print("[Saved]", feat_path)
        print("[Saved]", csv_path)

    summary_df = pd.DataFrame(summary_rows).sort_values("count", ascending=False)
    summary_csv = os.path.join(args.out_dir, "repulsion_bank_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print("[Saved]", summary_csv)
    print(summary_df)

    config = {
        "topk_per_class": args.topk_per_class,
        "score_col": args.score_col,
        "classes": summary_df["superclass_name"].tolist(),
    }
    with open(os.path.join(args.out_dir, "repulsion_bank_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()