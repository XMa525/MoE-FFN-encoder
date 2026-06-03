from __future__ import annotations

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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return normalize(x, norm="l2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-npy",
        type=str,
        required=True,
        help="Path to gland_like_features.npy",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="Optional metadata csv aligned row-wise with feature-npy",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of gland-like subclasses",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    feats = np.load(args.feature_npy)  # [N, D]
    if feats.ndim != 2:
        raise ValueError(f"feature-npy must be 2D, got {feats.shape}")

    if args.l2_normalize:
        feats = l2_normalize(feats)

    meta: Optional[pd.DataFrame] = None
    if args.metadata_csv is not None:
        meta = pd.read_csv(args.metadata_csv)
        if len(meta) != len(feats):
            raise ValueError(
                f"metadata rows ({len(meta)}) != feature rows ({len(feats)})"
            )

    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        batch_size=args.batch_size,
        n_init="auto",
    )
    labels = kmeans.fit_predict(feats)
    centers = kmeans.cluster_centers_

    if args.l2_normalize:
        centers = l2_normalize(centers)

    np.save(os.path.join(args.out_dir, "gland_like_subclass_labels.npy"), labels)
    np.save(os.path.join(args.out_dir, "gland_like_subclass_centers.npy"), centers)

    df = pd.DataFrame({
        "row_idx": np.arange(len(feats), dtype=np.int64),
        "subclass_id": labels.astype(np.int64),
    })

    if meta is not None:
        df = pd.concat([df, meta.reset_index(drop=True)], axis=1)

    df.to_csv(os.path.join(args.out_dir, "gland_like_subclass_assignments.csv"), index=False)

    counts = pd.Series(labels).value_counts().sort_index()
    summary = {
        "num_samples": int(len(feats)),
        "feature_dim": int(feats.shape[1]),
        "n_clusters": int(args.n_clusters),
        "counts": {f"sub{i}": int(counts.get(i, 0)) for i in range(args.n_clusters)},
    }

    # silhouette 可能比较慢，大数据时只抽样
    try:
        if len(feats) > 20000:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(len(feats), size=20000, replace=False)
            sil = silhouette_score(feats[idx], labels[idx], metric="cosine")
        else:
            sil = silhouette_score(feats, labels, metric="cosine")
        summary["silhouette_cosine"] = float(sil)
    except Exception as e:
        summary["silhouette_cosine"] = None
        summary["silhouette_error"] = str(e)

    with open(os.path.join(args.out_dir, "gland_like_subclass_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[OK] saved to: {args.out_dir}")


if __name__ == "__main__":
    main()