from __future__ import annotations

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import argparse
import numpy as np
from sklearn.preprocessing import normalize


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return normalize(x, norm="l2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-npy",
        type=str,
        required=True,
        help="original gland_like_features.npy",
    )
    parser.add_argument(
        "--label-npy",
        type=str,
        required=True,
        help="gland_like_subclass_labels.npy",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="bank export dir for online HN",
    )
    parser.add_argument(
        "--subclass-ids",
        type=int,
        nargs="+",
        required=True,
        help="selected gland-like subclass ids to export",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="gland_like_sub",
        help='export name prefix, e.g. "gland_like_sub"',
    )
    parser.add_argument(
        "--l2-normalize",
        action="store_true",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    feats = np.load(args.feature_npy)
    labels = np.load(args.label_npy)

    if len(feats) != len(labels):
        raise ValueError(f"feature rows ({len(feats)}) != label rows ({len(labels)})")

    for sid in args.subclass_ids:
        mask = labels == sid
        sub_feats = feats[mask]
        if len(sub_feats) == 0:
            print(f"[WARN] subclass {sid} has 0 samples, skip")
            continue

        if args.l2_normalize:
            sub_feats = l2_normalize(sub_feats)

        out_path = os.path.join(args.out_dir, f"{args.prefix}{sid}_features.npy")
        np.save(out_path, sub_feats)
        print(f"[OK] subclass {sid}: {sub_feats.shape} -> {out_path}")


if __name__ == "__main__":
    main()