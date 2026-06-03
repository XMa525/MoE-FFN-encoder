#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)

    if "keep" not in df.columns:
        raise ValueError("No keep column found in manifest.")

    keep = df["keep"].astype(str).str.strip().str.lower()
    clean = df[keep.isin(["1", "y", "yes", "true", "keep"])].copy()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.out_csv, index=False)

    print(f"Kept {len(clean)} / {len(df)}")
    print(f"Saved to: {args.out_csv}")


if __name__ == "__main__":
    main()