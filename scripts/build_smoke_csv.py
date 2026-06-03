import pandas as pd
import numpy as np

in_csv = "../data/CAMELYON17/split/slides_split.csv"
out_csv = "../data/CAMELYON17/split/slides_split_proposal_v1.csv"

df = pd.read_csv(in_csv)
rng = np.random.default_rng(42)

targets = {
    "train": 100,
    "val": 40,
}

parts = []
for split, n_total in targets.items():
    sub = df[df["split"] == split].copy()
    if "label" not in sub.columns:
        raise ValueError("CSV must contain label column")

    # balanced by label
    groups = list(sub.groupby("label"))
    per_group = n_total // len(groups)
    rem = n_total % len(groups)

    picked = []
    for i, (lab, g) in enumerate(groups):
        take = min(len(g), per_group + (1 if i < rem else 0))
        picked.append(g.sample(n=take, random_state=42 + i))

    part = pd.concat(picked, axis=0).sample(frac=1, random_state=42)
    parts.append(part)

out = pd.concat(parts, axis=0).reset_index(drop=True)
out.to_csv(out_csv, index=False)

print(out["split"].value_counts())
print(pd.crosstab(out["split"], out["label"]))
print("saved:", out_csv)