#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split the noise cluster (-1) into subclusters.

Usage example:
python visualization/split_noise_cluster.py \
  --csv analysis_outputs/token_meta.csv \
  --feature_npy analysis_outputs/features_moe.npy \
  --cluster_col cluster_id_moe \
  --outdir analysis_outputs/noise_split_moe \
  --expert_id all \
  --umap_neighbors 30 \
  --min_cluster_size 80
"""

import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_umap(features, n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    emb = reducer.fit_transform(features)
    return emb


def run_hdbscan(embedding, min_cluster_size=80, min_samples=None):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embedding)
    return labels


def plot_umap_by_label(embedding, labels, save_path, title="UMAP by subcluster"):
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="tab20",
        s=6
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_umap_by_expert(embedding, expert_ids, save_path, title="UMAP by expert"):
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=expert_ids,
        cmap="tab10",
        s=6
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def build_subcluster_expert_table(df, subcluster_col="subcluster_id"):
    table = pd.crosstab(df[subcluster_col], df["expert_id"], normalize="index")
    return table


def plot_heatmap(table, save_path, title="Subcluster-Expert Heatmap"):
    plt.figure(figsize=(8, 6))
    plt.imshow(table.values, aspect="auto", cmap="YlGnBu")
    plt.colorbar(label="Proportion within subcluster")
    plt.xticks(range(len(table.columns)), [f"Expert {c}" for c in table.columns], rotation=45)
    plt.yticks(range(len(table.index)), [f"{idx}" for idx in table.index])
    plt.xlabel("Expert")
    plt.ylabel("Noise Subcluster")
    plt.title(title)

    # annotate values
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = table.values[i, j]
            if val > 0:
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def summarize_subclusters(df, subcluster_col="subcluster_id"):
    rows = []
    for sid, sub in df.groupby(subcluster_col):
        exp_counts = sub["expert_id"].value_counts(normalize=True)
        dominant_expert = int(exp_counts.idxmax())
        dominant_ratio = float(exp_counts.max())

        rows.append({
            "subcluster_id": int(sid),
            "token_count": int(len(sub)),
            "unique_patch_count": int(sub["patch_path"].nunique()),
            "unique_sample_count": int(sub["sample_category"].nunique()),
            "dominant_expert": dominant_expert,
            "dominant_ratio": dominant_ratio,
        })

    summary = pd.DataFrame(rows).sort_values("token_count", ascending=False)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--feature_npy", type=str, required=True)
    parser.add_argument("--cluster_col", type=str, default="cluster_id_moe",
                        choices=["cluster_id_moe", "cluster_id_final"])
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--expert_id", type=str, default="all",
                        help='Use "all" or a specific expert id, e.g. 0')
    parser.add_argument("--umap_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--min_cluster_size", type=int, default=80)
    parser.add_argument("--min_samples", type=int, default=None)
    parser.add_argument("--standardize", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # load
    df = pd.read_csv(args.csv)
    feats = np.load(args.feature_npy)

    assert len(df) == len(feats), f"Length mismatch: csv={len(df)}, feats={len(feats)}"

    # select noise
    df_noise = df[df[args.cluster_col] == -1].copy()

    if args.expert_id != "all":
        expert_id = int(args.expert_id)
        df_noise = df_noise[df_noise["expert_id"] == expert_id].copy()
        tag = f"noise_only_expert{expert_id}"
    else:
        tag = "noise_all_experts"

    if len(df_noise) == 0:
        raise ValueError("No noise tokens found under this setting.")

    noise_idx = df_noise.index.to_numpy()
    noise_feats = feats[noise_idx]

    print(f"[INFO] Selected noise tokens: {len(df_noise)}")
    print(f"[INFO] Unique patches: {df_noise['patch_path'].nunique()}")
    print(f"[INFO] Expert distribution in selected noise:")
    print(df_noise["expert_id"].value_counts(normalize=True).sort_index())

    if args.standardize:
        noise_feats = StandardScaler().fit_transform(noise_feats)

    # run umap + hdbscan
    emb = run_umap(
        noise_feats,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )

    sub_labels = run_hdbscan(
        emb,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )

    df_noise["subcluster_id"] = sub_labels

    # save filtered token table
    df_noise.to_csv(os.path.join(args.outdir, f"{tag}_token_meta.csv"), index=False)

    # summaries
    summary = summarize_subclusters(df_noise, subcluster_col="subcluster_id")
    summary.to_csv(os.path.join(args.outdir, f"{tag}_summary.csv"), index=False)

    expert_table = build_subcluster_expert_table(df_noise, subcluster_col="subcluster_id")
    expert_table.to_csv(os.path.join(args.outdir, f"{tag}_subcluster_expert_table.csv"))

    # plots
    plot_umap_by_label(
        emb,
        sub_labels,
        save_path=os.path.join(args.outdir, f"{tag}_umap_by_subcluster.png"),
        title=f"Noise split UMAP by subcluster ({tag})"
    )

    plot_umap_by_expert(
        emb,
        df_noise["expert_id"].to_numpy(),
        save_path=os.path.join(args.outdir, f"{tag}_umap_by_expert.png"),
        title=f"Noise split UMAP by expert ({tag})"
    )

    plot_heatmap(
        expert_table,
        save_path=os.path.join(args.outdir, f"{tag}_subcluster_expert_heatmap.png"),
        title=f"Noise Subcluster × Expert Heatmap ({tag})"
    )

    # text summary
    with open(os.path.join(args.outdir, f"{tag}_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"csv = {args.csv}\n")
        f.write(f"feature_npy = {args.feature_npy}\n")
        f.write(f"cluster_col = {args.cluster_col}\n")
        f.write(f"expert_id = {args.expert_id}\n")
        f.write(f"selected_noise_tokens = {len(df_noise)}\n")
        f.write(f"unique_patches = {df_noise['patch_path'].nunique()}\n\n")
        f.write("[Expert distribution inside selected noise]\n")
        exp_dist = df_noise["expert_id"].value_counts(normalize=True).sort_index()
        for eid, ratio in exp_dist.items():
            f.write(f"expert {eid}: {ratio:.4f}\n")

        f.write("\n[Subcluster summary]\n")
        for _, row in summary.iterrows():
            f.write(
                f"subcluster={int(row['subcluster_id'])}, "
                f"tokens={int(row['token_count'])}, "
                f"patches={int(row['unique_patch_count'])}, "
                f"samples={int(row['unique_sample_count'])}, "
                f"dominant_expert={int(row['dominant_expert'])}, "
                f"dominant_ratio={row['dominant_ratio']:.4f}\n"
            )

    print(f"[DONE] Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()