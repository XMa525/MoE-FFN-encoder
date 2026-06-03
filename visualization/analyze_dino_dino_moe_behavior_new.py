#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Utils
# =========================================================
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def entropy_np(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Entropy over the last dimension.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, eps, None)
    return x / np.clip(x.sum(axis=1, keepdims=True), eps, None)


def normalize_cols(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, eps, None)
    return x / np.clip(x.sum(axis=0, keepdims=True), eps, None)


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8, base: float = 2.0) -> float:
    """
    Jensen-Shannon divergence.

    Returns divergence, not distance.
    Range is [0, 1] when base=2.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    p = p / np.clip(p.sum(), eps, None)
    q = q / np.clip(q.sum(), eps, None)

    m = 0.5 * (p + q)

    if base == 2.0:
        log_fn = np.log2
    else:
        log_fn = np.log

    kl_pm = np.sum(p * (log_fn(p) - log_fn(m)))
    kl_qm = np.sum(q * (log_fn(q) - log_fn(m)))

    return float(0.5 * (kl_pm + kl_qm))


def pairwise_jsd_matrix(P: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    P: [N, K], each row is a probability distribution.
    """
    P = normalize_rows(P, eps=eps)
    n = P.shape[0]
    mat = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            mat[i, j] = js_divergence(P[i], P[j], eps=eps, base=2.0)

    return mat


# =========================================================
# Expert-cluster tables
# =========================================================
def infer_expert_prob_cols(token_df: pd.DataFrame) -> List[str]:
    cols = [
        c for c in token_df.columns
        if c.startswith("expert_prob_") and c.replace("expert_prob_", "").isdigit()
    ]
    cols = sorted(cols, key=lambda x: int(x.replace("expert_prob_", "")))
    return cols


def expert_cluster_tables(
    token_df: pd.DataFrame,
    cluster_col: str,
    expert_prob_cols: List[str],
    n_clusters: int,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """
    Build soft and hard expert-cluster statistics.

    soft_mass[e, c] = sum token routing probability to expert e among tokens in cluster c.
    hard_counts[e, c] = number of tokens whose hard expert is e among tokens in cluster c.
    """
    if cluster_col not in token_df.columns:
        raise KeyError(f"Missing cluster column: {cluster_col}")

    for c in expert_prob_cols:
        if c not in token_df.columns:
            raise KeyError(f"Missing expert probability column: {c}")

    expert_probs = token_df[expert_prob_cols].values.astype(np.float64)
    expert_probs = np.clip(expert_probs, 0.0, None)
    expert_probs = expert_probs / np.clip(expert_probs.sum(axis=1, keepdims=True), eps, None)

    clusters = token_df[cluster_col].values.astype(np.int64)

    n_experts = len(expert_prob_cols)
    expert_ids = np.argmax(expert_probs, axis=1)

    soft_mass = np.zeros((n_experts, n_clusters), dtype=np.float64)
    hard_counts = np.zeros((n_experts, n_clusters), dtype=np.float64)
    cluster_counts = np.zeros((n_clusters,), dtype=np.float64)

    for c in range(n_clusters):
        m = clusters == c
        cluster_counts[c] = float(m.sum())

        if m.sum() == 0:
            continue

        soft_mass[:, c] = expert_probs[m].sum(axis=0)

        for e in range(n_experts):
            hard_counts[e, c] = float(np.sum(expert_ids[m] == e))

    soft_p_expert_given_cluster = normalize_cols(soft_mass, eps=eps)
    soft_p_cluster_given_expert = normalize_rows(soft_mass, eps=eps)

    hard_p_expert_given_cluster = normalize_cols(hard_counts, eps=eps)
    hard_p_cluster_given_expert = normalize_rows(hard_counts, eps=eps)

    return {
        "soft_mass": soft_mass.astype(np.float32),
        "hard_counts": hard_counts.astype(np.float32),
        "cluster_counts": cluster_counts.astype(np.float32),
        "soft_p_expert_given_cluster": soft_p_expert_given_cluster.astype(np.float32),
        "soft_p_cluster_given_expert": soft_p_cluster_given_expert.astype(np.float32),
        "hard_p_expert_given_cluster": hard_p_expert_given_cluster.astype(np.float32),
        "hard_p_cluster_given_expert": hard_p_cluster_given_expert.astype(np.float32),
    }


# =========================================================
# Plot helpers
# =========================================================
def save_heatmap(
    mat: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    xticklabels: List[str],
    yticklabels: List[str],
    cmap: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
) -> None:
    ensure_dir(out_path.parent)

    fig_w = max(6.0, 0.60 * len(xticklabels))
    fig_h = max(4.5, 0.55 * len(yticklabels))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(yticklabels)))
    ax.set_yticklabels(yticklabels)

    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def save_entropy_bar(
    entropy_norm: np.ndarray,
    effective_clusters: np.ndarray,
    dominant_cluster_prob: np.ndarray,
    out_path: Path,
    prefix: str,
) -> None:
    ensure_dir(out_path.parent)

    n_experts = len(entropy_norm)
    x = np.arange(n_experts)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, entropy_norm)

    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i}" for i in range(n_experts)])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Normalized cluster entropy")
    ax.set_title(f"{prefix}: expert cluster entropy")

    for i, v in enumerate(entropy_norm):
        ax.text(
            i,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)

    # Additional effective-cluster bar
    eff_path = out_path.with_name(out_path.stem.replace("entropy_bar", "effective_clusters_bar") + out_path.suffix)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, effective_clusters)

    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i}" for i in range(n_experts)])
    ax.set_ylabel("Effective number of clusters")
    ax.set_title(f"{prefix}: effective clusters per expert")

    ymax = max(1.0, float(np.max(effective_clusters)) * 1.2)
    ax.set_ylim(0.0, ymax)

    for i, v in enumerate(effective_clusters):
        ax.text(
            i,
            v,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(eff_path, dpi=240)
    plt.close(fig)

    # Dominant cluster probability bar
    dom_path = out_path.with_name(out_path.stem.replace("entropy_bar", "dominant_cluster_prob_bar") + out_path.suffix)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, dominant_cluster_prob)

    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i}" for i in range(n_experts)])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Dominant cluster probability")
    ax.set_title(f"{prefix}: dominant cluster strength")

    for i, v in enumerate(dominant_cluster_prob):
        ax.text(
            i,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(dom_path, dpi=240)
    plt.close(fig)


# =========================================================
# Main stats function
# =========================================================
def compute_and_save_expert_cluster_stats(
    token_df: pd.DataFrame,
    out_dir: Path,
    cluster_col: str,
    prefix: str,
    expert_prob_cols: List[str],
    n_clusters: int | None = None,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    ensure_dir(out_dir)

    if cluster_col not in token_df.columns:
        raise KeyError(f"Cluster column not found: {cluster_col}")

    if n_clusters is None:
        n_clusters = int(token_df[cluster_col].max()) + 1

    n_experts = len(expert_prob_cols)

    tables = expert_cluster_tables(
        token_df=token_df,
        cluster_col=cluster_col,
        expert_prob_cols=expert_prob_cols,
        n_clusters=n_clusters,
        eps=eps,
    )

    p_cluster_given_expert = tables["soft_p_cluster_given_expert"].astype(np.float64)
    p_expert_given_cluster = tables["soft_p_expert_given_cluster"].astype(np.float64)
    hard_p_cluster_given_expert = tables["hard_p_cluster_given_expert"].astype(np.float64)
    hard_p_expert_given_cluster = tables["hard_p_expert_given_cluster"].astype(np.float64)
    hard_counts = tables["hard_counts"].astype(np.float64)
    soft_mass = tables["soft_mass"].astype(np.float64)
    cluster_counts = tables["cluster_counts"].astype(np.float64)

    p_cluster_given_expert = normalize_rows(p_cluster_given_expert, eps=eps)

    expert_entropy = entropy_np(p_cluster_given_expert, eps=eps).astype(np.float32)
    expert_entropy_norm = expert_entropy / float(np.log(max(2, n_clusters)))

    effective_clusters = (1.0 / np.sum(p_cluster_given_expert ** 2, axis=1)).astype(np.float32)

    dominant_cluster = np.argmax(p_cluster_given_expert, axis=1).astype(np.int64)
    dominant_cluster_prob = np.max(p_cluster_given_expert, axis=1).astype(np.float32)

    jsd_mat = pairwise_jsd_matrix(p_cluster_given_expert, eps=eps)

    upper = jsd_mat[np.triu_indices(n_experts, k=1)]
    mean_pairwise_jsd = float(np.mean(upper)) if len(upper) > 0 else 0.0
    median_pairwise_jsd = float(np.median(upper)) if len(upper) > 0 else 0.0
    min_pairwise_jsd = float(np.min(upper)) if len(upper) > 0 else 0.0
    max_pairwise_jsd = float(np.max(upper)) if len(upper) > 0 else 0.0

    expert_names = [f"E{i}" for i in range(n_experts)]
    cluster_names = [f"c{i}" for i in range(n_clusters)]

    # Save csv matrices
    pd.DataFrame(
        p_cluster_given_expert,
        index=expert_names,
        columns=cluster_names,
    ).to_csv(out_dir / f"{prefix}_p_cluster_given_expert.csv")

    pd.DataFrame(
        p_expert_given_cluster,
        index=expert_names,
        columns=cluster_names,
    ).to_csv(out_dir / f"{prefix}_p_expert_given_cluster.csv")

    pd.DataFrame(
        hard_p_cluster_given_expert,
        index=expert_names,
        columns=cluster_names,
    ).to_csv(out_dir / f"{prefix}_hard_p_cluster_given_expert.csv")

    pd.DataFrame(
        hard_p_expert_given_cluster,
        index=expert_names,
        columns=cluster_names,
    ).to_csv(out_dir / f"{prefix}_hard_p_expert_given_cluster.csv")

    pd.DataFrame(
        soft_mass,
        index=expert_names,
        columns=cluster_names,
    ).to_csv(out_dir / f"{prefix}_soft_mass.csv")

    pd.DataFrame(
        hard_counts,
        index=expert_names,
        columns=cluster_names,
    ).to_csv(out_dir / f"{prefix}_hard_counts.csv")

    pd.DataFrame({
        "cluster": cluster_names,
        "count": cluster_counts.astype(np.int64),
    }).to_csv(out_dir / f"{prefix}_cluster_counts.csv", index=False)

    pd.DataFrame(
        jsd_mat,
        index=expert_names,
        columns=expert_names,
    ).to_csv(out_dir / f"{prefix}_expert_cluster_jsd_matrix.csv")

    entropy_df = pd.DataFrame({
        "expert": expert_names,
        "cluster_entropy": expert_entropy,
        "cluster_entropy_norm": expert_entropy_norm,
        "effective_clusters": effective_clusters,
        "dominant_cluster": [f"c{i}" for i in dominant_cluster],
        "dominant_cluster_prob": dominant_cluster_prob,
    })
    entropy_df.to_csv(out_dir / f"{prefix}_expert_cluster_entropy.csv", index=False)

    # Save plots
    save_heatmap(
        p_cluster_given_expert,
        out_dir / f"{prefix}_p_cluster_given_expert.png",
        title=f"{prefix}: P(cluster | expert)",
        xlabel="token cluster",
        ylabel="expert",
        xticklabels=cluster_names,
        yticklabels=expert_names,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        annotate=True,
    )

    save_heatmap(
        p_expert_given_cluster,
        out_dir / f"{prefix}_p_expert_given_cluster.png",
        title=f"{prefix}: P(expert | cluster)",
        xlabel="token cluster",
        ylabel="expert",
        xticklabels=cluster_names,
        yticklabels=expert_names,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        annotate=True,
    )

    save_heatmap(
        jsd_mat,
        out_dir / f"{prefix}_expert_cluster_jsd_matrix.png",
        title=f"{prefix}: pairwise JSD between experts based on P(cluster | expert)",
        xlabel="expert",
        ylabel="expert",
        xticklabels=expert_names,
        yticklabels=expert_names,
        cmap="Oranges",
        vmin=0.0,
        vmax=max(1e-6, float(jsd_mat.max())),
        annotate=True,
    )

    save_entropy_bar(
        entropy_norm=expert_entropy_norm,
        effective_clusters=effective_clusters,
        dominant_cluster_prob=dominant_cluster_prob,
        out_path=out_dir / f"{prefix}_expert_cluster_entropy_bar.png",
        prefix=prefix,
    )

    summary: Dict[str, Any] = {
        "prefix": prefix,
        "cluster_col": cluster_col,
        "n_tokens": int(len(token_df)),
        "n_experts": int(n_experts),
        "n_clusters": int(n_clusters),

        "mean_expert_cluster_entropy": float(np.mean(expert_entropy)),
        "mean_expert_cluster_entropy_norm": float(np.mean(expert_entropy_norm)),
        "mean_effective_clusters": float(np.mean(effective_clusters)),

        "mean_pairwise_expert_jsd": mean_pairwise_jsd,
        "median_pairwise_expert_jsd": median_pairwise_jsd,
        "min_pairwise_expert_jsd": min_pairwise_jsd,
        "max_pairwise_expert_jsd": max_pairwise_jsd,

        "expert_entropy": {
            f"E{i}": float(expert_entropy[i]) for i in range(n_experts)
        },
        "expert_entropy_norm": {
            f"E{i}": float(expert_entropy_norm[i]) for i in range(n_experts)
        },
        "effective_clusters": {
            f"E{i}": float(effective_clusters[i]) for i in range(n_experts)
        },
        "dominant_cluster": {
            f"E{i}": f"c{int(dominant_cluster[i])}" for i in range(n_experts)
        },
        "dominant_cluster_prob": {
            f"E{i}": float(dominant_cluster_prob[i]) for i in range(n_experts)
        },
    }

    with open(out_dir / f"{prefix}_expert_cluster_stats_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        "Compute expert-cluster entropy and pairwise JSD from token_level_moe_analysis.csv"
    )

    parser.add_argument(
        "--token_csv",
        type=str,
        required=True,
        help="Path to token_level_moe_analysis.csv generated by analyze_moe_backbone_token_behavior.py",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for expert-cluster stats.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["last_moe", "final"],
        choices=["last_moe", "final", "frozen_last", "frozen_final"],
        help="Which cluster columns to analyze. For example: last_moe final",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Optional number of clusters. If omitted, inferred from max cluster id + 1 for each layer.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
    )

    args = parser.parse_args()

    token_csv = Path(args.token_csv)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if not token_csv.exists():
        raise FileNotFoundError(f"token_csv not found: {token_csv}")

    print(f"[Load] {token_csv}")
    token_df = pd.read_csv(token_csv)

    expert_prob_cols = infer_expert_prob_cols(token_df)
    if len(expert_prob_cols) == 0:
        raise RuntimeError(
            "No expert_prob_{e} columns found. "
            "Expected columns like expert_prob_0, expert_prob_1, ..."
        )

    print(f"[Info] found expert probability columns: {expert_prob_cols}")

    all_summary: Dict[str, Any] = {}

    for layer in args.layers:
        cluster_col = f"{layer}_cluster"

        if cluster_col not in token_df.columns:
            print(f"[WARN] skip layer={layer}: missing column {cluster_col}")
            continue

        if args.n_clusters is None:
            n_clusters = int(token_df[cluster_col].max()) + 1
        else:
            n_clusters = int(args.n_clusters)

        print(
            f"[Analyze] layer={layer}, cluster_col={cluster_col}, "
            f"n_clusters={n_clusters}, n_experts={len(expert_prob_cols)}"
        )

        layer_out_dir = out_dir
        summary = compute_and_save_expert_cluster_stats(
            token_df=token_df,
            out_dir=layer_out_dir,
            cluster_col=cluster_col,
            prefix=layer,
            expert_prob_cols=expert_prob_cols,
            n_clusters=n_clusters,
            eps=args.eps,
        )

        all_summary[layer] = summary

    with open(out_dir / "expert_cluster_stats_all_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print(f"[Done] saved expert-cluster stats to: {out_dir}")


if __name__ == "__main__":
    main()