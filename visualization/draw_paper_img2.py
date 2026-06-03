#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
per_slide_dir = Path(
    "analysis_outputs/parotid_dino_vs_dino_moe_visual_v1/"
    "abmil_attention_compare_v1/per_slide"
)

save_dir = Path("outputs/abmil_attention_figures")
save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / "ABMIL_attention_rescue_suppression.png"


# =========================
# Cases
# =========================
cases = [
    {
        "case_id": "F23-00599A01_H01",
        "panel": "(A)",
        "title": "Positive rescue case",
        "label": "y=1",
        "prob_text": "p: 0.420 \u2192 0.652",
    },
    {
        "case_id": "F24-00820A01_H01",
        "panel": "(B)",
        "title": "False-positive suppression case",
        "label": "y=0",
        "prob_text": "p: 0.802 \u2192 0.269",
    },
]


# =========================
# Style
# =========================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.linewidth"] = 0.8


# =========================
# Helpers
# =========================
def find_xy_columns(df):
    """
    Try to infer x/y coordinate columns from common names.
    """
    candidates = [
        ("x", "y"),
        ("coord_x", "coord_y"),
        ("slide_x", "slide_y"),
        ("patch_x", "patch_y"),
        ("tsne_x", "tsne_y"),
        ("umap_x", "umap_y"),
        ("X", "Y"),
    ]

    for x_col, y_col in candidates:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col

    # fallback: use first two numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]

    raise ValueError(f"Cannot infer x/y columns. Available columns: {df.columns.tolist()}")


def find_attention_column(df):
    """
    Try to infer attention score column from common names.
    """
    candidates = [
        "attention",
        "attn",
        "attention_score",
        "attn_score",
        "score",
        "weight",
        "alpha",
        "prob",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove likely coordinate columns.
    xy_cols = set()
    try:
        x_col, y_col = find_xy_columns(df)
        xy_cols.update([x_col, y_col])
    except Exception:
        pass

    remain = [c for c in numeric_cols if c not in xy_cols]
    if len(remain) >= 1:
        return remain[-1]

    raise ValueError(f"Cannot infer attention column. Available columns: {df.columns.tolist()}")


def load_attention_csv(path):
    df = pd.read_csv(path)

    required_cols = ["coord_x", "coord_y", "attention"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}. Available columns: {df.columns.tolist()}")

    out = df[["coord_x", "coord_y", "attention"]].copy()
    out.columns = ["x", "y", "attn"]
    out = out.dropna()

    # Normalize attention into [0, 1] for consistent visualization.
    attn = out["attn"].to_numpy(dtype=float)
    lo, hi = np.nanpercentile(attn, [1, 99])
    attn_norm = (attn - lo) / (hi - lo + 1e-8)
    out["attn_norm"] = np.clip(attn_norm, 0, 1)

    return out


def get_case_paths(case_id):
    frozen_path = per_slide_dir / f"{case_id}_frozen_attention_points.csv"
    adapted_path = per_slide_dir / f"{case_id}_adapted_attention_points.csv"

    if not frozen_path.exists():
        raise FileNotFoundError(f"Frozen CSV not found: {frozen_path}")
    if not adapted_path.exists():
        raise FileNotFoundError(f"Adapted CSV not found: {adapted_path}")

    return frozen_path, adapted_path


def set_same_limits(ax_list, dfs, pad_ratio=0.05):
    all_x = np.concatenate([df["x"].to_numpy() for df in dfs])
    all_y = np.concatenate([df["y"].to_numpy() for df in dfs])

    xmin, xmax = np.nanmin(all_x), np.nanmax(all_x)
    ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)

    xpad = (xmax - xmin) * pad_ratio
    ypad = (ymax - ymin) * pad_ratio

    for ax in ax_list:
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymax + ypad, ymin - ypad)  # invert y for slide-like coordinates
        ax.set_aspect("equal", adjustable="box")


def plot_attention(ax, df, title):
    # Base layer: all patches
    ax.scatter(
        df["x"],
        df["y"],
        s=7,
        c="#D9D5CA",
        alpha=0.62,
        linewidths=0,
        rasterized=True,
    )

    # High-attention layer
    strong = df["attn_norm"] >= 0.12
    ax.scatter(
        df.loc[strong, "x"],
        df.loc[strong, "y"],
        s=8 + 45 * df.loc[strong, "attn_norm"],
        c=df.loc[strong, "attn_norm"],
        cmap="copper_r",
        vmin=0,
        vmax=1,
        alpha=0.78,
        linewidths=0,
        rasterized=True,
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.8)

# =========================
# Main
# =========================
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(9.6, 6.4),
    dpi=300,
)

for row_idx, case in enumerate(cases):
    frozen_path, adapted_path = get_case_paths(case["case_id"])

    frozen_df = load_attention_csv(frozen_path)
    adapted_df = load_attention_csv(adapted_path)

    ax_frozen = axes[row_idx, 0]
    ax_adapted = axes[row_idx, 1]

    plot_attention(ax_frozen, frozen_df, "w/ frozen encoder")
    plot_attention(ax_adapted, adapted_df, "w/ MoE-adapted encoder")

    set_same_limits(
        [ax_frozen, ax_adapted],
        [frozen_df, adapted_df],
        pad_ratio=0.04,
    )

    # Row title
    row_title = (
        f"{case['panel']} {case['title']} | "
        f"{case['label']}, {case['prob_text']}"
    )
    fig.text(
        0.035,
        0.965 - row_idx * 0.49,
        row_title,
        ha="left",
        va="center",
        fontsize=13,
        fontweight="bold",
    )

    # Shared row subtitle
    fig.text(
        0.50,
        0.922 - row_idx * 0.49,
        "ABMIL attention map",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

plt.subplots_adjust(
    left=0.05,
    right=0.98,
    top=0.88,
    bottom=0.06,
    wspace=0.06,
    hspace=0.42,
)

plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"[INFO] Saved figure to: {save_path}")