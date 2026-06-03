#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap


# =========================
# Paths
# =========================
frozen_csv = Path("analysis_outputs/dino_moe_token_analysis_parotid_tsne/frozen_compare/frozen_last_tsne_plot_points.csv")
moe_csv = Path("analysis_outputs/dino_moe_token_analysis_parotid_tsne/figures/last_moe_tsne_plot_points.csv")

save_dir = Path("outputs/representation_figures")
save_dir.mkdir(parents=True, exist_ok=True)

tsne_save_path = save_dir / "tsne_last_moe_representation_morandi.png"
heatmap_save_path = save_dir / "expert_cluster_heatmap_morandi.png"


# =========================
# Style
# =========================
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.linewidth"] = 0.8


# Morandi / low-saturation palette for 12 clusters
cluster_colors = [
    "#86A6C8",  # muted blue
    "#D8A15D",  # muted orange
    "#89A57C",  # muted green
    "#C96F6F",  # muted red
    "#B89AAF",  # muted mauve
    "#AFC4D6",  # pale blue
    "#C8B98D",  # khaki
    "#8E83AE",  # muted purple
    "#D9A79C",  # dusty pink
    "#6F9A91",  # teal gray
    "#B8B5A6",  # warm gray
    "#A98C7A",  # taupe
]

# Expert colors, fewer and more stable
expert_colors = {
    0: "#5F86A8",  # blue gray
    1: "#D49A50",  # muted orange
    2: "#7E9B74",  # muted green
    3: "#B09AA8",  # muted purple gray
}


# =========================
# Helpers
# =========================
def set_equal_axis(ax, x, y, pad_ratio=0.06):
    """
    Keep t-SNE geometry from being horizontally or vertically stretched.
    This sets a square data range around the point cloud.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0

    half = max(xmax - xmin, ymax - ymin) / 2.0
    half = half * (1.0 + pad_ratio)

    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    ax.set_aspect("equal", adjustable="box")


# =========================
# Load data
# =========================
frozen_df = pd.read_csv(frozen_csv)
moe_df = pd.read_csv(moe_csv)

# Required columns:
# frozen_df: frozen_last_x, frozen_last_y, frozen_last_cluster
# moe_df: last_moe_x, last_moe_y, last_moe_cluster, expert_id

# Make labels c0, c1, ...
cluster_ids = sorted(moe_df["last_moe_cluster"].dropna().unique().astype(int).tolist())
cluster_to_label = {c: f"c{c}" for c in cluster_ids}
cluster_to_color = {
    c: cluster_colors[i % len(cluster_colors)]
    for i, c in enumerate(cluster_ids)
}


# =========================
# Plot t-SNE panels
# =========================
fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.0), dpi=300)

panel_cfgs = [
    {
        "ax": axes[0],
        "df": frozen_df,
        "x": "frozen_last_x",
        "y": "frozen_last_y",
        "color_col": "frozen_last_cluster",
        "title": "Frozen DINO high-layer token features (cluster)",
        "mode": "cluster",
    },
    {
        "ax": axes[1],
        "df": moe_df,
        "x": "last_moe_x",
        "y": "last_moe_y",
        "color_col": "last_moe_cluster",
        "title": "MoE last-layer token features (cluster)",
        "mode": "cluster",
    },
    {
        "ax": axes[2],
        "df": moe_df,
        "x": "last_moe_x",
        "y": "last_moe_y",
        "color_col": "expert_id",
        "title": "MoE last-layer token features (expert)",
        "mode": "expert",
    },
]

for cfg in panel_cfgs:
    ax = cfg["ax"]
    df = cfg["df"]

    if cfg["mode"] == "cluster":
        for c in cluster_ids:
            sub = df[df[cfg["color_col"]] == c]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub[cfg["x"]],
                sub[cfg["y"]],
                s=3.0,
                c=cluster_to_color[c],
                alpha=0.78,
                linewidths=0,
                rasterized=True,
            )

    elif cfg["mode"] == "expert":
        for e in sorted(expert_colors.keys()):
            sub = df[df[cfg["color_col"]] == e]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub[cfg["x"]],
                sub[cfg["y"]],
                s=3.0,
                c=expert_colors[e],
                alpha=0.78,
                linewidths=0,
                rasterized=True,
            )

    # Important: prevent t-SNE panels from being stretched
    set_equal_axis(ax, df[cfg["x"]], df[cfg["y"]], pad_ratio=0.06)

    ax.set_title(cfg["title"], fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#777777")
        spine.set_linewidth(0.8)


# =========================
# Compact legends inside each panel
# =========================

cluster_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markersize=4.0,
        markerfacecolor=cluster_to_color[c],
        markeredgecolor="none",
        label=f"c{c}",
    )
    for c in cluster_ids
]

# Only add cluster legend to the middle panel
leg = axes[1].legend(
    handles=cluster_handles,
    title="Cluster",
    loc="upper right",
    bbox_to_anchor=(0.985, 0.985),
    ncol=2,
    frameon=True,
    fancybox=False,
    framealpha=0.88,
    edgecolor="none",
    fontsize=7.2,
    title_fontsize=8.0,
    handletextpad=0.3,
    columnspacing=0.6,
    borderpad=0.35,
    labelspacing=0.25,
)
leg.get_frame().set_facecolor("white")

expert_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        markersize=4.5,
        markerfacecolor=expert_colors[e],
        markeredgecolor="none",
        label=f"E{e}",
    )
    for e in sorted(expert_colors.keys())
]

leg = axes[2].legend(
    handles=expert_handles,
    title="Expert",
    loc="upper right",
    bbox_to_anchor=(0.985, 0.985),
    ncol=1,
    frameon=True,
    fancybox=False,
    framealpha=0.88,
    edgecolor="none",
    fontsize=7.5,
    title_fontsize=8.2,
    handletextpad=0.3,
    borderpad=0.35,
    labelspacing=0.25,
)
leg.get_frame().set_facecolor("white")


plt.subplots_adjust(
    left=0.025,
    right=0.985,
    top=0.84,
    bottom=0.08,
    wspace=0.10,
)
plt.savefig(tsne_save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved t-SNE figure to: {tsne_save_path}")


# =========================
# Heatmap data
# =========================
heatmap_values = np.array([
    [0.9559342265129089, 0.09768553078174591, 0.0040761916898190975, 0.9991319179534912, 0.0030341341625899076, 0.5016064047813416, 0.8626657724380493, 0.998970627784729, 0.01836910843849182, 0.9997234344482422, 0.999753475189209, 0.0],
    [0.0, 0.8975298404693604, 0.0, 0.0, 2.5300148990936577e-05, 0.4975326359272003, 0.0009194589802064002, 0.0, 0.01277912687510252, 8.455281204078346e-05, 0.00021717455820180476, 0.5032719969749451],
    [0.0, 9.764471542439424e-06, 0.9959237575531006, 0.0006223595119081438, 0.9964762926101685, 0.0, 0.12699976563453674, 0.0003941400209441781, 0.3742411434650421, 0.0, 0.0, 0.49672800302505493],
    [0.04406575858592987, 0.004774820059537888, 0.0, 0.00024575938005000353, 0.0004642514104489237, 0.0008609561482444406, 0.009415017440915108, 0.0006352643249556422, 0.5946105718612671, 0.00019201044051442295, 2.9361342967604287e-05, 0.0],
])

expert_labels = ["E0", "E1", "E2", "E3"]
cluster_labels = [f"c{i}" for i in range(12)]

morandi_blue_cmap = LinearSegmentedColormap.from_list(
    "morandi_blue",
    ["#F4F6F7", "#DCE7ED", "#AFC5D2", "#7899AE", "#315E7E"],
    N=256,
)


# =========================
# Plot heatmap
# =========================
fig, ax = plt.subplots(figsize=(7.8, 2.8), dpi=300)

im = ax.imshow(
    heatmap_values,
    cmap=morandi_blue_cmap,
    vmin=0,
    vmax=1,
    aspect="auto",
)

ax.set_title("Expert preference across token clusters", fontsize=12, fontweight="bold", pad=8)
ax.set_xlabel("Token cluster", fontsize=10)
ax.set_ylabel("Expert", fontsize=10)

ax.set_xticks(np.arange(len(cluster_labels)))
ax.set_xticklabels(cluster_labels, fontsize=9)

ax.set_yticks(np.arange(len(expert_labels)))
ax.set_yticklabels(expert_labels, fontsize=9)

ax.set_xticks(np.arange(-0.5, len(cluster_labels), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(expert_labels), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
ax.tick_params(which="minor", bottom=False, left=False)

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("#777777")
    spine.set_linewidth(0.8)

cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
cbar.set_label(r"$P(\mathrm{expert}\mid\mathrm{cluster})$", fontsize=10)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(heatmap_save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved heatmap figure to: {heatmap_save_path}")