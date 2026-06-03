import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os

# =========================
# Data
# =========================
shared_alpha = ["0", "0.05", "0.1"]
x = np.arange(len(shared_alpha))

bracs_f1 = np.array([0.7095, 0.6926, 0.6849])
parotid_f1 = np.array([0.7996, 0.8598, 0.8083])

# These are standard deviations over seeds
bracs_f1_std = np.array([0.020, 0.045, 0.037])
parotid_f1_std = np.array([0.087, 0.034, 0.075])

# Convert STD to SEM for visualization
n_seeds = 3
bracs_f1_sem = bracs_f1_std / np.sqrt(n_seeds)
parotid_f1_sem = parotid_f1_std / np.sqrt(n_seeds)

default_alpha = "0.05"
default_idx = shared_alpha.index(default_alpha)

# =========================
# Style
# =========================
color_bracs = "tab:blue"
color_parotid = "tab:orange"

plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"]

fig, ax1 = plt.subplots(figsize=(5.8, 4.0))
ax2 = ax1.twinx()

# =========================
# SEM bands
# =========================
ax1.fill_between(
    x,
    bracs_f1 - bracs_f1_sem,
    bracs_f1 + bracs_f1_sem,
    color=color_bracs,
    alpha=0.08,
    linewidth=0,
    zorder=1,
)

ax2.fill_between(
    x,
    parotid_f1 - parotid_f1_sem,
    parotid_f1 + parotid_f1_sem,
    color=color_parotid,
    alpha=0.08,
    linewidth=0,
    zorder=1,
)

# =========================
# Lines
# =========================
ax1.plot(
    x,
    bracs_f1,
    color=color_bracs,
    marker="o",
    markersize=7,
    linewidth=2.2,
    label="BRACS",
    zorder=4,
)

ax2.plot(
    x,
    parotid_f1,
    color=color_parotid,
    marker="^",
    markersize=7,
    linewidth=2.2,
    label="Parotid",
    zorder=4,
)

# Default stars
ax1.scatter(
    x[default_idx],
    bracs_f1[default_idx],
    marker="*",
    s=300,
    color=color_bracs,
    edgecolors="black",
    linewidths=0.9,
    zorder=10,
)

ax2.scatter(
    x[default_idx],
    parotid_f1[default_idx],
    marker="*",
    s=300,
    color=color_parotid,
    edgecolors="black",
    linewidths=0.9,
    zorder=10,
)

# =========================
# Axes limits / ticks
# =========================
ax1.set_ylim(0.65, 0.85)
ax2.set_ylim(0.60, 0.90)

ax1.set_xticks(x)
ax1.set_xticklabels(shared_alpha)
ax1.set_xlabel(r"Shared $\alpha$")
ax1.set_xlim(-0.25, len(x) - 1 + 0.25)

ax1.set_ylabel("F1", color=color_bracs)
# ax2.set_ylabel("F1", color=color_parotid)

ax1.tick_params(axis="y", colors=color_bracs)
ax2.tick_params(axis="y", colors=color_parotid)

# =========================
# Grid
# =========================
ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)

# =========================
# Remove normal spines
# =========================
for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(False)

ax1.spines["left"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax1.spines["bottom"].set_linewidth(1.2)

# =========================
# Draw arrow axes
# =========================
ax1.annotate(
    "",
    xy=(0, 1.03),
    xycoords=("axes fraction", "axes fraction"),
    xytext=(0, 0),
    textcoords=("axes fraction", "axes fraction"),
    arrowprops=dict(
        arrowstyle="-|>",
        color=color_bracs,
        lw=1.5,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=14,
    ),
)

ax2.annotate(
    "",
    xy=(1, 1.03),
    xycoords=("axes fraction", "axes fraction"),
    xytext=(1, 0),
    textcoords=("axes fraction", "axes fraction"),
    arrowprops=dict(
        arrowstyle="-|>",
        color=color_parotid,
        lw=1.5,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=14,
    ),
)

ax1.annotate(
    "",
    xy=(1.02, 0),
    xycoords=("axes fraction", "axes fraction"),
    xytext=(0, 0),
    textcoords=("axes fraction", "axes fraction"),
    arrowprops=dict(
        arrowstyle="-",
        color="black",
        lw=1.2,
        shrinkA=0,
        shrinkB=0,
    ),
)

# =========================
# Legend below whole figure
# =========================
legend_handles = [
    Line2D(
        [0], [0],
        color=color_bracs,
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="BRACS",
    ),
    Line2D(
        [0], [0],
        color=color_parotid,
        marker="^",
        linewidth=2.2,
        markersize=7,
        label="Parotid",
    ),
    Line2D(
        [0], [0],
        color="gray",
        marker="*",
        linestyle="None",
        markeredgecolor="black",
        markersize=14,
        label="Default",
    ),
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0),
    ncol=3,
    frameon=False,
)

plt.subplots_adjust(bottom=0.28)

# =========================
# Save
# =========================
save_dir = "outputs/insertion_result"
os.makedirs(save_dir, exist_ok=True)

plt.savefig(
    os.path.join(save_dir, "shared_alpha_f1_with_sem_band.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.show()