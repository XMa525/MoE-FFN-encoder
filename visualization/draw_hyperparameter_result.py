import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import os

# =========================
# Global style for paper figure
# =========================
color_bracs = "tab:blue"
color_parotid = "tab:orange"

plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans", "DejaVu Sans", "Arial", "Helvetica"]

# =========================
# Helper function
# =========================
def draw_dual_axis_panel(
    ax1,
    x,
    xticklabels,
    bracs_f1,
    parotid_f1,
    bracs_sem,
    parotid_sem,
    default_idx,
    xlabel,
    ylim_left,
    ylim_right,
    panel_title=None,
    show_left_ylabel=False,
    show_right_ylabel=False,
):
    ax2 = ax1.twinx()

    # SEM bands
    ax1.fill_between(
        x,
        bracs_f1 - bracs_sem,
        bracs_f1 + bracs_sem,
        color=color_bracs,
        alpha=0.08,
        linewidth=0,
        zorder=1,
    )

    ax2.fill_between(
        x,
        parotid_f1 - parotid_sem,
        parotid_f1 + parotid_sem,
        color=color_parotid,
        alpha=0.08,
        linewidth=0,
        zorder=1,
    )

    # Lines
    ax1.plot(
        x,
        bracs_f1,
        color=color_bracs,
        marker="o",
        markersize=7.5,
        linewidth=2.4,
        zorder=4,
    )

    ax2.plot(
        x,
        parotid_f1,
        color=color_parotid,
        marker="^",
        markersize=7.5,
        linewidth=2.4,
        zorder=4,
    )

    # Default stars
    ax1.scatter(
        x[default_idx],
        bracs_f1[default_idx],
        marker="*",
        s=330,
        color=color_bracs,
        edgecolors="black",
        linewidths=0.9,
        zorder=10,
    )

    ax2.scatter(
        x[default_idx],
        parotid_f1[default_idx],
        marker="*",
        s=330,
        color=color_parotid,
        edgecolors="black",
        linewidths=0.9,
        zorder=10,
    )

    # Axes limits / ticks
    ax1.set_ylim(*ylim_left)
    ax2.set_ylim(*ylim_right)

    ax1.set_xticks(x)
    ax1.set_xticklabels(xticklabels)
    ax1.set_xlabel(xlabel)
    ax1.set_xlim(-0.28, len(x) - 1 + 0.28)

    if show_left_ylabel:
        ax1.set_ylabel("F1", color=color_bracs)
    else:
        ax1.set_ylabel("")

    if show_right_ylabel:
        ax2.set_ylabel("F1", color=color_parotid)
    else:
        ax2.set_ylabel("")

    ax1.tick_params(axis="y", colors=color_bracs, width=1.1, length=4)
    ax2.tick_params(axis="y", colors=color_parotid, width=1.1, length=4)
    ax1.tick_params(axis="x", width=1.1, length=4)

    # Grid
    ax1.grid(True, linestyle="--", linewidth=0.6, alpha=0.55, zorder=0)

    # Remove normal spines
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)

    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax1.spines["bottom"].set_linewidth(1.2)

    # Draw arrow axes
    ax1.annotate(
        "",
        xy=(0, 1.035),
        xycoords=("axes fraction", "axes fraction"),
        xytext=(0, 0),
        textcoords=("axes fraction", "axes fraction"),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color_bracs,
            lw=1.6,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=14,
        ),
    )

    ax2.annotate(
        "",
        xy=(1, 1.035),
        xycoords=("axes fraction", "axes fraction"),
        xytext=(1, 0),
        textcoords=("axes fraction", "axes fraction"),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color_parotid,
            lw=1.6,
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

    if panel_title is not None:
        ax1.set_title(panel_title, fontsize=12, pad=8)

    return ax1, ax2


# =========================
# Data 1: Insertion layer
# =========================
layers = ["17-18", "19-20", "21-22", "22-23"]
x_layers = np.arange(len(layers))

bracs_layer_f1 = np.array([0.4994, 0.5868, 0.6926, 0.6780])
parotid_layer_f1 = np.array([0.7950, 0.8508, 0.8598, 0.8070])

bracs_layer_std = np.array([0.139, 0.129, 0.045, 0.088])
parotid_layer_std = np.array([0.080, 0.087, 0.034, 0.046])

n_seeds_layer = 5
bracs_layer_sem = bracs_layer_std / np.sqrt(n_seeds_layer)
parotid_layer_sem = parotid_layer_std / np.sqrt(n_seeds_layer)

default_layer_idx = layers.index("21-22")

# =========================
# Data 2: Shared alpha
# =========================
shared_alpha = ["0", "0.05", "0.1"]
x_alpha = np.arange(len(shared_alpha))

bracs_alpha_f1 = np.array([0.7095, 0.6926, 0.6849])
parotid_alpha_f1 = np.array([0.7996, 0.8598, 0.8083])

bracs_alpha_std = np.array([0.020, 0.045, 0.037])
parotid_alpha_std = np.array([0.087, 0.034, 0.075])

n_seeds_alpha = 3
bracs_alpha_sem = bracs_alpha_std / np.sqrt(n_seeds_alpha)
parotid_alpha_sem = parotid_alpha_std / np.sqrt(n_seeds_alpha)

default_alpha_idx = shared_alpha.index("0.05")

# =========================
# Data 3: Expert number
# =========================
expert_nums = ["3", "4", "5", "6"]
x_expert = np.arange(len(expert_nums))

bracs_expert_f1 = np.array([0.6338, 0.6926, 0.5449, 0.6280])
parotid_expert_f1 = np.array([0.7658, 0.8598, 0.7811, 0.8098])

bracs_expert_std = np.array([0.062, 0.045, 0.113, 0.053])
parotid_expert_std = np.array([0.037, 0.034, 0.025, 0.057])

n_seeds_expert = 5
bracs_expert_sem = bracs_expert_std / np.sqrt(n_seeds_expert)
parotid_expert_sem = parotid_expert_std / np.sqrt(n_seeds_expert)

default_expert_idx = expert_nums.index("4")

# =========================
# Create combined figure
# =========================
fig, axes = plt.subplots(
    1,
    3,
    figsize=(15.8, 3.9),
    gridspec_kw={"wspace": 0.30},
)

# Panel A: insertion layer
draw_dual_axis_panel(
    ax1=axes[0],
    x=x_layers,
    xticklabels=layers,
    bracs_f1=bracs_layer_f1,
    parotid_f1=parotid_layer_f1,
    bracs_sem=bracs_layer_sem,
    parotid_sem=parotid_layer_sem,
    default_idx=default_layer_idx,
    xlabel="Insertion layers",
    ylim_left=(0.45, 0.85),
    ylim_right=(0.60, 0.90),
    panel_title=None,
    show_left_ylabel=True,
    show_right_ylabel=False,
)

# Panel B: shared alpha
draw_dual_axis_panel(
    ax1=axes[1],
    x=x_alpha,
    xticklabels=shared_alpha,
    bracs_f1=bracs_alpha_f1,
    parotid_f1=parotid_alpha_f1,
    bracs_sem=bracs_alpha_sem,
    parotid_sem=parotid_alpha_sem,
    default_idx=default_alpha_idx,
    xlabel=r"Shared $\alpha$",
    ylim_left=(0.65, 0.85),
    ylim_right=(0.60, 0.90),
    panel_title=None,
    show_left_ylabel=True,
    show_right_ylabel=False,
)

# Panel C: expert number
draw_dual_axis_panel(
    ax1=axes[2],
    x=x_expert,
    xticklabels=expert_nums,
    bracs_f1=bracs_expert_f1,
    parotid_f1=parotid_expert_f1,
    bracs_sem=bracs_expert_sem,
    parotid_sem=parotid_expert_sem,
    default_idx=default_expert_idx,
    xlabel="Number of routed experts",
    ylim_left=(0.50, 0.90),
    ylim_right=(0.60, 0.90),
    panel_title=None,
    show_left_ylabel=True,
    show_right_ylabel=False,
)

# =========================
# Global legend
# =========================
legend_handles = [
    Line2D(
        [0], [0],
        color=color_bracs,
        marker="o",
        linewidth=2.4,
        markersize=7.5,
        label="BRACS",
    ),
    Line2D(
        [0], [0],
        color=color_parotid,
        marker="^",
        linewidth=2.4,
        markersize=7.5,
        label="PAROTID",
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
    bbox_to_anchor=(0.5, -0.01),
    ncol=3,
    frameon=False,
)

plt.subplots_adjust(left=0.055, right=0.985, bottom=0.30, top=0.96)

# =========================
# Save
# =========================
save_dir = "outputs/hyperparameter_result"
os.makedirs(save_dir, exist_ok=True)

plt.savefig(
    os.path.join(save_dir, "hyperparameter_f1_three_panels.png"),
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    os.path.join(save_dir, "hyperparameter_f1_three_panels.pdf"),
    bbox_inches="tight",
)

plt.show()