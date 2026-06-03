import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def save_fig(fig, path: str, dpi: int = 300):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def detect_slide_metric_columns(df: pd.DataFrame, prefix: str) -> Dict[str, str]:
    """
    Resolve common metric names produced by previous analysis scripts.
    prefix: e.g. baseline / current / wsi
    """
    candidates = {
        "topk_mean": [f"topk_mean_score_{prefix}", f"topk_mean_{prefix}", "topk_mean_score"],
        "topk_max": [f"topk_max_score_{prefix}", f"topk_max_{prefix}", "topk_max_score"],
        "topk_min": [f"topk_min_score_{prefix}", f"topk_min_{prefix}", "topk_min_score"],
        "mean_all": [f"score_mean_all_{prefix}", f"mean_score_all_{prefix}", "score_mean_all"],
    }
    out = {}
    for k, cols in candidates.items():
        out[k] = next((c for c in cols if c in df.columns), None)
    return out


def add_delta_if_needed(
    merged: pd.DataFrame,
    old_prefix: str,
    new_prefix: str,
    metric_pairs: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    for metric_name, (old_col, new_col) in metric_pairs.items():
        if old_col in merged.columns and new_col in merged.columns:
            merged[f"delta_{metric_name}_{new_prefix}_minus_{old_prefix}"] = merged[new_col] - merged[old_col]
    return merged


def summarise_series(x: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "median": np.nan, "min": np.nan, "max": np.nan}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "median": float(x.median()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


# =========================================================
# Plotters for paper-style figures
# =========================================================

def plot_scatter_compare(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    title: str,
    out_path: str,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    neg = df[df[label_col] == 0]
    pos = df[df[label_col] == 1]

    if len(neg) > 0:
        ax.scatter(neg[x_col], neg[y_col], alpha=0.75, s=20, label="Negative")
    if len(pos) > 0:
        ax.scatter(pos[x_col], pos[y_col], alpha=0.75, s=20, label="Positive")

    all_vals = pd.concat([df[x_col], df[y_col]], axis=0).dropna()
    if len(all_vals) > 0:
        lo = float(all_vals.min())
        hi = float(all_vals.max())
        pad = max((hi - lo) * 0.05, 1e-4)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

    ax.set_title(title)
    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.legend(frameon=False)
    save_fig(fig, out_path)


def plot_delta_box(
    df: pd.DataFrame,
    delta_col: str,
    label_col: str,
    title: str,
    out_path: str,
    ylabel: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    data = []
    labels = []
    for lv, name in [(0, "Negative"), (1, "Positive")]:
        vals = pd.to_numeric(df.loc[df[label_col] == lv, delta_col], errors="coerce").dropna()
        if len(vals) > 0:
            data.append(vals.values)
            labels.append(name)

    if len(data) == 0:
        raise ValueError(f"No valid data for {delta_col}")

    ax.boxplot(data, labels=labels, showfliers=True)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_ylabel(ylabel or delta_col)
    save_fig(fig, out_path)


def plot_grouped_bar(
    stats: Dict[str, Dict[str, float]],
    title: str,
    out_path: str,
    ylabel: str,
):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    keys = list(stats.keys())
    means = [stats[k]["mean"] for k in keys]
    stds = [stats[k]["std"] for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=15)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    save_fig(fig, out_path)


def plot_epoch_curves(
    epoch_df: pd.DataFrame,
    y_cols: List[str],
    title: str,
    out_path: str,
    x_col: str = "epoch",
):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for c in y_cols:
        if c in epoch_df.columns:
            ax.plot(epoch_df[x_col], epoch_df[c], marker="o", label=c)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.legend(frameon=False)
    save_fig(fig, out_path)


def plot_hist_compare(
    arrays: List[np.ndarray],
    labels: List[str],
    title: str,
    out_path: str,
    bins: int = 40,
    xlabel: str = "value",
):
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for arr, lab in zip(arrays, labels):
        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        ax.hist(arr, bins=bins, alpha=0.45, density=True, label=lab)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    save_fig(fig, out_path)


# =========================================================
# Core report builder
# =========================================================

class DistillClosingReport:
    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        ensure_dir(out_dir)
        self.fig_dir = os.path.join(out_dir, "figures")
        ensure_dir(self.fig_dir)
        self.table_dir = os.path.join(out_dir, "tables")
        ensure_dir(self.table_dir)
        self.summary_lines: List[str] = []

    def log(self, msg: str):
        print(msg)
        self.summary_lines.append(msg)

    def save_summary(self):
        with open(os.path.join(self.out_dir, "closing_summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.summary_lines))

    # -----------------------------------------------------
    # 1) Slide-level evidence comparison
    # -----------------------------------------------------
    def build_slide_level_section(
        self,
        baseline_slide_csv: str,
        current_slide_csv: str,
        baseline_name: str = "baseline",
        current_name: str = "current",
    ) -> pd.DataFrame:
        df_old = read_csv_safe(baseline_slide_csv)
        df_new = read_csv_safe(current_slide_csv)

        merged = df_old.merge(
            df_new,
            on=["slide_id", "slide_label"],
            suffixes=(f"_{baseline_name}", f"_{current_name}"),
            how="inner",
        )

        old_map = detect_slide_metric_columns(merged, baseline_name)
        new_map = detect_slide_metric_columns(merged, current_name)

        metric_pairs = {}
        for k in ["topk_mean", "topk_max", "topk_min", "mean_all"]:
            if old_map[k] is not None and new_map[k] is not None:
                metric_pairs[k] = (old_map[k], new_map[k])
        merged = add_delta_if_needed(merged, baseline_name, current_name, metric_pairs)

        merged.to_csv(os.path.join(self.table_dir, "slide_level_merged.csv"), index=False)

        # scatter figures
        if "topk_mean" in metric_pairs:
            x_col, y_col = metric_pairs["topk_mean"]
            plot_scatter_compare(
                merged, x_col, y_col, "slide_label",
                title="Slide-level top-k mean evidence",
                out_path=os.path.join(self.fig_dir, "fig_slide_topk_mean_scatter.png"),
                xlabel=f"{baseline_name} top-k mean",
                ylabel=f"{current_name} top-k mean",
            )
        if "topk_max" in metric_pairs:
            x_col, y_col = metric_pairs["topk_max"]
            plot_scatter_compare(
                merged, x_col, y_col, "slide_label",
                title="Slide-level top-k max evidence",
                out_path=os.path.join(self.fig_dir, "fig_slide_topk_max_scatter.png"),
                xlabel=f"{baseline_name} top-k max",
                ylabel=f"{current_name} top-k max",
            )

        # delta box plots
        for metric in ["topk_mean", "topk_max", "mean_all"]:
            dcol = f"delta_{metric}_{current_name}_minus_{baseline_name}"
            if dcol in merged.columns:
                plot_delta_box(
                    merged,
                    delta_col=dcol,
                    label_col="slide_label",
                    title=f"Delta {metric} ({current_name} - {baseline_name})",
                    out_path=os.path.join(self.fig_dir, f"fig_delta_{metric}_box.png"),
                    ylabel=f"Δ {metric}",
                )

        # summary table
        rows = []
        for lv, group_name in [(0, "negative"), (1, "positive")]:
            sdf = merged[merged["slide_label"] == lv]
            for metric in ["topk_mean", "topk_max", "mean_all"]:
                dcol = f"delta_{metric}_{current_name}_minus_{baseline_name}"
                if dcol in sdf.columns:
                    s = summarise_series(sdf[dcol])
                    rows.append({"group": group_name, "metric": dcol, **s})
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(os.path.join(self.table_dir, "slide_delta_summary.csv"), index=False)
        self.log("[Slide-level] built evidence comparison figures and summary table.")
        return merged

    # -----------------------------------------------------
    # 2) Ranking / residual-HN statistics over epochs
    # -----------------------------------------------------
    def build_epoch_stat_section(
        self,
        epoch_log_csv: str,
    ):
        df = read_csv_safe(epoch_log_csv)
        df.to_csv(os.path.join(self.table_dir, "epoch_log_copy.csv"), index=False)

        candidate_groups = [
            ["train_cond_rank_neg_selected_gap_mean", "val_cond_rank_neg_selected_gap_mean"],
            ["train_residual_hn_gap_mean", "val_residual_hn_gap_mean"],
            ["train_residual_hn_num_after_expert_filter", "val_residual_hn_num_after_expert_filter"],
            ["train_cond_rank_pos_selected_gap_mean", "val_cond_rank_pos_selected_gap_mean"],
        ]
        titles = [
            "Negative selected gap over epochs",
            "Residual-HN gap over epochs",
            "Residual-HN count over epochs",
            "Positive selected gap over epochs",
        ]
        names = [
            "fig_epoch_neg_selected_gap.png",
            "fig_epoch_residual_gap.png",
            "fig_epoch_residual_count.png",
            "fig_epoch_pos_selected_gap.png",
        ]

        for cols, title, name in zip(candidate_groups, titles, names):
            valid = [c for c in cols if c in df.columns]
            if valid and "epoch" in df.columns:
                plot_epoch_curves(df, valid, title, os.path.join(self.fig_dir, name))

        self.log("[Epoch-level] built ranking and residual-HN trend figures.")

    # -----------------------------------------------------
    # 3) Patch-level selected-token distributions
    # -----------------------------------------------------
    def build_patch_distribution_section(
        self,
        baseline_top_patch_csv: str,
        current_top_patch_csv: str,
        baseline_name: str = "baseline",
        current_name: str = "current",
    ):
        df_old = read_csv_safe(baseline_top_patch_csv)
        df_new = read_csv_safe(current_top_patch_csv)

        def maybe_plot(metric: str, label_val: int, suffix: str):
            if metric not in df_old.columns or metric not in df_new.columns:
                return
            arr_old = df_old.loc[df_old["slide_label"] == label_val, metric].to_numpy()
            arr_new = df_new.loc[df_new["slide_label"] == label_val, metric].to_numpy()
            label_name = "negative" if label_val == 0 else "positive"
            plot_hist_compare(
                [arr_old, arr_new],
                [baseline_name, current_name],
                title=f"{label_name.capitalize()} top-patch {metric} distribution",
                out_path=os.path.join(self.fig_dir, f"fig_{label_name}_{metric}_hist.png"),
                xlabel=metric,
            )

        for metric in ["score", "sim_tumor", "sim_other_max"]:
            maybe_plot(metric, 0, "neg")
            maybe_plot(metric, 1, "pos")

        self.log("[Patch-level] built top-patch score distribution figures.")

    # -----------------------------------------------------
    # 4) Expert usage comparison
    # -----------------------------------------------------
    def build_expert_usage_section(
        self,
        usage_csv: str,
        split_col: str = "split",
    ):
        """
        Expects a CSV with columns like:
        split, soft_frac_e0, soft_frac_e1, ..., hard_frac_e0, hard_frac_e1, ...
        One row per checkpoint / epoch / run is acceptable.
        """
        df = read_csv_safe(usage_csv)
        df.to_csv(os.path.join(self.table_dir, "expert_usage_copy.csv"), index=False)

        soft_cols = [c for c in df.columns if c.startswith("soft_frac_e")]
        hard_cols = [c for c in df.columns if c.startswith("hard_frac_e")]

        if soft_cols:
            stats = {c: summarise_series(df[c]) for c in soft_cols}
            plot_grouped_bar(
                stats,
                title="Soft expert usage",
                out_path=os.path.join(self.fig_dir, "fig_soft_expert_usage.png"),
                ylabel="Mean fraction",
            )
        if hard_cols:
            stats = {c: summarise_series(df[c]) for c in hard_cols}
            plot_grouped_bar(
                stats,
                title="Hard expert usage",
                out_path=os.path.join(self.fig_dir, "fig_hard_expert_usage.png"),
                ylabel="Mean fraction",
            )

        self.log("[Expert usage] built expert usage bar figures.")

    # -----------------------------------------------------
    # 5) Automatic closing checklist summary
    # -----------------------------------------------------
    def build_closing_checklist(
        self,
        merged_slide_df: Optional[pd.DataFrame] = None,
        epoch_log_csv: Optional[str] = None,
        current_name: str = "current",
        baseline_name: str = "baseline",
    ):
        lines = []

        if merged_slide_df is not None:
            for metric in ["topk_mean", "topk_max"]:
                dcol = f"delta_{metric}_{current_name}_minus_{baseline_name}"
                if dcol in merged_slide_df.columns:
                    neg = merged_slide_df.loc[merged_slide_df["slide_label"] == 0, dcol].dropna()
                    pos = merged_slide_df.loc[merged_slide_df["slide_label"] == 1, dcol].dropna()
                    if len(neg) > 0:
                        lines.append(f"Negative {metric} delta mean: {neg.mean():.6f}")
                    if len(pos) > 0:
                        lines.append(f"Positive {metric} delta mean: {pos.mean():.6f}")

        if epoch_log_csv is not None and os.path.exists(epoch_log_csv):
            df = pd.read_csv(epoch_log_csv)
            for col in [
                "val_cond_rank_neg_selected_gap_mean",
                "val_residual_hn_gap_mean",
                "val_residual_hn_num_after_expert_filter",
                "val_cond_rank_pos_selected_gap_mean",
            ]:
                if col in df.columns:
                    lines.append(f"Best/last {col}: {df[col].dropna().iloc[-1]:.6f}")

        checklist_path = os.path.join(self.table_dir, "closing_checklist.txt")
        with open(checklist_path, "w", encoding="utf-8") as f:
            f.write("Distillation closing checklist\n")
            f.write("=" * 32 + "\n")
            for line in lines:
                f.write(line + "\n")

        self.log("[Checklist] exported closing checklist summary.")


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True)

    # slide-level compare
    parser.add_argument("--baseline-slide-csv", type=str, default="")
    parser.add_argument("--current-slide-csv", type=str, default="")
    parser.add_argument("--baseline-name", type=str, default="baseline")
    parser.add_argument("--current-name", type=str, default="current")

    # top patch compare
    parser.add_argument("--baseline-top-patch-csv", type=str, default="")
    parser.add_argument("--current-top-patch-csv", type=str, default="")

    # epoch log
    parser.add_argument("--epoch-log-csv", type=str, default="")

    # expert usage
    parser.add_argument("--expert-usage-csv", type=str, default="")

    args = parser.parse_args()

    report = DistillClosingReport(args.out_dir)
    merged_slide_df = None

    if args.baseline_slide_csv and args.current_slide_csv:
        merged_slide_df = report.build_slide_level_section(
            baseline_slide_csv=args.baseline_slide_csv,
            current_slide_csv=args.current_slide_csv,
            baseline_name=args.baseline_name,
            current_name=args.current_name,
        )

    if args.epoch_log_csv:
        report.build_epoch_stat_section(args.epoch_log_csv)

    if args.baseline_top_patch_csv and args.current_top_patch_csv:
        report.build_patch_distribution_section(
            baseline_top_patch_csv=args.baseline_top_patch_csv,
            current_top_patch_csv=args.current_top_patch_csv,
            baseline_name=args.baseline_name,
            current_name=args.current_name,
        )

    if args.expert_usage_csv:
        report.build_expert_usage_section(args.expert_usage_csv)

    report.build_closing_checklist(
        merged_slide_df=merged_slide_df,
        epoch_log_csv=args.epoch_log_csv or None,
        current_name=args.current_name,
        baseline_name=args.baseline_name,
    )
    report.save_summary()
    print(f"[Done] outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
