# analysis/plot_cluster_stats.py
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_global_cluster_freq(freq_json_path, save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(freq_json_path)
    k = data["k"]
    freq = np.array(data["global_cluster_freq"])
    cnt = np.array(data["global_cluster_count"])
    total_tokens = data["total_tokens"]

    x = np.arange(k)

    plt.figure(figsize=(8, 4))
    plt.bar(x, freq)
    plt.xticks(x, [f"C{i}" for i in range(k)])
    plt.ylabel("Frequency")
    plt.title(f"Global cluster frequency (k={k}, total_tokens={total_tokens:,})")
    for i, v in enumerate(freq):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_dir / "global_cluster_freq_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(x, cnt)
    plt.xticks(x, [f"C{i}" for i in range(k)])
    plt.ylabel("Token count")
    plt.title(f"Global cluster count (k={k})")
    for i, v in enumerate(cnt):
        plt.text(i, v + cnt.max() * 0.01, f"{int(v):,}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_dir / "global_cluster_count_bar.png", dpi=200)
    plt.close()

    print(f"[Saved] {save_dir/'global_cluster_freq_bar.png'}")
    print(f"[Saved] {save_dir/'global_cluster_count_bar.png'}")


def print_extract_meta(meta_json_path):
    meta = load_json(meta_json_path)
    print("=== Extract Meta ===")
    for k, v in meta.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    meta_json = "outputs/token_clustering_layer24/extract_meta.json"
    freq_json = "outputs/token_clustering_layer24/assignments/global_cluster_freq_k6.json"
    save_dir = "outputs/token_clustering_layer24/analysis"

    print_extract_meta(meta_json)
    plot_global_cluster_freq(freq_json, save_dir)