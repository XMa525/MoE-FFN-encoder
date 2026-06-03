import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# Project hooks (TODO)
# =========================
# 你需要把下面这几行替换成你工程里的真实 import
#
# 例如可能是：
# from distillation.dataset.spider_dataset import SpiderPatchDataset
# from distillation.trainers.train_distill import get_val_transform
# from distillation.modeling.xxx import build_teacher, build_student
# from distillation.distiller import MoEDistiller
#
# 这里先留空，避免误导你直接运行失败。

SpiderPatchDataset = None
MoEDistiller = None


def build_distiller_and_loader(args, device):
    """
    TODO: 你需要按工程补这部分
    返回:
        distiller: 已 load ckpt 且 eval() 的 MoEDistiller
        loader:    DataLoader, batch 输出 (images, organs, offline_cluster_ids)
    """
    raise NotImplementedError(
        "请在 build_distiller_and_loader() 里接入你自己的 student/teacher/distiller/dataloader 构建逻辑。"
    )


# =========================
# Utils
# =========================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_rows(mat, eps=1e-8):
    row_sum = mat.sum(axis=1, keepdims=True)
    return mat / np.clip(row_sum, eps, None)


def plot_heatmap(mat, row_labels, col_labels, title, save_path, vmin=0.0, vmax=1.0):
    plt.figure(figsize=(1.4 * len(col_labels) + 2, 0.8 * len(row_labels) + 2))
    im = plt.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=vmax)
    plt.colorbar(im)
    plt.xticks(np.arange(len(col_labels)), col_labels)
    plt.yticks(np.arange(len(row_labels)), row_labels)
    plt.xlabel("Expert")
    plt.ylabel("Cluster")
    plt.title(title)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                plt.text(
                    j, i, f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if val < 0.6 else "white"
                )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# =========================
# Prior analysis
# =========================

def get_learned_cluster_expert_prior(distiller):
    """
    distiller.cluster_expert_logits: [K, E]
    """
    if not hasattr(distiller, "cluster_expert_logits"):
        raise AttributeError("distiller does not have cluster_expert_logits")

    logits = distiller.cluster_expert_logits.detach().cpu().float()
    prior = torch.softmax(logits, dim=-1).numpy()
    return prior


def analyze_prior(distiller, save_dir):
    prior = get_learned_cluster_expert_prior(distiller)
    K, E = prior.shape

    row_labels = [f"C{i}" for i in range(K)]
    col_labels = [f"E{i}" for i in range(E)]

    np.save(os.path.join(save_dir, "cluster_expert_prior.npy"), prior)

    plot_heatmap(
        prior,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Learned Cluster→Expert Prior",
        save_path=os.path.join(save_dir, "cluster_expert_prior_heatmap.png"),
    )

    dominant_expert = prior.argmax(axis=1)
    dominant_conf = prior.max(axis=1)

    plt.figure(figsize=(1.2 * K + 2, 4))
    plt.bar(np.arange(K), dominant_conf)
    plt.xticks(np.arange(K), [f"C{i}->E{dominant_expert[i]}" for i in range(K)])
    plt.ylabel("Max prior prob")
    plt.title("Dominant Expert Confidence per Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cluster_expert_prior_confidence.png"), dpi=200)
    plt.close()

    save_json(
        {
            "cluster_expert_prior": prior.tolist(),
            "dominant_expert": dominant_expert.tolist(),
            "dominant_confidence": dominant_conf.tolist(),
        },
        os.path.join(save_dir, "cluster_expert_prior_summary.json"),
    )

    return prior


# =========================
# Usage analysis
# =========================

@torch.no_grad()
def get_last_dispatch_from_gate_info(gate_info_list, B, N):
    """
    返回最后一层:
        dispatch_weight: [B, N, E]
        dispatch_mask:   [B, N, E]
    """
    last_gate_info = gate_info_list[-1]
    dispatch_weight = last_gate_info["dispatch_weight"]   # [B*(N+1) or B*N, E]
    dispatch_mask = last_gate_info["dispatch_mask"]       # [B*(N+1) or B*N, E]

    E = dispatch_weight.shape[-1]
    tokens_per_batch = dispatch_weight.shape[0] // B

    dispatch_weight = dispatch_weight.view(B, tokens_per_batch, E)
    dispatch_mask = dispatch_mask.view(B, tokens_per_batch, E)

    # 去掉 CLS
    if tokens_per_batch == N + 1:
        dispatch_weight = dispatch_weight[:, 1:, :]
        dispatch_mask = dispatch_mask[:, 1:, :]

    return dispatch_weight, dispatch_mask


@torch.no_grad()
def collect_group_expert_usage(
    distiller,
    loader,
    device,
    num_clusters,
    max_batches=None,
    export_token_csv=False,
    csv_path=None,
):
    soft_group_expert = None   # [K, E]
    hard_group_expert = None   # [K, E]
    group_token_count = np.zeros(num_clusters, dtype=np.int64)

    valid_token_total = 0
    bg_token_total = 0
    total_token_total = 0

    token_rows = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Collect group×expert usage")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, organs, offline_cluster_ids = batch
        images = images.to(device, non_blocking=True)
        offline_cluster_ids = offline_cluster_ids.to(device, non_blocking=True).long()

        # 这里默认你现在的 distiller.forward 返回:
        #   loss, loss_dict, gate_info_list
        loss, loss_dict, gate_info_list = distiller(
            images,
            is_eval=True,
            offline_cluster_ids=offline_cluster_ids,
            epoch=0,
        )

        B, N = offline_cluster_ids.shape
        dispatch_weight, dispatch_mask = get_last_dispatch_from_gate_info(gate_info_list, B, N)
        E = dispatch_weight.shape[-1]

        if soft_group_expert is None:
            soft_group_expert = np.zeros((num_clusters, E), dtype=np.float64)
            hard_group_expert = np.zeros((num_clusters, E), dtype=np.float64)

        valid_mask = distiller.build_valid_cluster_mask(offline_cluster_ids)  # [B, N]
        valid_token_total += int(valid_mask.sum().item())
        bg_token_total += int((~valid_mask).sum().item())
        total_token_total += int(offline_cluster_ids.numel())

        cluster_np = offline_cluster_ids.cpu().numpy()
        soft_np = dispatch_weight.float().cpu().numpy()
        hard_np = dispatch_mask.float().cpu().numpy()

        # 按 group 聚合 usage
        for g in range(num_clusters):
            mask = (cluster_np == g)   # [B, N]
            cnt = int(mask.sum())
            if cnt == 0:
                continue

            group_token_count[g] += cnt
            soft_group_expert[g] += soft_np[mask].sum(axis=0)
            hard_group_expert[g] += hard_np[mask].sum(axis=0)

        # 可选导出 token 级 csv
        if export_token_csv:
            hard_expert = hard_np.argmax(axis=-1)   # [B, N]
            soft_expert = soft_np.argmax(axis=-1)   # [B, N]
            organs_np = organs.numpy() if torch.is_tensor(organs) else np.asarray(organs)
            valid_np = valid_mask.cpu().numpy()

            for b in range(B):
                for t in range(N):
                    row = {
                        "batch_idx": batch_idx,
                        "sample_idx_in_batch": b,
                        "token_idx": t,
                        "organ": int(organs_np[b]),
                        "offline_cluster_id": int(cluster_np[b, t]),
                        "hard_expert_id": int(hard_expert[b, t]),
                        "soft_expert_argmax": int(soft_expert[b, t]),
                        "is_valid_group": int(valid_np[b, t]),
                    }
                    # soft prob 全导出
                    for e in range(E):
                        row[f"soft_prob_e{e}"] = float(soft_np[b, t, e])
                        row[f"hard_mask_e{e}"] = float(hard_np[b, t, e])
                    token_rows.append(row)

    if soft_group_expert is None:
        raise RuntimeError("No batch collected. Check loader.")

    soft_group_expert = normalize_rows(soft_group_expert)
    hard_group_expert = normalize_rows(hard_group_expert)

    stats = {
        "group_token_count": group_token_count.tolist(),
        "group_valid_token_ratio": float(valid_token_total / max(total_token_total, 1)),
        "group_bg_token_ratio": float(bg_token_total / max(total_token_total, 1)),
        "num_tokens_total": int(total_token_total),
        "num_tokens_valid": int(valid_token_total),
        "num_tokens_bg": int(bg_token_total),
    }

    if export_token_csv and csv_path is not None:
        df = pd.DataFrame(token_rows)
        df.to_csv(csv_path, index=False)

    return soft_group_expert, hard_group_expert, stats


def save_usage_results(soft_mat, hard_mat, stats, save_dir):
    ensure_dir(save_dir)

    K, E = soft_mat.shape
    row_labels = [f"C{i} (n={stats['group_token_count'][i]})" for i in range(K)]
    col_labels = [f"E{i}" for i in range(E)]

    np.save(os.path.join(save_dir, "group_expert_soft.npy"), soft_mat)
    np.save(os.path.join(save_dir, "group_expert_hard.npy"), hard_mat)
    save_json(stats, os.path.join(save_dir, "group_expert_stats.json"))

    plot_heatmap(
        soft_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Actual Group×Expert Soft Usage",
        save_path=os.path.join(save_dir, "group_expert_soft_heatmap.png"),
    )

    plot_heatmap(
        hard_mat,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Actual Group×Expert Hard Usage",
        save_path=os.path.join(save_dir, "group_expert_hard_heatmap.png"),
    )

    # dominant expert confidence
    dominant_soft = soft_mat.argmax(axis=1)
    conf_soft = soft_mat.max(axis=1)

    plt.figure(figsize=(1.2 * K + 2, 4))
    plt.bar(np.arange(K), conf_soft)
    plt.xticks(np.arange(K), [f"C{i}->E{dominant_soft[i]}" for i in range(K)])
    plt.ylabel("Max soft usage")
    plt.title("Dominant Expert Confidence by Group (Soft)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "group_expert_soft_confidence.png"), dpi=200)
    plt.close()


# =========================
# Alignment analysis
# =========================

def analyze_prior_usage_alignment(prior, soft_mat, hard_mat, save_dir):
    """
    prior:    [K, E]
    soft_mat: [K, E]
    hard_mat: [K, E]
    """
    K, E = prior.shape

    # 每个 cluster 的 prior dominant expert 和 actual dominant expert
    prior_argmax = prior.argmax(axis=1)
    soft_argmax = soft_mat.argmax(axis=1)
    hard_argmax = hard_mat.argmax(axis=1)

    prior_soft_match = (prior_argmax == soft_argmax).astype(np.int32)
    prior_hard_match = (prior_argmax == hard_argmax).astype(np.int32)

    summary = {
        "prior_argmax": prior_argmax.tolist(),
        "soft_argmax": soft_argmax.tolist(),
        "hard_argmax": hard_argmax.tolist(),
        "prior_soft_match": prior_soft_match.tolist(),
        "prior_hard_match": prior_hard_match.tolist(),
        "prior_soft_match_ratio": float(prior_soft_match.mean()),
        "prior_hard_match_ratio": float(prior_hard_match.mean()),
    }

    save_json(summary, os.path.join(save_dir, "prior_usage_alignment.json"))

    # 简单柱状图
    plt.figure(figsize=(6, 4))
    plt.bar(["prior-soft", "prior-hard"], [
        summary["prior_soft_match_ratio"],
        summary["prior_hard_match_ratio"],
    ])
    plt.ylim(0, 1)
    plt.ylabel("Match ratio")
    plt.title("Prior vs Actual Usage Alignment")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prior_usage_alignment_bar.png"), dpi=200)
    plt.close()

    return summary


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_clusters", type=int, default=6)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--export_token_csv", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.save_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: 用你工程里的逻辑替换
    distiller, loader = build_distiller_and_loader(args, device)

    distiller.eval()

    # 1) learned prior
    prior = analyze_prior(distiller, args.save_dir)

    # 2) actual usage
    csv_path = os.path.join(args.save_dir, "group_expert_token_meta.csv") if args.export_token_csv else None
    soft_mat, hard_mat, stats = collect_group_expert_usage(
        distiller=distiller,
        loader=loader,
        device=device,
        num_clusters=args.num_clusters,
        max_batches=args.max_batches,
        export_token_csv=args.export_token_csv,
        csv_path=csv_path,
    )
    save_usage_results(soft_mat, hard_mat, stats, args.save_dir)

    # 3) prior vs usage alignment
    analyze_prior_usage_alignment(prior, soft_mat, hard_mat, args.save_dir)

    print(f"[Done] results saved to: {args.save_dir}")


if __name__ == "__main__":
    main()