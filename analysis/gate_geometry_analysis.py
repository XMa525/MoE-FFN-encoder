import os
import yaml
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision.transforms.v2 as T

import umap
from tqdm import tqdm
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ====== 改成你自己的项目导入 ======
from models.encoders.moe_encoder import MoEEncoder
from distillation.dataset.spider_dataset import SpiderPatchDataset


# =========================================================
# utils
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(config_path, ckpt_path, device="cuda"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 兼容 {"student_state_dict": ...} 和纯 state_dict
    if isinstance(ckpt, dict) and "student_state_dict" in ckpt:
        state_dict = ckpt["student_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, cfg


def resolve_real_moe_layer_ids(model):
    # 你现在的 moe_layers_idx 可能是 [-3, -2]
    num_blocks = len(model.base_encoder.model.encoder.layer)
    real_ids = []
    for idx in model.moe_layers_idx:
        real_ids.append(idx if idx >= 0 else num_blocks + idx)
    return real_ids


def get_moe_blocks(model):
    real_ids = resolve_real_moe_layer_ids(model)
    blocks = model.base_encoder.model.encoder.layer
    return real_ids, [blocks[i] for i in real_ids]


def reshape_gate_tensor(x, B, N):
    """
    x: [B*(N+1), D] or [B*N, D]
    return: [B, N, D]
    """
    if x is None:
        return None

    D = x.shape[-1]
    tokens_per_batch = x.shape[0] // B
    x = x.view(B, tokens_per_batch, D)

    # 去掉 cls
    if tokens_per_batch == N + 1:
        x = x[:, 1:, :]
    return x


def reshape_gate_vector(x, B, N):
    """
    x: [B*(N+1)] or [B*N]
    return: [B, N]
    """
    if x is None:
        return None
    tokens_per_batch = x.shape[0] // B
    x = x.view(B, tokens_per_batch)
    if tokens_per_batch == N + 1:
        x = x[:, 1:]
    return x


def flatten_by_mask(x, mask=None):
    """
    x: [B, N, D] or [B, N]
    mask: [B, N] bool
    """
    if x is None:
        return None

    if mask is None:
        if x.ndim == 3:
            return x.reshape(-1, x.shape[-1])
        elif x.ndim == 2:
            return x.reshape(-1)
        else:
            raise ValueError("Unsupported shape")

    if x.ndim == 3:
        return x[mask]
    elif x.ndim == 2:
        return x[mask]
    else:
        raise ValueError("Unsupported shape")


def safe_silhouette(X, y):
    uniq = np.unique(y)
    if len(uniq) < 2:
        return np.nan
    counts = [np.sum(y == c) for c in uniq]
    if min(counts) < 2:
        return np.nan
    try:
        return silhouette_score(X, y)
    except Exception:
        return np.nan


def compute_top2_gap(score):
    """
    score: [M, E]
    """
    if score is None or score.shape[1] < 2:
        return None
    top2 = np.partition(score, -2, axis=1)[:, -2:]
    top1 = top2[:, 1]
    top2v = top2[:, 0]
    gap = top1 - top2v
    return gap


def compute_centroids(X, y):
    uniq = np.unique(y)
    centers = []
    labels = []
    for c in uniq:
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        centers.append(X[idx].mean(axis=0))
        labels.append(c)
    if len(centers) == 0:
        return None, None
    return np.stack(centers, axis=0), np.array(labels)


def cosine_matrix_np(X):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X @ X.T


def linear_probe_acc(X, y, test_size=0.3, seed=42):
    uniq = np.unique(y)
    if len(uniq) < 2:
        return np.nan

    counts = [np.sum(y == c) for c in uniq]
    if min(counts) < 2:
        return np.nan

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(
            max_iter=2000,
            multi_class="auto",
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        return clf.score(X_test, y_test)
    except Exception:
        return np.nan


# =========================================================
# collector
# =========================================================
@torch.no_grad()
def collect_gate_geometry_data(
    model,
    dataloader,
    device="cuda",
    max_batches=50,
    sample_per_cluster=300,
    include_noise=True,
):
    real_ids, moe_blocks = get_moe_blocks(model)
    print("Real MoE layer ids:", real_ids)

    # 每层一个桶
    buckets = []
    for _ in real_ids:
        buckets.append(defaultdict(list))

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collect Gate Geometry")):
        if batch_idx >= max_batches:
            break

        # 兼容你的 dataset 输出
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device, non_blocking=True)
            cluster_ids = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
        elif isinstance(batch, dict):
            images = batch["image"].to(device, non_blocking=True)
            cluster_ids = batch.get("offline_cluster_ids", None)
            if cluster_ids is not None:
                cluster_ids = cluster_ids.to(device, non_blocking=True)
        else:
            raise ValueError("Unsupported batch format")

        # forward
        _, gate_info_list, feature_dict, moe_feature_list = model(
            images,
            return_gates=True,
            return_features=True,
            is_eval=True,
            offline_cluster_ids=cluster_ids,
        )

        B = images.shape[0]
        N = moe_feature_list[0].shape[1] - 1  # 去 cls 后 patch token 数

        # 每层收集
        for li, (block, gate_info, moe_feat) in enumerate(zip(moe_blocks, gate_info_list, moe_feature_list)):
            gate = block.mlp.gate

            # feature = MoE 层输出 patch features
            feat_patch = moe_feat[:, 1:, :]  # [B, N, D]

            # gate cached tensors
            last_input = reshape_gate_tensor(gate.last_input, B, N) if gate.last_input is not None else None
            last_proj = reshape_gate_tensor(gate.last_proj, B, N) if gate.last_proj is not None else None
            last_sim = reshape_gate_tensor(gate.last_sim, B, N) if gate.last_sim is not None else None
            last_score = reshape_gate_tensor(gate.last_score, B, N) if gate.last_score is not None else None

            dispatch_weight = reshape_gate_tensor(gate_info["dispatch_weight"], B, N)
            dispatch_mask = reshape_gate_tensor(gate_info["dispatch_mask"].float(), B, N)

            hard_expert = dispatch_weight.argmax(dim=-1)  # [B, N]

            if cluster_ids is None:
                cid = torch.full((B, N), -999, device=device, dtype=torch.long)
            else:
                cid = cluster_ids

            # 采样：每个 cluster 最多 sample_per_cluster
            cid_np = cid.cpu().numpy()
            hard_np = hard_expert.cpu().numpy()

            for b in range(B):
                unique_c = np.unique(cid_np[b])
                for c in unique_c:
                    if (not include_noise) and (c < 0):
                        continue

                    idx = np.where(cid_np[b] == c)[0]
                    if len(idx) == 0:
                        continue
                    if len(idx) > sample_per_cluster:
                        idx = np.random.choice(idx, sample_per_cluster, replace=False)

                    buckets[li]["cluster_id"].append(cid_np[b, idx])
                    buckets[li]["expert_id"].append(hard_np[b, idx])

                    buckets[li]["feature"].append(feat_patch[b, idx].detach().cpu().numpy())

                    if last_input is not None:
                        buckets[li]["input"].append(last_input[b, idx].detach().cpu().numpy())
                    if last_proj is not None:
                        buckets[li]["proj"].append(last_proj[b, idx].detach().cpu().numpy())
                    if last_sim is not None:
                        buckets[li]["sim"].append(last_sim[b, idx].detach().cpu().numpy())
                    if last_score is not None:
                        buckets[li]["score"].append(last_score[b, idx].detach().cpu().numpy())

                    buckets[li]["dispatch_weight"].append(dispatch_weight[b, idx].detach().cpu().numpy())
                    buckets[li]["dispatch_mask"].append(dispatch_mask[b, idx].detach().cpu().numpy())

    # concat
    outputs = []
    for li, bucket in enumerate(buckets):
        out = {}
        for k, v in bucket.items():
            if len(v) == 0:
                out[k] = None
                continue
            out[k] = np.concatenate(v, axis=0)
        out["layer_id"] = real_ids[li]
        outputs.append(out)

    return outputs


# =========================================================
# analysis
# =========================================================
# def analyze_space(X, cluster_id, expert_id, name):
#     stats = {}

#     if X is None:
#         return None

#     if X.ndim == 1:
#         X = X[:, None]

#     # by cluster
#     valid_cluster_mask = cluster_id >= 0
#     if valid_cluster_mask.sum() > 10:
#         Xc = X[valid_cluster_mask]
#         yc = cluster_id[valid_cluster_mask]
#         stats["cluster_silhouette"] = safe_silhouette(Xc, yc)
#         stats["cluster_probe_acc"] = linear_probe_acc(Xc, yc)

#         centers, labels = compute_centroids(Xc, yc)
#         if centers is not None and len(labels) >= 2:
#             sim_mat = cosine_matrix_np(centers)
#             stats["cluster_center_avg_cos"] = float(
#                 sim_mat[np.triu_indices_from(sim_mat, k=1)].mean()
#             )
#         else:
#             stats["cluster_center_avg_cos"] = np.nan
#     else:
#         stats["cluster_silhouette"] = np.nan
#         stats["cluster_probe_acc"] = np.nan
#         stats["cluster_center_avg_cos"] = np.nan

#     # by expert
#     valid_expert_mask = expert_id >= 0
#     if valid_expert_mask.sum() > 10:
#         Xe = X[valid_expert_mask]
#         ye = expert_id[valid_expert_mask]
#         stats["expert_silhouette"] = safe_silhouette(Xe, ye)
#         stats["expert_probe_acc"] = linear_probe_acc(Xe, ye)

#         centers, labels = compute_centroids(Xe, ye)
#         if centers is not None and len(labels) >= 2:
#             sim_mat = cosine_matrix_np(centers)
#             stats["expert_center_avg_cos"] = float(
#                 sim_mat[np.triu_indices_from(sim_mat, k=1)].mean()
#             )
#         else:
#             stats["expert_center_avg_cos"] = np.nan
#     else:
#         stats["expert_silhouette"] = np.nan
#         stats["expert_probe_acc"] = np.nan
#         stats["expert_center_avg_cos"] = np.nan

#     return stats
def analyze_space(X, cluster_id, expert_id, name, max_eval_points=3000, run_probe=False):
    stats = {}

    if X is None:
        return None

    if X.ndim == 1:
        X = X[:, None]

    # 统一抽样，避免 silhouette / probe 太慢
    if X.shape[0] > max_eval_points:
        idx = np.random.choice(X.shape[0], max_eval_points, replace=False)
        X = X[idx]
        cluster_id = cluster_id[idx]
        expert_id = expert_id[idx]

    # by cluster
    valid_cluster_mask = cluster_id >= 0
    if valid_cluster_mask.sum() > 10:
        Xc = X[valid_cluster_mask]
        yc = cluster_id[valid_cluster_mask]
        stats["cluster_silhouette"] = safe_silhouette(Xc, yc)

        if run_probe:
            stats["cluster_probe_acc"] = linear_probe_acc(Xc, yc)
        else:
            stats["cluster_probe_acc"] = np.nan

        centers, labels = compute_centroids(Xc, yc)
        if centers is not None and len(labels) >= 2:
            sim_mat = cosine_matrix_np(centers)
            stats["cluster_center_avg_cos"] = float(
                sim_mat[np.triu_indices_from(sim_mat, k=1)].mean()
            )
        else:
            stats["cluster_center_avg_cos"] = np.nan
    else:
        stats["cluster_silhouette"] = np.nan
        stats["cluster_probe_acc"] = np.nan
        stats["cluster_center_avg_cos"] = np.nan

    # by expert
    valid_expert_mask = expert_id >= 0
    if valid_expert_mask.sum() > 10:
        Xe = X[valid_expert_mask]
        ye = expert_id[valid_expert_mask]
        stats["expert_silhouette"] = safe_silhouette(Xe, ye)

        if run_probe:
            stats["expert_probe_acc"] = linear_probe_acc(Xe, ye)
        else:
            stats["expert_probe_acc"] = np.nan

        centers, labels = compute_centroids(Xe, ye)
        if centers is not None and len(labels) >= 2:
            sim_mat = cosine_matrix_np(centers)
            stats["expert_center_avg_cos"] = float(
                sim_mat[np.triu_indices_from(sim_mat, k=1)].mean()
            )
        else:
            stats["expert_center_avg_cos"] = np.nan
    else:
        stats["expert_silhouette"] = np.nan
        stats["expert_probe_acc"] = np.nan
        stats["expert_center_avg_cos"] = np.nan

    return stats


def save_stats_txt(stats_dict, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for layer_name, layer_stats in stats_dict.items():
            f.write(f"===== {layer_name} =====\n")
            for space_name, stats in layer_stats.items():
                f.write(f"[{space_name}]\n")
                if stats is None:
                    f.write("None\n\n")
                    continue
                for k, v in stats.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")


# =========================================================
# plotting
# =========================================================
def plot_umap(X, y, title, save_path, max_points=3000, cmap="tab10"):
    if X is None:
        return
    if X.shape[0] > max_points:
        idx = np.random.choice(X.shape[0], max_points, replace=False)
        X = X[idx]
        y = y[idx]

    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    emb = reducer.fit_transform(X)

    plt.figure(figsize=(8, 8))
    uniq = np.unique(y)
    for c in uniq:
        idx = y == c
        label = f"{c} (n={idx.sum()})"
        if c < 0:
            plt.scatter(emb[idx, 0], emb[idx, 1], s=4, alpha=0.35, label=f"noise {label}")
        else:
            plt.scatter(emb[idx, 0], emb[idx, 1], s=6, alpha=0.8, label=label)

    plt.legend(markerscale=3, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_score_gap_hist(score, cluster_id, save_path, title):
    if score is None:
        return
    gap = compute_top2_gap(score)
    if gap is None:
        return

    plt.figure(figsize=(7, 5))
    plt.hist(gap, bins=60, alpha=0.8)
    plt.title(title)
    plt.xlabel("Top1 - Top2 score gap")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

    # per cluster summary
    txt_path = str(save_path).replace(".png", ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Global gap mean: {gap.mean():.6f}\n")
        f.write(f"Global gap std: {gap.std():.6f}\n\n")
        for c in np.unique(cluster_id):
            idx = cluster_id == c
            if idx.sum() == 0:
                continue
            f.write(f"Cluster {c}: n={idx.sum()}, mean={gap[idx].mean():.6f}, std={gap[idx].std():.6f}\n")


def plot_centroid_cosine(X, labels, save_path, title):
    centers, uniq = compute_centroids(X, labels)
    if centers is None or len(uniq) < 2:
        return
    sim = cosine_matrix_np(centers)

    plt.figure(figsize=(6, 5))
    plt.imshow(sim, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(range(len(uniq)), uniq, rotation=45)
    plt.yticks(range(len(uniq)), uniq)
    plt.title(title)
    for i in range(len(uniq)):
        for j in range(len(uniq)):
            plt.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


# =========================================================
# main
# =========================================================
def build_dataloader(root, cluster_cache_path, batch_size=32, num_workers=8):
    transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    dataset = SpiderPatchDataset(
        root=root,
        transform=transform,
        cluster_cache_path=cluster_cache_path,
        num_patch_tokens=256,
        missing_cluster_mode="error",
        enable_tissue_filter=True,
        white_threshold=0.85,
        tissue_threshold=0.15,
        samples_cache_path="outputs/dataset_cache/samples_t015.pkl",
        rebuild_samples_cache=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def run_analysis(model, loader, save_dir, tag, device, max_batches=50):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    layer_outputs = collect_gate_geometry_data(
        model=model,
        dataloader=loader,
        device=device,
        max_batches=max_batches,
        sample_per_cluster=300,
        include_noise=True,
    )

    all_stats = {}

    for layer_out in layer_outputs:
        lid = layer_out["layer_id"]
        layer_name = f"{tag}_layer{lid}"
        layer_dir = save_dir / layer_name
        layer_dir.mkdir(parents=True, exist_ok=True)

        cluster_id = layer_out["cluster_id"]
        expert_id = layer_out["expert_id"]

        layer_stats = {}

        for space_name in [ "input", "sim", "score"]:
            print(f"[{layer_name}] analyzing space: {space_name}")
            X = layer_out.get(space_name, None)
            if X is None:
                layer_stats[space_name] = None
                continue

            #stats = analyze_space(X, cluster_id, expert_id, space_name)
            stats = analyze_space(
                X, cluster_id, expert_id, space_name,
                max_eval_points=3000,
                run_probe=False,
            )
            layer_stats[space_name] = stats
            print(f"[{layer_name}] stats done: {space_name}")

            # UMAP by cluster
            plot_umap(
                X, cluster_id,
                title=f"{tag} {space_name} UMAP by cluster (layer {lid})",
                save_path=layer_dir / f"{space_name}_umap_cluster.png"
            )

            # UMAP by expert
            plot_umap(
                X, expert_id,
                title=f"{tag} {space_name} UMAP by expert (layer {lid})",
                save_path=layer_dir / f"{space_name}_umap_expert.png"
            )

            # centroid cosine
            valid_cluster_mask = cluster_id >= 0
            if valid_cluster_mask.sum() > 10:
                plot_centroid_cosine(
                    X[valid_cluster_mask],
                    cluster_id[valid_cluster_mask],
                    save_path=layer_dir / f"{space_name}_cluster_centroid_cosine.png",
                    title=f"{tag} {space_name} cluster centroid cosine (layer {lid})"
                )

            plot_centroid_cosine(
                X,
                expert_id,
                save_path=layer_dir / f"{space_name}_expert_centroid_cosine.png",
                title=f"{tag} {space_name} expert centroid cosine (layer {lid})"
            )

            # score gap only for score space
            if space_name == "score":
                plot_score_gap_hist(
                    X, cluster_id,
                    save_path=layer_dir / "score_gap_hist.png",
                    title=f"{tag} score gap histogram (layer {lid})"
                )

        all_stats[layer_name] = layer_stats

    save_stats_txt(all_stats, save_dir / f"{tag}_geometry_stats.txt")
    print(f"Saved analysis to {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True, help="stage1 or stage2_smoke etc.")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="../data/raw")
    parser.add_argument("--cluster_cache_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    set_seed(42)
    device = args.device if torch.cuda.is_available() else "cpu"

    model, cfg = load_model(args.config, args.ckpt, device=device)
    loader = build_dataloader(
        root=args.data_root,
        cluster_cache_path=args.cluster_cache_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    run_analysis(
        model=model,
        loader=loader,
        save_dir=args.save_dir,
        tag=args.tag,
        device=device,
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()