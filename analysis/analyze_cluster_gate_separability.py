import os
import sys
import json
import random
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from models.encoders.moe_encoder import MoEEncoder
from distillation.dataset.spider_dataset import SpiderPatchDataset


# =========================
# 基础工具
# =========================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_cfg(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def save_cache(data, cache_path):
    ensure_dir(os.path.dirname(cache_path))
    cpu_data = {}
    for k, v in data.items():
        if torch.is_tensor(v):
            cpu_data[k] = v.detach().cpu()
        else:
            cpu_data[k] = v
    with open(cache_path, "wb") as f:
        pickle.dump(cpu_data, f)


def load_cache(cache_path):
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    return data


def remove_cls_if_needed(x, target_n):
    """
    x: [B, T, D]
    """
    if x.shape[1] == target_n + 1:
        return x[:, 1:, ...]
    return x


def reshape_gate_tensor(x, B, N):
    """
    x: [B*T, E] -> [B, N, E] or [B, N+1, E], then remove cls if needed
    """
    T_tokens = x.shape[0] // B
    E = x.shape[-1]
    x = x.view(B, T_tokens, E)
    x = remove_cls_if_needed(x, N)
    return x


# =========================
# 统计函数
# =========================

def compute_p_expert_given_cluster(dispatch, cluster_ids, valid_cluster_ids):
    """
    dispatch: [M, E]
    cluster_ids: [M]
    """
    E = dispatch.shape[-1]
    stats = {}
    mat = []

    for cid in valid_cluster_ids:
        mask = (cluster_ids == cid)
        n = int(mask.sum().item())
        stats[f"cluster_token_count_{cid}"] = n

        if n == 0:
            vec = torch.zeros(E, device=dispatch.device)
        else:
            vec = dispatch[mask].mean(dim=0)
            vec = vec / vec.sum().clamp_min(1e-8)

        stats[f"p_e_given_c_{cid}"] = [float(v) for v in vec.detach().cpu()]
        mat.append(vec.detach().cpu().numpy())

    return stats, np.stack(mat, axis=0)


def compute_p_cluster_given_expert(dispatch, cluster_ids, valid_cluster_ids):
    """
    dispatch: [M, E]
    cluster_ids: [M]
    """
    E = dispatch.shape[-1]
    stats = {}
    mat = []

    for e in range(E):
        mass_e = dispatch[:, e].sum()
        stats[f"expert_token_mass_{e}"] = float(mass_e.detach().cpu())

        row = []
        if mass_e.item() <= 0:
            for cid in valid_cluster_ids:
                stats[f"p_c{cid}_given_e{e}"] = 0.0
                row.append(0.0)
            stats[f"expert_purity_{e}"] = 0.0
            stats[f"expert_main_cluster_{e}"] = -1
        else:
            probs = []
            for cid in valid_cluster_ids:
                mask_c = (cluster_ids == cid)
                if mask_c.sum() == 0:
                    p = torch.tensor(0.0, device=dispatch.device)
                else:
                    p = dispatch[mask_c, e].sum() / mass_e.clamp_min(1e-8)

                p_val = float(p.detach().cpu())
                stats[f"p_c{cid}_given_e{e}"] = p_val
                probs.append(p_val)
                row.append(p_val)

            probs = np.array(probs)
            stats[f"expert_purity_{e}"] = float(probs.max())
            stats[f"expert_main_cluster_{e}"] = int(valid_cluster_ids[int(probs.argmax())])

        mat.append(row)

    return stats, np.array(mat)


def compute_distance_metrics(X, y):
    X_np = to_numpy(X)
    y_np = to_numpy(y)

    metrics = {}

    try:
        if len(np.unique(y_np)) >= 2 and X_np.shape[0] > len(np.unique(y_np)):
            metrics["silhouette"] = float(silhouette_score(X_np, y_np))
        else:
            metrics["silhouette"] = None
    except Exception:
        metrics["silhouette"] = None

    centers = {}
    intra_var = {}
    for cid in np.unique(y_np):
        Xc = X_np[y_np == cid]
        center = Xc.mean(axis=0)
        centers[int(cid)] = center
        intra_var[int(cid)] = float(((Xc - center) ** 2).sum(axis=1).mean())

    inter_dist = {}
    cids = sorted(list(centers.keys()))
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            ci, cj = cids[i], cids[j]
            dist = np.linalg.norm(centers[ci] - centers[cj])
            inter_dist[f"{ci}_{cj}"] = float(dist)

    metrics["intra_var"] = intra_var
    metrics["inter_center_dist"] = inter_dist
    return metrics


def run_linear_probe(X, y, seed=42):
    X_np = to_numpy(X)
    y_np = to_numpy(y)

    if len(np.unique(y_np)) < 2:
        return {"acc": None, "macro_f1": None}

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.3, random_state=seed, stratify=y_np
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return {
        "acc": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }


def reduce_to_2d(X, method="pca", seed=42):
    X_np = to_numpy(X)

    if method == "pca":
        emb2d = PCA(n_components=2, random_state=seed).fit_transform(X_np)
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("umap-learn not installed")
        emb2d = umap.UMAP(n_components=2, random_state=seed).fit_transform(X_np)
    else:
        raise ValueError(f"Unknown reducer: {method}")

    return emb2d


def plot_scatter(emb2d, labels, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(emb2d[:, 0], emb2d[:, 1], c=labels, s=4, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_heatmap(mat, row_labels, col_labels, title, save_path):
    plt.figure(figsize=(6, 4))
    plt.imshow(mat, aspect="auto")
    plt.xticks(range(len(col_labels)), col_labels)
    plt.yticks(range(len(row_labels)), row_labels)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def cosine_sim_matrix(A, B):
    A = F.normalize(A, dim=-1)
    B = F.normalize(B, dim=-1)
    return A @ B.t()


# =========================
# 构建模型和数据
# =========================

def build_model(cfg, ckpt_path, device):
    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"]).to(device)

    state = torch.load(ckpt_path, map_location="cpu")
    if "student_state_dict" in state:
        state_dict = state["student_state_dict"]
    elif "model" in state:
        state_dict = state["model"]
    elif "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def build_loader(cfg):
    val_transform = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    full_dataset = SpiderPatchDataset(
        root="../data/raw",
        transform=val_transform,
        cluster_cache_path="outputs/token_clustering_layer24_fullassign/path_to_cluster_ids.pkl",
        num_patch_tokens=256,
        missing_cluster_mode="error",
        enable_tissue_filter=True,
        white_threshold=0.85,
        tissue_threshold=0.15,
        samples_cache_path="outputs/dataset_cache/samples_t015.pkl",
        rebuild_samples_cache=False,
    )

    loader = DataLoader(
        full_dataset,
        batch_size=cfg.get("analysis", {}).get("batch_size", 128),
        shuffle=False,
        num_workers=cfg.get("analysis", {}).get("num_workers", 8),
        pin_memory=True,
        drop_last=False,
    )
    return loader


# =========================
# 数据收集
# =========================

@torch.no_grad()
def collect_analysis_data(model, loader, device, valid_cluster_ids, max_tokens_per_cluster=5000):
    """
    收集:
    - cluster
    - layer0/1 moe feature
    - layer0/1 sim
    - layer0/1 dispatch
    """
    model.eval()

    bucket = {
        "cluster": [],
        "l0_feat": [],
        "l1_feat": [],
        "l0_sim": [],
        "l1_sim": [],
        "l0_dispatch": [],
        "l1_dispatch": [],
    }

    cluster_counts = {cid: 0 for cid in valid_cluster_ids}

    for batch in tqdm(loader, desc="Collect analysis data"):
        images, organs, offline_cluster_ids = batch
        images = images.to(device, non_blocking=True)
        offline_cluster_ids = offline_cluster_ids.to(device, non_blocking=True)

        final_feats, gate_info_list, _, moe_feature_list = model(
            images,
            return_gates=True,
            return_features=True,
            is_eval=True,
        )

        B, N = offline_cluster_ids.shape

        l0_feat = remove_cls_if_needed(moe_feature_list[0], N)
        l1_feat = remove_cls_if_needed(moe_feature_list[1], N)

        l0_sim = reshape_gate_tensor(gate_info_list[0]["sim"], B, N)
        l1_sim = reshape_gate_tensor(gate_info_list[1]["sim"], B, N)

        l0_dispatch = reshape_gate_tensor(gate_info_list[0]["dispatch_weight"], B, N)
        l1_dispatch = reshape_gate_tensor(gate_info_list[1]["dispatch_weight"], B, N)

        for cid in valid_cluster_ids:
            remain = max_tokens_per_cluster - cluster_counts[cid]
            if remain <= 0:
                continue

            mask = (offline_cluster_ids == cid)
            idx = mask.nonzero(as_tuple=False)
            if idx.shape[0] == 0:
                continue

            idx = idx[:remain]
            b_idx, n_idx = idx[:, 0], idx[:, 1]

            bucket["cluster"].append(offline_cluster_ids[b_idx, n_idx].detach().cpu())
            bucket["l0_feat"].append(l0_feat[b_idx, n_idx].detach().cpu())
            bucket["l1_feat"].append(l1_feat[b_idx, n_idx].detach().cpu())
            bucket["l0_sim"].append(l0_sim[b_idx, n_idx].detach().cpu())
            bucket["l1_sim"].append(l1_sim[b_idx, n_idx].detach().cpu())
            bucket["l0_dispatch"].append(l0_dispatch[b_idx, n_idx].detach().cpu())
            bucket["l1_dispatch"].append(l1_dispatch[b_idx, n_idx].detach().cpu())

            cluster_counts[cid] += idx.shape[0]

        if all(cluster_counts[cid] >= max_tokens_per_cluster for cid in valid_cluster_ids):
            break

    for k in bucket:
        if len(bucket[k]) == 0:
            raise RuntimeError(f"No data collected for key={k}")
        bucket[k] = torch.cat(bucket[k], dim=0)

    return bucket


# =========================
# 分析函数
# =========================

def analyze_embedding_space(X, y, name, out_dir, reducer="pca", seed=42):
    ensure_dir(out_dir)

    metrics = {
        "linear_probe": run_linear_probe(X, y, seed=seed),
        "distance": compute_distance_metrics(X, y),
    }

    emb2d = reduce_to_2d(X, method=reducer, seed=seed)
    plot_scatter(
        emb2d=emb2d,
        labels=to_numpy(y),
        title=f"{name} ({reducer})",
        save_path=os.path.join(out_dir, f"{name}_{reducer}.png"),
    )

    return metrics


def analyze_dispatch_space(dispatch, cluster_ids, valid_cluster_ids, name, out_dir):
    ensure_dir(out_dir)

    p_e_stats, mat_e_given_c = compute_p_expert_given_cluster(
        dispatch, cluster_ids, valid_cluster_ids
    )
    p_c_stats, mat_c_given_e = compute_p_cluster_given_expert(
        dispatch, cluster_ids, valid_cluster_ids
    )

    plot_heatmap(
        mat_e_given_c,
        row_labels=[f"c{cid}" for cid in valid_cluster_ids],
        col_labels=[f"e{i}" for i in range(mat_e_given_c.shape[1])],
        title=f"{name}: P(expert|cluster)",
        save_path=os.path.join(out_dir, f"{name}_p_e_given_c.png"),
    )

    plot_heatmap(
        mat_c_given_e,
        row_labels=[f"e{i}" for i in range(mat_c_given_e.shape[0])],
        col_labels=[f"c{cid}" for cid in valid_cluster_ids],
        title=f"{name}: P(cluster|expert)",
        save_path=os.path.join(out_dir, f"{name}_p_c_given_e.png"),
    )

    stats = {}
    stats.update(p_e_stats)
    stats.update(p_c_stats)
    return stats


def analyze_prototype_vs_cluster_center(X, y, gate_vectors, valid_cluster_ids, name, out_dir):
    """
    X: [M, D]  feature space
    gate_vectors: [E, D]
    """
    ensure_dir(out_dir)

    centers = []
    row_labels = []

    for cid in valid_cluster_ids:
        mask = (y == cid)
        Xc = X[mask]
        center = Xc.mean(dim=0, keepdim=True)
        centers.append(center)
        row_labels.append(f"c{cid}")

    centers = torch.cat(centers, dim=0)
    sim_mat = cosine_sim_matrix(centers, gate_vectors).detach().cpu().numpy()

    plot_heatmap(
        sim_mat,
        row_labels=row_labels,
        col_labels=[f"e{i}" for i in range(gate_vectors.shape[0])],
        title=f"{name}: cosine(cluster_center, prototype)",
        save_path=os.path.join(out_dir, f"{name}_cluster_center_vs_proto.png"),
    )

    return sim_mat.tolist()


# =========================
# 主程序
# =========================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/phase2.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cache_path", type=str, default=None)
    parser.add_argument("--force_recollect", action="store_true")
    parser.add_argument("--max_tokens_per_cluster", type=int, default=5000)
    parser.add_argument("--reducer", type=str, default="pca", choices=["pca", "umap"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out_dir)

    cfg = load_cfg(args.config)
    cfg.setdefault("analysis", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_cluster_ids = cfg["moe_loss"]["valid_cluster_ids"]

    model = build_model(cfg, args.ckpt, device)
    loader = build_loader(cfg)

    if args.cache_path is None:
        args.cache_path = os.path.join(args.out_dir, "analysis_cache.pkl")

    if os.path.exists(args.cache_path) and (not args.force_recollect):
        print(f"[Info] Loading cached analysis data from: {args.cache_path}")
        data = load_cache(args.cache_path)
    else:
        print("[Info] Collecting analysis data from model forward ...")
        data = collect_analysis_data(
            model=model,
            loader=loader,
            device=device,
            valid_cluster_ids=valid_cluster_ids,
            max_tokens_per_cluster=args.max_tokens_per_cluster,
        )
        print(f"[Info] Saving analysis cache to: {args.cache_path}")
        save_cache(data, args.cache_path)

    y = data["cluster"]
    results = {}

    results["layer0_feature"] = analyze_embedding_space(
        data["l0_feat"], y, "layer0_feature", args.out_dir, reducer=args.reducer, seed=args.seed
    )
    results["layer1_feature"] = analyze_embedding_space(
        data["l1_feat"], y, "layer1_feature", args.out_dir, reducer=args.reducer, seed=args.seed
    )
    results["layer0_sim"] = analyze_embedding_space(
        data["l0_sim"], y, "layer0_sim", args.out_dir, reducer=args.reducer, seed=args.seed
    )
    results["layer1_sim"] = analyze_embedding_space(
        data["l1_sim"], y, "layer1_sim", args.out_dir, reducer=args.reducer, seed=args.seed
    )

    results["layer0_dispatch"] = analyze_dispatch_space(
        data["l0_dispatch"], y, valid_cluster_ids, "layer0_dispatch", args.out_dir
    )
    results["layer1_dispatch"] = analyze_dispatch_space(
        data["l1_dispatch"], y, valid_cluster_ids, "layer1_dispatch", args.out_dir
    )

    try:
        moe_layers = cfg["moe_encoder"]["moe_layers"]

        if hasattr(model, "blocks"):
            blocks = model.blocks
        elif hasattr(model, "encoder") and hasattr(model.encoder, "blocks"):
            blocks = model.encoder.blocks
        else:
            raise AttributeError("Cannot find transformer blocks on model")

        blk0 = blocks[moe_layers[0]]
        blk1 = blocks[moe_layers[1]]

        gate_vec_l0 = blk0.mlp.gate.gate_vectors.detach().cpu()
        gate_vec_l1 = blk1.mlp.gate.gate_vectors.detach().cpu()

        results["layer0_proto_vs_center"] = analyze_prototype_vs_cluster_center(
            data["l0_feat"], y, gate_vec_l0, valid_cluster_ids, "layer0_feature", args.out_dir
        )
        results["layer1_proto_vs_center"] = analyze_prototype_vs_cluster_center(
            data["l1_feat"], y, gate_vec_l1, valid_cluster_ids, "layer1_feature", args.out_dir
        )
    except Exception as e:
        results["prototype_center_error"] = str(e)

    save_json(results, os.path.join(args.out_dir, "metrics.json"))
    print(f"Done. Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()