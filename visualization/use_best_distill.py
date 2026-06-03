import os
import glob
import random
from collections import Counter


os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import yaml
import torch
import torchvision.transforms as T
from PIL import Image
import json 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import umap
import torch.nn.functional as F

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

from models.encoders.moe_encoder import MoEEncoder


# =========================================================
# 0) 全局配置
# =========================================================
# CONFIG_PATH = "configs/phase2.yaml"
# CKPT_PATH = "results/distilled_best_model/moe_encoder_best.pth"
CONFIG_PATH = "configs/stage2_roleproto.yaml"
#CKPT_PATH = "results/stage2_best_model/moe_encoder_stage2_best.pth"
FULL_CKPT_PATH = "distill_checkpoints_stage2_tcga_roleproto_v1/best_full.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# 数据采样
BASE_DIR = "../data/raw"
CATEGORIES = ["SPIDER-breast", "SPIDER-colorectal", "SPIDER-skin", "SPIDER-thorax"]
N_SAMPLES_PER_CAT = 20

# 主分析空间:
#   "moe"   -> 最后一个 MoE 层输出特征
#   "final" -> 最终 encoder 输出特征
PRIMARY_FEATURE_SPACE = "moe"

# 聚类方式:
#   "kmeans" / "hdbscan" / "dbscan"
CLUSTER_METHOD = "hdbscan"

# KMeans 参数
N_CLUSTERS = 4
SUB_N_CLUSTERS = 4

# HDBSCAN / DBSCAN 参数
HDBSCAN_MIN_CLUSTER_SIZE = 200
HDBSCAN_MIN_SAMPLES = 10

DBSCAN_EPS = 0.8
DBSCAN_MIN_SAMPLES = 20

# UMAP 参数
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# 可视化开关
DRAW_HEATMAP = True
DRAW_OVERLAY = False
DRAW_CONTRAST_ANALYSIS = False
DRAW_OVERLAP_ANALYSIS = True
KNN_K = 10
MIN_EXPERT_SAMPLES = 0   # 少于这个数的 expert 不参与 overlap 统计
# =========================================================
# Specialization audit
# =========================================================
DRAW_SPECIALIZATION_AUDIT = True
DRAW_SAME_TOKEN_PAIR_HIST = True
DRAW_EXPERT_OUTPUT_PROTO_HEATMAP = True
DRAW_EXPERT_CLUSTER_TABLE = True
MIN_ACTIVE_PAIR_TOKENS = 10

# overlay 设置
PATCH_GRID_SIZE = 16
MAX_OVERLAY_IMAGES = 8
OVERLAY_ALPHA = 0.35

# 分析哪个 MoE 层:
#   "first" -> 第一个 MoE 层（你现在想看的 layer 9）
#   "last"  -> 最后一个 MoE 层（当前 block 10）
#   "both"  -> 两层都分析
MOE_ANALYSIS_LAYER = "last"

# 统一选择后续导出/overlay 默认使用哪个 MoE 层
PRIMARY_MOE_LAYER = "first"   # "first" or "last"

# =========================================================
# Role-prototype-aware analysis
# =========================================================
DRAW_ROLE_PROTO_ANALYSIS = True

# 是否画 expert × role prototype 平均相似度热图
DRAW_EXPERT_ROLE_AFFINITY_HEATMAP = True

# 是否画 expert × nearest-role ratio 热图
DRAW_EXPERT_ROLE_RATIO_HEATMAP = True

# 是否画 UMAP 按 nearest role 着色
DRAW_NEAREST_ROLE_UMAP = True

# 是否专门分析 free expert 吸收了什么
DRAW_FREE_EXPERT_RESIDUE = True

# 如果 role prototype 在 teacher space，需要 projection head
ROLE_PROTO_DIR = "outputs/role_proto_init_virchow2_layer26_405/3role_v1"
FREE_EXPERT_ID = 3

# 如果你后面把 distiller.state_dict 一起存了，就用它
#DISTILLER_CKPT_PATH = "distill_checkpoints_stage2_roleproto_static/epoch_5.pth"

# 若没有 projection head，自动关闭严格 role affinity
STRICT_ROLE_SPACE = True

# =========================================================
# 1) 工具函数
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(config_path, ckpt_path, device="cuda"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    print("Best model loaded")
    print(f"Current moe_layers_idx = {model.moe_layers_idx}")
    return model, cfg

def load_stage2_bundle(config_path, full_ckpt_path, device="cuda"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not os.path.exists(full_ckpt_path):
        raise FileNotFoundError(f"Full checkpoint not found: {full_ckpt_path}")

    ckpt = torch.load(full_ckpt_path, map_location="cpu")

    if "student_state_dict" not in ckpt:
        raise KeyError("student_state_dict not found in full checkpoint")
    if "distiller_state_dict" not in ckpt:
        raise KeyError("distiller_state_dict not found in full checkpoint")

    # 1) load student / encoder
    model = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    model.load_state_dict(ckpt["student_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    # 2) load proj_l12 from distiller
    distiller_sd = ckpt["distiller_state_dict"]
    if "proj_l12.weight" not in distiller_sd or "proj_l12.bias" not in distiller_sd:
        raise KeyError("proj_l12 not found in distiller_state_dict")

    proj_out_dim, proj_in_dim = distiller_sd["proj_l12.weight"].shape
    role_proj_head = torch.nn.Linear(proj_in_dim, proj_out_dim)
    role_proj_head.load_state_dict({
        "weight": distiller_sd["proj_l12.weight"],
        "bias": distiller_sd["proj_l12.bias"],
    })
    role_proj_head = role_proj_head.to(device)
    role_proj_head.eval()

    print("Loaded matched student + proj_l12 from the same best_full checkpoint")
    print(f"Current moe_layers_idx = {model.moe_layers_idx}")
    print(f"proj_l12 shape: {proj_in_dim} -> {proj_out_dim}")

    return model, role_proj_head, cfg

def get_last_moe_real_idx(model):
    real_indices = sorted(model.moe_layer_map.keys())
    return real_indices[-1]


def sample_images(base_dir, categories, n_samples_per_cat):
    sample_imgs = []
    sample_meta = []

    for cat in categories:
        img_dir = os.path.join(base_dir, cat, cat, "images")
        all_imgs = glob.glob(os.path.join(img_dir, "*.png"))

        if len(all_imgs) < n_samples_per_cat:
            raise ValueError(f"{cat} 中的图片数量不足 {n_samples_per_cat} 张")

        sampled = random.sample(all_imgs, n_samples_per_cat)
        sample_imgs.extend(sampled)
        sample_meta.extend([cat] * len(sampled))

    print(f"总共采样 {len(sample_imgs)} 张图片")
    return sample_imgs, sample_meta


def build_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(), 
    ])


def run_model_and_collect(model, img_tensor):
    """
    return:
        final_feats:    [1, seq_len, D]
        gate_info_list: dict
        moe_feature_list: [1, seq_len, D]
    """
    with torch.no_grad():
        final_feats, gate_info_list, _, moe_feature_list = model(
            img_tensor,
            return_gates=True,
            return_features=True,
            is_eval=True,
        )

    if len(gate_info_list) == 0:
        raise RuntimeError("gate_info_list 为空，请检查 moe_layers 配置或模型 forward。")

    if len(moe_feature_list) == 0:
        raise RuntimeError("moe_feature_list 为空，请检查 MoEEncoder.forward 是否已改写。")

    last_gate_info = gate_info_list[-1]
    last_moe_feats = moe_feature_list[-1]
    return final_feats,gate_info_list, moe_feature_list


def get_expert_assignment_from_gate_info(gate_info, seq_len):
    dispatch = gate_info["dispatch_weight"]  # [B*seq_len, E]
    total_tokens, num_experts = dispatch.shape
    B = total_tokens // seq_len

    if B != 1:
        raise ValueError(f"当前函数默认一次处理 1 张图，但检测到 B={B}")

    gates = dispatch.reshape(B, seq_len, num_experts)[0]  # [seq_len, E]
    gates = gates[1:]  # remove CLS
    expert_id = gates.argmax(dim=1)  # [num_patches]
    return expert_id


def prepare_feature_space(features, name="feature"):
    features_norm = normalize(features, norm="l2")
    n_components = min(32, features_norm.shape[1], features_norm.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    features_pca = pca.fit_transform(features_norm)

    print(f"{name} PCA features shape: {features_pca.shape}")
    return features_norm, features_pca


def cluster_features(features_pca, method="kmeans"):
    if method == "kmeans":
        clusterer = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = clusterer.fit_predict(features_pca)

    elif method == "hdbscan":
        if not HAS_HDBSCAN:
            raise ImportError("当前环境未安装 hdbscan，请 pip install hdbscan 或改用 kmeans/dbscan")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric="euclidean"
        )
        cluster_labels = clusterer.fit_predict(features_pca)

    elif method == "dbscan":
        clusterer = DBSCAN(
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES,
            metric="euclidean"
        )
        cluster_labels = clusterer.fit_predict(features_pca)

    else:
        raise ValueError(f"Unknown CLUSTER_METHOD: {method}")

    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    noise_ratio = float((cluster_labels == -1).mean()) if -1 in cluster_labels else 0.0
    print(f"[{method}] number of clusters = {n_clusters_found}, noise ratio = {noise_ratio:.4f}")
    return cluster_labels


def build_umap(features_norm):
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_SEED
    )
    embedding = reducer.fit_transform(features_norm)
    print("embedding shape:", embedding.shape)
    return embedding



##导出token级别的 metadata csv，供后续分析使用
EXPORT_TOKEN_META = True
TOKEN_META_SAVE_PATH = "analysis_outputs/token_meta.csv"


def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def token_idx_to_row_col(token_idx, grid_size=16):
    row = int(token_idx) // grid_size
    col = int(token_idx) % grid_size
    return row, col


def build_token_meta_records(
    per_image_paths,
    sample_meta,
    per_image_expert_ids,
    per_image_cluster_labels_moe=None,
    per_image_cluster_labels_final=None,
    grid_size=16,
):
    """
    构建 token 级别的 metadata 表。
    一行 = 一个 token（不含 CLS）。
    """
    records = []

    num_images = len(per_image_paths)
    assert len(per_image_expert_ids) == num_images

    if per_image_cluster_labels_moe is not None:
        assert len(per_image_cluster_labels_moe) == num_images
    if per_image_cluster_labels_final is not None:
        assert len(per_image_cluster_labels_final) == num_images

    for img_idx in range(num_images):
        patch_path = per_image_paths[img_idx]
        category = sample_meta[img_idx]
        expert_ids = per_image_expert_ids[img_idx]  # [num_tokens]

        cluster_labels_moe = None
        cluster_labels_final = None

        if per_image_cluster_labels_moe is not None:
            cluster_labels_moe = per_image_cluster_labels_moe[img_idx]
        if per_image_cluster_labels_final is not None:
            cluster_labels_final = per_image_cluster_labels_final[img_idx]

        num_tokens = len(expert_ids)

        for token_idx in range(num_tokens):
            row, col = token_idx_to_row_col(token_idx, grid_size=grid_size)

            rec = {
                "image_index": img_idx,
                "patch_path": patch_path,
                "patch_name": os.path.basename(patch_path),
                "sample_category": category,
                "token_idx": int(token_idx),
                "token_row": int(row),
                "token_col": int(col),
                "expert_id": int(expert_ids[token_idx]),
            }

            if cluster_labels_moe is not None:
                rec["cluster_id_moe"] = int(cluster_labels_moe[token_idx])

            if cluster_labels_final is not None:
                rec["cluster_id_final"] = int(cluster_labels_final[token_idx])

            records.append(rec)

    df = pd.DataFrame(records)
    return df


def export_token_meta_csv(
    save_path,
    per_image_paths,
    sample_meta,
    per_image_expert_ids,
    per_image_cluster_labels_moe=None,
    per_image_cluster_labels_final=None,
    grid_size=16,
):
    df = build_token_meta_records(
        per_image_paths=per_image_paths,
        sample_meta=sample_meta,
        per_image_expert_ids=per_image_expert_ids,
        per_image_cluster_labels_moe=per_image_cluster_labels_moe,
        per_image_cluster_labels_final=per_image_cluster_labels_final,
        grid_size=grid_size,
    )
    ensure_parent_dir(save_path)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] token metadata csv -> {save_path}")
    print(df.head())
    return df

def load_role_prototypes(role_proto_dir):
    proto_path = os.path.join(role_proto_dir, "role_prototypes_init.npy")
    names_path = os.path.join(role_proto_dir, "role_names.json")

    if (not os.path.exists(proto_path)) or (not os.path.exists(names_path)):
        print(f"[RoleProto] Missing files under {role_proto_dir}, skip role analysis.")
        return None, None

    protos = np.load(proto_path).astype(np.float32)
    with open(names_path, "r", encoding="utf-8") as f:
        role_names = json.load(f)

    protos = normalize(protos, norm="l2")
    return protos, role_names

def load_role_projection_from_ckpt(distiller_ckpt_path, in_dim=384, out_dim=1280, device="cpu"):
    if not os.path.exists(distiller_ckpt_path):
        print(f"[RoleProto] distiller ckpt not found: {distiller_ckpt_path}")
        return None

    ckpt = torch.load(distiller_ckpt_path, map_location=device)

    proj = torch.nn.Linear(in_dim, out_dim)
    loaded = False

    if "distiller_state_dict" in ckpt:
        sd = ckpt["distiller_state_dict"]
        if "proj_l12.weight" in sd and "proj_l12.bias" in sd:
            proj.load_state_dict({
                "weight": sd["proj_l12.weight"],
                "bias": sd["proj_l12.bias"],
            })
            loaded = True

    if (not loaded) and ("proj_l12_state_dict" in ckpt):
        proj.load_state_dict(ckpt["proj_l12_state_dict"])
        loaded = True

    if not loaded:
        print("[RoleProto] proj_l12 not found in checkpoint, cannot do strict role-space analysis.")
        return None

    proj = proj.to(device)
    proj.eval()
    return proj

@torch.no_grad()
def project_features_to_role_space(features, proj_head, device="cpu", batch_size=4096):
    """
    features: np.ndarray [N, D_student]
    returns: np.ndarray [N, D_teacher]
    """
    if proj_head is None:
        return None

    outs = []
    for start in range(0, len(features), batch_size):
        x = torch.from_numpy(features[start:start+batch_size]).float().to(device)
        y = proj_head(x)
        y = F.normalize(y, dim=-1)
        outs.append(y.cpu().numpy())

    return np.concatenate(outs, axis=0)
def compute_role_affinity(features_role_space, role_prototypes):
    """
    features_role_space: [N, D]
    role_prototypes: [R, D]
    """
    feats = normalize(features_role_space, norm="l2")
    protos = normalize(role_prototypes, norm="l2")
    return feats @ protos.T   # [N, R]
def nearest_role_assignment(role_affinity, role_names):
    idx = role_affinity.argmax(axis=1)
    labels = np.array([role_names[i] for i in idx], dtype=object)
    return idx, labels
# =========================================================
# 2) 图 A：UMAP
# =========================================================
def plot_cluster_umap(embedding, cluster_labels, title):
    plt.figure(figsize=(8, 8))
    unique_labels = sorted(np.unique(cluster_labels).tolist())

    normal_labels = [x for x in unique_labels if x != -1]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(normal_labels), 1)))

    # 先画噪声
    if -1 in unique_labels:
        idx = cluster_labels == -1
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c="lightgray",
            s=6,
            alpha=0.5,
            label=f"noise -1 (n={idx.sum()})"
        )

    # 再画正常簇
    for lbl, col in zip(normal_labels, colors):
        idx = cluster_labels == lbl
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c=[col],
            s=6,
            alpha=0.75,
            label=f"cluster {lbl} (n={idx.sum()})"
        )

    plt.legend(markerscale=2)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_expert_umap(embedding, expert_ids, title):
    plt.figure(figsize=(8, 8))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=expert_ids,
        cmap="tab10",
        s=6,
        alpha=0.75
    )
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_pca_2d(features_pca, labels, title, cmap="tab10"):
    plt.figure(figsize=(8, 8))
    if -1 in labels:
        normal_mask = labels != -1
        noise_mask = labels == -1

        plt.scatter(
            features_pca[noise_mask, 0],
            features_pca[noise_mask, 1],
            c="lightgray",
            s=5,
            alpha=0.5,
            label=f"noise -1 (n={noise_mask.sum()})"
        )
        plt.scatter(
            features_pca[normal_mask, 0],
            features_pca[normal_mask, 1],
            c=labels[normal_mask],
            cmap=cmap,
            s=5,
            alpha=0.75
        )
        plt.legend()
    else:
        plt.scatter(
            features_pca[:, 0],
            features_pca[:, 1],
            c=labels,
            cmap=cmap,
            s=5,
            alpha=0.75
        )

    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================================================
# 3) 图 B：cluster × expert 热图
# =========================================================
def plot_cluster_expert_heatmap(cluster_labels, expert_ids, title):
    unique_clusters = sorted(np.unique(cluster_labels).tolist())
    unique_experts = sorted(np.unique(expert_ids).tolist())

    mat = np.zeros((len(unique_clusters), len(unique_experts)), dtype=np.float32)
    cluster_sizes = []

    for i, c in enumerate(unique_clusters):
        idx = cluster_labels == c
        e = expert_ids[idx]
        cluster_sizes.append(idx.sum())

        cnt = Counter(e.tolist())
        total = len(e)
        for j, ex in enumerate(unique_experts):
            mat[i, j] = cnt.get(ex, 0) / max(total, 1)

    plt.figure(figsize=(1.6 * len(unique_experts) + 3, 0.65 * len(unique_clusters) + 3))
    im = plt.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=1.0)

    plt.xticks(range(len(unique_experts)), [f"Expert {e}" for e in unique_experts], rotation=30)

    ytick_labels = []
    for i, c in enumerate(unique_clusters):
        if c == -1:
            ytick_labels.append(f"Noise (n={cluster_sizes[i]})")
        else:
            ytick_labels.append(f"Cluster {c} (n={cluster_sizes[i]})")

    plt.yticks(range(len(unique_clusters)), ytick_labels)

    plt.xlabel("Expert")
    plt.ylabel("Feature Cluster")
    plt.title(title)
    plt.colorbar(im, label="Proportion within cluster")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if val >= 0.05:
                plt.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=8,
                    color="black" if val < 0.6 else "white"
                )

    plt.tight_layout()
    plt.show()


# =========================================================
# 4) purity / 二次聚类
# =========================================================
def cluster_purity_analysis(cluster_labels, expert_ids):
    print("\n===== Cluster Purity Analysis =====")
    unique_labels = sorted(np.unique(cluster_labels).tolist())

    purity_stats = []
    for c in unique_labels:
        idx = cluster_labels == c
        e = expert_ids[idx]
        cnt = Counter(e.tolist())
        total = len(e)

        ratios = {k: v / total for k, v in sorted(cnt.items())}
        dominant_expert, dominant_count = cnt.most_common(1)[0]
        dominant_purity = dominant_count / total

        purity_stats.append({
            "cluster": int(c),
            "size": int(total),
            "dominant_expert": int(dominant_expert),
            "dominant_purity": float(dominant_purity),
            "expert_ratios": ratios
        })

    for stat in sorted(purity_stats, key=lambda x: x["size"], reverse=True):
        cluster_name = "Noise" if stat["cluster"] == -1 else f"cluster={stat['cluster']}"
        print(
            f"{cluster_name}, size={stat['size']}, "
            f"dominant_expert={stat['dominant_expert']}, "
            f"dominant_purity={stat['dominant_purity']:.3f}, "
            f"ratios={stat['expert_ratios']}"
        )


def subcluster_largest_cluster(features_norm, embedding, cluster_labels, expert_ids, sub_n_clusters=4, title_prefix=""):
    unique_labels = [x for x in np.unique(cluster_labels).tolist() if x != -1]
    if len(unique_labels) == 0:
        print("\nNo non-noise clusters found. Skip largest cluster sub-clustering.")
        return

    largest_cluster = max(unique_labels, key=lambda x: np.sum(cluster_labels == x))
    largest_idx = cluster_labels == largest_cluster

    largest_features = features_norm[largest_idx]
    largest_experts = expert_ids[largest_idx]
    largest_embedding = embedding[largest_idx]

    print(f"\nLargest cluster = {largest_cluster}, size = {largest_idx.sum()}")

    if largest_features.shape[0] < sub_n_clusters:
        print("Largest cluster too small for sub-clustering, skip.")
        return

    sub_kmeans = KMeans(n_clusters=sub_n_clusters, random_state=RANDOM_SEED, n_init=10)
    sub_labels = sub_kmeans.fit_predict(largest_features)

    plt.figure(figsize=(8, 8))
    sub_colors = plt.cm.Set2(np.linspace(0, 1, sub_n_clusters))
    for s, col in zip(range(sub_n_clusters), sub_colors):
        idx = sub_labels == s
        plt.scatter(
            largest_embedding[idx, 0],
            largest_embedding[idx, 1],
            c=[col],
            s=6,
            alpha=0.75,
            label=f"subcluster {s} (n={idx.sum()})"
        )
    plt.legend(markerscale=2)
    plt.title(f"{title_prefix} Largest Cluster with Sub-clusters")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.scatter(
        largest_embedding[:, 0],
        largest_embedding[:, 1],
        c=largest_experts,
        cmap="tab10",
        s=6,
        alpha=0.75
    )
    plt.title(f"{title_prefix} Largest Cluster colored by Expert")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    print("\n===== Largest Cluster Sub-cluster Purity =====")
    for s in range(sub_n_clusters):
        idx = sub_labels == s
        e = largest_experts[idx]
        cnt = Counter(e.tolist())
        total = len(e)
        dominant_expert, dominant_count = cnt.most_common(1)[0]
        print(
            f"subcluster={s}, size={total}, dominant_expert={dominant_expert}, "
            f"dominant_purity={dominant_count/total:.3f}, ratios={dict(cnt)}"
        )


# =========================================================
# 5) 图 C：overlay
# =========================================================
def labels_to_grid(labels_1d, grid_size=16):
    if len(labels_1d) != grid_size * grid_size:
        raise ValueError(f"labels 数量 {len(labels_1d)} 与 grid_size={grid_size} 不匹配")
    return np.array(labels_1d).reshape(grid_size, grid_size)


def plot_patch_overlay(
    image_path,
    patch_labels,
    title="Patch Overlay",
    grid_size=16,
    alpha=0.35,
    cmap_name="tab10"
):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_np = np.array(img)

    label_grid = labels_to_grid(patch_labels, grid_size=grid_size)
    unique_labels = sorted(np.unique(patch_labels).tolist())

    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_grid = np.vectorize(label_to_idx.get)(label_grid)

    cmap = plt.cm.get_cmap(cmap_name, len(unique_labels))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)

    # 关键：保存这个 im 对象，colorbar 直接绑它
    im = ax.imshow(
        idx_grid,
        cmap=cmap,
        alpha=alpha,
        interpolation="nearest",
        extent=(0, 224, 224, 0)
    )

    patch_size = 224 // grid_size
    for x in range(0, 225, patch_size):
        ax.axvline(x, color="white", linewidth=0.25, alpha=0.35)
    for y in range(0, 225, patch_size):
        ax.axhline(y, color="white", linewidth=0.25, alpha=0.35)

    # 关键：显式指定 ax=ax
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    cbar.set_ticks(range(len(unique_labels)))
    tick_names = ["Noise" if lab == -1 else str(lab) for lab in unique_labels]
    cbar.set_ticklabels(tick_names)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# =========================================================
# 3.5) Prototype analysis
# =========================================================
def get_gate_vectors_from_block(block):
    """
    取出某个 MoE block 里的 gate prototype 向量
    """
    if not hasattr(block, "mlp"):
        raise AttributeError("block has no mlp")
    if not hasattr(block.mlp, "gate"):
        raise AttributeError("block.mlp has no gate")
    if not hasattr(block.mlp.gate, "gate_vectors"):
        raise AttributeError("block.mlp.gate has no gate_vectors")
    return block.mlp.gate.gate_vectors.detach().cpu()   # [E, D]


def analyze_prototype_similarity(gate_vectors, title_prefix=""):
    """
    gate_vectors: [E, D]
    """
    gv = gate_vectors.detach().cpu()
    gv_norm = F.normalize(gv, dim=-1)
    sim_mat = gv_norm @ gv_norm.t()   # [E, E]

    E = sim_mat.shape[0]
    off_diag = []
    for i in range(E):
        for j in range(E):
            if i != j:
                off_diag.append(sim_mat[i, j].item())

    print(f"\n===== Prototype Similarity Analysis: {title_prefix} =====")
    print("Prototype-prototype cosine similarity matrix:")
    print(sim_mat.numpy())
    print(f"Off-diagonal mean = {np.mean(off_diag):.4f}")
    print(f"Off-diagonal max  = {np.max(off_diag):.4f}")
    print(f"Off-diagonal min  = {np.min(off_diag):.4f}")

    return sim_mat.numpy(), {
        "off_diag_mean": float(np.mean(off_diag)),
        "off_diag_max": float(np.max(off_diag)),
        "off_diag_min": float(np.min(off_diag)),
    }


def plot_prototype_similarity_heatmap(sim_mat, title):
    E = sim_mat.shape[0]

    plt.figure(figsize=(6, 5))
    im = plt.imshow(sim_mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    plt.xticks(range(E), [f"Expert {i}" for i in range(E)])
    plt.yticks(range(E), [f"Expert {i}" for i in range(E)])
    plt.title(title)
    plt.colorbar(im, label="Cosine similarity")

    for i in range(E):
        for j in range(E):
            val = sim_mat[i, j]
            plt.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=9,
                color="white" if abs(val) > 0.5 else "black"
            )

    plt.tight_layout()
    plt.show()


def analyze_prototype_norms(gate_vectors, title_prefix=""):
    gv = gate_vectors.detach().cpu()
    norms = gv.norm(dim=-1)   # [E]

    print(f"\n===== Prototype Norm Analysis: {title_prefix} =====")
    for i, n in enumerate(norms.tolist()):
        print(f"Expert {i}: norm = {n:.4f}")

    print(f"Norm mean = {norms.mean().item():.4f}")
    print(f"Norm std  = {norms.std().item():.4f}")
    print(f"Norm min  = {norms.min().item():.4f}")
    print(f"Norm max  = {norms.max().item():.4f}")

    return norms.numpy(), {
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std().item()),
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
    }


def plot_prototype_norms(norms, title):
    E = len(norms)
    plt.figure(figsize=(6, 4))
    plt.bar(range(E), norms)
    plt.xticks(range(E), [f"Expert {i}" for i in range(E)])
    plt.ylabel("L2 norm")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_prototype_pca(gate_vectors, title):
    gv = gate_vectors.detach().cpu().numpy()   # [E, D]

    if gv.shape[0] < 2:
        print("Too few prototypes for PCA.")
        return

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    emb = pca.fit_transform(gv)

    plt.figure(figsize=(6, 6))
    for i in range(emb.shape[0]):
        plt.scatter(emb[i, 0], emb[i, 1], s=100)
        plt.text(emb[i, 0], emb[i, 1], f"E{i}", fontsize=11)

    plt.title(title)
    plt.tight_layout()
    plt.show()
def run_prototype_analysis(block, title_prefix=""):
    gate_vectors = get_gate_vectors_from_block(block)

    sim_mat, sim_stats = analyze_prototype_similarity(
        gate_vectors, title_prefix=title_prefix
    )
    plot_prototype_similarity_heatmap(
        sim_mat,
        title=f"Prototype-Prototype Cosine ({title_prefix})"
    )

    norms, norm_stats = analyze_prototype_norms(
        gate_vectors, title_prefix=title_prefix
    )
    plot_prototype_norms(
        norms,
        title=f"Prototype Norms ({title_prefix})"
    )

    plot_prototype_pca(
        gate_vectors,
        title=f"Prototype PCA ({title_prefix})"
    )

    return {
        "sim_mat": sim_mat,
        "sim_stats": sim_stats,
        "norms": norms,
        "norm_stats": norm_stats,
    }

def compute_expert_centers_and_scatter(features_norm, expert_ids, min_samples=20):
    """
    features_norm: [N, D]
    expert_ids:    [N]
    """
    unique_experts = sorted(np.unique(expert_ids).tolist())
    valid_experts = []

    centers = []
    scatters = []
    counts = []

    for e in unique_experts:
        idx = (expert_ids == e)
        n = int(idx.sum())
        if n < min_samples:
            continue

        feats_e = features_norm[idx]                  # [n, D]
        center_e = feats_e.mean(axis=0, keepdims=True)
        center_e = normalize(center_e, norm="l2")[0]  # [D]

        scatter_e = np.mean(np.sum((feats_e - center_e[None, :]) ** 2, axis=1))

        valid_experts.append(e)
        centers.append(center_e)
        scatters.append(scatter_e)
        counts.append(n)

    if len(valid_experts) == 0:
        return None

    centers = np.stack(centers, axis=0)   # [E, D]
    scatters = np.array(scatters)
    counts = np.array(counts)

    return {
        "valid_experts": valid_experts,
        "centers": centers,
        "scatters": scatters,
        "counts": counts,
    }
def analyze_expert_overlap(features_norm, expert_ids, title_prefix="", min_samples=20):
    stats = compute_expert_centers_and_scatter(features_norm, expert_ids, min_samples=min_samples)
    if stats is None:
        print(f"[{title_prefix}] No experts with >= {min_samples} samples. Skip overlap analysis.")
        return None

    valid_experts = stats["valid_experts"]
    centers = stats["centers"]       # [E, D]
    scatters = stats["scatters"]     # [E]
    counts = stats["counts"]         # [E]

    center_sim = centers @ centers.T   # cosine sim, because normalized

    E = len(valid_experts)
    inter_dist = np.zeros((E, E), dtype=np.float32)
    overlap_ratio = np.zeros((E, E), dtype=np.float32)

    for i in range(E):
        for j in range(E):
            dist = np.linalg.norm(centers[i] - centers[j])
            inter_dist[i, j] = dist
            if i == j:
                overlap_ratio[i, j] = 0.0
            else:
                overlap_ratio[i, j] = (scatters[i] + scatters[j]) / max(dist ** 2, 1e-8)

    off_diag_sims = []
    off_diag_ratios = []
    for i in range(E):
        for j in range(E):
            if i != j:
                off_diag_sims.append(center_sim[i, j])
                off_diag_ratios.append(overlap_ratio[i, j])

    print(f"\n===== Expert Overlap Analysis: {title_prefix} =====")
    print("valid experts:", valid_experts)
    print("counts:", counts.tolist())
    print("center cosine similarity matrix:")
    print(center_sim)
    print(f"off-diagonal center cosine mean = {np.mean(off_diag_sims):.4f}")
    print(f"off-diagonal center cosine max  = {np.max(off_diag_sims):.4f}")
    print(f"mean overlap ratio              = {np.mean(off_diag_ratios):.4f}")
    print(f"max overlap ratio               = {np.max(off_diag_ratios):.4f}")

    return {
        "valid_experts": valid_experts,
        "counts": counts,
        "center_sim": center_sim,
        "inter_dist": inter_dist,
        "overlap_ratio": overlap_ratio,
        "off_diag_center_sim_mean": float(np.mean(off_diag_sims)),
        "off_diag_center_sim_max": float(np.max(off_diag_sims)),
        "mean_overlap_ratio": float(np.mean(off_diag_ratios)),
        "max_overlap_ratio": float(np.max(off_diag_ratios)),
    }
def plot_matrix_heatmap(mat, labels, title, cmap="coolwarm", vmin=None, vmax=None, fmt=".2f"):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)

    plt.xticks(range(len(labels)), [f"E{x}" for x in labels])
    plt.yticks(range(len(labels)), [f"E{x}" for x in labels])
    plt.title(title)
    plt.colorbar(im)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            plt.text(
                j, i, format(val, fmt),
                ha="center", va="center",
                fontsize=9,
                color="white" if (vmax is not None and abs(val - vmax) < (vmax - (vmin or 0)) * 0.35) else "black"
            )

    plt.tight_layout()
    plt.show()
def compute_expert_silhouette(features_norm, expert_ids, min_samples=20):
    unique_experts = sorted(np.unique(expert_ids).tolist())
    keep_mask = np.zeros_like(expert_ids, dtype=bool)

    valid_experts = []
    for e in unique_experts:
        idx = (expert_ids == e)
        if idx.sum() >= min_samples:
            keep_mask |= idx
            valid_experts.append(e)

    feats = features_norm[keep_mask]
    labels = expert_ids[keep_mask]

    if len(np.unique(labels)) < 2:
        return None

    score = silhouette_score(feats, labels, metric="euclidean")
    print(f"Silhouette score by expert = {score:.4f}")
    return score
def compute_knn_purity(features_norm, expert_ids, k=10, min_samples=20):
    unique_experts = sorted(np.unique(expert_ids).tolist())
    keep_mask = np.zeros_like(expert_ids, dtype=bool)

    for e in unique_experts:
        idx = (expert_ids == e)
        if idx.sum() >= min_samples:
            keep_mask |= idx

    feats = features_norm[keep_mask]
    labels = expert_ids[keep_mask]

    if len(np.unique(labels)) < 2 or feats.shape[0] <= k:
        return None

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(feats)
    indices = nbrs.kneighbors(feats, return_distance=False)   # [N, k+1]

    # 第一个是自己
    neighbor_idx = indices[:, 1:]
    neighbor_labels = labels[neighbor_idx]   # [N, k]
    same_ratio = (neighbor_labels == labels[:, None]).mean(axis=1)

    purity = float(same_ratio.mean())
    print(f"kNN purity by expert (k={k}) = {purity:.4f}")
    return purity
def print_expert_token_count_with_filter(expert_ids, min_samples=20, title_prefix=""):
    unique_ids, counts = np.unique(expert_ids, return_counts=True)

    print(f"\n===== Expert Token Count (raw): {title_prefix} =====")
    for e, c in zip(unique_ids, counts):
        print(f"Expert {int(e)}: {int(c)}")

    print(f"----- Experts kept with MIN_EXPERT_SAMPLES={min_samples} -----")
    for e, c in zip(unique_ids, counts):
        if c >= min_samples:
            print(f"Keep Expert {int(e)}: {int(c)}")
        else:
            print(f"Drop Expert {int(e)}: {int(c)}")

def get_layer_expert_outputs_from_gate_info(gate_info, seq_len):
    """
    gate_info["expert_outputs"]: [B*seq_len, E, D]
    return: [B, seq_len-1, E, D]  (remove CLS)
    """
    expert_outputs = gate_info["expert_outputs"]
    total_tokens, E, D = expert_outputs.shape
    B = total_tokens // seq_len
    expert_outputs = expert_outputs.view(B, seq_len, E, D)
    expert_outputs = expert_outputs[:, 1:, :, :]
    return expert_outputs


def get_layer_dispatch_from_gate_info(gate_info, seq_len):
    """
    return:
        dispatch_weight: [B, seq_len-1, E]
        dispatch_mask:   [B, seq_len-1, E]
    """
    dispatch_weight = gate_info["dispatch_weight"]
    dispatch_mask = gate_info["dispatch_mask"]

    total_tokens, E = dispatch_weight.shape
    B = total_tokens // seq_len

    dispatch_weight = dispatch_weight.view(B, seq_len, E)[:, 1:, :]
    dispatch_mask = dispatch_mask.view(B, seq_len, E)[:, 1:, :]
    return dispatch_weight, dispatch_mask

def analyze_same_token_active_pair_similarity(
    expert_outputs, dispatch_mask, title_prefix=""
):
    """
    expert_outputs: [B, N, E, D]
    dispatch_mask:  [B, N, E]
    """
    B, N, E, D = expert_outputs.shape
    expert_outputs = F.normalize(expert_outputs, dim=-1)

    pair_sims = []
    pair_sum = np.zeros((E, E), dtype=np.float32)
    pair_cnt = np.zeros((E, E), dtype=np.int32)

    for b in range(B):
        for n in range(N):
            active_idx = torch.nonzero(dispatch_mask[b, n] > 0, as_tuple=False).squeeze(-1)
            if active_idx.numel() < 2:
                continue

            feats = expert_outputs[b, n, active_idx, :]    # [K, D]
            sim_mat = feats @ feats.t()                    # [K, K]

            K = sim_mat.shape[0]
            for i in range(K):
                for j in range(i + 1, K):
                    ei = int(active_idx[i].item())
                    ej = int(active_idx[j].item())
                    sij = float(sim_mat[i, j].detach().cpu())

                    pair_sims.append(sij)
                    pair_sum[ei, ej] += sij
                    pair_sum[ej, ei] += sij
                    pair_cnt[ei, ej] += 1
                    pair_cnt[ej, ei] += 1

    pair_mean = np.zeros_like(pair_sum)
    valid = pair_cnt > 0
    pair_mean[valid] = pair_sum[valid] / pair_cnt[valid]

    print(f"\n===== Same-token Active Pair Similarity: {title_prefix} =====")
    print(f"num active pairs = {len(pair_sims)}")
    if len(pair_sims) > 0:
        print(f"mean pair cosine = {np.mean(pair_sims):.4f}")
        print(f"std  pair cosine = {np.std(pair_sims):.4f}")
        print(f"min  pair cosine = {np.min(pair_sims):.4f}")
        print(f"max  pair cosine = {np.max(pair_sims):.4f}")

    return {
        "pair_sims": np.array(pair_sims, dtype=np.float32),
        "pair_mean": pair_mean,
        "pair_cnt": pair_cnt,
    }


def plot_same_token_pair_hist(pair_sims, title):
    if len(pair_sims) == 0:
        print("No active expert pairs found. Skip histogram.")
        return

    plt.figure(figsize=(6, 4))
    plt.hist(pair_sims, bins=40, alpha=0.85)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def compute_expert_output_prototypes_numpy(features_bn, dispatch_weight_bn):
    """
    features_bn:        [B, N, D]
    dispatch_weight_bn: [B, N, E]
    """
    B, N, D = features_bn.shape
    E = dispatch_weight_bn.shape[-1]

    feat_flat = features_bn.reshape(B * N, D)
    weight_flat = dispatch_weight_bn.reshape(B * N, E)

    protos = []
    masses = []
    for e in range(E):
        w = weight_flat[:, e:e+1]
        mass = w.sum().clamp_min(1e-6)
        proto = (feat_flat * w).sum(dim=0) / mass
        protos.append(proto)
        masses.append(mass)

    protos = torch.stack(protos, dim=0)   # [E, D]
    masses = torch.stack(masses, dim=0)   # [E]
    protos = F.normalize(protos, dim=-1)
    sim_mat = protos @ protos.t()

    return (
        protos.detach().cpu().numpy(),
        masses.detach().cpu().numpy(),
        sim_mat.detach().cpu().numpy(),
    )

def build_expert_cluster_ratio_table(cluster_labels, expert_ids, num_experts=None):
    unique_clusters = sorted(np.unique(cluster_labels).tolist())
    if num_experts is None:
        num_experts = int(np.max(expert_ids)) + 1

    table = np.zeros((num_experts, len(unique_clusters)), dtype=np.float32)

    for ci, c in enumerate(unique_clusters):
        idx = (cluster_labels == c)
        e = expert_ids[idx]
        total = len(e)
        if total == 0:
            continue

        cnt = Counter(e.tolist())
        for ex, n in cnt.items():
            table[int(ex), ci] = n / total

    return table, unique_clusters


def plot_expert_cluster_ratio_heatmap(table, cluster_ids, title):
    plt.figure(figsize=(8, 5))
    im = plt.imshow(table, cmap="YlGnBu", vmin=0.0, vmax=1.0)

    plt.xticks(range(len(cluster_ids)), [f"C{c}" if c != -1 else "Noise" for c in cluster_ids], rotation=45)
    plt.yticks(range(table.shape[0]), [f"E{i}" for i in range(table.shape[0])])
    plt.title(title)
    plt.colorbar(im)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            plt.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()

def build_expert_role_affinity_table(role_affinity, expert_ids, num_experts=None):
    if num_experts is None:
        num_experts = int(np.max(expert_ids)) + 1

    R = role_affinity.shape[1]
    table = np.zeros((num_experts, R), dtype=np.float32)

    for e in range(num_experts):
        idx = expert_ids == e
        if idx.sum() == 0:
            continue
        table[e] = role_affinity[idx].mean(axis=0)

    return table
def plot_expert_role_affinity_heatmap(table, role_names, title):
    plt.figure(figsize=(7, 5))
    im = plt.imshow(table, cmap="coolwarm", aspect="auto", vmin=-1.0, vmax=1.0)
    plt.xticks(range(len(role_names)), role_names, rotation=30)
    plt.yticks(range(table.shape[0]), [f"E{i}" for i in range(table.shape[0])])
    plt.title(title)
    plt.colorbar(im, label="Mean cosine to role prototype")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            plt.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()

def build_expert_role_ratio_table(nearest_role_ids, expert_ids, num_roles, num_experts=None):
    if num_experts is None:
        num_experts = int(np.max(expert_ids)) + 1

    table = np.zeros((num_experts, num_roles), dtype=np.float32)
    for e in range(num_experts):
        idx = expert_ids == e
        if idx.sum() == 0:
            continue
        total = idx.sum()
        for r in range(num_roles):
            table[e, r] = np.sum(nearest_role_ids[idx] == r) / max(total, 1)
    return table

def plot_expert_role_ratio_heatmap(table, role_names, title):
    plt.figure(figsize=(7, 5))
    im = plt.imshow(table, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
    plt.xticks(range(len(role_names)), role_names, rotation=30)
    plt.yticks(range(table.shape[0]), [f"E{i}" for i in range(table.shape[0])])
    plt.title(title)
    plt.colorbar(im, label="Ratio")

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            plt.text(j, i, f"{table[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()
def plot_role_umap(embedding, nearest_role_ids, role_names, title):
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(role_names)))

    for rid, col in enumerate(colors):
        idx = nearest_role_ids == rid
        plt.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            c=[col],
            s=6,
            alpha=0.75,
            label=f"{role_names[rid]} (n={idx.sum()})"
        )

    plt.legend(markerscale=2)
    plt.title(title)
    plt.tight_layout()
    plt.show()
def analyze_free_expert_residue(role_affinity, expert_ids, role_names, free_expert_id=3, title_prefix=""):
    idx = expert_ids == free_expert_id
    if idx.sum() == 0:
        print(f"[{title_prefix}] No tokens assigned to free expert E{free_expert_id}.")
        return None

    aff = role_affinity[idx]
    mean_aff = aff.mean(axis=0)
    std_aff = aff.std(axis=0)

    print(f"\n===== Free Expert Residue Analysis: {title_prefix} =====")
    print(f"free expert id = {free_expert_id}, token count = {idx.sum()}")
    for i, name in enumerate(role_names):
        print(f"{name}: mean={mean_aff[i]:.4f}, std={std_aff[i]:.4f}")

    return {
        "count": int(idx.sum()),
        "mean_affinity": {role_names[i]: float(mean_aff[i]) for i in range(len(role_names))},
        "std_affinity": {role_names[i]: float(std_aff[i]) for i in range(len(role_names))},
    }
# =========================================================
# 6) 主分析流程
# =========================================================
def run_analysis(
    features,
    expert_ids,
    title_prefix="",
    dispatch_weight=None,   # [B, N, E] or None
    dispatch_mask=None,     # [B, N, E] or None
    expert_outputs=None,    # [B, N, E, D] or None
    role_prototypes=None,      # [R, D_teacher] or None
    role_names=None,           # list[str] or None
    role_proj_head=None,       # nn.Linear or None
    free_expert_id=3,
):
    print(f"\n===== Running analysis for: {title_prefix} =====")
    print("total tokens:", features.shape)

    features_norm, features_pca = prepare_feature_space(features, name=title_prefix)
    cluster_labels = cluster_features(features_pca, method=CLUSTER_METHOD)
    embedding = build_umap(features_norm)

    print_expert_token_count_with_filter(
        expert_ids,
        min_samples=MIN_EXPERT_SAMPLES,
        title_prefix=title_prefix
    )

    # 图 A1: UMAP by feature cluster
    plot_cluster_umap(
        embedding,
        cluster_labels,
        title=f"UMAP colored by feature clusters ({title_prefix})"
    )

    # 图 A2: UMAP by expert
    plot_expert_umap(
        embedding,
        expert_ids,
        title=f"UMAP colored by expert ({title_prefix})"
    )

    # 先停掉 PCA 2D，避免图太多
    plot_pca_2d(
        features_pca,
        cluster_labels,
        title=f"PCA 2D colored by feature clusters ({title_prefix})"
    )

    # cluster purity
    cluster_purity_analysis(cluster_labels, expert_ids)

    # overlap analysis
    overlap_stats = None
    sil_score = None
    knn_purity = None
    if DRAW_OVERLAP_ANALYSIS:
        overlap_stats = analyze_expert_overlap(
            features_norm=features_norm,
            expert_ids=expert_ids,
            title_prefix=title_prefix,
            min_samples=MIN_EXPERT_SAMPLES
        )

        sil_score = compute_expert_silhouette(
            features_norm=features_norm,
            expert_ids=expert_ids,
            min_samples=MIN_EXPERT_SAMPLES
        )

        knn_purity = compute_knn_purity(
            features_norm=features_norm,
            expert_ids=expert_ids,
            k=KNN_K,
            min_samples=MIN_EXPERT_SAMPLES
        )

        if overlap_stats is not None:
            plot_matrix_heatmap(
                overlap_stats["center_sim"],
                overlap_stats["valid_experts"],
                title=f"Expert Center Cosine Similarity ({title_prefix})",
                cmap="coolwarm",
                vmin=-1.0,
                vmax=1.0,
                fmt=".2f",
            )

            plot_matrix_heatmap(
                overlap_stats["overlap_ratio"],
                overlap_stats["valid_experts"],
                title=f"Expert Overlap Ratio ({title_prefix})",
                cmap="YlOrRd",
                vmin=0.0,
                vmax=max(1.0, overlap_stats["overlap_ratio"].max()),
                fmt=".2f",
            )

    # cluster x expert heatmap
    if DRAW_HEATMAP:
        plot_cluster_expert_heatmap(
            cluster_labels,
            expert_ids,
            title=f"Cluster x Expert Heatmap ({title_prefix})"
        )

    # =========================================================
    # Specialization audit
    # =========================================================
    specialization_results = None
    if DRAW_SPECIALIZATION_AUDIT:
        specialization_results = {}

        # 1) same-token active expert outputs similarity
        if (expert_outputs is not None) and (dispatch_mask is not None):
            pair_stats = analyze_same_token_active_pair_similarity(
                expert_outputs=expert_outputs,
                dispatch_mask=dispatch_mask,
                title_prefix=title_prefix
            )
            specialization_results["pair_stats"] = pair_stats

            if DRAW_SAME_TOKEN_PAIR_HIST:
                plot_same_token_pair_hist(
                    pair_stats["pair_sims"],
                    title=f"Same-token active-pair cosine ({title_prefix})"
                )

            plot_matrix_heatmap(
                pair_stats["pair_mean"],
                list(range(pair_stats["pair_mean"].shape[0])),
                title=f"Same-token Expert Pair Cosine Mean ({title_prefix})",
                cmap="coolwarm",
                vmin=0.0,
                vmax=1.0,
                fmt=".2f",
            )

        # 2) expert output prototype cosine
        if dispatch_weight is not None:
            B, N, E = dispatch_weight.shape
            D = features.shape[-1]
            features_bn = torch.from_numpy(features).float().view(B, N, D)

            _, masses, proto_sim = compute_expert_output_prototypes_numpy(
                features_bn, dispatch_weight
            )
            specialization_results["proto_sim"] = proto_sim
            specialization_results["proto_masses"] = masses

            if DRAW_EXPERT_OUTPUT_PROTO_HEATMAP:
                plot_matrix_heatmap(
                    proto_sim,
                    list(range(proto_sim.shape[0])),
                    title=f"Expert Output Prototype Cosine ({title_prefix})",
                    cmap="coolwarm",
                    vmin=-1.0,
                    vmax=1.0,
                    fmt=".2f",
                )
                print(f"[{title_prefix}] expert output masses = {masses}")

        # 3) expert x feature-cluster ratio
        if DRAW_EXPERT_CLUSTER_TABLE:
            table, cluster_ids = build_expert_cluster_ratio_table(
                cluster_labels=cluster_labels,
                expert_ids=expert_ids,
                num_experts=int(np.max(expert_ids)) + 1,
            )
            specialization_results["expert_cluster_table"] = table
            specialization_results["cluster_ids"] = cluster_ids

            plot_expert_cluster_ratio_heatmap(
                table,
                cluster_ids,
                title=f"Expert x Feature-cluster ratio ({title_prefix})"
            )

    overlap_results = {
        "overlap_stats": overlap_stats if DRAW_OVERLAP_ANALYSIS else None,
        "silhouette": sil_score if DRAW_OVERLAP_ANALYSIS else None,
        "knn_purity": knn_purity if DRAW_OVERLAP_ANALYSIS else None,
    }

    # =========================================================
    # Role-prototype-aware analysis
    # =========================================================
    if DRAW_ROLE_PROTO_ANALYSIS:
        if (role_prototypes is None) or (role_names is None):
            print(f"[{title_prefix}] role prototypes not provided, skip role-aware analysis.")
        elif STRICT_ROLE_SPACE and (role_proj_head is None):
            print(f"[{title_prefix}] strict role-space enabled but projection head missing, skip role-aware analysis.")
        else:
            if role_proj_head is not None:
                features_role_space = project_features_to_role_space(
                    features,
                    role_proj_head,
                    device=DEVICE,
                )
            else:
                # 只有在 feature dim 和 prototype dim 本来就一致时才允许
                if features.shape[1] != role_prototypes.shape[1]:
                    print(f"[{title_prefix}] feature dim {features.shape[1]} != role dim {role_prototypes.shape[1]}, skip role-aware analysis.")
                    features_role_space = None
                else:
                    features_role_space = normalize(features, norm="l2")

            if features_role_space is not None:
                role_affinity = compute_role_affinity(features_role_space, role_prototypes)
                nearest_role_ids, nearest_role_labels = nearest_role_assignment(role_affinity, role_names)

                if DRAW_EXPERT_ROLE_AFFINITY_HEATMAP:
                    aff_table = build_expert_role_affinity_table(role_affinity, expert_ids)
                    plot_expert_role_affinity_heatmap(
                        aff_table,
                        role_names,
                        title=f"Expert x Role Affinity ({title_prefix})"
                    )

                if DRAW_EXPERT_ROLE_RATIO_HEATMAP:
                    ratio_table = build_expert_role_ratio_table(
                        nearest_role_ids,
                        expert_ids,
                        num_roles=len(role_names),
                    )
                    plot_expert_role_ratio_heatmap(
                        ratio_table,
                        role_names,
                        title=f"Expert x Nearest-Role Ratio ({title_prefix})"
                    )

                if DRAW_NEAREST_ROLE_UMAP:
                    plot_role_umap(
                        embedding,
                        nearest_role_ids,
                        role_names,
                        title=f"UMAP colored by nearest role ({title_prefix})"
                    )

                if DRAW_FREE_EXPERT_RESIDUE:
                    free_stats = analyze_free_expert_residue(
                        role_affinity=role_affinity,
                        expert_ids=expert_ids,
                        role_names=role_names,
                        free_expert_id=free_expert_id,
                        title_prefix=title_prefix,
                    )
                    if specialization_results is None:
                        specialization_results = {}
                    specialization_results["free_expert_role_stats"] = free_stats

                if specialization_results is None:
                    specialization_results = {}
                specialization_results["role_affinity"] = role_affinity
                specialization_results["nearest_role_ids"] = nearest_role_ids
                specialization_results["nearest_role_labels"] = nearest_role_labels


    return cluster_labels, embedding, overlap_results, specialization_results 


def main():
    seed_everything(RANDOM_SEED)
    os.makedirs("analysis_outputs", exist_ok=True)

    # model, cfg = load_model(CONFIG_PATH, CKPT_PATH, device=DEVICE)
    # role_prototypes = None
    # role_names = None
    # role_proj_head = None

    # if DRAW_ROLE_PROTO_ANALYSIS:
    #     role_prototypes, role_names = load_role_prototypes(ROLE_PROTO_DIR)

    #     # 严格 role-space 需要 projection head
    #     if role_prototypes is not None:
    #         role_proj_head = load_role_projection_from_ckpt(
    #             DISTILLER_CKPT_PATH,
    #             in_dim=384,
    #             out_dim=role_prototypes.shape[1],
    #             device=DEVICE,
    #         )
    model, role_proj_head, cfg = load_stage2_bundle(
        CONFIG_PATH,
        FULL_CKPT_PATH,
        device=DEVICE
    )

    role_prototypes = None
    role_names = None

    if DRAW_ROLE_PROTO_ANALYSIS:
        role_prototypes, role_names = load_role_prototypes(ROLE_PROTO_DIR)

    last_moe_real_idx = get_last_moe_real_idx(model)

    print(f"当前最后一个 MoE block 的真实索引为: {last_moe_real_idx}")
    print(f"主分析空间: {PRIMARY_FEATURE_SPACE}")
    print(f"聚类方式: {CLUSTER_METHOD}")

    sample_imgs, sample_meta = sample_images(BASE_DIR, CATEGORIES, N_SAMPLES_PER_CAT)
    transform = build_transform()

    all_features_moe_first = []
    all_features_moe_last = []
    all_features_final = []

    all_expert_ids_first = []
    all_expert_ids_last = []

    per_image_expert_ids_first = []
    per_image_expert_ids_last = []

    per_image_features_moe_first = []
    per_image_features_moe_last = []
    per_image_features_final = []

    # 新增：specialization audit 所需缓存
    per_image_dispatch_weight_first = []
    per_image_dispatch_mask_first = []
    per_image_expert_outputs_first = []

    per_image_dispatch_weight_last = []
    per_image_dispatch_mask_last = []
    per_image_expert_outputs_last = []

    for img_idx, img_path in enumerate(sample_imgs):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        final_feats, gate_info_list, moe_feature_list = run_model_and_collect(model, img_tensor)

        seq_len = final_feats.shape[1]

        # 第一个/最后一个 MoE 层
        first_gate_info = gate_info_list[0]
        last_gate_info = gate_info_list[-1]

        first_moe_feats = moe_feature_list[0]
        last_moe_feats = moe_feature_list[-1]

        expert_id_first = get_expert_assignment_from_gate_info(first_gate_info, seq_len=seq_len)
        expert_id_last = get_expert_assignment_from_gate_info(last_gate_info, seq_len=seq_len)

        # 特征
        features_moe_first = first_moe_feats[0, 1:]   # [256, D]
        features_moe_last = last_moe_feats[0, 1:]     # [256, D]
        features_final = final_feats[0, 1:]           # [256, D]

        # ===== 从 gate_info 取 dispatch / expert_outputs =====
        dispatch_weight_first, dispatch_mask_first = get_layer_dispatch_from_gate_info(
            first_gate_info, seq_len
        )
        expert_outputs_first = get_layer_expert_outputs_from_gate_info(
            first_gate_info, seq_len
        )

        dispatch_weight_last, dispatch_mask_last = get_layer_dispatch_from_gate_info(
            last_gate_info, seq_len
        )
        expert_outputs_last = get_layer_expert_outputs_from_gate_info(
            last_gate_info, seq_len
        )
        # 汇总到总池
        all_features_moe_first.append(features_moe_first.cpu())
        all_features_moe_last.append(features_moe_last.cpu())
        all_features_final.append(features_final.cpu())

        all_expert_ids_first.append(expert_id_first.cpu())
        all_expert_ids_last.append(expert_id_last.cpu())

        # 保存每张图结果
        per_image_expert_ids_first.append(expert_id_first.cpu().numpy())
        per_image_expert_ids_last.append(expert_id_last.cpu().numpy())

        per_image_features_moe_first.append(features_moe_first.cpu().numpy())
        per_image_features_moe_last.append(features_moe_last.cpu().numpy())
        per_image_features_final.append(features_final.cpu().numpy())

        # specialization audit 所需缓存
        per_image_dispatch_weight_first.append(dispatch_weight_first.cpu())
        per_image_dispatch_mask_first.append(dispatch_mask_first.cpu())
        per_image_expert_outputs_first.append(expert_outputs_first.cpu())

        per_image_dispatch_weight_last.append(dispatch_weight_last.cpu())
        per_image_dispatch_mask_last.append(dispatch_mask_last.cpu())
        per_image_expert_outputs_last.append(expert_outputs_last.cpu())


    features_moe_first = torch.cat(all_features_moe_first, dim=0).numpy()
    features_moe_last = torch.cat(all_features_moe_last, dim=0).numpy()
    features_final = torch.cat(all_features_final, dim=0).numpy()

    expert_ids_first = torch.cat(all_expert_ids_first, dim=0).numpy()
    expert_ids_last = torch.cat(all_expert_ids_last, dim=0).numpy()

    dispatch_weight_first_all = torch.cat(per_image_dispatch_weight_first, dim=0)   # [B, N, E]
    dispatch_mask_first_all = torch.cat(per_image_dispatch_mask_first, dim=0)       # [B, N, E]
    expert_outputs_first_all = torch.cat(per_image_expert_outputs_first, dim=0)     # [B, N, E, D]

    dispatch_weight_last_all = torch.cat(per_image_dispatch_weight_last, dim=0)
    dispatch_mask_last_all = torch.cat(per_image_dispatch_mask_last, dim=0)
    expert_outputs_last_all = torch.cat(per_image_expert_outputs_last, dim=0)


    np.save("analysis_outputs/features_moe_first.npy", features_moe_first)
    np.save("analysis_outputs/features_moe_last.npy", features_moe_last)
    np.save("analysis_outputs/features_final.npy", features_final)

    print("\n================ Summary ================")
    print("First MoE-layer feature shape :", features_moe_first.shape)
    print("Last  MoE-layer feature shape :", features_moe_last.shape)
    print("Final feature shape           :", features_final.shape)
    print("Expert id first shape         :", expert_ids_first.shape)
    print("Expert id last shape          :", expert_ids_last.shape)
    print("========================================\n")

    cluster_labels_moe_first_all = None
    cluster_labels_moe_last_all = None
    cluster_labels_final_all = None
    embedding_moe_first = None
    embedding_moe_last = None
    embedding_final = None

    overlap_moe_first = None
    overlap_moe_last = None
    overlap_final = None

    spec_audit_first = None
    spec_audit_last = None
    spec_audit_final = None
    if MOE_ANALYSIS_LAYER in ["first", "both"]:
        print("\n===== Running analysis on First MoE feature space =====")
        cluster_labels_moe_first_all, embedding_moe_first, overlap_moe_first, spec_audit_first = run_analysis(
            features=features_moe_first,
            expert_ids=expert_ids_first,
            title_prefix="MoE-layer features (first MoE block / layer 9)",
            dispatch_weight=dispatch_weight_first_all,
            dispatch_mask=dispatch_mask_first_all,
            expert_outputs=expert_outputs_first_all,
            role_prototypes=role_prototypes,
            role_names=role_names,
            role_proj_head=role_proj_head,
            free_expert_id=FREE_EXPERT_ID,
        )

    if MOE_ANALYSIS_LAYER in ["last", "both"]:
        print("\n===== Running analysis on Last MoE feature space =====")
        cluster_labels_moe_last_all, embedding_moe_last, overlap_moe_last, spec_audit_last = run_analysis(
            features=features_moe_last,
            expert_ids=expert_ids_last,
            title_prefix=f"MoE-layer features (last MoE block / block {last_moe_real_idx})",
            dispatch_weight=dispatch_weight_last_all,
            dispatch_mask=dispatch_mask_last_all,
            expert_outputs=expert_outputs_last_all,
            role_prototypes=role_prototypes,
            role_names=role_names,
            role_proj_head=role_proj_head,
            free_expert_id=FREE_EXPERT_ID,
        )

    print("\n===== Running analysis on Final feature space =====")
    cluster_labels_final_all, embedding_final, overlap_final, spec_audit_final = run_analysis(
        features=features_final,
        expert_ids=expert_ids_last,
        title_prefix=f"Final features aligned to last MoE routing (block {last_moe_real_idx})",
        role_prototypes=role_prototypes,
        role_names=role_names,
        role_proj_head=role_proj_head,
        free_expert_id=FREE_EXPERT_ID,
    )

    # cluster_labels_moe_first_all = None
    # cluster_labels_moe_last_all = None
    # cluster_labels_final_all = None

    # 切回每张图自己的 cluster labels
    per_image_cluster_labels_moe_first = []
    per_image_cluster_labels_moe_last = []
    per_image_cluster_labels_final = []

    if cluster_labels_moe_first_all is not None:
        offset = 0
        for feat in per_image_features_moe_first:
            n = feat.shape[0]
            per_image_cluster_labels_moe_first.append(cluster_labels_moe_first_all[offset:offset + n])
            offset += n

    if cluster_labels_moe_last_all is not None:
        offset = 0
        for feat in per_image_features_moe_last:
            n = feat.shape[0]
            per_image_cluster_labels_moe_last.append(cluster_labels_moe_last_all[offset:offset + n])
            offset += n

    if cluster_labels_final_all is not None:
        offset = 0
        for feat in per_image_features_final:
            n = feat.shape[0]
            per_image_cluster_labels_final.append(cluster_labels_final_all[offset:offset + n])
            offset += n

    
    
    if PRIMARY_MOE_LAYER == "first":
        per_image_expert_ids = per_image_expert_ids_first
        per_image_cluster_labels_moe = per_image_cluster_labels_moe_first
    elif PRIMARY_MOE_LAYER == "last":
        per_image_expert_ids = per_image_expert_ids_last
        per_image_cluster_labels_moe = per_image_cluster_labels_moe_last
    else:
        raise ValueError("PRIMARY_MOE_LAYER must be 'first' or 'last'")

    if EXPORT_TOKEN_META:
        export_token_meta_csv(
            save_path=TOKEN_META_SAVE_PATH,
            per_image_paths=sample_imgs,
            sample_meta=sample_meta,
            per_image_expert_ids=per_image_expert_ids,
            per_image_cluster_labels_moe=per_image_cluster_labels_moe,
            per_image_cluster_labels_final=per_image_cluster_labels_final,
            grid_size=PATCH_GRID_SIZE,
        )

    # 图 C1：expert overlay
    if DRAW_OVERLAY:
        print("\n===== Drawing Expert Overlays =====")
        num_show = min(MAX_OVERLAY_IMAGES, len(sample_imgs))

        for i in range(num_show):
            plot_patch_overlay(
                image_path=sample_imgs[i],
                patch_labels=per_image_expert_ids[i],
                title=f"Expert Overlay - img {i} - {sample_meta[i]}",
                grid_size=PATCH_GRID_SIZE,
                alpha=OVERLAY_ALPHA,
                cmap_name="tab10"
            )

        print("\n===== Drawing MoE Cluster Overlays =====")
        for i in range(num_show):
            plot_patch_overlay(
                image_path=sample_imgs[i],
                patch_labels=per_image_cluster_labels_moe[i],
                title=f"MoE Cluster Overlay - img {i} - {sample_meta[i]}",
                grid_size=PATCH_GRID_SIZE,
                alpha=OVERLAY_ALPHA,
                cmap_name="tab10"
            )

        print("\n===== Drawing Final Cluster Overlays =====")
        for i in range(num_show):
            plot_patch_overlay(
                image_path=sample_imgs[i],
                patch_labels=per_image_cluster_labels_final[i],
                title=f"Final Cluster Overlay - img {i} - {sample_meta[i]}",
                grid_size=PATCH_GRID_SIZE,
                alpha=OVERLAY_ALPHA,
                cmap_name="tab10"
            )

        # =========================================================
    # Prototype analysis (first / last MoE block)
    # =========================================================
    # print("\n===== Prototype Analysis =====")
    # real_indices = sorted(model.moe_layer_map.keys())
    # first_moe_real_idx = real_indices[0]
    # last_moe_real_idx = real_indices[-1]

    # first_block = model.blocks[first_moe_real_idx]
    # last_block = model.blocks[last_moe_real_idx]

    # proto_stats_first = run_prototype_analysis(
    #     first_block,
    #     title_prefix=f"First MoE block (layer {first_moe_real_idx})"
    # )

    # proto_stats_last = run_prototype_analysis(
    #     last_block,
    #     title_prefix=f"Last MoE block (layer {last_moe_real_idx})"
    # )

if __name__ == "__main__":
    main()