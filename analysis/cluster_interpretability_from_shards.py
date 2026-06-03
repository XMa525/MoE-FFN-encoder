# analysis/cluster_interpretability_from_shards.py

import os
import json
import math
import glob
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================================================
# 基础工具
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_cluster_centers(center_path):
    """
    支持 .npy / .pt
    返回 numpy [K, D]
    """
    if center_path.endswith(".npy"):
        centers = np.load(center_path)
    elif center_path.endswith(".pt"):
        obj = torch.load(center_path, map_location="cpu")
        if isinstance(obj, dict):
            if "centers" in obj:
                centers = obj["centers"]
            elif "cluster_centers" in obj:
                centers = obj["cluster_centers"]
            else:
                raise KeyError(f"Cannot find centers in {center_path}, keys={obj.keys()}")
        else:
            centers = obj
        if torch.is_tensor(centers):
            centers = centers.cpu().numpy()
    else:
        raise ValueError(f"Unsupported center file: {center_path}")

    centers = np.asarray(centers, dtype=np.float32)
    return centers


def load_one_shard(shard_path):
    """
    读取 .npz shard
    """
    obj = np.load(shard_path, allow_pickle=True)

    possible_keys = {
        "features": ["features", "feats", "x"],
        "patch_ids": ["patch_ids", "patchid", "patch_id"],
        "paths": ["paths", "img_paths", "path"],
        "organs": ["organs", "organ"],
        "token_x": ["token_x", "xs", "xpos"],
        "token_y": ["token_y", "ys", "ypos"],
    }

    out = {}
    keys = list(obj.keys())

    for std_key, candidates in possible_keys.items():
        found = None
        for k in candidates:
            if k in obj:
                found = obj[k]
                break
        if found is None:
            raise KeyError(f"{std_key} not found in shard {shard_path}, keys={keys}")
        out[std_key] = found

    return out

def assign_clusters(feats_np, centers_np, batch_size=200000):
    """
    feats_np: [M, D]
    centers_np: [K, D]
    return cluster_ids: [M]
    """
    feats = torch.from_numpy(feats_np).float()
    centers = torch.from_numpy(centers_np).float()

    all_ids = []
    for i in range(0, len(feats), batch_size):
        chunk = feats[i:i + batch_size]
        dist = torch.cdist(chunk, centers)   # [m, K]
        cid = dist.argmin(dim=1)
        all_ids.append(cid.cpu().numpy())

    return np.concatenate(all_ids, axis=0)


def make_palette(k):
    cmap = plt.get_cmap("tab10")
    colors = []
    for i in range(k):
        c = cmap(i % 10)
        colors.append(tuple(int(255 * x) for x in c[:3]))
    return colors


def overlay_cluster_grid_on_patch(img, token_x, token_y, cluster_ids, grid_size=16, alpha=0.35):
    """
    img: PIL.Image
    token_x, token_y, cluster_ids: [N]
    """
    img = img.convert("RGB")
    w, h = img.size
    cell_w = w / grid_size
    cell_h = h / grid_size

    k = int(cluster_ids.max()) + 1
    palette = make_palette(k)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for x, y, cid in zip(token_x, token_y, cluster_ids):
        cid = int(cid)
        color = palette[cid] + (int(255 * alpha),)
        x0 = int(x * cell_w)
        y0 = int(y * cell_h)
        x1 = int((x + 1) * cell_w)
        y1 = int((y + 1) * cell_h)
        draw.rectangle([x0, y0, x1, y1], fill=color)

    merged = Image.alpha_composite(img.convert("RGBA"), overlay)
    return merged.convert("RGB")


def safe_open_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[Warn] failed to open {path}: {e}")
        return None


# =========================================================
# 主分析逻辑
# =========================================================

def analyze_cluster_interpretability(
    shard_dir,
    center_path,
    save_dir,
    grid_size=16,
    topk_patch_per_cluster=36,
    max_shards=None,
):
    ensure_dir(save_dir)
    ensure_dir(os.path.join(save_dir, "montage"))
    ensure_dir(os.path.join(save_dir, "overlay_examples"))

    centers_np = load_cluster_centers(center_path)
    num_clusters = centers_np.shape[0]

    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]

    print(f"[Info] found {len(shard_paths)} shards")

    # 全局统计
    global_token_count = np.zeros(num_clusters, dtype=np.int64)
    organ_cluster_count = defaultdict(lambda: np.zeros(num_clusters, dtype=np.int64))

    # patch 级聚合
    # patch_info[patch_id] = {
    #   path, organ, token_x, token_y, cluster_ids
    # }
    patch_info = {}

    for shard_path in tqdm(shard_paths, desc="Reading shards"):
        data = load_one_shard(shard_path)

        feats = data["features"].astype(np.float32)      # [M, D]
        patch_ids = data["patch_ids"]                    # [M]
        paths = data["paths"]                            # [M]
        organs = data["organs"]                          # [M]
        token_x = data["token_x"]                        # [M]
        token_y = data["token_y"]                        # [M]

        cluster_ids = assign_clusters(feats, centers_np)  # [M]

        # 全局 token 统计
        global_token_count += np.bincount(cluster_ids, minlength=num_clusters)

        # organ 统计
        for organ, cid in zip(organs, cluster_ids):
            organ_cluster_count[int(organ)][int(cid)] += 1

        # patch 聚合
        unique_patch_ids = np.unique(patch_ids)
        for pid in unique_patch_ids:
            mask = (patch_ids == pid)

            pid_int = int(pid)
            patch_info[pid_int] = {
                "path": str(paths[mask][0]),
                "organ": int(organs[mask][0]),
                "token_x": token_x[mask].astype(np.int32),
                "token_y": token_y[mask].astype(np.int32),
                "cluster_ids": cluster_ids[mask].astype(np.int32),
            }

    # =====================================================
    # 1) 全局 cluster 频率
    # =====================================================
    total_tokens = int(global_token_count.sum())
    global_freq = global_token_count / max(total_tokens, 1)

    freq_json = {
        "num_clusters": int(num_clusters),
        "total_tokens": total_tokens,
        "global_cluster_count": global_token_count.tolist(),
        "global_cluster_freq": global_freq.tolist(),
    }
    with open(os.path.join(save_dir, "cluster_global_stats.json"), "w") as f:
        json.dump(freq_json, f, indent=2)

    plt.figure(figsize=(8, 4))
    xs = np.arange(num_clusters)
    plt.bar(xs, global_freq)
    plt.xticks(xs, [f"C{i}" for i in range(num_clusters)])
    plt.ylabel("Frequency")
    plt.title("Global cluster frequency")
    for i, v in enumerate(global_freq):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "global_cluster_freq_bar.png"), dpi=200)
    plt.close()

    # =====================================================
    # 2) organ-wise cluster 分布
    # =====================================================
    organ_ids = sorted(organ_cluster_count.keys())
    organ_mat = []
    for organ in organ_ids:
        cnt = organ_cluster_count[organ]
        freq = cnt / max(cnt.sum(), 1)
        organ_mat.append(freq)
    organ_mat = np.stack(organ_mat, axis=0) if len(organ_mat) > 0 else np.zeros((0, num_clusters))

    np.save(os.path.join(save_dir, "organ_cluster_freq.npy"), organ_mat)

    if organ_mat.shape[0] > 0:
        plt.figure(figsize=(8, max(4, len(organ_ids) * 0.45)))
        plt.imshow(organ_mat, aspect="auto")
        plt.colorbar()
        plt.xticks(np.arange(num_clusters), [f"C{i}" for i in range(num_clusters)])
        plt.yticks(np.arange(len(organ_ids)), [str(o) for o in organ_ids])
        plt.xlabel("Cluster")
        plt.ylabel("Organ")
        plt.title("Organ-wise cluster frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "organ_cluster_heatmap.png"), dpi=200)
        plt.close()

    # =====================================================
    # 3) patch 级 composition，选出每个 cluster 最具代表性的 patch
    # =====================================================
    cluster_to_patch_candidates = defaultdict(list)
    per_patch_comp = []

    for pid, info in tqdm(patch_info.items(), desc="Aggregating patch composition"):
        cids = info["cluster_ids"]
        comp = np.bincount(cids, minlength=num_clusters).astype(np.float32)
        comp = comp / comp.sum()

        dominant_cluster = int(comp.argmax())
        dominant_ratio = float(comp[dominant_cluster])

        record = {
            "patch_id": int(pid),
            "path": info["path"],
            "organ": int(info["organ"]),
            "composition": comp.tolist(),
            "dominant_cluster": dominant_cluster,
            "dominant_ratio": dominant_ratio,
        }
        per_patch_comp.append(record)

        for cid in range(num_clusters):
            ratio = float(comp[cid])
            cluster_to_patch_candidates[cid].append({
                "patch_id": int(pid),
                "path": info["path"],
                "organ": int(info["organ"]),
                "ratio": ratio,
            })

    with open(os.path.join(save_dir, "per_patch_cluster_composition.json"), "w") as f:
        json.dump(per_patch_comp, f, indent=2)

    # =====================================================
    # 4) 每个 cluster 的代表 patch montage + overlay 示例
    # =====================================================
    for cid in range(num_clusters):
        candidates = sorted(
            cluster_to_patch_candidates[cid],
            key=lambda x: x["ratio"],
            reverse=True
        )[:topk_patch_per_cluster]

        thumbs = []
        saved_overlay = 0

        for item in candidates:
            path = item["path"]
            img = safe_open_image(path)
            if img is None:
                continue

            # montage 用原图
            thumbs.append(img.resize((128, 128)))

            # overlay 示例存前几张
            if saved_overlay < 8:
                # 找到 patch_info
                target_pid = item["patch_id"]
                info = patch_info[target_pid]
                overlay = overlay_cluster_grid_on_patch(
                    img=img,
                    token_x=info["token_x"],
                    token_y=info["token_y"],
                    cluster_ids=info["cluster_ids"],
                    grid_size=grid_size,
                    alpha=0.35,
                )
                overlay.save(os.path.join(save_dir, "overlay_examples", f"cluster{cid}_ex{saved_overlay}.png"))
                saved_overlay += 1

        # montage
        if len(thumbs) > 0:
            cols = 6
            rows = math.ceil(len(thumbs) / cols)
            canvas = Image.new("RGB", (cols * 128, rows * 128), (255, 255, 255))
            for i, thumb in enumerate(thumbs):
                x = (i % cols) * 128
                y = (i // cols) * 128
                canvas.paste(thumb, (x, y))
            canvas.save(os.path.join(save_dir, "montage", f"cluster_{cid}_montage.png"))

    print(f"[Done] results saved to: {save_dir}")


if __name__ == "__main__":
    analyze_cluster_interpretability(
        shard_dir="outputs/token_clustering_layer24/feature_shards",
        center_path="outputs/token_clustering_layer24/kmeans/kmeans_centers_k6.npy",
        save_dir="outputs/token_clustering_layer24/interpretability_k6",
        grid_size=16,
        topk_patch_per_cluster=36,
        max_shards=None,
    )