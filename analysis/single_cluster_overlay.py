# analysis/debug_single_cluster_overlay.py

import os
import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageStat
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================================================
# 基础工具
# =========================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_cluster_centers(center_path):
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

    return np.asarray(centers, dtype=np.float32)


def load_one_shard(shard_path):
    if shard_path.endswith(".npz"):
        obj = np.load(shard_path, allow_pickle=True)
        keys = obj.files
        get_item = lambda k: obj[k]
    elif shard_path.endswith(".pt") or shard_path.endswith(".pth"):
        obj = torch.load(shard_path, map_location="cpu")
        keys = list(obj.keys())
        get_item = lambda k: obj[k]
    else:
        raise ValueError(f"Unsupported shard format: {shard_path}")

    possible_keys = {
        "features": ["features", "feats", "x"],
        "patch_ids": ["patch_ids", "patchid", "patch_id"],
        "paths": ["paths", "img_paths", "path"],
        "organs": ["organs", "organ"],
        "token_x": ["token_x", "xs", "xpos"],
        "token_y": ["token_y", "ys", "ypos"],
    }

    out = {}
    for std_key, candidates in possible_keys.items():
        found = None
        for k in candidates:
            if k in keys:
                found = get_item(k)
                break
        if found is None:
            raise KeyError(f"{std_key} not found in {shard_path}, keys={keys}")

        if torch.is_tensor(found):
            found = found.cpu().numpy()
        out[std_key] = np.asarray(found)

    return out


def assign_clusters(feats_np, centers_np, batch_size=200000):
    feats = torch.from_numpy(feats_np).float()
    centers = torch.from_numpy(centers_np).float()

    all_ids = []
    for i in range(0, len(feats), batch_size):
        chunk = feats[i:i + batch_size]
        dist = torch.cdist(chunk, centers)
        cid = dist.argmin(dim=1)
        all_ids.append(cid.cpu().numpy())

    return np.concatenate(all_ids, axis=0)


def safe_open_image(path):
    try:
        img = Image.open(path)
        img.load()  # 强制触发解码，避免懒加载问题
        return img.convert("RGB")
    except Exception as e:
        print(f"[Warn] failed to open {path}: {e}")
        return None


def image_blank_ratio(img, white_thresh=245):
    """
    粗略估计图中接近白色区域占比，帮助判断 C5 是否是空白/边缘 patch
    """
    arr = np.asarray(img).astype(np.uint8)
    white_mask = (arr[..., 0] >= white_thresh) & (arr[..., 1] >= white_thresh) & (arr[..., 2] >= white_thresh)
    return float(white_mask.mean())


def make_palette(k):
    cmap = plt.get_cmap("tab10")
    colors = []
    for i in range(k):
        c = cmap(i % 10)
        colors.append(tuple(int(255 * x) for x in c[:3]))
    return colors


# =========================================================
# Overlay 函数
# =========================================================

def overlay_all_clusters(img, token_x, token_y, cluster_ids, grid_size=16, alpha=0.35):
    """
    把所有 cluster 都染色
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


def overlay_single_cluster(img, token_x, token_y, cluster_ids, target_cluster, grid_size=16, alpha=0.55):
    """
    只高亮目标 cluster，其它 token 不染色
    """
    img = img.convert("RGB")
    w, h = img.size
    cell_w = w / grid_size
    cell_h = h / grid_size

    # 目标 cluster 固定用红色，更容易看
    color = (255, 0, 0, int(255 * alpha))

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    hit = 0
    for x, y, cid in zip(token_x, token_y, cluster_ids):
        if int(cid) != int(target_cluster):
            continue
        hit += 1
        x0 = int(x * cell_w)
        y0 = int(y * cell_h)
        x1 = int((x + 1) * cell_w)
        y1 = int((y + 1) * cell_h)
        draw.rectangle([x0, y0, x1, y1], fill=color)

    merged = Image.alpha_composite(img.convert("RGBA"), overlay)
    return merged.convert("RGB"), hit


# =========================================================
# 主流程
# =========================================================

def debug_single_cluster_overlay(
    shard_dir,
    center_path,
    save_dir,
    target_cluster,
    grid_size=16,
    topk_patch=12,
    max_shards=None,
):
    ensure_dir(save_dir)

    centers_np = load_cluster_centers(center_path)
    num_clusters = centers_np.shape[0]

    all_files = sorted(glob.glob(os.path.join(shard_dir, "*")))
    shard_paths = [
        p for p in all_files
        if os.path.isfile(p) and (p.endswith(".npz") or p.endswith(".pt") or p.endswith(".pth"))
    ]
    if max_shards is not None:
        shard_paths = shard_paths[:max_shards]

    print("[Debug] shard_dir =", shard_dir)
    print("[Debug] num shard files =", len(shard_paths))
    print("[Debug] first 5 shard files =", shard_paths[:5])
    assert len(shard_paths) > 0, f"No shard files found under: {shard_dir}"

    # patch 级信息
    patch_records = []

    for shard_path in tqdm(shard_paths, desc="Reading shards"):
        data = load_one_shard(shard_path)

        feats = data["features"].astype(np.float32)
        patch_ids = data["patch_ids"]
        paths = data["paths"]
        organs = data["organs"]
        token_x = data["token_x"]
        token_y = data["token_y"]

        cluster_ids = assign_clusters(feats, centers_np)

        unique_patch_ids = np.unique(patch_ids)
        for pid in unique_patch_ids:
            mask = (patch_ids == pid)

            cids = cluster_ids[mask].astype(np.int32)
            comp = np.bincount(cids, minlength=num_clusters).astype(np.float32)
            comp = comp / comp.sum()

            record = {
                "patch_id": int(pid),
                "path": str(paths[mask][0]),
                "organ": int(organs[mask][0]),
                "token_x": token_x[mask].astype(np.int32),
                "token_y": token_y[mask].astype(np.int32),
                "cluster_ids": cids,
                "composition": comp,
                "target_ratio": float(comp[target_cluster]),
                "dominant_cluster": int(comp.argmax()),
            }
            patch_records.append(record)

    # 按 target cluster 占比排序
    patch_records = sorted(patch_records, key=lambda x: x["target_ratio"], reverse=True)

    # 取 topk
    selected = patch_records[:topk_patch]
    print(f"[Debug] selected top-{topk_patch} patches for cluster {target_cluster}")

    debug_meta = []

    for i, rec in enumerate(selected):
        path = rec["path"]
        img = safe_open_image(path)

        item_dir = os.path.join(save_dir, f"cluster{target_cluster}_sample{i:02d}")
        ensure_dir(item_dir)

        meta = {
            "index": i,
            "path": path,
            "organ": int(rec["organ"]),
            "patch_id": int(rec["patch_id"]),
            "target_cluster": int(target_cluster),
            "target_ratio": float(rec["target_ratio"]),
            "dominant_cluster": int(rec["dominant_cluster"]),
            "composition": [float(x) for x in rec["composition"]],
            "image_open_ok": img is not None,
        }

        if img is None:
            with open(os.path.join(item_dir, "debug.json"), "w") as f:
                json.dump(meta, f, indent=2)
            debug_meta.append(meta)
            continue

        # 原图信息
        meta["image_mode"] = img.mode
        meta["image_size"] = list(img.size)
        meta["blank_ratio_white245"] = image_blank_ratio(img, white_thresh=245)

        # 保存 raw
        raw_path = os.path.join(item_dir, "raw.png")
        img.save(raw_path)

        # 保存所有 cluster overlay
        all_overlay = overlay_all_clusters(
            img=img,
            token_x=rec["token_x"],
            token_y=rec["token_y"],
            cluster_ids=rec["cluster_ids"],
            grid_size=grid_size,
            alpha=0.35
        )
        all_overlay_path = os.path.join(item_dir, "all_clusters_overlay.png")
        all_overlay.save(all_overlay_path)

        # 保存单 cluster overlay
        single_overlay, hit = overlay_single_cluster(
            img=img,
            token_x=rec["token_x"],
            token_y=rec["token_y"],
            cluster_ids=rec["cluster_ids"],
            target_cluster=target_cluster,
            grid_size=grid_size,
            alpha=0.55
        )
        single_overlay_path = os.path.join(item_dir, "single_cluster_overlay.png")
        single_overlay.save(single_overlay_path)

        meta["target_token_count"] = int(hit)
        meta["raw_path"] = raw_path
        meta["all_overlay_path"] = all_overlay_path
        meta["single_overlay_path"] = single_overlay_path

        with open(os.path.join(item_dir, "debug.json"), "w") as f:
            json.dump(meta, f, indent=2)

        debug_meta.append(meta)

    with open(os.path.join(save_dir, f"cluster{target_cluster}_summary.json"), "w") as f:
        json.dump(debug_meta, f, indent=2)

    print(f"[Done] saved debug outputs to: {save_dir}")


if __name__ == "__main__":
    debug_single_cluster_overlay(
        shard_dir="outputs/token_clustering_layer24/feature_shards",
        center_path="outputs/token_clustering_layer24/kmeans/kmeans_centers_k6.npy",
        save_dir="outputs/token_clustering_layer24/debug_single_cluster_c5",
        target_cluster=5,   # 这里改成你要看的 cluster
        grid_size=16,
        topk_patch=12,
        max_shards=None,
    )