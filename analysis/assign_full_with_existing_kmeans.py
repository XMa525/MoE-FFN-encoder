import os
import glob
import json
import pickle
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_cluster_centers(center_path):
    """
    支持 .npy / .pt
    返回 [K, D] numpy
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
    keys = obj.files

    def get_key(cands):
        for k in cands:
            if k in keys:
                return obj[k]
        raise KeyError(f"Missing keys {cands} in {shard_path}, available={keys}")

    data = {
        "features": get_key(["features", "feats", "x"]),
        "patch_ids": get_key(["patch_ids", "patchid", "patch_id"]),
        "paths": get_key(["paths", "img_paths", "path"]),
        "organs": get_key(["organs", "organ"]),
        "token_x": get_key(["token_x", "xs", "xpos"]),
        "token_y": get_key(["token_y", "ys", "ypos"]),
    }
    return data


def assign_clusters(feats_np, centers_np, batch_size=200000):
    """
    feats_np: [M, D]
    centers_np: [K, D]
    return:
        cluster_ids: [M]
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


def build_full_assign_and_cache(
    shard_dir,
    center_path,
    save_dir,
    grid_size=16,
    use_abs_path=True,
):
    ensure_dir(save_dir)

    centers_np = load_cluster_centers(center_path)
    num_clusters = centers_np.shape[0]

    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    assert len(shard_paths) > 0, f"No shard files found in {shard_dir}"

    print(f"[Info] found {len(shard_paths)} shards")
    print(f"[Info] centers shape = {centers_np.shape}")

    global_cluster_count = np.zeros(num_clusters, dtype=np.int64)
    organ_cluster_count = defaultdict(lambda: np.zeros(num_clusters, dtype=np.int64))

    path_to_cluster_ids = {}
    per_patch_comp = []

    for shard_path in tqdm(shard_paths, desc="Assigning full shards"):
        data = load_one_shard(shard_path)

        feats = data["features"].astype(np.float32)   # [M, D]
        patch_ids = data["patch_ids"]
        paths = data["paths"]
        organs = data["organs"]
        token_x = data["token_x"].astype(np.int32)
        token_y = data["token_y"].astype(np.int32)

        cluster_ids = assign_clusters(feats, centers_np)   # [M]

        # global token count
        global_cluster_count += np.bincount(cluster_ids, minlength=num_clusters)

        # organ token count
        for organ, cid in zip(organs, cluster_ids):
            organ_cluster_count[int(organ)][int(cid)] += 1

        # patch-level aggregation
        unique_patch_ids = np.unique(patch_ids)
        for pid in unique_patch_ids:
            mask = (patch_ids == pid)

            patch_path = str(paths[mask][0])
            if use_abs_path:
                patch_path = os.path.abspath(patch_path)

            organ = int(organs[mask][0])
            xs = token_x[mask]
            ys = token_y[mask]
            cids = cluster_ids[mask]

            flat = np.full(grid_size * grid_size, -1, dtype=np.int16)
            for x, y, cid in zip(xs, ys, cids):
                flat_idx = y * grid_size + x
                flat[flat_idx] = int(cid)

            if (flat < 0).any():
                raise ValueError(f"Incomplete token grid for patch: {patch_path}")

            # save cache
            path_to_cluster_ids[patch_path] = flat

            # patch composition
            comp = np.bincount(flat, minlength=num_clusters).astype(np.float32)
            comp = comp / comp.sum()

            per_patch_comp.append({
                "path": patch_path,
                "organ": organ,
                "composition": comp.tolist(),
                "dominant_cluster": int(comp.argmax()),
                "dominant_ratio": float(comp.max()),
            })

    # save global stats
    total_tokens = int(global_cluster_count.sum())
    global_cluster_freq = (global_cluster_count / max(total_tokens, 1)).tolist()

    with open(os.path.join(save_dir, "global_cluster_freq_k6.json"), "w") as f:
        json.dump({
            "num_clusters": int(num_clusters),
            "total_tokens": total_tokens,
            "global_cluster_count": global_cluster_count.tolist(),
            "global_cluster_freq": global_cluster_freq,
        }, f, indent=2)

    # save organ stats
    organ_ids = sorted(organ_cluster_count.keys())
    organ_cluster_freq = {}
    for organ in organ_ids:
        cnt = organ_cluster_count[organ]
        freq = cnt / max(cnt.sum(), 1)
        organ_cluster_freq[int(organ)] = freq.tolist()

    with open(os.path.join(save_dir, "organ_cluster_freq.json"), "w") as f:
        json.dump(organ_cluster_freq, f, indent=2)

    # save per-patch composition
    with open(os.path.join(save_dir, "per_patch_cluster_composition.json"), "w") as f:
        json.dump(per_patch_comp, f, indent=2)

    # save full cache
    with open(os.path.join(save_dir, "path_to_cluster_ids.pkl"), "wb") as f:
        pickle.dump(path_to_cluster_ids, f)

    print(f"[Done] total patches in cache = {len(path_to_cluster_ids)}")
    print(f"[Done] total tokens assigned = {total_tokens}")
    print(f"[Done] saved to {save_dir}")


if __name__ == "__main__":
    build_full_assign_and_cache(
        shard_dir="outputs/token_clustering_layer24/feature_shards",
        center_path="outputs/token_clustering_layer24/kmeans/kmeans_centers_k6.npy",
        save_dir="outputs/token_clustering_layer24_fullassign",
        grid_size=16,
        use_abs_path=True,
    )