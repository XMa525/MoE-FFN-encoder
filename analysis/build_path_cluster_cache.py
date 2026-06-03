import os
import glob
import pickle
import numpy as np
import torch
from tqdm import tqdm


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
    obj = np.load(shard_path, allow_pickle=True)
    keys = obj.files

    def get_key(cands):
        for k in cands:
            if k in keys:
                return obj[k]
        raise KeyError(f"Missing keys {cands} in {shard_path}, available={keys}")

    return {
        "features": get_key(["features", "feats", "x"]),
        "patch_ids": get_key(["patch_ids", "patchid", "patch_id"]),
        "paths": get_key(["paths", "img_paths", "path"]),
        "token_x": get_key(["token_x", "xs", "xpos"]),
        "token_y": get_key(["token_y", "ys", "ypos"]),
    }


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


def build_path_cluster_cache(
    shard_dir,
    center_path,
    save_path,
    grid_size=16,
):
    centers_np = load_cluster_centers(center_path)

    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "*.npz")))
    assert len(shard_paths) > 0, f"No npz shards found in {shard_dir}"

    path_to_cluster_ids = {}

    for shard_path in tqdm(shard_paths, desc="Processing shards"):
        data = load_one_shard(shard_path)

        feats = data["features"].astype(np.float32)
        patch_ids = data["patch_ids"]
        paths = data["paths"]
        token_x = data["token_x"].astype(np.int32)
        token_y = data["token_y"].astype(np.int32)

        cluster_ids = assign_clusters(feats, centers_np)

        unique_patch_ids = np.unique(patch_ids)
        for pid in unique_patch_ids:
            mask = (patch_ids == pid)

            patch_path = str(paths[mask][0])
            xs = token_x[mask]
            ys = token_y[mask]
            cids = cluster_ids[mask]

            flat = np.full(grid_size * grid_size, -1, dtype=np.int16)
            for x, y, cid in zip(xs, ys, cids):
                flat_idx = y * grid_size + x
                flat[flat_idx] = int(cid)

            if (flat < 0).any():
                raise ValueError(f"Incomplete token grid for patch: {patch_path}")

            path_to_cluster_ids[patch_path] = flat

    with open(save_path, "wb") as f:
        pickle.dump(path_to_cluster_ids, f)

    print(f"[Done] saved cache to {save_path}, total patches={len(path_to_cluster_ids)}")


if __name__ == "__main__":
    build_path_cluster_cache(
        shard_dir="outputs/token_clustering_layer24/feature_shards",
        center_path="outputs/token_clustering_layer24/kmeans/kmeans_centers_k6.npy",
        save_path="outputs/token_clustering_layer24/path_to_cluster_ids.pkl",
        grid_size=16,
    )