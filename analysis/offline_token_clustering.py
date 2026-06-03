import os
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import random
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from sklearn.cluster import MiniBatchKMeans

from models.distill_teacher.virchow2 import Virchow2FeatureExtractor
from distillation.dataset.spider_dataset import SpiderPatchDataset


# =========================================================
# 配置
# =========================================================
class Args:
    # 数据
    root = "../data/raw"
    output_dir = "outputs/token_clustering_layer24"
    batch_size = 64
    num_workers = 8

    # 第一阶段探索版：只先跑一部分 patch
    max_images = 10**9

    # feature
    target_layer = 24
    image_size = 224
    patch_size = 14
    grid_size = image_size // patch_size   # 16
    token_dim = 1280

    # 聚类
    k = 6
    sample_tokens_for_kmeans = 200000
    kmeans_batch_size = 4096
    random_seed = 42

    # 存储
    shard_size_images = 5000   # 调大，减少频繁写盘

    # 背景过滤
    white_threshold = 0.85
    tissue_threshold = 0.15

    # 控制流程
    extract_only = True
    assign_only = False


# =========================================================
# 基础工具
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_transform():
    # 与你蒸馏验证/teacher推理一致
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])


def simple_tissue_ratio(images: torch.Tensor, white_threshold: float = 0.85) -> torch.Tensor:
    """
    images: [B, 3, H, W], float in [0,1]
    返回每张 patch 的非白区域占比
    """
    gray = images.mean(dim=1)  # [B,H,W]
    non_white = (gray < white_threshold).float()
    ratio = non_white.flatten(1).mean(dim=1)
    return ratio


# =========================================================
# 精确提取 teacher 第 target_layer 层 patch tokens
# =========================================================
@torch.no_grad()
def extract_teacher_layer_tokens(teacher_model, images: torch.Tensor, target_layer: int) -> torch.Tensor:
    """
    精确提取 teacher 第 target_layer 层的 patch tokens
    返回:
        [B, 256, D]
    """
    model = teacher_model
    B = images.shape[0]

    # 1) patch embedding
    x = model.patch_embed(images)   # [B, num_patches, D]

    # 2) 位置编码 + prefix tokens
    # timm 的 _pos_embed 会自己把 cls/reg token 拼进去
    if hasattr(model, "_pos_embed"):
        x = model._pos_embed(x)
    else:
        # fallback：只有没有 _pos_embed 时，才手动拼 prefix
        cls_token = model.cls_token.expand(B, -1, -1)
        num_reg = getattr(model, "reg_tokens", 0)

        if num_reg > 0 and hasattr(model, "reg_token") and model.reg_token is not None:
            reg_tokens = model.reg_token.expand(B, -1, -1)
            x = torch.cat([cls_token, reg_tokens, x], dim=1)
        else:
            x = torch.cat([cls_token, x], dim=1)

        if hasattr(model, "pos_embed") and model.pos_embed is not None:
            x = x + model.pos_embed[:, :x.shape[1], :]
        if hasattr(model, "pos_drop"):
            x = model.pos_drop(x)

    # 3) patch drop / norm_pre
    if hasattr(model, "patch_drop"):
        x = model.patch_drop(x)
    if hasattr(model, "norm_pre"):
        x = model.norm_pre(x)

    # 4) 逐层前向到 target_layer
    assert hasattr(model, "blocks"), "teacher_model 没有 blocks"
    assert len(model.blocks) >= target_layer, \
        f"model.blocks层数不足: {len(model.blocks)} < {target_layer}"

    for i, blk in enumerate(model.blocks, start=1):
        x = blk(x)
        if i == target_layer:
            break

    # 5) 自动反推 prefix token 数，不硬编码
    num_patches = model.patch_embed.num_patches
    total_tokens = x.shape[1]
    num_prefix = total_tokens - num_patches

    # print(
    #     f"[Debug] layer{target_layer}: total_tokens={total_tokens}, "
    #     f"num_patches={num_patches}, inferred_prefix={num_prefix}"
    # )

    assert num_prefix >= 0, \
        f"prefix token 数非法: total={total_tokens}, num_patches={num_patches}"

    patch_tokens = x[:, num_prefix:, :]

    assert patch_tokens.shape[1] == num_patches, \
        f"patch token数仍不匹配: got {patch_tokens.shape[1]}, expected {num_patches}"

    return patch_tokens


# =========================================================
# 特征提取 + 分片保存
# =========================================================
@torch.no_grad()
def extract_and_save_shards(args: Args, device: str):
    ensure_dir(args.output_dir)
    shard_dir = os.path.join(args.output_dir, "feature_shards")
    ensure_dir(shard_dir)

    transform = build_transform()
    dataset = SpiderPatchDataset(root=args.root, transform=transform)

    if args.max_images is None:
        effective_len = len(dataset)
    else:
        effective_len = min(len(dataset), args.max_images)

    dataset.samples = dataset.samples[:effective_len]
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    teacher_wrapper = Virchow2FeatureExtractor(device=device)
    teacher_model = teacher_wrapper.model
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    print(f"[Info] dataset size = {len(dataset)}")
    print(f"[Info] output dir = {args.output_dir}")
    print(f"[Info] max_images = {args.max_images}")

    shard_features = []
    shard_patch_ids = []
    shard_paths = []
    shard_organs = []
    shard_token_x = []
    shard_token_y = []

    kept_patches = 0
    skipped_patches = 0
    shard_idx = 0
    running_idx = 0

    for batch in tqdm(loader, desc="Extract teacher layer24 tokens"):
        if running_idx >= args.max_images:
            print(f"[Info] Reached max_images={args.max_images}, stop extraction.")
            break

        images, organs,_ = batch
        bsz = images.size(0)

        # 处理最后一个超出 max_images 的 batch
        if args.max_images is not None:
            if running_idx >= args.max_images:
                print(f"[Info] Reached max_images={args.max_images}, stop extraction.")
                break

            remain = args.max_images - running_idx
            if remain <= 0:
                break
            if bsz > remain:
                images = images[:remain]
                organs = organs[:remain]
                bsz = remain

        images = images.to(device, non_blocking=True)

        # patch path 从 dataset.samples 直接取，不改原dataset
        batch_paths = [dataset.samples[running_idx + i][0] for i in range(bsz)]

        # ===== patch-level 背景过滤 =====
        tissue_ratio = simple_tissue_ratio(images, white_threshold=args.white_threshold)
        keep_mask = tissue_ratio >= args.tissue_threshold

        if keep_mask.sum().item() == 0:
            skipped_patches += bsz
            running_idx += bsz
            continue

        keep_mask_cpu = keep_mask.cpu()

        kept_images = images[keep_mask]
        kept_organs = organs[keep_mask_cpu].numpy()
        keep_list = keep_mask_cpu.tolist()
        kept_paths = [p for p, k in zip(batch_paths, keep_list) if k]
        kept_ids = [running_idx + i for i, k in enumerate(keep_list) if k]

        tokens = extract_teacher_layer_tokens(
            teacher_model=teacher_model,
            images=kept_images,
            target_layer=args.target_layer
        )  # [B, N, D]

        B, N, D = tokens.shape
        assert N == args.grid_size * args.grid_size, \
            f"token数不匹配: got {N}, expected {args.grid_size * args.grid_size}"
        assert D == args.token_dim, \
            f"特征维度不匹配: got {D}, expected {args.token_dim}"

        tokens = tokens.float().cpu().numpy()

        ys, xs = np.divmod(np.arange(N), args.grid_size)

        for b in range(B):
            patch_tokens = tokens[b]  # [N,D]
            patch_id = kept_ids[b]

            shard_features.append(patch_tokens)
            shard_patch_ids.append(np.full((N,), patch_id, dtype=np.int64))
            shard_paths.append(np.array([kept_paths[b]] * N, dtype=object))
            shard_organs.append(np.full((N,), kept_organs[b], dtype=np.int16))
            shard_token_x.append(xs.astype(np.int16))
            shard_token_y.append(ys.astype(np.int16))

        kept_patches += B
        skipped_patches += (bsz - B)
        running_idx += bsz

        # 到分片大小就保存
        if kept_patches > 0 and kept_patches % args.shard_size_images < B:
            save_one_shard(
                shard_dir=shard_dir,
                shard_idx=shard_idx,
                feat_list=shard_features,
                patchid_list=shard_patch_ids,
                path_list=shard_paths,
                organ_list=shard_organs,
                x_list=shard_token_x,
                y_list=shard_token_y,
            )
            shard_features, shard_patch_ids, shard_paths = [], [], []
            shard_organs, shard_token_x, shard_token_y = [], [], []
            shard_idx += 1

    # flush last shard
    if len(shard_features) > 0:
        save_one_shard(
            shard_dir=shard_dir,
            shard_idx=shard_idx,
            feat_list=shard_features,
            patchid_list=shard_patch_ids,
            path_list=shard_paths,
            organ_list=shard_organs,
            x_list=shard_token_x,
            y_list=shard_token_y,
        )

    meta = {
        "dataset_size": len(dataset),
        "max_images": args.max_images,
        "kept_patches": kept_patches,
        "skipped_patches": skipped_patches,
        "target_layer": args.target_layer,
        "grid_size": args.grid_size,
        "token_dim": args.token_dim,
        "white_threshold": args.white_threshold,
        "tissue_threshold": args.tissue_threshold,
        "shard_size_images": args.shard_size_images,
    }
    with open(os.path.join(args.output_dir, "extract_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[Done] feature extraction finished.")
    print(json.dumps(meta, indent=2))


def save_one_shard(
    shard_dir,
    shard_idx,
    feat_list,
    patchid_list,
    path_list,
    organ_list,
    x_list,
    y_list
):
    feats = np.concatenate(feat_list, axis=0)        # [num_tokens, D]
    patch_ids = np.concatenate(patchid_list, axis=0) # [num_tokens]
    paths = np.concatenate(path_list, axis=0)        # [num_tokens], object
    organs = np.concatenate(organ_list, axis=0)
    token_x = np.concatenate(x_list, axis=0)
    token_y = np.concatenate(y_list, axis=0)

    save_path = os.path.join(shard_dir, f"shard_{shard_idx:04d}.npz")
    np.savez(
        save_path,
        features=feats.astype(np.float16),  # 节省磁盘与写盘时间
        patch_ids=patch_ids.astype(np.int64),
        paths=paths,
        organs=organs.astype(np.int16),
        token_x=token_x.astype(np.int16),
        token_y=token_y.astype(np.int16),
    )
    print(f"[Saved] {save_path} | tokens={len(patch_ids)}")


# =========================================================
# KMeans 训练
# =========================================================
def list_shards(shard_dir: str) -> List[str]:
    return sorted(str(p) for p in Path(shard_dir).glob("shard_*.npz"))


def reservoir_sample_features(shard_paths: List[str], sample_size: int, seed: int = 42) -> np.ndarray:
    """
    从所有 shard 中抽样 token feature
    """
    rng = np.random.default_rng(seed)
    sampled = None
    total_seen = 0

    for sp in tqdm(shard_paths, desc="Sampling features for KMeans"):
        data = np.load(sp, allow_pickle=True)
        feats = data["features"].astype(np.float32)  # 从 float16 恢复到 float32 做聚类

        if sampled is None:
            d = feats.shape[1]
            sampled = np.zeros((sample_size, d), dtype=np.float32)

        for row in feats:
            if total_seen < sample_size:
                sampled[total_seen] = row
            else:
                j = rng.integers(0, total_seen + 1)
                if j < sample_size:
                    sampled[j] = row
            total_seen += 1

    if total_seen < sample_size:
        sampled = sampled[:total_seen]

    return sampled


def fit_kmeans(args: Args):
    print("[Info] Start fitting kmeans...")
    shard_dir = os.path.join(args.output_dir, "feature_shards")
    kmeans_dir = os.path.join(args.output_dir, "kmeans")
    ensure_dir(kmeans_dir)

    shard_paths = list_shards(shard_dir)
    if len(shard_paths) == 0:
        raise RuntimeError(f"No shard files found in {shard_dir}")

    sampled = reservoir_sample_features(
        shard_paths=shard_paths,
        sample_size=args.sample_tokens_for_kmeans,
        seed=args.random_seed
    )

    sampled = sampled / (np.linalg.norm(sampled, axis=1, keepdims=True) + 1e-8)

    print(f"[Info] sampled tokens for kmeans: {sampled.shape}")

    kmeans = MiniBatchKMeans(
        n_clusters=args.k,
        batch_size=args.kmeans_batch_size,
        random_state=args.random_seed,
        n_init=10,
        verbose=1
    )
    kmeans.fit(sampled)

    joblib.dump(kmeans, os.path.join(kmeans_dir, f"kmeans_k{args.k}.pkl"))
    np.save(os.path.join(kmeans_dir, f"kmeans_centers_k{args.k}.npy"), kmeans.cluster_centers_)
    print(f"[Done] saved kmeans model to {kmeans_dir}")
    print("[Info] KMeans saved successfully.")


# =========================================================
# 全量 token 打 cluster + patch 直方图
# =========================================================
def assign_clusters_and_build_hist(args: Args):
    shard_dir = os.path.join(args.output_dir, "feature_shards")
    assign_dir = os.path.join(args.output_dir, "assignments")
    kmeans_dir = os.path.join(args.output_dir, "kmeans")
    ensure_dir(assign_dir)

    kmeans = joblib.load(os.path.join(kmeans_dir, f"kmeans_k{args.k}.pkl"))
    shard_paths = list_shards(shard_dir)

    global_cluster_count = np.zeros((args.k,), dtype=np.int64)
    patch_hist: Dict[int, np.ndarray] = {}
    patch_meta: Dict[int, Dict[str, object]] = {}

    for sp in tqdm(shard_paths, desc="Assigning clusters"):
        data = np.load(sp, allow_pickle=True)
        feats = data["features"].astype(np.float32)
        patch_ids = data["patch_ids"]
        paths = data["paths"]
        organs = data["organs"]
        token_x = data["token_x"]
        token_y = data["token_y"]

        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        cluster_ids = kmeans.predict(feats)

        # 保存 token 级 assignment
        out_path = os.path.join(assign_dir, Path(sp).stem + f"_cluster_k{args.k}.npz")
        np.savez_compressed(
            out_path,
            patch_ids=patch_ids.astype(np.int64),
            paths=paths,
            organs=organs.astype(np.int16),
            token_x=token_x.astype(np.int16),
            token_y=token_y.astype(np.int16),
            cluster_ids=cluster_ids.astype(np.int16),
        )

        # 全局频率
        binc = np.bincount(cluster_ids, minlength=args.k)
        global_cluster_count += binc

        # patch histogram
        unique_patch_ids = np.unique(patch_ids)
        for pid in unique_patch_ids:
            mask = (patch_ids == pid)
            hist = np.bincount(cluster_ids[mask], minlength=args.k).astype(np.int64)
            if pid not in patch_hist:
                patch_hist[pid] = hist
                patch_meta[pid] = {
                    "path": str(paths[mask][0]),
                    "organ": int(organs[mask][0])
                }
            else:
                patch_hist[pid] += hist

    total_tokens = int(global_cluster_count.sum())
    freq = (global_cluster_count / max(total_tokens, 1)).tolist()
    with open(os.path.join(assign_dir, f"global_cluster_freq_k{args.k}.json"), "w") as f:
        json.dump({
            "k": args.k,
            "global_cluster_count": global_cluster_count.tolist(),
            "global_cluster_freq": freq,
            "total_tokens": total_tokens,
        }, f, indent=2)

    # 保存 patch histogram
    csv_path = os.path.join(assign_dir, f"patch_cluster_hist_k{args.k}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = ["patch_id", "path", "organ"] + [f"cluster_{i}" for i in range(args.k)]
        f.write(",".join(header) + "\n")
        for pid in sorted(patch_hist.keys()):
            meta = patch_meta[pid]
            hist = patch_hist[pid]
            row = [str(pid), meta["path"], str(meta["organ"])] + [str(int(x)) for x in hist]
            f.write(",".join(row) + "\n")

    print("[Done] cluster assignment & patch histogram saved.")


# =========================================================
# main
# =========================================================
def main():
    args = Args()
    seed_everything(args.random_seed)
    ensure_dir(args.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] device = {device}")

    if not args.assign_only:
        extract_and_save_shards(args, device)

    if not args.extract_only:
        fit_kmeans(args)
        assign_clusters_and_build_hist(args)

    print("[All Done]")


if __name__ == "__main__":
    main()