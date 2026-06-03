#!/usr/bin/env python3
"""
Batch semantic annotation of pathology patches/tiles using CONCH,
with direct support for the user's SpiderPatchDataset.

What this script does
---------------------
1. Load patches either from:
   - a recursive image root,
   - a CSV/TXT image list,
   - or directly from SpiderPatchDataset samples.
2. Run CONCH zero-shot image-text matching with a small pathology prompt bank.
3. Save per-patch semantic pseudo-labels and scores.
4. Optionally save normalized image embeddings for later inspection.
5. When using SpiderPatchDataset, also save organ_id and organ_to_id mapping.

Typical use with SpiderPatchDataset
-----------------------------------
python conch_distill_dataset_semantic_labeling.py \
    --use-spider-dataset \
    --spider-root /data/maxinyu/path/to/spider_root \
    --spider-dataset-file /data/maxinyu/path/to/spider_dataset.py \
    --ckpt /data/maxinyu/WSI_WORKSPACE/CONCH/CONCH_hf/pytorch_model.bin \
    --output-dir /data/maxinyu/WSI_WORKSPACE/conch_semantic_outputs \
    --class-set tissue4 \
    --batch-size 128 \
    --num-workers 8 \
    --samples-cache-path /data/maxinyu/cache/spider_samples.pkl \
    --cluster-cache-path /data/maxinyu/cache/offline_cluster_cache.pkl \
    --enable-tissue-filter \
    --white-threshold 0.85 \
    --tissue-threshold 0.15 \
    --amp

Notes
-----
- For zero-shot semantic annotation, this script uses projected + normalized image/text embeddings.
- For your later role prototype construction, you will likely use CONCH semantic labels only as
  grouping priors, then compute role prototypes in your own teacher feature space.
- When SpiderPatchDataset is used, this script reuses dataset.samples instead of re-scanning
  directories on its own.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


DEFAULT_PROMPT_BANKS: Dict[str, Dict[str, List[str]]] = {
    "tissue4": {
        "tumor": [
            "tumor tissue in a histopathology image",
            "malignant epithelial region in a pathology patch",
            "neoplastic tumor cells in an H&E pathology tile",
            "tumor-dominant histology region",
        ],
        "stroma": [
            "stromal tissue in a histopathology image",
            "fibrous stromal region in a pathology patch",
            "connective tissue stroma in an H&E tile",
            "stroma-dominant histology region",
        ],
        "immune": [
            "immune cell infiltration in a histopathology image",
            "lymphocyte-rich inflammatory region in a pathology patch",
            "immune reactive tissue in an H&E pathology tile",
            "inflammatory infiltrate in histology",
        ],
        "ambiguous": [
            "mixed tumor stroma boundary region in a histopathology image",
            "heterogeneous pathological region with mixed tissue patterns",
            "tumor stromal interface in an H&E pathology patch",
            "ambiguous mixed tissue region in histology",
        ],
    },
    "tissue5": {
        "tumor": [
            "tumor tissue in a histopathology image",
            "malignant epithelial region in a pathology patch",
            "neoplastic tumor cells in an H&E pathology tile",
            "tumor-dominant histology region",
        ],
        "stroma": [
            "stromal tissue in a histopathology image",
            "fibrous stromal region in a pathology patch",
            "connective tissue stroma in an H&E tile",
            "stroma-dominant histology region",
        ],
        "immune": [
            "immune cell infiltration in a histopathology image",
            "lymphocyte-rich inflammatory region in a pathology patch",
            "immune reactive tissue in an H&E pathology tile",
            "inflammatory infiltrate in histology",
        ],
        "background": [
            "blank background in a pathology patch",
            "tissue-free background region in a histopathology image",
            "low-information background area in histology",
            "artifact or empty background in a pathology tile",
        ],
        "ambiguous": [
            "mixed tumor stroma boundary region in a histopathology image",
            "heterogeneous pathological region with mixed tissue patterns",
            "tumor stromal interface in an H&E pathology patch",
            "ambiguous mixed tissue region in histology",
        ],
    },
}


class PatchDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[str],
        preprocess,
        organ_ids: Optional[Sequence[int]] = None,
        canonical_paths: Optional[Sequence[str]] = None,
    ):
        self.image_paths = list(image_paths)
        self.preprocess = preprocess
        self.organ_ids = list(organ_ids) if organ_ids is not None else None
        self.canonical_paths = list(canonical_paths) if canonical_paths is not None else None

        if self.organ_ids is not None and len(self.organ_ids) != len(self.image_paths):
            raise ValueError("organ_ids length must match image_paths length")
        if self.canonical_paths is not None and len(self.canonical_paths) != len(self.image_paths):
            raise ValueError("canonical_paths length must match image_paths length")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.preprocess(image)

        organ_id = -1 if self.organ_ids is None else int(self.organ_ids[idx])
        canonical_path = path if self.canonical_paths is None else self.canonical_paths[idx]
        return image, path, organ_id, canonical_path


class SpiderDatasetProxy:
    def __init__(
        self,
        image_paths: List[str],
        organ_ids: List[int],
        canonical_paths: List[str],
        organ_to_id: Dict[str, int],
        num_samples_before_cluster_align: Optional[int] = None,
    ):
        self.image_paths = image_paths
        self.organ_ids = organ_ids
        self.canonical_paths = canonical_paths
        self.organ_to_id = organ_to_id
        self.num_samples_before_cluster_align = num_samples_before_cluster_align


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CONCH zero-shot semantic annotation for patch datasets")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-root", type=str, help="Root directory containing patch images recursively")
    src.add_argument("--image-list", type=str, help="CSV/TSV/TXT file with image paths")
    src.add_argument("--use-spider-dataset", action="store_true", help="Load samples through SpiderPatchDataset")

    parser.add_argument("--image-col", type=str, default="patch_path", help="Column name if using --image-list CSV/TSV")
    parser.add_argument("--ckpt", type=str, required=True, help="Local path to CONCH pytorch_model.bin")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")

    parser.add_argument("--class-set", type=str, default="tissue4", choices=sorted(DEFAULT_PROMPT_BANKS.keys()))
    parser.add_argument("--prompt-json", type=str, default=None, help="Optional custom prompt bank JSON")
    parser.add_argument("--model-name", type=str, default="conch_ViT-B-16")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA")
    parser.add_argument("--save-image-features", action="store_true", help="Save normalized image embeddings")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for debugging")

    parser.add_argument("--temperature", type=float, default=100.0, help="Scale applied before softmax for confidence scoring")
    parser.add_argument("--slide-id-mode", type=str, default="parent", choices=["parent", "stem", "none"],
                        help="How to derive slide_id from patch path")

    parser.add_argument("--spider-root", type=str, default=None, help="SpiderPatchDataset root")
    parser.add_argument("--spider-dataset-file", type=str, default=None, help="Path to python file defining SpiderPatchDataset")
    parser.add_argument("--spider-class-name", type=str, default="SpiderPatchDataset", help="Dataset class name")
    parser.add_argument("--cluster-cache-path", type=str, default=None, help="Optional offline cluster cache passed to SpiderPatchDataset")
    parser.add_argument("--num-patch-tokens", type=int, default=256)
    parser.add_argument("--missing-cluster-mode", type=str, default="error", choices=["error", "zeros"])
    parser.add_argument("--enable-tissue-filter", action="store_true")
    parser.add_argument("--white-threshold", type=float, default=0.85)
    parser.add_argument("--tissue-threshold", type=float, default=0.15)
    parser.add_argument("--samples-cache-path", type=str, default=None)
    parser.add_argument("--rebuild-samples-cache", action="store_true")

    return parser.parse_args()


def load_prompt_bank(args: argparse.Namespace) -> Dict[str, List[str]]:
    if args.prompt_json is not None:
        with open(args.prompt_json, "r", encoding="utf-8") as f:
            prompt_bank = json.load(f)
        if not isinstance(prompt_bank, dict) or not all(isinstance(v, list) for v in prompt_bank.values()):
            raise ValueError("Custom prompt JSON must be a dict: {class_name: [prompt1, prompt2, ...]}")
        return prompt_bank
    return DEFAULT_PROMPT_BANKS[args.class_set]


def discover_images_from_root(image_root: str) -> Tuple[List[str], Optional[List[int]], Optional[List[str]], Optional[Dict[str, int]], Dict[str, object]]:
    root = Path(image_root)
    if not root.exists():
        raise FileNotFoundError(f"image root does not exist: {root}")
    image_paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    image_paths.sort()
    meta = {"source": "image_root", "image_root": str(root)}
    return image_paths, None, None, None, meta


def discover_images_from_list(image_list: str, image_col: str) -> Tuple[List[str], Optional[List[int]], Optional[List[str]], Optional[Dict[str, int]], Dict[str, object]]:
    path = Path(image_list)
    if not path.exists():
        raise FileNotFoundError(f"image list does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if image_col not in df.columns:
            raise KeyError(f"column '{image_col}' not found in {path}. Available: {list(df.columns)}")
        image_paths = df[image_col].astype(str).tolist()
    else:
        with open(path, "r", encoding="utf-8") as f:
            image_paths = [line.strip() for line in f if line.strip()]

    meta = {"source": "image_list", "image_list": str(path), "image_col": image_col}
    return image_paths, None, None, None, meta


def load_spider_dataset_class(dataset_file: str, class_name: str):
    dataset_file = os.path.abspath(dataset_file)
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Spider dataset file does not exist: {dataset_file}")

    spec = importlib.util.spec_from_file_location("spider_dataset_module", dataset_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load python module from: {dataset_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        raise AttributeError(f"Class '{class_name}' not found in {dataset_file}")
    return getattr(module, class_name)


def discover_images_from_spider_dataset(args: argparse.Namespace) -> Tuple[List[str], List[int], List[str], Dict[str, int], Dict[str, object]]:
    if args.spider_root is None:
        raise ValueError("--spider-root is required when using --use-spider-dataset")
    if args.spider_dataset_file is None:
        raise ValueError("--spider-dataset-file is required when using --use-spider-dataset")

    dataset_cls = load_spider_dataset_class(args.spider_dataset_file, args.spider_class_name)

    dataset = dataset_cls(
        root=args.spider_root,
        transform=None,
        cluster_cache_path=args.cluster_cache_path,
        num_patch_tokens=args.num_patch_tokens,
        missing_cluster_mode=args.missing_cluster_mode,
        enable_tissue_filter=args.enable_tissue_filter,
        white_threshold=args.white_threshold,
        tissue_threshold=args.tissue_threshold,
        samples_cache_path=args.samples_cache_path,
        rebuild_samples_cache=args.rebuild_samples_cache,
    )

    image_paths: List[str] = []
    organ_ids: List[int] = []
    canonical_paths: List[str] = []

    canonicalize = getattr(dataset, "_canonicalize_path", None)
    if canonicalize is None:
        canonicalize = lambda p: os.path.normpath(p).replace("\\", "/")

    num_after_align = len(dataset.samples)
    num_before_align = getattr(dataset, "_num_samples_before_cluster_align", None)

    for path, organ_id in dataset.samples:
        image_paths.append(path)
        organ_ids.append(int(organ_id))
        canonical_paths.append(canonicalize(path))

    organ_to_id = getattr(dataset, "organ_to_id", None)
    if organ_to_id is None:
        organ_to_id = {}

    meta = {
        "source": "spider_dataset",
        "spider_root": args.spider_root,
        "spider_dataset_file": args.spider_dataset_file,
        "spider_class_name": args.spider_class_name,
        "cluster_cache_path": args.cluster_cache_path,
        "num_patch_tokens": args.num_patch_tokens,
        "missing_cluster_mode": args.missing_cluster_mode,
        "enable_tissue_filter": args.enable_tissue_filter,
        "white_threshold": args.white_threshold,
        "tissue_threshold": args.tissue_threshold,
        "samples_cache_path": args.samples_cache_path,
        "rebuild_samples_cache": args.rebuild_samples_cache,
        "num_samples_after_align": num_after_align,
        "num_samples_before_align": num_before_align,
    }

    return image_paths, organ_ids, canonical_paths, organ_to_id, meta


def derive_slide_id(image_path: str, mode: str) -> str:
    p = Path(image_path)
    if mode == "parent":
        return p.parent.name
    if mode == "stem":
        return p.stem
    return ""


def build_text_features(model, tokenizer, prompt_bank, device):
    class_names = list(prompt_bank.keys())
    class_features = []

    for class_name in class_names:
        prompts = prompt_bank[class_name]
        text_tokens = tokenize(texts=prompts, tokenizer=tokenizer).to(device)

        with torch.inference_mode():
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            class_feature = text_features.mean(dim=0, keepdim=True)
            class_feature = F.normalize(class_feature, dim=-1)

        class_features.append(class_feature)

    text_matrix = torch.cat(class_features, dim=0)
    return text_matrix, class_names, prompt_bank


def compute_entropy(prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -(prob * np.log(prob + eps)).sum(axis=1)


def top1_top2_margin(prob: np.ndarray) -> np.ndarray:
    if prob.shape[1] == 1:
        return np.ones((prob.shape[0],), dtype=np.float32)
    part = np.partition(prob, kth=-2, axis=1)
    top2 = part[:, -2:]
    return top2[:, 1] - top2[:, 0]


def collate_keep_metadata(batch):
    images, paths, organ_ids, canonical_paths = zip(*batch)
    images = torch.stack(images, dim=0)
    organ_ids = torch.as_tensor(organ_ids, dtype=torch.long)
    return images, list(paths), organ_ids, list(canonical_paths)


def ensure_parent_dir(path: Optional[str]) -> None:
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def maybe_record_spider_pre_align_count(args: argparse.Namespace) -> Optional[int]:
    if not args.use_spider_dataset:
        return None
    if args.spider_root is None or args.spider_dataset_file is None:
        return None

    if args.cluster_cache_path is None:
        return None

    dataset_cls = load_spider_dataset_class(args.spider_dataset_file, args.spider_class_name)
    dataset_no_align = dataset_cls(
        root=args.spider_root,
        transform=None,
        cluster_cache_path=None,
        num_patch_tokens=args.num_patch_tokens,
        missing_cluster_mode=args.missing_cluster_mode,
        enable_tissue_filter=args.enable_tissue_filter,
        white_threshold=args.white_threshold,
        tissue_threshold=args.tissue_threshold,
        samples_cache_path=args.samples_cache_path,
        rebuild_samples_cache=False,
    )
    return len(dataset_no_align.samples)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ensure_parent_dir(args.samples_cache_path)

    if args.use_spider_dataset:
        pre_align_count = maybe_record_spider_pre_align_count(args)
        image_paths, organ_ids, canonical_paths, organ_to_id, source_meta = discover_images_from_spider_dataset(args)
        if pre_align_count is not None:
            source_meta["num_samples_before_align"] = pre_align_count
    elif args.image_root:
        image_paths, organ_ids, canonical_paths, organ_to_id, source_meta = discover_images_from_root(args.image_root)
    else:
        image_paths, organ_ids, canonical_paths, organ_to_id, source_meta = discover_images_from_list(args.image_list, args.image_col)

    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
        if organ_ids is not None:
            organ_ids = organ_ids[: args.max_images]
        if canonical_paths is not None:
            canonical_paths = canonical_paths[: args.max_images]

    if len(image_paths) == 0:
        raise RuntimeError("No images found.")

    prompt_bank = load_prompt_bank(args)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model, preprocess = create_model_from_pretrained(args.model_name, checkpoint_path=args.ckpt)
    model.eval().to(device)
    tokenizer = get_tokenizer()

    text_features, class_names, prompt_bank = build_text_features(model, tokenizer, prompt_bank, device)

    dataset = PatchDataset(
        image_paths=image_paths,
        preprocess=preprocess,
        organ_ids=organ_ids,
        canonical_paths=canonical_paths,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_keep_metadata,
    )

    all_rows = []
    all_image_features = [] if args.save_image_features else None
    autocast_enabled = bool(args.amp and device.type == "cuda")

    for images, batch_paths, batch_organ_ids, batch_canonical_paths in tqdm(loader, desc="Annotating patches"):
        images = images.to(device, non_blocking=True)

        with torch.inference_mode():
            if autocast_enabled:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    image_features = model.encode_image(images, proj_contrast=True, normalize=True)
            else:
                image_features = model.encode_image(images, proj_contrast=True, normalize=True)

            image_features = F.normalize(image_features, dim=-1)
            logits = args.temperature * (image_features @ text_features.T)
            prob = torch.softmax(logits, dim=-1)

        prob_np = prob.detach().cpu().numpy().astype(np.float32)
        feat_np = image_features.detach().cpu().numpy().astype(np.float32)
        batch_organ_ids_np = batch_organ_ids.cpu().numpy()

        pred_idx = prob_np.argmax(axis=1)
        pred_conf = prob_np.max(axis=1)
        ent = compute_entropy(prob_np)
        margin = top1_top2_margin(prob_np)

        for i, path in enumerate(batch_paths):
            row = {
                "patch_path": path,
                "canonical_path": batch_canonical_paths[i],
                "organ_id": int(batch_organ_ids_np[i]),
                "slide_id": derive_slide_id(path, args.slide_id_mode),
                "pred_label": class_names[int(pred_idx[i])],
                "pred_confidence": float(pred_conf[i]),
                "entropy": float(ent[i]),
                "margin_top1_top2": float(margin[i]),
            }
            for c, cname in enumerate(class_names):
                row[f"score_{cname}"] = float(prob_np[i, c])
            all_rows.append(row)

        if args.save_image_features:
            all_image_features.append(feat_np)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.output_dir, "patch_semantic_predictions.csv")
    df.to_csv(csv_path, index=False)

    np.save(os.path.join(args.output_dir, "text_features.npy"), text_features.detach().cpu().numpy().astype(np.float32))
    with open(os.path.join(args.output_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "prompt_bank.json"), "w", encoding="utf-8") as f:
        json.dump(prompt_bank, f, ensure_ascii=False, indent=2)

    if organ_to_id is not None:
        with open(os.path.join(args.output_dir, "organ_to_id.json"), "w", encoding="utf-8") as f:
            json.dump(organ_to_id, f, ensure_ascii=False, indent=2)
        id_to_organ = {int(v): k for k, v in organ_to_id.items()}
        df["organ_name"] = df["organ_id"].map(lambda x: id_to_organ.get(int(x), ""))
        df.to_csv(csv_path, index=False)

    if args.save_image_features:
        image_feature_array = np.concatenate(all_image_features, axis=0)
        np.save(os.path.join(args.output_dir, "image_features.npy"), image_feature_array)

    summary = {
        "num_images": int(len(df)),
        "class_counts": df["pred_label"].value_counts().to_dict(),
        "mean_confidence": float(df["pred_confidence"].mean()),
        "mean_entropy": float(df["entropy"].mean()),
        "settings": {
            "class_set": args.class_set,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "slide_id_mode": args.slide_id_mode,
            "amp": args.amp,
            "save_image_features": args.save_image_features,
            "ckpt": args.ckpt,
            "source_meta": source_meta,
        },
    }
    with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Saved predictions to: {csv_path}")
    print("Class counts:")
    print(df["pred_label"].value_counts())
    if organ_to_id is not None and "organ_name" in df.columns:
        print("\nCounts by organ and predicted label:")
        print(pd.crosstab(df["organ_name"], df["pred_label"]))


if __name__ == "__main__":
    main()
