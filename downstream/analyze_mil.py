import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from torchmil.models import ABMIL, DSMIL


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_auc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def compute_binary_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    auc = safe_auc(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": float(auc),
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    rows = []
    for th in thresholds:
        m = compute_binary_metrics(y_true, y_prob, threshold=float(th))
        m["threshold"] = float(th)
        rows.append(m)

    df = pd.DataFrame(rows)
    best_f1_idx = df["f1"].idxmax()
    best_youden_idx = (df["sensitivity"] + df["specificity"] - 1.0).idxmax()

    return {
        "table": df,
        "best_f1": df.loc[best_f1_idx].to_dict(),
        "best_youden": df.loc[best_youden_idx].to_dict(),
    }


# =========================================================
# Dataset
# =========================================================
class BagFeatureDataset(Dataset):
    """
    每个样本 = 一张 slide 的 bag features
    读取 .pt 格式：
    {
        "features": Tensor[N, D],
        "coords": Tensor[N, 2],   # optional
        "label": int,
        "slide_id": str,
        ...
    }
    """
    def __init__(
        self,
        slides_csv: str,
        feature_dir: str,
        split: str,
        max_instances: int = None,
        shuffle_instances: bool = False,
    ):
        self.df = pd.read_csv(slides_csv)
        if "label" not in self.df.columns:
            if "slide_binary_label" in self.df.columns:
                self.df["label"] = self.df["slide_binary_label"]
            else:
                raise ValueError("slides_csv 需要包含 label 或 slide_binary_label 列")

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.feature_dir = feature_dir
        self.max_instances = max_instances
        self.shuffle_instances = shuffle_instances

        self.samples = []
        for _, row in self.df.iterrows():
            slide_id = row["slide_id"]
            label = int(row["label"])
            bag_path = os.path.join(self.feature_dir, f"{slide_id}.pt")
            if not os.path.exists(bag_path):
                print(f"[Warn] missing bag file, skip: {bag_path}")
                continue

            self.samples.append({
                "slide_id": slide_id,
                "label": label,
                "bag_path": bag_path,
            })

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found for split={split}")

        first_obj = torch.load(self.samples[0]["bag_path"], map_location="cpu")
        self.feat_dim = int(first_obj["features"].shape[1])

        print(f"[{split}] num_samples={len(self.samples)}, feat_dim={self.feat_dim}")

    def __len__(self):
        return len(self.samples)

    def _maybe_subsample_instances(self, feats: torch.Tensor):
        n = feats.shape[0]
        if self.max_instances is None or n <= self.max_instances:
            keep_idx = torch.arange(n)
            return feats, keep_idx

        if self.shuffle_instances:
            keep_idx = torch.randperm(n)[:self.max_instances]
        else:
            keep_idx = torch.arange(self.max_instances)

        return feats[keep_idx], keep_idx

    def __getitem__(self, idx):
        sample = self.samples[idx]
        obj = torch.load(sample["bag_path"], map_location="cpu")

        feats = obj["features"].float()   # [N, D]
        coords = obj["coords"].float() if "coords" in obj else None
        feats, keep_idx = self._maybe_subsample_instances(feats)

        if coords is not None:
            coords = coords[keep_idx]

        return {
            "features": feats,
            "coords": coords,
            "label": int(sample["label"]),
            "slide_id": sample["slide_id"],
            "bag_path": sample["bag_path"],
        }


def bag_collate_fn(batch):
    return {
        "features": [x["features"] for x in batch],
        "coords": [x["coords"] for x in batch],
        "labels": torch.tensor([x["label"] for x in batch], dtype=torch.long),
        "slide_ids": [x["slide_id"] for x in batch],
        "bag_paths": [x["bag_path"] for x in batch],
    }


# =========================================================
# Model wrappers
# =========================================================
class ABMILWrapper(nn.Module):
    def __init__(self, in_dim: int, device: str):
        super().__init__()
        self.model = ABMIL(
            in_shape=(in_dim,),
            att_dim=256,
            att_act="tanh",
            gated=False
        )
        self.device = device
        self.to(device)

    def forward(self, bag_feats: torch.Tensor):
        out = self.model(bag_feats)

        if isinstance(out, torch.Tensor):
            return out

        if isinstance(out, dict):
            for key in ["logits", "pred", "scores", "output"]:
                if key in out:
                    return out[key]

        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    return item

        raise TypeError(f"Unsupported ABMIL output type: {type(out)}")


class DSMILWrapper(nn.Module):
    def __init__(self, in_dim: int, device: str):
        super().__init__()
        self.model = DSMIL(
            in_shape=(in_dim,),
            att_dim=128,
            nonlinear_q=False,
            nonlinear_v=False,
            dropout=0.0,
        )
        self.device = device
        self.to(device)

    def forward(self, bag_feats: torch.Tensor):
        if bag_feats.ndim == 2:
            bag_feats = bag_feats.unsqueeze(0)

        out = self.model(bag_feats)

        if isinstance(out, torch.Tensor):
            return out

        if isinstance(out, dict):
            for key in ["logits", "pred", "scores", "output", "Y_pred"]:
                if key in out:
                    return out[key]

        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    return item

        raise TypeError(f"Unsupported DSMIL output type: {type(out)}")


class MeanPoolMIL(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1, device: str = "cuda"):
        super().__init__()
        self.classifier = nn.Linear(in_dim, out_dim)
        self.device = device
        self.to(device)

    def forward(self, bag_feats: torch.Tensor):
        if bag_feats.ndim == 3:
            bag_repr = bag_feats.mean(dim=1)
        elif bag_feats.ndim == 2:
            bag_repr = bag_feats.mean(dim=0, keepdim=True)
        else:
            raise ValueError(f"Unsupported bag_feats shape: {bag_feats.shape}")
        logits = self.classifier(bag_repr)
        return logits


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 0:
        logits = logits.view(1)

    if logits.ndim == 1:
        return torch.sigmoid(logits)

    if logits.ndim == 2 and logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])

    if logits.ndim == 2 and logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1]

    raise ValueError(f"Unsupported logits shape: {logits.shape}")


def build_model(mil_model: str, in_dim: int, device: str):
    if mil_model == "abmil":
        return ABMILWrapper(in_dim=in_dim, device=device)
    elif mil_model == "dsmil":
        return DSMILWrapper(in_dim=in_dim, device=device)
    elif mil_model == "meanpool":
        return MeanPoolMIL(in_dim=in_dim, out_dim=1, device=device)
    else:
        raise ValueError(f"Unknown mil_model: {mil_model}")


# =========================================================
# Feature summarization
# =========================================================
def summarize_bag_features(feats: np.ndarray) -> Dict[str, np.ndarray]:
    """
    feats: [N, D]
    返回几种不依赖 MIL 的 bag summary
    """
    mean_feat = feats.mean(axis=0)
    max_feat = feats.max(axis=0)
    std_feat = feats.std(axis=0)
    mean_max = np.concatenate([mean_feat, max_feat], axis=0)
    mean_std = np.concatenate([mean_feat, std_feat], axis=0)

    return {
        "mean": mean_feat,
        "max": max_feat,
        "std": std_feat,
        "mean_max": mean_max,
        "mean_std": mean_std,
    }


def compute_centroid_evidence(
    feats: np.ndarray,
    pos_centroid: np.ndarray,
    neg_centroid: np.ndarray,
):
    """
    用 instance 到正负 centroid 的余弦相似度差，
    粗略衡量 bag 内“正类证据是否稀疏”
    """
    pos_sim = cosine_similarity(feats, pos_centroid[None, :]).reshape(-1)
    neg_sim = cosine_similarity(feats, neg_centroid[None, :]).reshape(-1)
    delta = pos_sim - neg_sim  # >0 越像正类

    out = {
        "inst_delta_mean": float(delta.mean()),
        "inst_delta_std": float(delta.std()),
        "inst_delta_max": float(delta.max()),
        "inst_delta_top5_mean": float(np.sort(delta)[-min(5, len(delta)):].mean()),
        "inst_delta_top10_mean": float(np.sort(delta)[-min(10, len(delta)):].mean()),
        "inst_pos_like_ratio": float((delta > 0).mean()),
    }
    return out


# =========================================================
# Inference
# =========================================================
@torch.no_grad()
def infer_split(model, loader, device="cuda"):
    model.eval()

    rows = []
    for batch in tqdm(loader, desc="Inference", leave=False):
        bags = batch["features"]
        labels = batch["labels"].numpy().tolist()
        slide_ids = batch["slide_ids"]
        bag_paths = batch["bag_paths"]

        for i, bag_feats in enumerate(bags):
            bag_feats = bag_feats.to(device)
            bag_input = bag_feats.unsqueeze(0)  # [1, N, D]

            logits = model(bag_input)
            prob = float(logits_to_prob(logits).detach().cpu().numpy()[0])

            feats_np = bag_feats.detach().cpu().numpy()
            summaries = summarize_bag_features(feats_np)

            row = {
                "slide_id": slide_ids[i],
                "bag_path": bag_paths[i],
                "label": int(labels[i]),
                "y_prob": prob,
                "num_instances": int(feats_np.shape[0]),
            }

            for k, v in summaries.items():
                row[f"bagfeat_{k}"] = v

            rows.append(row)

    return rows


# =========================================================
# Probe
# =========================================================
def train_and_eval_probe(train_X, train_y, eval_X, eval_y, threshold=0.5):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ))
    ])
    clf.fit(train_X, train_y)
    eval_prob = clf.predict_proba(eval_X)[:, 1]
    metrics = compute_binary_metrics(eval_y, eval_prob, threshold=threshold)
    return clf, eval_prob, metrics


# =========================================================
# Visualization / analysis
# =========================================================
def save_score_overlap_plot(df: pd.DataFrame, out_path: str, title: str):
    pos = df[df["label"] == 1]["y_prob"].values
    neg = df[df["label"] == 0]["y_prob"].values

    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, 1, 31)
    plt.hist(neg, bins=bins, alpha=0.6, label="negative", density=True)
    plt.hist(pos, bins=bins, alpha=0.6, label="positive", density=True)
    plt.xlabel("Bag predicted probability")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_threshold_curve(df_th: pd.DataFrame, out_path: str, title: str):
    plt.figure(figsize=(7, 5))
    plt.plot(df_th["threshold"], df_th["f1"], label="F1")
    plt.plot(df_th["threshold"], df_th["sensitivity"], label="Sensitivity")
    plt.plot(df_th["threshold"], df_th["specificity"], label="Specificity")
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_pca_scatter(X: np.ndarray, y: np.ndarray, out_path: str, title: str):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(7, 6))
    neg = y == 0
    pos = y == 1
    plt.scatter(X2[neg, 0], X2[neg, 1], s=16, alpha=0.7, label="negative")
    plt.scatter(X2[pos, 0], X2[pos, 1], s=16, alpha=0.7, label="positive")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def compute_class_separation_ratio(X: np.ndarray, y: np.ndarray):
    """
    粗略的类间/类内距离比
    """
    X0 = X[y == 0]
    X1 = X[y == 1]

    if len(X0) < 2 or len(X1) < 2:
        return float("nan")

    c0 = X0.mean(axis=0)
    c1 = X1.mean(axis=0)

    inter = np.linalg.norm(c0 - c1)
    intra0 = np.mean(np.linalg.norm(X0 - c0[None, :], axis=1))
    intra1 = np.mean(np.linalg.norm(X1 - c1[None, :], axis=1))
    intra = 0.5 * (intra0 + intra1)

    return float(inter / (intra + 1e-8))


def export_centroid_evidence(train_df: pd.DataFrame, eval_df: pd.DataFrame, out_csv: str):
    """
    基于 train 集 bag mean centroid，分析 eval 集每张 slide 的 patch-level 正类证据稀疏度
    """
    train_pos = np.stack(train_df[train_df["label"] == 1]["bagfeat_mean"].values, axis=0)
    train_neg = np.stack(train_df[train_df["label"] == 0]["bagfeat_mean"].values, axis=0)

    pos_centroid = train_pos.mean(axis=0)
    neg_centroid = train_neg.mean(axis=0)

    rows = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evidence analysis", leave=False):
        obj = torch.load(row["bag_path"], map_location="cpu")
        feats = obj["features"].float().numpy()

        ev = compute_centroid_evidence(feats, pos_centroid, neg_centroid)
        out = {
            "slide_id": row["slide_id"],
            "label": int(row["label"]),
            "y_prob": float(row["y_prob"]),
            "num_instances": int(feats.shape[0]),
            **ev,
        }
        rows.append(out)

    df_ev = pd.DataFrame(rows)
    df_ev.to_csv(out_csv, index=False)
    return df_ev


def summarize_evidence_df(df_ev: pd.DataFrame):
    summary = {}

    for label_value, name in [(0, "neg"), (1, "pos")]:
        sub = df_ev[df_ev["label"] == label_value]
        if len(sub) == 0:
            continue
        summary[name] = {
            "count": int(len(sub)),
            "inst_delta_mean_mean": float(sub["inst_delta_mean"].mean()),
            "inst_delta_max_mean": float(sub["inst_delta_max"].mean()),
            "inst_delta_top10_mean_mean": float(sub["inst_delta_top10_mean"].mean()),
            "inst_pos_like_ratio_mean": float(sub["inst_pos_like_ratio"].mean()),
        }

    return summary


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Analyze frozen-feature MIL failure modes")
    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--mil_model", type=str, default="abmil", choices=["abmil", "dsmil", "meanpool"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # datasets
    train_set = BagFeatureDataset(
        slides_csv=args.slides_csv,
        feature_dir=args.feature_dir,
        split="train",
        max_instances=args.max_instances,
        shuffle_instances=False,
    )
    val_set = BagFeatureDataset(
        slides_csv=args.slides_csv,
        feature_dir=args.feature_dir,
        split="val",
        max_instances=args.max_instances,
        shuffle_instances=False,
    )
    test_set = BagFeatureDataset(
        slides_csv=args.slides_csv,
        feature_dir=args.feature_dir,
        split="test",
        max_instances=args.max_instances,
        shuffle_instances=False,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=bag_collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=bag_collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=bag_collate_fn, pin_memory=True
    )

    # model
    model = build_model(args.mil_model, in_dim=train_set.feat_dim, device=str(device))
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # inference
    train_rows = infer_split(model, train_loader, device=str(device))
    val_rows = infer_split(model, val_loader, device=str(device))
    test_rows = infer_split(model, test_loader, device=str(device))

    train_df = pd.DataFrame(train_rows)
    val_df = pd.DataFrame(val_rows)
    test_df = pd.DataFrame(test_rows)

    train_df.to_pickle(os.path.join(args.out_dir, "train_inference.pkl"))
    val_df.to_pickle(os.path.join(args.out_dir, "val_inference.pkl"))
    test_df.to_pickle(os.path.join(args.out_dir, "test_inference.pkl"))

    # -------------------------
    # 1) score overlap + threshold sweep
    # -------------------------
    val_metrics_05 = compute_binary_metrics(val_df["label"], val_df["y_prob"], threshold=0.5)
    test_metrics_05 = compute_binary_metrics(test_df["label"], test_df["y_prob"], threshold=0.5)

    val_sweep = threshold_sweep(val_df["label"], val_df["y_prob"])
    test_sweep = threshold_sweep(test_df["label"], test_df["y_prob"])

    val_sweep["table"].to_csv(os.path.join(args.out_dir, "val_threshold_sweep.csv"), index=False)
    test_sweep["table"].to_csv(os.path.join(args.out_dir, "test_threshold_sweep.csv"), index=False)

    save_score_overlap_plot(
        val_df,
        os.path.join(args.out_dir, "val_score_overlap.png"),
        "Validation score overlap"
    )
    save_score_overlap_plot(
        test_df,
        os.path.join(args.out_dir, "test_score_overlap.png"),
        "Test score overlap"
    )
    save_threshold_curve(
        val_sweep["table"],
        os.path.join(args.out_dir, "val_threshold_curve.png"),
        "Validation threshold sweep"
    )
    save_threshold_curve(
        test_sweep["table"],
        os.path.join(args.out_dir, "test_threshold_curve.png"),
        "Test threshold sweep"
    )

    # -------------------------
    # 2) bag embedding separation
    #    先用 frozen bag mean feature
    # -------------------------
    train_X_mean = np.stack(train_df["bagfeat_mean"].values, axis=0)
    train_y = train_df["label"].values.astype(int)

    val_X_mean = np.stack(val_df["bagfeat_mean"].values, axis=0)
    val_y = val_df["label"].values.astype(int)

    test_X_mean = np.stack(test_df["bagfeat_mean"].values, axis=0)
    test_y = test_df["label"].values.astype(int)

    save_pca_scatter(
        val_X_mean, val_y,
        os.path.join(args.out_dir, "val_bagmean_pca.png"),
        "Validation bag-mean feature PCA"
    )
    save_pca_scatter(
        test_X_mean, test_y,
        os.path.join(args.out_dir, "test_bagmean_pca.png"),
        "Test bag-mean feature PCA"
    )

    val_sep_ratio = compute_class_separation_ratio(val_X_mean, val_y)
    test_sep_ratio = compute_class_separation_ratio(test_X_mean, test_y)

    # -------------------------
    # 3) simple frozen-feature linear probe
    # -------------------------
    probe, val_probe_prob, val_probe_metrics = train_and_eval_probe(
        train_X_mean, train_y, val_X_mean, val_y, threshold=0.5
    )
    _, test_probe_prob, test_probe_metrics = train_and_eval_probe(
        train_X_mean, train_y, test_X_mean, test_y, threshold=0.5
    )

    pd.DataFrame({
        "slide_id": val_df["slide_id"].values,
        "label": val_y,
        "probe_prob": val_probe_prob,
    }).to_csv(os.path.join(args.out_dir, "val_probe_predictions.csv"), index=False)

    pd.DataFrame({
        "slide_id": test_df["slide_id"].values,
        "label": test_y,
        "probe_prob": test_probe_prob,
    }).to_csv(os.path.join(args.out_dir, "test_probe_predictions.csv"), index=False)

    # -------------------------
    # 4) positive evidence sparsity
    # -------------------------
    val_ev = export_centroid_evidence(
        train_df, val_df, os.path.join(args.out_dir, "val_centroid_evidence.csv")
    )
    test_ev = export_centroid_evidence(
        train_df, test_df, os.path.join(args.out_dir, "test_centroid_evidence.csv")
    )

    val_ev_summary = summarize_evidence_df(val_ev)
    test_ev_summary = summarize_evidence_df(test_ev)

    # -------------------------
    # Final summary
    # -------------------------
    summary = {
        "val_mil_metrics_at_0.5": val_metrics_05,
        "test_mil_metrics_at_0.5": test_metrics_05,
        "val_best_f1_threshold": val_sweep["best_f1"],
        "test_best_f1_threshold": test_sweep["best_f1"],
        "val_best_youden_threshold": val_sweep["best_youden"],
        "test_best_youden_threshold": test_sweep["best_youden"],

        "val_bagmean_separation_ratio": val_sep_ratio,
        "test_bagmean_separation_ratio": test_sep_ratio,

        "val_probe_metrics_bagmean": val_probe_metrics,
        "test_probe_metrics_bagmean": test_probe_metrics,

        "val_centroid_evidence_summary": val_ev_summary,
        "test_centroid_evidence_summary": test_ev_summary,
    }

    with open(os.path.join(args.out_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== Analysis Summary ==========")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()