#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

from torchmil.models import ABMIL, DSMIL

# ===== official TransMIL =====
import sys

TRANSMIL_REPO = "/data/maxinyu/WSI_WORKSPACE/TransMIL"
if TRANSMIL_REPO not in sys.path:
    sys.path.insert(0, TRANSMIL_REPO)

try:
    # official repo: models/TransMIL.py
    from models.TransMIL import TransMIL as OfficialTransMIL
    TRANSMIL_AVAILABLE = True
    TRANSMIL_IMPORT_ERROR = ""
except Exception as e:
    OfficialTransMIL = None
    TRANSMIL_AVAILABLE = False
    TRANSMIL_IMPORT_ERROR = str(e)

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


# =========================================================
# Metrics
# =========================================================
def compute_binary_metrics(y_true, y_prob, threshold=0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": float(auc),
        "acc": float(acc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_tumor_role_aux(role_probs: torch.Tensor, topk: int = 50) -> torch.Tensor:
    """
    role_probs: [N, R]
    Assumption:
        column 0 = tumor / atypical role
        column 1 = negative / stroma role

    return: [3]
        [tumor_role_ratio, tumor_role_topk_mean, tumor_stroma_margin]
    """
    if role_probs.ndim != 2 or role_probs.shape[1] < 2:
        raise ValueError(f"role_probs should be [N, R], got {role_probs.shape}")

    tumor_prob = role_probs[:, 0]
    stroma_prob = role_probs[:, 1]

    role_id = torch.argmax(role_probs, dim=1)
    tumor_role_ratio = (role_id == 0).float().mean()

    k = min(topk, role_probs.shape[0])
    tumor_role_topk_mean = torch.topk(tumor_prob, k=k).values.mean()

    tumor_stroma_margin = (tumor_prob - stroma_prob).mean()

    aux = torch.stack(
        [
            tumor_role_ratio,
            tumor_role_topk_mean,
            tumor_stroma_margin,
        ],
        dim=0,
    )

    return aux.float()


# =========================================================
# Dataset
# =========================================================
class BagFeatureDataset(Dataset):
    def __init__(
        self,
        slides_csv: str,
        feature_dir: str,
        split: str,
        max_instances: int = None,
        shuffle_instances: bool = False,
        use_role_aux: bool = False,
        role_topk: int = 50,
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
        self.use_role_aux = use_role_aux
        self.role_topk = role_topk

        self.samples = []
        for _, row in self.df.iterrows():
            slide_id = row["slide_id"]
            label = int(row["label"])
            bag_path = os.path.join(self.feature_dir, f"{slide_id}.pt")
            if not os.path.exists(bag_path):
                print(f"[Warn] missing bag file, skip: {bag_path}")
                continue

            self.samples.append(
                {
                    "slide_id": slide_id,
                    "label": label,
                    "bag_path": bag_path,
                }
            )

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found for split={split} in {feature_dir}")

        first_obj = torch.load(self.samples[0]["bag_path"], map_location="cpu")
        self.feat_dim = int(first_obj["features"].shape[1])

        labels = [x["label"] for x in self.samples]
        print(
            f"[{split}] num_samples={len(self.samples)}, "
            f"feat_dim={self.feat_dim}, "
            f"label_counts={pd.Series(labels).value_counts().to_dict()}"
        )

    def __len__(self):
        return len(self.samples)

    def _maybe_subsample_instances(self, feats: torch.Tensor):
        n = feats.shape[0]

        if self.max_instances is None or n <= self.max_instances:
            keep_idx = torch.arange(n)
            return feats, keep_idx

        if self.shuffle_instances:
            keep_idx = torch.randperm(n)[: self.max_instances]
        else:
            keep_idx = torch.arange(self.max_instances)

        feats = feats[keep_idx]
        return feats, keep_idx

    def __getitem__(self, idx):
        sample = self.samples[idx]
        obj = torch.load(sample["bag_path"], map_location="cpu")

        feats = obj["features"].float()
        feats, keep_idx = self._maybe_subsample_instances(feats)

        role_aux = None
        if self.use_role_aux:
            if "role_probs" not in obj:
                raise KeyError(
                    f"{sample['bag_path']} missing 'role_probs', but use_role_aux=True"
                )
            role_probs = obj["role_probs"].float()
            role_probs = role_probs[keep_idx]
            role_aux = build_tumor_role_aux(role_probs, topk=self.role_topk)

        label = int(sample["label"])
        slide_id = sample["slide_id"]

        return {
            "features": feats,
            "label": label,
            "slide_id": slide_id,
            "role_aux": role_aux,
        }


def bag_collate_fn(batch):
    features = [item["features"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    slide_ids = [item["slide_id"] for item in batch]

    if batch[0]["role_aux"] is None:
        role_aux = None
    else:
        role_aux = torch.stack([item["role_aux"] for item in batch], dim=0)

    return {
        "features": features,
        "labels": labels,
        "slide_ids": slide_ids,
        "role_aux": role_aux,
    }


# =========================================================
# Model wrapper
# =========================================================
class ABMILWrapper(nn.Module):
    def __init__(self, in_dim: int, device: str, att_dim: int = 128, gated: bool = False):
        super().__init__()
        self.model = ABMIL(
            in_shape=(in_dim,),
            att_dim=att_dim,
            att_act="tanh",
            gated=gated,
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
    def __init__(
        self,
        in_dim: int,
        device: str,
        att_dim: int = 128,
        nonlinear_q: bool = False,
        nonlinear_v: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model = DSMIL(
            in_shape=(in_dim,),
            att_dim=att_dim,
            nonlinear_q=nonlinear_q,
            nonlinear_v=nonlinear_v,
            dropout=dropout,
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


class TransMILWrapper(nn.Module):
    """
    Wrapper around official TransMIL implementation.

    Official repo notes:
    - class name: TransMIL
    - forward style: model(data=bag_feats)
    - internal input is hard-coded to 1024-d before transformer
      via nn.Linear(1024, 512)

    So here we optionally project arbitrary input dim -> 1024 first,
    while keeping the official TransMIL body untouched.
    """
    def __init__(
        self,
        in_dim: int,
        device: str,
        num_classes: int = 2,
        use_input_proj: bool = True,
        input_proj_dim: int = 1024,
    ):
        super().__init__()

        if not TRANSMIL_AVAILABLE:
            raise ImportError(
                "Official TransMIL import failed. "
                f"Please set TRANSMIL_REPO to the cloned official repo path. "
                f"Original error: {TRANSMIL_IMPORT_ERROR}"
            )

        self.device = device
        self.input_proj_dim = input_proj_dim

        # Official TransMIL hardcodes 1024-d input.
        if in_dim == input_proj_dim and not use_input_proj:
            self.input_proj = nn.Identity()
        elif in_dim == input_proj_dim:
            self.input_proj = nn.Identity()
        else:
            #self.input_proj = nn.Linear(in_dim, input_proj_dim)
            self.input_proj = nn.Sequential(
                nn.Linear(in_dim, input_proj_dim),
                nn.LayerNorm(input_proj_dim),
                nn.Dropout(0.1),
            )

        self.model = OfficialTransMIL(n_classes=num_classes)
        self.to(device)

    def forward(self, bag_feats: torch.Tensor):
        """
        bag_feats:
            [B, N, D] or [N, D]
        """
        if bag_feats.ndim == 2:
            bag_feats = bag_feats.unsqueeze(0)

        if bag_feats.ndim != 3:
            raise ValueError(f"TransMIL expects [B,N,D] or [N,D], got {bag_feats.shape}")

        bag_feats = self.input_proj(bag_feats)

        # official TransMIL forward signature: model(data=...)
        out = self.model(data=bag_feats)

        if isinstance(out, torch.Tensor):
            return out

        if isinstance(out, dict):
            for key in ["logits", "Y_prob", "Y_hat", "pred", "scores", "output"]:
                if key in out:
                    # prefer logits when available
                    if "logits" in out:
                        return out["logits"]
                    return out[key]

        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    return item

        raise TypeError(f"Unsupported TransMIL output type: {type(out)}")

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


class MILWithRoleAux(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        aux_dim: int = 3,
        hidden_dim: int = 16,
        device: str = "cuda",
    ):
        super().__init__()
        self.base_model = base_model
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.device = device
        self.to(device)

    def forward(self, bag_feats: torch.Tensor, role_aux: torch.Tensor = None):
        logits = self.base_model(bag_feats)

        if role_aux is None:
            return logits

        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        elif logits.ndim == 2 and logits.shape[1] == 2:
            raise ValueError(
                "MILWithRoleAux currently expects one-logit binary output, not [B,2]."
            )

        aux_bias = self.aux_mlp(role_aux)
        logits = logits + aux_bias
        return logits


# =========================================================
# Train / Eval
# =========================================================
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


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: float = None,
) -> torch.Tensor:
    if logits.ndim == 1:
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=logits.device)
            return nn.BCEWithLogitsLoss(pos_weight=pw)(logits, labels.float())
        return nn.BCEWithLogitsLoss()(logits, labels.float())

    if logits.ndim == 2 and logits.shape[1] == 1:
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=logits.device)
            return nn.BCEWithLogitsLoss(pos_weight=pw)(logits[:, 0], labels.float())
        return nn.BCEWithLogitsLoss()(logits[:, 0], labels.float())

    if logits.ndim == 2 and logits.shape[1] == 2:
        return nn.CrossEntropyLoss()(logits, labels.long())

    raise ValueError(f"Unsupported logits shape for loss: {logits.shape}")


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer=None,
    device="cuda",
    desc="train",
    pos_weight: float = None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_count = 0

    y_true_all = []
    y_prob_all = []
    slide_ids_all = []

    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        bags = batch["features"]
        labels = batch["labels"].to(device)
        slide_ids = batch["slide_ids"]
        role_aux_batch = batch["role_aux"]

        if role_aux_batch is not None:
            role_aux_batch = role_aux_batch.to(device)

        batch_losses = []

        if is_train:
            optimizer.zero_grad()

        for i, bag_feats in enumerate(bags):
            bag_feats = bag_feats.to(device)
            label_i = labels[i : i + 1]
            bag_input = bag_feats.unsqueeze(0)

            role_aux_i = None
            if role_aux_batch is not None:
                role_aux_i = role_aux_batch[i : i + 1]

            if role_aux_i is not None:
                logits = model(bag_input, role_aux=role_aux_i)
            else:
                logits = model(bag_input)

            if logits.ndim == 0:
                logits = logits.view(1)
            elif logits.ndim == 1 and logits.shape[0] != 1:
                logits = logits.unsqueeze(0)

            loss = compute_loss(logits, label_i, pos_weight=pos_weight)
            batch_losses.append(loss)

            prob = logits_to_prob(logits).detach().cpu().numpy()[0]

            y_true_all.append(int(label_i.item()))
            y_prob_all.append(float(prob))
            slide_ids_all.append(slide_ids[i])

        batch_loss = torch.stack(batch_losses).mean()

        if is_train:
            batch_loss.backward()
            optimizer.step()

        batch_size_now = len(bags)
        total_loss += batch_loss.item() * batch_size_now
        total_count += batch_size_now

        if len(set(y_true_all)) > 1:
            metrics_tmp = compute_binary_metrics(y_true_all, y_prob_all)
        else:
            metrics_tmp = {
                "auc": float("nan"),
                "acc": float("nan"),
                "f1": float("nan"),
                "sensitivity": float("nan"),
                "specificity": float("nan"),
            }

        pbar.set_postfix(loss=f"{batch_loss.item():.4f}", auc=f"{metrics_tmp['auc']:.4f}")

    avg_loss = total_loss / max(total_count, 1)
    metrics = compute_binary_metrics(y_true_all, y_prob_all)

    return avg_loss, metrics, {
        "slide_ids": slide_ids_all,
        "y_true": y_true_all,
        "y_prob": y_prob_all,
    }


# =========================================================
# Checkpoint selection
# =========================================================
def select_balanced_early_auc_epoch(
    history_df: pd.DataFrame,
    auc_tol: float = 0.03,
    min_sens: float = 0.30,
    min_spec: float = 0.60,
    min_f1: float = 1e-8,
    min_epoch: int = 1,
) -> Dict[str, float]:
    """
    BRACS-style small-validation checkpoint selection.

    Rule:
    1. Only consider epochs >= min_epoch.
    2. Keep epochs with:
        val_f1 >= min_f1
        val_sensitivity >= min_sens
        val_specificity >= min_spec
    3. Among valid epochs, find max val_auc.
    4. Select the earliest epoch whose val_auc >= max_valid_auc - auc_tol.
    5. If no valid epoch exists, fallback to highest validation balanced accuracy
       among epochs >= min_epoch.
    """
    df = history_df.copy()

    required = ["epoch", "val_auc", "val_f1", "val_sensitivity", "val_specificity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for balanced_early_auc selection: {missing}")

    df = df[df["epoch"] >= min_epoch].copy()
    if len(df) == 0:
        raise ValueError(
            f"No epoch satisfies epoch >= min_epoch={min_epoch}. "
            f"Please lower --min_select_epoch or train for more epochs."
        )

    df["val_bacc"] = 0.5 * (df["val_sensitivity"] + df["val_specificity"])

    valid = df[
        (df["val_f1"] >= min_f1)
        & (df["val_sensitivity"] >= min_sens)
        & (df["val_specificity"] >= min_spec)
        & (~df["val_auc"].isna())
    ].copy()

    if len(valid) == 0:
        fallback = df.sort_values(
            ["val_bacc", "val_auc", "epoch"],
            ascending=[False, False, True],
        ).iloc[0]

        return {
            "epoch": int(fallback["epoch"]),
            "selection_mode": "fallback_val_bacc",
            "selected_val_auc": float(fallback["val_auc"]),
            "selected_val_bacc": float(fallback["val_bacc"]),
            "selected_val_sens": float(fallback["val_sensitivity"]),
            "selected_val_spec": float(fallback["val_specificity"]),
            "selected_val_f1": float(fallback["val_f1"]),
            "max_valid_auc": float("nan"),
            "auc_tol": float(auc_tol),
            "min_sens": float(min_sens),
            "min_spec": float(min_spec),
            "min_f1": float(min_f1),
            "min_epoch": int(min_epoch),
        }

    max_valid_auc = float(valid["val_auc"].max())
    eligible = valid[valid["val_auc"] >= max_valid_auc - auc_tol].copy()
    selected = eligible.sort_values("epoch", ascending=True).iloc[0]

    return {
        "epoch": int(selected["epoch"]),
        "selection_mode": "balanced_early_auc",
        "selected_val_auc": float(selected["val_auc"]),
        "selected_val_bacc": float(selected["val_bacc"]),
        "selected_val_sens": float(selected["val_sensitivity"]),
        "selected_val_spec": float(selected["val_specificity"]),
        "selected_val_f1": float(selected["val_f1"]),
        "max_valid_auc": float(max_valid_auc),
        "auc_tol": float(auc_tol),
        "min_sens": float(min_sens),
        "min_spec": float(min_spec),
        "min_f1": float(min_f1),
        "min_epoch": int(min_epoch),
    }


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    val_metrics: Dict[str, float],
    feat_dim: int,
    args,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "val_metrics": val_metrics,
            "feat_dim": feat_dim,
            "args": vars(args),
        },
        path,
    )


def save_predictions_csv(preds: Dict, path: str):
    pd.DataFrame(
        {
            "slide_id": preds["slide_ids"],
            "y_true": preds["y_true"],
            "y_prob": preds["y_prob"],
        }
    ).to_csv(path, index=False)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Train MIL on extracted bag features")

    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--shuffle_instances", action="store_true")

    parser.add_argument(
        "--monitor",
        type=str,
        default="auc",
        choices=["auc", "acc", "f1", "balanced_early_auc"],
    )
    parser.add_argument(
        "--early_auc_tol",
        type=float,
        default=0.03,
        help="Tolerance for balanced_early_auc: select earliest epoch within best_auc - tol.",
    )
    parser.add_argument(
        "--min_val_sens",
        type=float,
        default=0.30,
        help="Minimum validation sensitivity for balanced_early_auc.",
    )
    parser.add_argument(
        "--min_val_spec",
        type=float,
        default=0.60,
        help="Minimum validation specificity for balanced_early_auc.",
    )
    parser.add_argument(
        "--min_val_f1",
        type=float,
        default=1e-8,
        help="Minimum validation F1 for balanced_early_auc.",
    )
    parser.add_argument(
        "--min_select_epoch",
        type=int,
        default=1,
        help="Minimum epoch allowed for balanced_early_auc checkpoint selection.",
    )
    parser.add_argument(
        "--save_all_epoch_ckpts",
        action="store_true",
        help="Save every epoch checkpoint. Automatically enabled when monitor=balanced_early_auc.",
    )

    parser.add_argument(
        "--mil_model",
        type=str,
        default="abmil",
        choices=["abmil", "meanpool", "dsmil", "transmil"],
        help="MIL aggregator type",
    )

    parser.add_argument(
        "--transmil_input_proj_dim",
        type=int,
        default=1024,
        help="Official TransMIL expects 1024-d input; project input features to this dim first.",
    )
    parser.add_argument(
        "--no_transmil_input_proj",
        action="store_true",
        help="Disable external input projection for TransMIL. Only use this when feat_dim already matches transmil_input_proj_dim.",
    )

    parser.add_argument("--att_dim", type=int, default=128, help="ABMIL / DSMIL attention dim")
    parser.add_argument("--abmil_gated", action="store_true")
    parser.add_argument("--dsmil_dropout", type=float, default=0.0)
    parser.add_argument("--dsmil_nonlinear_q", action="store_true")
    parser.add_argument("--dsmil_nonlinear_v", action="store_true")

    parser.add_argument("--pos_weight", type=float, default=None)

    parser.add_argument("--use_role_aux", action="store_true")
    parser.add_argument("--role_topk", type=int, default=50)
    parser.add_argument("--role_aux_hidden_dim", type=int, default=16)

    parser.add_argument("--save_last_ckpt", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_set = BagFeatureDataset(
        slides_csv=args.slides_csv,
        feature_dir=args.feature_dir,
        split="train",
        max_instances=args.max_instances,
        shuffle_instances=args.shuffle_instances,
        use_role_aux=args.use_role_aux,
        role_topk=args.role_topk,
    )
    val_set = BagFeatureDataset(
        slides_csv=args.slides_csv,
        feature_dir=args.feature_dir,
        split="val",
        max_instances=args.max_instances,
        shuffle_instances=False,
        use_role_aux=args.use_role_aux,
        role_topk=args.role_topk,
    )
    test_set = BagFeatureDataset(
        slides_csv=args.slides_csv,
        feature_dir=args.feature_dir,
        split="test",
        max_instances=args.max_instances,
        shuffle_instances=False,
        use_role_aux=args.use_role_aux,
        role_topk=args.role_topk,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=bag_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=bag_collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=bag_collate_fn,
        pin_memory=True,
    )

    if args.mil_model == "abmil":
        model = ABMILWrapper(
            in_dim=train_set.feat_dim,
            device=str(device),
            att_dim=args.att_dim,
            gated=args.abmil_gated,
        )
    elif args.mil_model == "dsmil":
        model = DSMILWrapper(
            in_dim=train_set.feat_dim,
            device=str(device),
            att_dim=args.att_dim,
            nonlinear_q=args.dsmil_nonlinear_q,
            nonlinear_v=args.dsmil_nonlinear_v,
            dropout=args.dsmil_dropout,
        )
    elif args.mil_model == "transmil":
        model = TransMILWrapper(
            in_dim=train_set.feat_dim,
            device=str(device),
            num_classes=2,
            use_input_proj=not args.no_transmil_input_proj,
            input_proj_dim=args.transmil_input_proj_dim,
        )
    elif args.mil_model == "meanpool":
        model = MeanPoolMIL(
            in_dim=train_set.feat_dim,
            out_dim=1,
            device=str(device),
        )
    else:
        raise ValueError(f"Unknown mil_model: {args.mil_model}")

    if args.use_role_aux:
        model = MILWithRoleAux(
            base_model=model,
            aux_dim=3,
            hidden_dim=args.role_aux_hidden_dim,
            device=str(device),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_score = -float("inf")
    best_epoch = -1
    wait = 0
    selection_info = None

    best_ckpt_path = os.path.join(args.out_dir, "best_model.pt")
    last_ckpt_path = os.path.join(args.out_dir, "last_model.pt")
    epoch_ckpt_dir = os.path.join(args.out_dir, "epoch_ckpts")

    if args.monitor == "balanced_early_auc" or args.save_all_epoch_ckpts:
        ensure_dir(epoch_ckpt_dir)

    print(f"[INFO] feature_dim = {train_set.feat_dim}")
    print(f"[INFO] save_dir = {args.out_dir}")
    print(f"[INFO] monitor = {args.monitor}")
    if args.monitor == "balanced_early_auc":
        print(
            "[INFO] balanced_early_auc settings: "
            f"tol={args.early_auc_tol}, "
            f"min_sens={args.min_val_sens}, "
            f"min_spec={args.min_val_spec}, "
            f"min_f1={args.min_val_f1}, "
            f"min_select_epoch={args.min_select_epoch}"
        )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _ = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=str(device),
            desc=f"Train {epoch}/{args.epochs}",
            pos_weight=args.pos_weight,
        )

        val_loss, val_metrics, val_preds = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=str(device),
            desc=f"Val   {epoch}/{args.epochs}",
            pos_weight=args.pos_weight,
        )

        if args.monitor in ["auc", "acc", "f1"]:
            score = val_metrics[args.monitor]
        elif args.monitor == "balanced_early_auc":
            score = val_metrics["auc"]
        else:
            raise ValueError(f"Unknown monitor: {args.monitor}")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{
                f"train_{k}": v
                for k, v in train_metrics.items()
                if isinstance(v, (int, float))
            },
            **{
                f"val_{k}": v
                for k, v in val_metrics.items()
                if isinstance(v, (int, float))
            },
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_sens={val_metrics['sensitivity']:.4f} "
            f"val_spec={val_metrics['specificity']:.4f}"
        )

        if args.monitor == "balanced_early_auc" or args.save_all_epoch_ckpts:
            epoch_ckpt_path = os.path.join(epoch_ckpt_dir, f"epoch_{epoch:03d}.pt")
            save_checkpoint(
                path=epoch_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_metrics=val_metrics,
                feat_dim=train_set.feat_dim,
                args=args,
            )

        if args.monitor in ["auc", "acc", "f1"]:
            if score > best_score:
                best_score = score
                best_epoch = epoch
                wait = 0

                save_checkpoint(
                    path=best_ckpt_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    val_metrics=val_metrics,
                    feat_dim=train_set.feat_dim,
                    args=args,
                )

                save_predictions_csv(
                    val_preds,
                    os.path.join(args.out_dir, "best_val_predictions.csv"),
                )

                print(f"[Best] epoch={epoch}, {args.monitor}={best_score:.4f}")
            else:
                wait += 1

        elif args.monitor == "balanced_early_auc":
            wait = 0

        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "train_history.csv"), index=False)

        if args.save_last_ckpt:
            save_checkpoint(
                path=last_ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                val_metrics=val_metrics,
                feat_dim=train_set.feat_dim,
                args=args,
            )

        if args.monitor in ["auc", "acc", "f1"] and wait >= args.patience:
            print(
                f"[Early Stop] no improvement for {args.patience} epochs. "
                f"Stop at epoch {epoch}."
            )
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(args.out_dir, "train_history.csv"), index=False)

    if args.monitor == "balanced_early_auc":
        selection_info = select_balanced_early_auc_epoch(
            history_df=history_df,
            auc_tol=args.early_auc_tol,
            min_sens=args.min_val_sens,
            min_spec=args.min_val_spec,
            min_f1=args.min_val_f1,
            min_epoch=args.min_select_epoch,
        )

        best_epoch = int(selection_info["epoch"])
        best_score = float(selection_info["selected_val_auc"])

        selected_ckpt_path = os.path.join(epoch_ckpt_dir, f"epoch_{best_epoch:03d}.pt")
        if not os.path.exists(selected_ckpt_path):
            raise FileNotFoundError(f"Selected checkpoint not found: {selected_ckpt_path}")

        ckpt = torch.load(selected_ckpt_path, map_location=device)
        torch.save(ckpt, best_ckpt_path)

        with open(os.path.join(args.out_dir, "best_epoch_selection.json"), "w") as f:
            json.dump(selection_info, f, indent=2)

        print(
            f"[BalancedEarlyAUC Best] epoch={best_epoch}, "
            f"selected_val_auc={selection_info['selected_val_auc']:.4f}, "
            f"max_valid_auc={selection_info['max_valid_auc']:.4f}, "
            f"selected_val_sens={selection_info['selected_val_sens']:.4f}, "
            f"selected_val_spec={selection_info['selected_val_spec']:.4f}, "
            f"min_epoch={selection_info['min_epoch']}, "
            f"mode={selection_info['selection_mode']}"
        )

        model.load_state_dict(ckpt["model_state_dict"])

        selected_val_loss, selected_val_metrics, selected_val_preds = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=str(device),
            desc="Selected Val",
            pos_weight=args.pos_weight,
        )

        save_predictions_csv(
            selected_val_preds,
            os.path.join(args.out_dir, "best_val_predictions.csv"),
        )

        with open(os.path.join(args.out_dir, "selected_val_metrics.json"), "w") as f:
            json.dump(
                {
                    "selected_val_loss": selected_val_loss,
                    "selected_val_metrics": selected_val_metrics,
                    "selection_info": selection_info,
                },
                f,
                indent=2,
            )

    else:
        if not os.path.exists(best_ckpt_path):
            raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_metrics, test_preds = run_one_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=str(device),
        desc="Test",
        pos_weight=args.pos_weight,
    )

    with open(os.path.join(args.out_dir, "final_test_metrics.json"), "w") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_monitor": args.monitor,
                "best_val_score": best_score,
                "selection_info": selection_info,
                "test_loss": test_loss,
                "test_metrics": test_metrics,
                "args": vars(args),
            },
            f,
            indent=2,
        )

    save_predictions_csv(
        test_preds,
        os.path.join(args.out_dir, "test_predictions.csv"),
    )

    print("\n========== Final Test ==========")
    print(f"best_epoch: {best_epoch}")
    print(f"best_monitor: {args.monitor}")
    print(f"best_val_score: {best_score:.4f}")
    print(f"test_loss: {test_loss:.4f}")
    print(f"test_auc:  {test_metrics['auc']:.4f}")
    print(f"test_acc:  {test_metrics['acc']:.4f}")
    print(f"test_f1:   {test_metrics['f1']:.4f}")
    print(f"test_sens: {test_metrics['sensitivity']:.4f}")
    print(f"test_spec: {test_metrics['specificity']:.4f}")


if __name__ == "__main__":
    main()