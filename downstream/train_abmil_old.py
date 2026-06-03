import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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

from torchmil.models import ABMIL,DSMIL


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


def safe_item(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return x


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
    role_probs: [N, 3] -> columns: tumor, stroma, ambiguous
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

    aux = torch.stack([
        tumor_role_ratio,
        tumor_role_topk_mean,
        tumor_stroma_margin,
    ], dim=0)

    return aux.float()

# =========================================================
# Dataset
# =========================================================
class BagFeatureDataset(Dataset):
    """
    每个样本 = 一张 slide 的 bag features
    读取 .pt 格式：
    {
        "features": Tensor[N, D],
        "coords": Tensor[N, 2],
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

            self.samples.append({
                "slide_id": slide_id,
                "label": label,
                "bag_path": bag_path,
            })

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found for split={split} in {feature_dir}")

        # 推断 feature dim
        first_obj = torch.load(self.samples[0]["bag_path"], map_location="cpu")
        self.feat_dim = int(first_obj["features"].shape[1])

        print(f"[{split}] num_samples={len(self.samples)}, feat_dim={self.feat_dim}")

    def __len__(self):
        return len(self.samples)

    def _maybe_subsample_instances(self, feats: torch.Tensor):
        """
        return:
            feats_sub: [M, D]
            keep_idx: [M]
        """
        n = feats.shape[0]

        if self.max_instances is None or n <= self.max_instances:
            keep_idx = torch.arange(n)
            return feats, keep_idx

        if self.shuffle_instances:
            keep_idx = torch.randperm(n)[:self.max_instances]
        else:
            keep_idx = torch.arange(self.max_instances)

        feats = feats[keep_idx]
        return feats, keep_idx

    def __getitem__(self, idx):
        sample = self.samples[idx]
        obj = torch.load(sample["bag_path"], map_location="cpu")

        feats = obj["features"].float()   # [N, D]
        feats,keep_idx = self._maybe_subsample_instances(feats)

        role_aux = None
        if self.use_role_aux:
            if "role_probs" not in obj:
                raise KeyError(
                    f"{sample['bag_path']} missing 'role_probs', but use_role_aux=True"
                )
            role_probs = obj["role_probs"].float()   # [N, 3]
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
    """
    ABMIL 一般按 batch_size=1 最稳。
    这里仍保留通用接口，返回 list[Tensor] 形式的 bags。
    """
    features = [item["features"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    slide_ids = [item["slide_id"] for item in batch]

    if batch[0]["role_aux"] is None:
        role_aux = None
    else:
        role_aux = torch.stack([item["role_aux"] for item in batch], dim=0)  # [B, 3]

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
    """
    统一处理不同 torchmil 版本下的 forward 输出格式。
    """
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
        """
        bag_feats: [N, D] or [1, N, D]
        返回 logits shape 尽量整理成 [1] 或 [1, 2]
        """
        out = self.model(bag_feats)

        # 常见情况 1: 直接返回 tensor
        if isinstance(out, torch.Tensor):
            return out

        # 常见情况 2: 返回 dict
        if isinstance(out, dict):
            for key in ["logits", "pred", "scores", "output"]:
                if key in out:
                    return out[key]

        # 常见情况 3: tuple/list
        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    return item

        raise TypeError(f"Unsupported ABMIL output type: {type(out)}")

class DSMILWrapper(nn.Module):
    """
    兼容 torchmil 的 DSMIL 输出格式。
    官方文档里 DSMIL forward(X, mask=None, return_att=False, return_inst_pred=False)
    返回 bag label logits，shape 通常是 (batch_size,)。
    """
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
        """
        bag_feats: [1, N, D] or [N, D]
        """
        if bag_feats.ndim == 2:
            bag_feats = bag_feats.unsqueeze(0)  # [1, N, D]

        out = self.model(bag_feats)

        # 常见情况 1: 直接返回 tensor
        if isinstance(out, torch.Tensor):
            return out

        # 常见情况 2: dict
        if isinstance(out, dict):
            for key in ["logits", "pred", "scores", "output", "Y_pred"]:
                if key in out:
                    return out[key]

        # 常见情况 3: tuple/list
        if isinstance(out, (tuple, list)):
            for item in out:
                if isinstance(item, torch.Tensor):
                    return item

        raise TypeError(f"Unsupported DSMIL output type: {type(out)}")

class MeanPoolMIL(nn.Module):
    """
    最简单的 MIL baseline:
    bag_feats: [1, N, D] 或 [N, D]
    先对 N 个 instance 做 mean pooling，再接线性分类器
    """
    def __init__(self, in_dim: int, out_dim: int = 1, device: str = "cuda"):
        super().__init__()
        self.classifier = nn.Linear(in_dim, out_dim)
        self.device = device
        self.to(device)

    def forward(self, bag_feats: torch.Tensor):
        # 支持 [1, N, D] 或 [N, D]
        if bag_feats.ndim == 3:
            # [B, N, D]，你当前 batch_size=1，所以 B 通常为 1
            bag_repr = bag_feats.mean(dim=1)   # [B, D]
        elif bag_feats.ndim == 2:
            bag_repr = bag_feats.mean(dim=0, keepdim=True)  # [1, D]
        else:
            raise ValueError(f"Unsupported bag_feats shape: {bag_feats.shape}")

        logits = self.classifier(bag_repr)  # [B, 1]
        return logits


class MILWithRoleAux(nn.Module):
    """
    Wrap any MIL model that outputs bag logits.
    Final logit = base_logit + aux_bias
    """
    def __init__(self, base_model: nn.Module, aux_dim: int = 3, hidden_dim: int = 16, device: str = "cuda"):
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

        # 统一成 [B,1] 方便加 bias
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)          # [B] -> [B,1]
        elif logits.ndim == 2 and logits.shape[1] == 2:
            # 如果是两类 logits，不建议直接这样用
            raise ValueError("MILWithRoleAux currently expects one-logit binary output, not [B,2].")

        aux_bias = self.aux_mlp(role_aux)          # [B,1]
        logits = logits + aux_bias
        return logits
# =========================================================
# Train / Eval
# =========================================================
def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    """
    兼容：
    - binary one-logit: [B] or [B,1]
    - 2-class logits: [B,2]
    """
    if logits.ndim == 0:
        logits = logits.view(1)

    if logits.ndim == 1:
        # [B]
        prob = torch.sigmoid(logits)
        return prob

    if logits.ndim == 2 and logits.shape[1] == 1:
        prob = torch.sigmoid(logits[:, 0])
        return prob

    if logits.ndim == 2 and logits.shape[1] == 2:
        prob = torch.softmax(logits, dim=1)[:, 1]
        return prob

    raise ValueError(f"Unsupported logits shape: {logits.shape}")


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, pos_weight: float = None) -> torch.Tensor:
    """
    兼容 1-logit binary / 2-class logits
    """
    if logits.ndim == 1:
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=logits.device)
            return nn.BCEWithLogitsLoss(pos_weight=pw)(logits, labels.float())
        else:
            return nn.BCEWithLogitsLoss()(logits, labels.float())

    if logits.ndim == 2 and logits.shape[1] == 1:
        if pos_weight is not None:
            pw = torch.tensor([pos_weight], device=logits.device)
            return nn.BCEWithLogitsLoss(pos_weight=pw)(logits[:, 0], labels.float())
        else:
            return nn.BCEWithLogitsLoss()(logits[:, 0], labels.float())

    if logits.ndim == 2 and logits.shape[1] == 2:
        # CrossEntropy 这里先不加 class weight，保持简单
        return nn.CrossEntropyLoss()(logits, labels.long())

    raise ValueError(f"Unsupported logits shape for loss: {logits.shape}")


def run_one_epoch(
    model: ABMILWrapper,
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
        bags = batch["features"]     # list of [Ni, D]
        labels = batch["labels"].to(device)  # [B]
        slide_ids = batch["slide_ids"]
        role_aux_batch = batch["role_aux"]
        if role_aux_batch is not None:
            role_aux_batch = role_aux_batch.to(device)

        # 为了稳妥，建议 batch_size=1；这里仍支持逐个 bag 处理
        batch_losses = []
        batch_probs = []

        if is_train:
            optimizer.zero_grad()

        for i, bag_feats in enumerate(bags):
            bag_feats = bag_feats.to(device)         # [N, D]
            label_i = labels[i:i+1]                  # [1]

            # 给某些实现一个 batch 维度：[1, N, D]
            bag_input = bag_feats.unsqueeze(0)

            #logits = model(bag_input)
            role_aux_i = None
            if role_aux_batch is not None:
                role_aux_i = role_aux_batch[i:i+1]   # [1, 3]

            if role_aux_i is not None:
                logits = model(bag_input, role_aux=role_aux_i)
            else:
                logits = model(bag_input)

            if logits.ndim == 0:
                logits = logits.view(1)
            elif logits.ndim == 1 and logits.shape[0] != 1:
                # 某些实现可能给 [2] 之类
                logits = logits.unsqueeze(0)

            loss = compute_loss(logits, label_i,pos_weight=pos_weight)
            batch_losses.append(loss)

            prob = logits_to_prob(logits).detach().cpu().numpy()[0]
            batch_probs.append(prob)

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

        metrics_tmp = compute_binary_metrics(y_true_all, y_prob_all) if len(set(y_true_all)) > 1 else {
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
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Train official ABMIL on extracted bag features")

    parser.add_argument("--slides_csv", type=str, required=True,
                        help="slides_split.csv or a csv containing slide_id / label / split")
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory containing slide-level bag .pt files")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--shuffle_instances", action="store_true")

    parser.add_argument("--monitor", type=str, default="auc", choices=["auc", "acc", "f1"])
    parser.add_argument(
        "--mil_model",
        type=str,
        default="abmil",
        choices=["abmil", "meanpool","dsmil"],
        help="MIL aggregator type"
    )
    parser.add_argument("--pos_weight", type=float, default=None,
                    help="Positive class weight for BCE loss. If None, no reweighting.")
    parser.add_argument("--use_role_aux", action="store_true",
                    help="Use tumor-like role auxiliary features")
    parser.add_argument("--role_topk", type=int, default=50,
                        help="Top-k patches for tumor_role_topk_mean")
    parser.add_argument("--role_aux_hidden_dim", type=int, default=16)

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

    #model = ABMILWrapper(in_dim=train_set.feat_dim, device=str(device))
    if args.mil_model == "abmil":
        model = ABMILWrapper(in_dim=train_set.feat_dim, device=str(device))
    elif args.mil_model == "dsmil":
        model = DSMILWrapper(in_dim=train_set.feat_dim, device=str(device))
    elif args.mil_model == "meanpool":
        model = MeanPoolMIL(in_dim=train_set.feat_dim, out_dim=1, device=str(device))
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
    best_ckpt_path = os.path.join(args.out_dir, "best_abmil.pt")

    print(f"[INFO] feature_dim = {train_set.feat_dim}")
    print(f"[INFO] save_dir = {args.out_dir}")

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
        )

        score = val_metrics[args.monitor]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))},
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
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

        if score > best_score:
            best_score = score
            best_epoch = epoch

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "feat_dim": train_set.feat_dim,
                "args": vars(args),
            }, best_ckpt_path)

            pd.DataFrame({
                "slide_id": val_preds["slide_ids"],
                "y_true": val_preds["y_true"],
                "y_prob": val_preds["y_prob"],
            }).to_csv(os.path.join(args.out_dir, "best_val_predictions.csv"), index=False)

            print(f"[Best] epoch={epoch}, {args.monitor}={best_score:.4f}")

        pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "train_history.csv"), index=False)

    # ===== Test with best checkpoint =====
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_metrics, test_preds = run_one_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=str(device),
        desc="Test",
    )

    with open(os.path.join(args.out_dir, "final_test_metrics.json"), "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_monitor": args.monitor,
            "best_val_score": best_score,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
        }, f, indent=2)

    pd.DataFrame({
        "slide_id": test_preds["slide_ids"],
        "y_true": test_preds["y_true"],
        "y_prob": test_preds["y_prob"],
    }).to_csv(os.path.join(args.out_dir, "test_predictions.csv"), index=False)

    print("\n========== Final Test ==========")
    print(f"best_epoch: {best_epoch}")
    print(f"test_loss: {test_loss:.4f}")
    print(f"test_auc:  {test_metrics['auc']:.4f}")
    print(f"test_acc:  {test_metrics['acc']:.4f}")
    print(f"test_f1:   {test_metrics['f1']:.4f}")
    print(f"test_sens: {test_metrics['sensitivity']:.4f}")
    print(f"test_spec: {test_metrics['specificity']:.4f}")


if __name__ == "__main__":
    main()