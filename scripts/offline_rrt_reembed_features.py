#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm


# =========================================================
# Try import RRTEncoder
# =========================================================
RRT_IMPORT_OK = False
RRT_IMPORT_MSG = ""
import sys
sys.path.insert(0, "/data/maxinyu/WSI_WORKSPACE/RRT-MIL")


try:
    from rrt import RRTEncoder
    RRT_IMPORT_OK = True
except Exception as e1:
    try:
        from modules.rrt import RRTEncoder  # type: ignore
        RRT_IMPORT_OK = True
    except Exception as e2:
        RRT_IMPORT_MSG = f"Import failed:\nrrt: {e1}\nmodels.rrt: {e2}"


# =========================================================
# Utils
# =========================================================
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pt_feature(path: str | Path) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {path}, got {type(obj)}")

    if "features" not in obj:
        raise KeyError(f"'features' not found in {path}")

    feats = obj["features"]
    if not torch.is_tensor(feats):
        feats = torch.tensor(feats, dtype=torch.float32)
    feats = feats.float().cpu()

    out = dict(obj)
    out["features"] = feats
    return out


def save_pt_feature(path: str | Path, obj: Dict[str, Any]):
    torch.save(obj, path)


# =========================================================
# RRT re-embedding model
# =========================================================
class RRTReEmbed(nn.Module):
    """
    features [N, in_dim]
      -> Linear(in_dim, proj_dim)
      -> Dropout
      -> RRTEncoder
      -> output [N, proj_dim]
    """
    def __init__(
        self,
        in_dim: int,
        proj_dim: int = 512,
        dropout: float = 0.25,
        # ---- RRT args ----
        n_layers: int = 2,
        n_heads: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        with_cls_token: bool = False,
    ):
        super().__init__()

        self.fc = nn.Linear(in_dim, proj_dim)
        self.dp = nn.Dropout(dropout)

        if not RRT_IMPORT_OK:
            raise ImportError(
                "Failed to import RRTEncoder.\n"
                f"{RRT_IMPORT_MSG}\n"
                "Please modify the import path in this script to match your RRT-MIL repo."
            )

        # 这里尽量按通用接口写。如果你本地 RRTEncoder 参数名略有不同，改这里即可。
        self.rrt = RRTEncoder(
            mlp_dim=proj_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
        
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: [N, in_dim]
        return: [N, proj_dim]
        """
        if feats.ndim != 2:
            raise ValueError(f"Expected feats [N, D], got {tuple(feats.shape)}")

        x = self.fc(feats)      # [N, proj_dim]
        x = self.dp(x)

        # RRT repo 常见输入是 [1, N, D]
        x = x.unsqueeze(0)      # [1, N, D]
        x = self.rrt(x)         # expected [1, N, D] or tuple/dict
        if isinstance(x, tuple):
            x = x[0]
        elif isinstance(x, dict):
            # 尽量兼容不同 repo 返回风格
            if "x" in x:
                x = x["x"]
            elif "features" in x:
                x = x["features"]
            elif "tokens" in x:
                x = x["tokens"]
            else:
                raise TypeError(f"Unsupported RRT dict output keys: {x.keys()}")

        if not torch.is_tensor(x):
            raise TypeError(f"Unsupported RRT output type: {type(x)}")

        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(f"Expected RRT output [1, N, D], got {tuple(x.shape)}")

        x = x.squeeze(0)        # [N, proj_dim]
        return x


# =========================================================
# Train one re-embedding model on train split features
# =========================================================
class BagBinaryHead(nn.Module):
    """
    一个很轻的 bag-level supervision 头：
    feature [N, D] -> attention pooling -> bag logit
    用来训练投影+RRT
    """
    def __init__(self, dim: int, att_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, att_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(att_dim, 1),
        )
        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: [N, D]
        a = self.att(feats).squeeze(-1)   # [N]
        a = torch.softmax(a, dim=0)
        bag_feat = torch.sum(feats * a.unsqueeze(-1), dim=0, keepdim=True)  # [1, D]
        logit = self.cls(bag_feat).squeeze(0).squeeze(-1)
        return logit


class RRTSlideClassifier(nn.Module):
    def __init__(self, reembed: RRTReEmbed, head: BagBinaryHead):
        super().__init__()
        self.reembed = reembed
        self.head = head

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.reembed(feats)
        logit = self.head(x)
        return logit


def evaluate_classifier(
    model: nn.Module,
    slide_paths: List[Path],
    labels: List[int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    losses = []
    y_true = []
    y_prob = []

    with torch.no_grad():
        for pt_path, y in zip(slide_paths, labels):
            obj = load_pt_feature(pt_path)
            feats = obj["features"].to(device)
            logit = model(feats)
            target = torch.tensor(float(y), device=device)
            loss = criterion(logit.view(1), target.view(1))

            prob = torch.sigmoid(logit).item()

            losses.append(float(loss.item()))
            y_true.append(int(y))
            y_prob.append(float(prob))

    y_true = torch.tensor(y_true).numpy()
    y_prob = torch.tensor(y_prob).numpy()

    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) > 1 else float("nan")
    except Exception:
        auc = float("nan")

    y_pred = (y_prob >= 0.5).astype(int)
    acc = float((y_pred == y_true).mean())

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)

    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "auc": auc,
        "acc": acc,
        "f1": float(f1),
    }


def train_rrt_reembed_model(
    train_slide_paths: List[Path],
    train_labels: List[int],
    val_slide_paths: List[Path],
    val_labels: List[int],
    in_dim: int,
    device: torch.device,
    proj_dim: int = 512,
    dropout: float = 0.25,
    epochs: int = 10,
    patience: int = 3,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    att_dim: int = 128,
    # rrt args
    n_layers: int = 2,
    n_heads: int = 8,
    mlp_ratio: float = 2.0,
    qkv_bias: bool = True,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    drop_path: float = 0.0,
    with_cls_token: bool = False,
):
    reembed = RRTReEmbed(
        in_dim=in_dim,
        proj_dim=proj_dim,
        dropout=dropout,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        attn_drop=attn_drop,
        proj_drop=proj_drop,
        drop_path=drop_path,
        with_cls_token=with_cls_token,
    )
    head = BagBinaryHead(dim=proj_dim, att_dim=att_dim, dropout=dropout)
    model = RRTSlideClassifier(reembed, head).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_state = None
    best_epoch = -1
    best_auc = -1.0
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        order = torch.randperm(len(train_slide_paths)).tolist()
        for idx in tqdm(order, desc=f"Train epoch {epoch}", leave=False):
            pt_path = train_slide_paths[idx]
            y = train_labels[idx]

            obj = load_pt_feature(pt_path)
            feats = obj["features"].to(device)

            target = torch.tensor(float(y), device=device)
            logit = model(feats)
            loss = criterion(logit.view(1), target.view(1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(float(loss.item()))

        val_res = evaluate_classifier(model, val_slide_paths, val_labels, device=device)
        rec = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / max(1, len(losses))),
            "val_loss": val_res["loss"],
            "val_auc": val_res["auc"],
            "val_acc": val_res["acc"],
            "val_f1": val_res["f1"],
        }
        history.append(rec)

        print(
            f"[Epoch {epoch}] "
            f"train_loss={rec['train_loss']:.4f} | "
            f"val_auc={val_res['auc']:.4f} | "
            f"val_acc={val_res['acc']:.4f} | "
            f"val_f1={val_res['f1']:.4f}"
        )

        score = val_res["auc"]
        if not math.isnan(score) and score > best_auc:
            best_auc = score
            best_epoch = epoch
            best_state = {
                "model_state_dict": model.state_dict(),
                "reembed_state_dict": model.reembed.state_dict(),
                "head_state_dict": model.head.state_dict(),
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"[Early Stop] no val AUC improvement for {patience} epochs.")
            break

    if best_state is None:
        raise RuntimeError("No valid best checkpoint found for RRT re-embedding model.")

    model.load_state_dict(best_state["model_state_dict"])

    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "history": history,
    }


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Offline RRT re-embedding on existing UNI/Virchow features")

    parser.add_argument("--slides_csv", type=str, required=True)
    parser.add_argument("--feature_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--train_splits", nargs="+", default=["train"])
    parser.add_argument("--val_splits", nargs="+", default=["val"])
    parser.add_argument("--apply_splits", nargs="+", default=["train", "val", "test"])

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--att_dim", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # RRT args
    parser.add_argument("--rrt_layers", type=int, default=2)
    parser.add_argument("--rrt_heads", type=int, default=8)
    parser.add_argument("--rrt_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--rrt_qkv_bias", action="store_true")
    parser.add_argument("--rrt_attn_drop", type=float, default=0.0)
    parser.add_argument("--rrt_proj_drop", type=float, default=0.0)
    parser.add_argument("--rrt_drop_path", type=float, default=0.0)
    parser.add_argument("--rrt_with_cls_token", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.slides_csv)
    if "slide_id" not in df.columns and "image_id" in df.columns:
        df["slide_id"] = df["image_id"]
    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("slides_csv must contain label or slide_binary_label")
    if "split" not in df.columns:
        raise ValueError("slides_csv must contain split")

    feature_dir = Path(args.feature_dir)
    out_dir = Path(args.out_dir)
    out_pt_dir = out_dir / "pt_files"
    ensure_dir(out_pt_dir)

    train_df = df[df["split"].isin(args.train_splits)].copy().reset_index(drop=True)
    val_df = df[df["split"].isin(args.val_splits)].copy().reset_index(drop=True)
    apply_df = df[df["split"].isin(args.apply_splits)].copy().reset_index(drop=True)

    # collect train/val feature paths
    train_paths, train_labels = [], []
    for _, row in train_df.iterrows():
        pt_path = feature_dir / f"{row['slide_id']}.pt"
        if pt_path.exists():
            train_paths.append(pt_path)
            train_labels.append(int(row["label"]))

    val_paths, val_labels = [], []
    for _, row in val_df.iterrows():
        pt_path = feature_dir / f"{row['slide_id']}.pt"
        if pt_path.exists():
            val_paths.append(pt_path)
            val_labels.append(int(row["label"]))

    if len(train_paths) == 0:
        raise RuntimeError("No train .pt files found.")
    if len(val_paths) == 0:
        raise RuntimeError("No val .pt files found.")

    # infer input dim
    first_obj = load_pt_feature(train_paths[0])
    in_dim = int(first_obj["features"].shape[1])
    print(f"[Info] inferred input_dim = {in_dim}")

    # train re-embedding model
    train_out = train_rrt_reembed_model(
        train_slide_paths=train_paths,
        train_labels=train_labels,
        val_slide_paths=val_paths,
        val_labels=val_labels,
        in_dim=in_dim,
        device=device,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        att_dim=args.att_dim,
        n_layers=args.rrt_layers,
        n_heads=args.rrt_heads,
        mlp_ratio=args.rrt_mlp_ratio,
        qkv_bias=args.rrt_qkv_bias,
        attn_drop=args.rrt_attn_drop,
        proj_drop=args.rrt_proj_drop,
        drop_path=args.rrt_drop_path,
        with_cls_token=args.rrt_with_cls_token,
    )

    model = train_out["model"]
    history = train_out["history"]

    # save train ckpt
    torch.save(
        {
            "best_epoch": train_out["best_epoch"],
            "best_val_auc": train_out["best_val_auc"],
            "best_state": train_out["best_state"],
            "args": vars(args),
            "input_dim": in_dim,
            "output_dim": args.proj_dim,
        },
        out_dir / "rrt_reembed_ckpt.pth",
    )
    pd.DataFrame(history).to_csv(out_dir / "rrt_reembed_train_history.csv", index=False)

    # apply to all selected splits
    model.eval()
    saved_rows = []

    with torch.no_grad():
        for _, row in tqdm(apply_df.iterrows(), total=len(apply_df), desc="Re-embed features"):
            slide_id = str(row["slide_id"])
            pt_path = feature_dir / f"{slide_id}.pt"
            if not pt_path.exists():
                continue

            obj = load_pt_feature(pt_path)
            feats = obj["features"].to(device)              # [N, in_dim]
            new_feats = model.reembed(feats).cpu()          # [N, proj_dim]

            out_obj = dict(obj)
            out_obj["features"] = new_feats
            out_obj["feat_dim"] = int(new_feats.shape[1])
            out_obj["rrt_reembed"] = {
                "input_dim": int(in_dim),
                "output_dim": int(new_feats.shape[1]),
                "source_feature_path": str(pt_path),
            }

            save_path = out_pt_dir / f"{slide_id}.pt"
            save_pt_feature(save_path, out_obj)

            saved_rows.append({
                "slide_id": slide_id,
                "label": int(row["label"]),
                "split": str(row["split"]),
                "source_pt": str(pt_path),
                "save_pt": str(save_path),
                "num_instances": int(new_feats.shape[0]),
                "feat_dim": int(new_feats.shape[1]),
            })

    pd.DataFrame(saved_rows).to_csv(out_dir / "reembed_feature_meta.csv", index=False)

    summary = {
        "input_dim": int(in_dim),
        "output_dim": int(args.proj_dim),
        "n_train": int(len(train_paths)),
        "n_val": int(len(val_paths)),
        "n_saved": int(len(saved_rows)),
        "best_epoch": int(train_out["best_epoch"]),
        "best_val_auc": float(train_out["best_val_auc"]),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Done] saved to: {out_dir}")


if __name__ == "__main__":
    main()