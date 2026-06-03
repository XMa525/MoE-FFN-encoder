#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    import umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# =========================================================
# Utils
# =========================================================
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def stable_seed(seed: int, slide_id: str) -> int:
    h = hashlib.md5(str(slide_id).encode("utf-8")).hexdigest()
    return int(seed) + (int(h[:8], 16) % 100000)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Data helpers
# =========================================================
def read_split_csv(csv_path: str, split_names: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "slide_id" not in df.columns and "image_id" in df.columns:
        df["slide_id"] = df["image_id"]
    if "label" not in df.columns:
        if "slide_binary_label" in df.columns:
            df["label"] = df["slide_binary_label"]
        else:
            raise ValueError("split csv missing label / slide_binary_label")
    required = {"slide_id", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split csv missing columns: {missing}")
    return df[df["split"].isin(split_names)].copy().reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, per_split_total: int, seed: int) -> pd.DataFrame:
    parts = []
    for split, sub in df.groupby("split"):
        if len(sub) <= per_split_total:
            parts.append(sub.copy())
            continue

        labels = sorted(sub["label"].unique().tolist())
        total = len(sub)
        alloc: Dict[int, int] = {}
        for y in labels:
            alloc[y] = max(1, round(per_split_total * int((sub["label"] == y).sum()) / total))

        cur = sum(alloc.values())
        while cur > per_split_total:
            y = max(alloc, key=alloc.get)
            if alloc[y] > 1:
                alloc[y] -= 1
                cur -= 1
            else:
                break

        while cur < per_split_total:
            y = max(labels, key=lambda yy: int((sub["label"] == yy).sum()) - alloc.get(yy, 0))
            alloc[y] += 1
            cur += 1

        split_parts = []
        for y in labels:
            ss = sub[sub["label"] == y]
            k = min(len(ss), alloc[y])
            split_parts.append(ss.sample(n=k, random_state=seed))
        parts.append(pd.concat(split_parts, axis=0))

    out = pd.concat(parts, axis=0).drop_duplicates("slide_id").reset_index(drop=True)
    return out


def load_bag_feature(feature_dir: str, slide_id: str) -> Dict[str, Any]:
    path = Path(feature_dir) / f"{slide_id}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    obj = torch.load(path, map_location="cpu", weights_only=False)

    feats = obj["features"]
    coords = obj.get("coords", None)
    role_probs = obj.get("role_probs", None)

    if torch.is_tensor(feats):
        feats = feats.cpu().numpy()
    if coords is not None and torch.is_tensor(coords):
        coords = coords.cpu().numpy()
    if role_probs is not None and torch.is_tensor(role_probs):
        role_probs = role_probs.cpu().numpy()

    return {
        "features": np.asarray(feats, dtype=np.float32),
        "coords": np.asarray(coords, dtype=np.int64) if coords is not None else None,
        "role_probs": np.asarray(role_probs, dtype=np.float32) if role_probs is not None else None,
        "num_instances": int(obj.get("num_instances", len(feats))),
        "slide_id": str(obj.get("slide_id", slide_id)),
        "label": int(obj.get("label", -1)),
    }


def maybe_subsample_instances(
    feats: np.ndarray,
    coords: Optional[np.ndarray],
    role_probs: Optional[np.ndarray],
    max_instances: Optional[int],
    shuffle_instances: bool,
    seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    n = len(feats)
    if max_instances is None or n <= max_instances:
        idx = np.arange(n)
    else:
        if shuffle_instances:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=max_instances, replace=False)
        else:
            idx = np.arange(max_instances)

    feats = feats[idx]
    coords = coords[idx] if coords is not None else None
    role_probs = role_probs[idx] if role_probs is not None else None
    return feats, coords, role_probs


def sample_matched_patches(
    frozen_feats: np.ndarray,
    frozen_coords: Optional[np.ndarray],
    adapt_feats: np.ndarray,
    adapt_coords: Optional[np.ndarray],
    slide_id: str,
    seed: int,
    max_patches: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    rng = np.random.default_rng(stable_seed(seed, slide_id))

    if frozen_coords is not None and adapt_coords is not None:
        f_map = {tuple(map(int, c)): i for i, c in enumerate(frozen_coords)}
        a_map = {tuple(map(int, c)): i for i, c in enumerate(adapt_coords)}
        common = list(set(f_map.keys()) & set(a_map.keys()))
        if len(common) > 0:
            if len(common) > max_patches:
                sel = rng.choice(len(common), size=max_patches, replace=False)
                common = [common[i] for i in sel]
            f_idx = np.array([f_map[c] for c in common], dtype=np.int64)
            a_idx = np.array([a_map[c] for c in common], dtype=np.int64)
            return frozen_feats[f_idx], adapt_feats[a_idx], np.asarray(common, dtype=np.int64)

    n = min(len(frozen_feats), len(adapt_feats), max_patches)
    f_idx = rng.choice(len(frozen_feats), size=n, replace=False) if len(frozen_feats) > n else np.arange(n)
    a_idx = rng.choice(len(adapt_feats), size=n, replace=False) if len(adapt_feats) > n else np.arange(n)
    coords = frozen_coords[f_idx] if frozen_coords is not None else None
    return frozen_feats[f_idx], adapt_feats[a_idx], coords


# =========================================================
# Model definitions compatible with user's training script
# =========================================================
def build_tumor_role_aux(role_probs: torch.Tensor, topk: int = 50) -> torch.Tensor:
    if role_probs.ndim != 2 or role_probs.shape[1] < 2:
        raise ValueError(f"role_probs should be [N, R], got {tuple(role_probs.shape)}")

    tumor_prob = role_probs[:, 0]
    stroma_prob = role_probs[:, 1]
    role_id = torch.argmax(role_probs, dim=1)
    tumor_role_ratio = (role_id == 0).float().mean()
    k = min(topk, role_probs.shape[0])
    tumor_role_topk_mean = torch.topk(tumor_prob, k=k).values.mean()
    tumor_stroma_margin = (tumor_prob - stroma_prob).mean()

    return torch.stack(
        [tumor_role_ratio, tumor_role_topk_mean, tumor_stroma_margin], dim=0
    ).float()


class TorchmilABMILWrapper(nn.Module):
    def __init__(self, in_dim: int, device: str, att_dim: int = 128, gated: bool = False):
        super().__init__()
        try:
            from torchmil.models import ABMIL  # type: ignore
        except Exception as e:
            raise ImportError("torchmil is required in the runtime environment for ABMIL analysis") from e

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
        try:
            from torchmil.models import DSMIL  # type: ignore
        except Exception as e:
            raise ImportError("torchmil is required in the runtime environment for DSMIL analysis") from e

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
            raise ValueError(f"Unsupported bag_feats shape: {tuple(bag_feats.shape)}")
        return self.classifier(bag_repr)


class MILWithRoleAux(nn.Module):
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

    def forward(self, bag_feats: torch.Tensor, role_aux: Optional[torch.Tensor] = None):
        logits = self.base_model(bag_feats)
        if role_aux is None:
            return logits
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        elif logits.ndim == 2 and logits.shape[1] == 2:
            raise ValueError("MILWithRoleAux expects one-logit binary output, not [B,2]")
        return logits + self.aux_mlp(role_aux)


# =========================================================
# Checkpoint/model helpers
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
    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def load_checkpoint(ckpt_path: str, device: torch.device) -> Dict[str, Any]:
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(obj)}")
    return obj


def build_model_from_ckpt(
    ckpt: Dict[str, Any],
    override_feat_dim: Optional[int],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    args = ckpt.get("args", {}) or {}
    feat_dim = int(override_feat_dim if override_feat_dim is not None else ckpt.get("feat_dim", args.get("feat_dim", 0)))
    if feat_dim <= 0:
        raise ValueError("Could not infer feat_dim from checkpoint; pass --feat-dim")

    mil_model = args.get("mil_model", "abmil")
    att_dim = int(args.get("att_dim", 128))
    abmil_gated = bool(args.get("abmil_gated", False))
    use_role_aux = bool(args.get("use_role_aux", False))
    role_aux_hidden_dim = int(args.get("role_aux_hidden_dim", 16))

    if mil_model == "abmil":
        model: nn.Module = TorchmilABMILWrapper(feat_dim, str(device), att_dim=att_dim, gated=abmil_gated)
    elif mil_model == "dsmil":
        model = DSMILWrapper(
            feat_dim,
            str(device),
            att_dim=att_dim,
            nonlinear_q=bool(args.get("dsmil_nonlinear_q", False)),
            nonlinear_v=bool(args.get("dsmil_nonlinear_v", False)),
            dropout=float(args.get("dsmil_dropout", 0.0)),
        )
    elif mil_model == "meanpool":
        model = MeanPoolMIL(feat_dim, 1, str(device))
    else:
        raise ValueError(f"Unsupported mil_model from ckpt: {mil_model}")

    if use_role_aux:
        model = MILWithRoleAux(model, aux_dim=3, hidden_dim=role_aux_hidden_dim, device=str(device))

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    return model, {
        "feat_dim": feat_dim,
        "mil_model": mil_model,
        "att_dim": att_dim,
        "abmil_gated": abmil_gated,
        "use_role_aux": use_role_aux,
        "role_aux_hidden_dim": role_aux_hidden_dim,
        "max_instances": args.get("max_instances", None),
        "shuffle_instances": args.get("shuffle_instances", False),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "raw_args": args,
    }


# =========================================================
# Attention extraction
# =========================================================
def _unwrap_to_base_abmil(model: nn.Module) -> nn.Module:
    if isinstance(model, MILWithRoleAux):
        model = model.base_model
    if isinstance(model, TorchmilABMILWrapper):
        return model.model
    return model


def _normalize_attention_tensor(x: torch.Tensor, n_instances: int) -> Optional[torch.Tensor]:
    if not torch.is_tensor(x):
        return None
    t = x.detach().float().cpu()
    if t.numel() == 0:
        return None

    if t.ndim == 1 and t.shape[0] == n_instances:
        out = t
    elif t.ndim == 2 and 1 in t.shape and n_instances in t.shape:
        out = t.reshape(-1)
    elif t.ndim == 2 and t.shape[0] == 1 and t.shape[1] == n_instances:
        out = t[0]
    elif t.ndim == 3 and t.shape[0] == 1 and t.shape[1] == n_instances:
        out = t[0].reshape(n_instances)
    else:
        return None

    if out.numel() != n_instances:
        return None
    return out


def _extract_from_output(out: Any, n_instances: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    att_logits = None
    att_weights = None
    source = "none"

    if isinstance(out, dict):
        for k in ["att_logits", "attention_logits", "A_raw", "A_logits", "inst_att_logits"]:
            if k in out:
                att_logits = _normalize_attention_tensor(out[k], n_instances)
                if att_logits is not None:
                    source = f"dict:{k}"
                    break
        for k in ["att_weights", "attention", "weights", "A", "inst_att", "attention_weights"]:
            if k in out:
                att_weights = _normalize_attention_tensor(out[k], n_instances)
                if att_weights is not None:
                    source = f"dict:{k}"
                    break

    elif isinstance(out, (tuple, list)):
        tensors = [x for x in out if torch.is_tensor(x)]
        for i, t in enumerate(tensors):
            cand = _normalize_attention_tensor(t, n_instances)
            if cand is None:
                continue
            s = float(cand.sum())
            if 0.9 <= s <= 1.1 and (cand >= 0).all():
                att_weights = cand
                source = f"tuple:{i}:weights"
            else:
                att_logits = cand
                source = f"tuple:{i}:logits"

    return att_logits, att_weights, source


def extract_abmil_attention(
    model: nn.Module,
    bag_feats: torch.Tensor,
    role_aux: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    n_instances = int(bag_feats.shape[1] if bag_feats.ndim == 3 else bag_feats.shape[0])

    base = _unwrap_to_base_abmil(model)
    direct_out = None
    try:
        direct_out = base(bag_feats)
    except Exception:
        direct_out = None

    att_logits, att_weights, source = _extract_from_output(direct_out, n_instances)

    debug_rows = []

    # Hook fallback
    captured: List[Tuple[str, torch.Tensor]] = []
    hooks = []
    try:
        for name, module in base.named_modules():
            lname = name.lower()
            if any(k in lname for k in ["att", "attention", "gate"]):
                def _make_hook(hname):
                    def _hook(m, i, o):
                        if torch.is_tensor(o):
                            captured.append((hname, o.detach().cpu()))
                        elif isinstance(o, (tuple, list)):
                            for j, oo in enumerate(o):
                                if torch.is_tensor(oo):
                                    captured.append((f"{hname}[{j}]", oo.detach().cpu()))
                    return _hook
                hooks.append(module.register_forward_hook(_make_hook(name)))
        _ = base(bag_feats)
    finally:
        for h in hooks:
            h.remove()

    hook_best_weights = None
    hook_best_logits = None
    hook_best_source = "hook:none"
    hook_best_std = -1.0

    for name, t in captured:
        cand = _normalize_attention_tensor(t, n_instances)
        if cand is None:
            debug_rows.append({
                "name": name,
                "shape": list(t.shape),
                "numel": int(t.numel()),
                "std": float(t.float().std().item()) if t.numel() > 1 else 0.0,
                "sum": float(t.float().sum().item()) if t.numel() > 0 else 0.0,
                "normalized_ok": 0,
            })
            continue

        std = float(cand.std().item()) if cand.numel() > 1 else 0.0
        s = float(cand.sum().item())
        debug_rows.append({
            "name": name,
            "shape": list(t.shape),
            "numel": int(t.numel()),
            "std": std,
            "sum": s,
            "normalized_ok": 1,
        })

        if 0.9 <= s <= 1.1 and (cand >= 0).all():
            if std > hook_best_std:
                hook_best_weights = cand
                hook_best_source = f"hook:{name}:weights"
                hook_best_std = std
        else:
            if std > hook_best_std:
                hook_best_logits = cand
                hook_best_source = f"hook:{name}:logits"
                hook_best_std = std

    # Prefer hook result if direct output is degenerate
    if hook_best_weights is not None:
        if att_weights is None or float(att_weights.std().item()) < 1e-8:
            att_weights = hook_best_weights
            source = hook_best_source

    if hook_best_logits is not None:
        if att_logits is None or float(att_logits.std().item()) < 1e-8:
            att_logits = hook_best_logits
            if att_weights is None:
                source = hook_best_source

    with torch.no_grad():
        if isinstance(model, MILWithRoleAux):
            logits = model(bag_feats, role_aux=role_aux)
        else:
            logits = model(bag_feats)
        prob = logits_to_prob(logits).reshape(-1)[0].detach().cpu()

    if att_weights is None and att_logits is not None:
        att_weights = torch.softmax(att_logits, dim=0)

    if att_logits is None and att_weights is not None:
        att_logits = torch.log(torch.clamp(att_weights, min=1e-12))

    if att_weights is None:
        att_weights = torch.full((n_instances,), 1.0 / max(1, n_instances), dtype=torch.float32)
        att_logits = torch.log(torch.clamp(att_weights, min=1e-12))
        source = "uniform_fallback"

    return {
        "att_logits": att_logits.detach().cpu(),
        "att_weights": att_weights.detach().cpu(),
        "bag_prob": prob,
        "bag_logits": logits.detach().cpu(),
        "attention_source": source,
        "debug_rows": debug_rows,
    }


# =========================================================
# Representation analysis
# =========================================================
def fit_reducer(X: np.ndarray, seed: int, n_neighbors: int, min_dist: float):
    if HAS_UMAP:
        return umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric="cosine",
            random_state=seed,
        ).fit(X)
    return PCA(n_components=2, random_state=seed).fit(X)


def transform_reducer(reducer, X: np.ndarray) -> np.ndarray:
    return reducer.transform(X)


def attention_entropy(a: np.ndarray, eps: float = 1e-12) -> float:
    a = np.clip(a, eps, 1.0)
    return float(-(a * np.log(a)).sum())


def topk_mass(a: np.ndarray, frac: float) -> float:
    k = max(1, int(math.ceil(len(a) * frac)))
    idx = np.argsort(a)[-k:]
    return float(a[idx].sum())


def centroid_distance(X: np.ndarray, y: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    c0 = X[y == 0].mean(axis=0)
    c1 = X[y == 1].mean(axis=0)
    return float(np.linalg.norm(c1 - c0))


def plot_umap(ax, emb: np.ndarray, labels: np.ndarray, title: str):
    uniq = sorted(np.unique(labels).tolist())
    for y in uniq:
        m = labels == y
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.65, label=f"label={y}")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8)


def plot_umap_cluster(ax, emb: np.ndarray, cluster_ids: np.ndarray, title: str, n_show_legend: int = 12):
    uniq = sorted(np.unique(cluster_ids).tolist())
    cmap = plt.get_cmap("tab20")
    for i, cid in enumerate(uniq):
        m = cluster_ids == cid
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.70, color=cmap(i % 20), label=f"c={cid}")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if len(uniq) <= n_show_legend:
        ax.legend(frameon=False, fontsize=7, ncol=2)


def plot_attention_map(ax, coords: np.ndarray, att: np.ndarray, title: str):
    x = coords[:, 0]
    y = coords[:, 1]

    q_low = np.quantile(att, 0.02)
    q_high = np.quantile(att, 0.98)
    if q_high <= q_low:
        q_low = float(np.min(att))
        q_high = float(np.max(att) + 1e-12)

    sc = ax.scatter(x, -y, c=att, s=12, cmap="inferno", vmin=q_low, vmax=q_high)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser("Compatible ABMIL feature/attention analysis v2")
    parser.add_argument("--frozen-dir", type=str, required=True)
    parser.add_argument("--adapted-dir", type=str, required=True)
    parser.add_argument("--split-csv", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--sample-slides-per-split", type=int, default=20)
    parser.add_argument("--max-patches-per-slide", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--frozen-abmil-ckpt", type=str, default="")
    parser.add_argument("--adapted-abmil-ckpt", type=str, default="")
    parser.add_argument("--feat-dim", type=int, default=None)

    parser.add_argument("--attn-slide-ids", nargs="*", default=[])
    parser.add_argument("--num-auto-attn-slides", type=int, default=6)
    parser.add_argument("--prefer-test-for-attn", action="store_true")
    parser.add_argument("--max-attn-patches", type=int, default=None)

    parser.add_argument("--patch-clusters", type=int, default=10)
    parser.add_argument("--slide-clusters", type=int, default=6)
    parser.add_argument("--pca-dim-before-cluster", type=int, default=32)

    args = parser.parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(Path(args.out_dir) / "attention_maps")
    set_seed(args.seed)

    df = read_split_csv(args.split_csv, args.splits)
    df = stratified_sample(df, args.sample_slides_per_split, args.seed)
    df.to_csv(Path(args.out_dir) / "sampled_slides.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoints
    load_info: Dict[str, Any] = {}
    frozen_model = adapted_model = None
    frozen_cfg = adapted_cfg = None

    if args.frozen_abmil_ckpt:
        ckpt = load_checkpoint(args.frozen_abmil_ckpt, device)
        frozen_model, frozen_cfg = build_model_from_ckpt(ckpt, args.feat_dim, device)
        load_info["frozen"] = frozen_cfg

    if args.adapted_abmil_ckpt:
        ckpt = load_checkpoint(args.adapted_abmil_ckpt, device)
        adapted_model, adapted_cfg = build_model_from_ckpt(ckpt, args.feat_dim, device)
        load_info["adapted"] = adapted_cfg

    with open(Path(args.out_dir) / "abmil_load_info.json", "w", encoding="utf-8") as f:
        json.dump(load_info, f, indent=2, ensure_ascii=False)

    # =====================================================
    # Patch-level / slide-level representation
    # =====================================================
    frozen_patch_X, adapt_patch_X = [], []
    patch_labels = []
    patch_rows = []

    slide_frozen = []
    slide_adapt = []
    slide_labels = []
    slide_meta = []

    for _, row in df.iterrows():
        slide_id = str(row["slide_id"])
        y = int(row["label"])

        f_obj = load_bag_feature(args.frozen_dir, slide_id)
        a_obj = load_bag_feature(args.adapted_dir, slide_id)

        f_feats, a_feats, coords = sample_matched_patches(
            f_obj["features"], f_obj["coords"],
            a_obj["features"], a_obj["coords"],
            slide_id, args.seed, args.max_patches_per_slide,
        )

        frozen_patch_X.append(f_feats)
        adapt_patch_X.append(a_feats)
        patch_labels.extend([y] * len(f_feats))

        for i in range(len(f_feats)):
            patch_rows.append({
                "slide_id": slide_id,
                "label": y,
                "coord_x": int(coords[i, 0]) if coords is not None else -1,
                "coord_y": int(coords[i, 1]) if coords is not None else -1,
            })

        f_feats_mean, _, _ = maybe_subsample_instances(
            f_obj["features"], f_obj["coords"], f_obj["role_probs"],
            args.max_patches_per_slide, False, stable_seed(args.seed, slide_id)
        )
        a_feats_mean, _, _ = maybe_subsample_instances(
            a_obj["features"], a_obj["coords"], a_obj["role_probs"],
            args.max_patches_per_slide, False, stable_seed(args.seed, slide_id)
        )

        slide_frozen.append(f_feats_mean.mean(axis=0))
        slide_adapt.append(a_feats_mean.mean(axis=0))
        slide_labels.append(y)
        slide_meta.append({
            "slide_id": slide_id,
            "label": y,
            "split": row["split"],
        })

    frozen_patch_X = np.concatenate(frozen_patch_X, axis=0)
    adapt_patch_X = np.concatenate(adapt_patch_X, axis=0)
    patch_labels_np = np.asarray(patch_labels, dtype=np.int64)

    slide_frozen = np.stack(slide_frozen, axis=0)
    slide_adapt = np.stack(slide_adapt, axis=0)
    slide_labels_np = np.asarray(slide_labels, dtype=np.int64)

    # UMAP by label
    patch_reducer = fit_reducer(
        np.concatenate([frozen_patch_X, adapt_patch_X], axis=0),
        args.seed, 15, 0.1
    )
    f_patch_emb = transform_reducer(patch_reducer, frozen_patch_X)
    a_patch_emb = transform_reducer(patch_reducer, adapt_patch_X)

    patch_df = pd.DataFrame(patch_rows)
    patch_df["frozen_umap_x"] = f_patch_emb[:, 0]
    patch_df["frozen_umap_y"] = f_patch_emb[:, 1]
    patch_df["adapted_umap_x"] = a_patch_emb[:, 0]
    patch_df["adapted_umap_y"] = a_patch_emb[:, 1]
    patch_df.to_csv(Path(args.out_dir) / "patch_umap_points.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_umap(axes[0], f_patch_emb, patch_labels_np, "Frozen patch UMAP")
    plot_umap(axes[1], a_patch_emb, patch_labels_np, "Adapted patch UMAP")
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "patch_umap_by_label.png", dpi=200)
    plt.close(fig)

    slide_reducer = fit_reducer(
        np.concatenate([slide_frozen, slide_adapt], axis=0),
        args.seed, 10, 0.15
    )
    f_slide_emb = transform_reducer(slide_reducer, slide_frozen)
    a_slide_emb = transform_reducer(slide_reducer, slide_adapt)

    slide_df = pd.DataFrame(slide_meta)
    slide_df["frozen_umap_x"] = f_slide_emb[:, 0]
    slide_df["frozen_umap_y"] = f_slide_emb[:, 1]
    slide_df["adapted_umap_x"] = a_slide_emb[:, 0]
    slide_df["adapted_umap_y"] = a_slide_emb[:, 1]
    slide_df.to_csv(Path(args.out_dir) / "slide_umap_points.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_umap(axes[0], f_slide_emb, slide_labels_np, "Frozen slide UMAP")
    plot_umap(axes[1], a_slide_emb, slide_labels_np, "Adapted slide UMAP")
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "slide_umap_by_label.png", dpi=200)
    plt.close(fig)

    # =====================================================
    # Unified clustering + UMAP coloring
    # =====================================================
    patch_all = np.concatenate([frozen_patch_X, adapt_patch_X], axis=0)
    patch_pca_dim = min(args.pca_dim_before_cluster, patch_all.shape[1], max(2, patch_all.shape[0] - 1))
    patch_pca = PCA(n_components=patch_pca_dim, random_state=args.seed)
    patch_all_pca = patch_pca.fit_transform(patch_all)

    patch_kmeans = KMeans(n_clusters=args.patch_clusters, random_state=args.seed, n_init=10)
    patch_cluster_all = patch_kmeans.fit_predict(patch_all_pca)
    patch_cluster_f = patch_cluster_all[:len(frozen_patch_X)]
    patch_cluster_a = patch_cluster_all[len(frozen_patch_X):]

    patch_df["frozen_cluster"] = patch_cluster_f
    patch_df["adapted_cluster"] = patch_cluster_a
    patch_df.to_csv(Path(args.out_dir) / "patch_umap_points.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_umap_cluster(axes[0], f_patch_emb, patch_cluster_f, "Frozen patch UMAP (cluster)")
    plot_umap_cluster(axes[1], a_patch_emb, patch_cluster_a, "Adapted patch UMAP (cluster)")
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "patch_umap_by_cluster.png", dpi=200)
    plt.close(fig)

    patch_occ_rows = []
    for version, cluster_ids in [("frozen", patch_cluster_f), ("adapted", patch_cluster_a)]:
        vc = pd.Series(cluster_ids).value_counts(normalize=False).sort_index()
        vc_ratio = pd.Series(cluster_ids).value_counts(normalize=True).sort_index()
        for cid in sorted(vc.index.tolist()):
            patch_occ_rows.append({
                "level": "patch",
                "version": version,
                "cluster_id": int(cid),
                "count": int(vc[cid]),
                "ratio": float(vc_ratio[cid]),
            })
    pd.DataFrame(patch_occ_rows).to_csv(Path(args.out_dir) / "patch_cluster_occupancy.csv", index=False)

    slide_all = np.concatenate([slide_frozen, slide_adapt], axis=0)
    slide_pca_dim = min(args.pca_dim_before_cluster, slide_all.shape[1], max(2, slide_all.shape[0] - 1))
    slide_pca = PCA(n_components=slide_pca_dim, random_state=args.seed)
    slide_all_pca = slide_pca.fit_transform(slide_all)

    slide_kmeans = KMeans(n_clusters=args.slide_clusters, random_state=args.seed, n_init=10)
    slide_cluster_all = slide_kmeans.fit_predict(slide_all_pca)
    slide_cluster_f = slide_cluster_all[:len(slide_frozen)]
    slide_cluster_a = slide_cluster_all[len(slide_frozen):]

    slide_df["frozen_cluster"] = slide_cluster_f
    slide_df["adapted_cluster"] = slide_cluster_a
    slide_df.to_csv(Path(args.out_dir) / "slide_umap_points.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_umap_cluster(axes[0], f_slide_emb, slide_cluster_f, "Frozen slide UMAP (cluster)")
    plot_umap_cluster(axes[1], a_slide_emb, slide_cluster_a, "Adapted slide UMAP (cluster)")
    fig.tight_layout()
    fig.savefig(Path(args.out_dir) / "slide_umap_by_cluster.png", dpi=200)
    plt.close(fig)

    slide_occ_rows = []
    for version, cluster_ids in [("frozen", slide_cluster_f), ("adapted", slide_cluster_a)]:
        vc = pd.Series(cluster_ids).value_counts(normalize=False).sort_index()
        vc_ratio = pd.Series(cluster_ids).value_counts(normalize=True).sort_index()
        for cid in sorted(vc.index.tolist()):
            slide_occ_rows.append({
                "level": "slide",
                "version": version,
                "cluster_id": int(cid),
                "count": int(vc[cid]),
                "ratio": float(vc_ratio[cid]),
            })
    pd.DataFrame(slide_occ_rows).to_csv(Path(args.out_dir) / "slide_cluster_occupancy.csv", index=False)

    # =====================================================
    # Metrics
    # =====================================================
    metrics = []
    for level, Xf, Xa, y in [
        ("patch", frozen_patch_X, adapt_patch_X, patch_labels_np),
        ("slide", slide_frozen, slide_adapt, slide_labels_np),
    ]:
        for version, X in [("frozen", Xf), ("adapted", Xa)]:
            rec = {
                "level": level,
                "version": version,
                "n": int(len(X)),
                "centroid_distance": centroid_distance(X, y),
            }
            if len(np.unique(y)) > 1 and len(X) >= 10:
                try:
                    rec["silhouette"] = float(silhouette_score(X, y, metric="cosine"))
                except Exception:
                    rec["silhouette"] = float("nan")
            else:
                rec["silhouette"] = float("nan")
            metrics.append(rec)
    pd.DataFrame(metrics).to_csv(Path(args.out_dir) / "representation_metrics.csv", index=False)

    # =====================================================
    # Attention analysis
    # =====================================================
    attn_rows = []
    debug_rows = []

    chosen_slide_ids = list(args.attn_slide_ids)
    if not chosen_slide_ids and (frozen_model is not None and adapted_model is not None):
        df_att = df.copy()
        if args.prefer_test_for_attn and "test" in set(df["split"].tolist()):
            df_att = df[df["split"] == "test"].copy()

        pos = df_att[df_att["label"] == 1]["slide_id"].tolist()
        neg = df_att[df_att["label"] == 0]["slide_id"].tolist()
        k_each = max(1, args.num_auto_attn_slides // 2)
        chosen_slide_ids = pos[:k_each] + neg[:k_each]

    if frozen_model is not None and adapted_model is not None:
        for slide_id in chosen_slide_ids:
            row = df[df["slide_id"] == slide_id].iloc[0]
            y = int(row["label"])
            split = str(row["split"])

            f_obj = load_bag_feature(args.frozen_dir, slide_id)
            a_obj = load_bag_feature(args.adapted_dir, slide_id)

            f_max = args.max_attn_patches if args.max_attn_patches is not None else (frozen_cfg or {}).get("max_instances", None)
            a_max = args.max_attn_patches if args.max_attn_patches is not None else (adapted_cfg or {}).get("max_instances", None)

            f_shuffle = bool((frozen_cfg or {}).get("shuffle_instances", False))
            a_shuffle = bool((adapted_cfg or {}).get("shuffle_instances", False))

            f_feats_np, f_coords, f_role_probs = maybe_subsample_instances(
                f_obj["features"], f_obj["coords"], f_obj["role_probs"],
                f_max, f_shuffle, stable_seed(args.seed, slide_id)
            )
            a_feats_np, a_coords, a_role_probs = maybe_subsample_instances(
                a_obj["features"], a_obj["coords"], a_obj["role_probs"],
                a_max, a_shuffle, stable_seed(args.seed, slide_id)
            )

            f_feats = torch.from_numpy(f_feats_np).float().to(device).unsqueeze(0)
            a_feats = torch.from_numpy(a_feats_np).float().to(device).unsqueeze(0)

            f_role_aux = None
            if (frozen_cfg or {}).get("use_role_aux", False):
                if f_role_probs is None:
                    raise KeyError(f"{slide_id} missing role_probs but frozen ABMIL ckpt expects role aux")
                topk = int((frozen_cfg or {}).get("raw_args", {}).get("role_topk", 50))
                f_role_aux = build_tumor_role_aux(torch.from_numpy(f_role_probs).float(), topk=topk).unsqueeze(0).to(device)

            a_role_aux = None
            if (adapted_cfg or {}).get("use_role_aux", False):
                if a_role_probs is None:
                    raise KeyError(f"{slide_id} missing role_probs but adapted ABMIL ckpt expects role aux")
                topk = int((adapted_cfg or {}).get("raw_args", {}).get("role_topk", 50))
                a_role_aux = build_tumor_role_aux(torch.from_numpy(a_role_probs).float(), topk=topk).unsqueeze(0).to(device)

            f_out = extract_abmil_attention(frozen_model, f_feats, f_role_aux)
            a_out = extract_abmil_attention(adapted_model, a_feats, a_role_aux)

            f_att = f_out["att_weights"].numpy()
            a_att = a_out["att_weights"].numpy()

            attn_rows.append({
                "slide_id": slide_id,
                "split": split,
                "label": y,

                "frozen_bag_prob": float(f_out["bag_prob"]),
                "adapted_bag_prob": float(a_out["bag_prob"]),

                "frozen_att_entropy": attention_entropy(f_att),
                "adapted_att_entropy": attention_entropy(a_att),

                "frozen_top1_mass": float(np.max(f_att)),
                "adapted_top1_mass": float(np.max(a_att)),
                "frozen_top5pct_mass": topk_mass(f_att, 0.05),
                "adapted_top5pct_mass": topk_mass(a_att, 0.05),

                "frozen_attention_std": float(np.std(f_att)),
                "adapted_attention_std": float(np.std(a_att)),
                "frozen_attention_min": float(np.min(f_att)),
                "adapted_attention_min": float(np.min(a_att)),
                "frozen_attention_max": float(np.max(f_att)),
                "adapted_attention_max": float(np.max(a_att)),

                "frozen_attention_source": f_out["attention_source"],
                "adapted_attention_source": a_out["attention_source"],

                "n_frozen_instances": int(len(f_att)),
                "n_adapted_instances": int(len(a_att)),
            })

            for rr in f_out["debug_rows"]:
                debug_rows.append({
                    "slide_id": slide_id,
                    "version": "frozen",
                    **rr,
                })
            for rr in a_out["debug_rows"]:
                debug_rows.append({
                    "slide_id": slide_id,
                    "version": "adapted",
                    **rr,
                })

            def _top_table(coords, att, tag):
                idx = np.argsort(att)[::-1]
                topn = min(32, len(idx))
                out_df = pd.DataFrame({
                    "rank": np.arange(topn),
                    "coord_x": [int(coords[i, 0]) if coords is not None else -1 for i in idx[:topn]],
                    "coord_y": [int(coords[i, 1]) if coords is not None else -1 for i in idx[:topn]],
                    "attention": att[idx[:topn]],
                })
                out_df.to_csv(
                    Path(args.out_dir) / "attention_maps" / f"{slide_id}_{tag}_top_attention.csv",
                    index=False,
                )

            _top_table(f_coords, f_att, "frozen")
            _top_table(a_coords, a_att, "adapted")

            if f_coords is not None and a_coords is not None:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                sc0 = plot_attention_map(
                    axes[0], f_coords, f_att,
                    f"Frozen {slide_id} y={y} p={float(f_out['bag_prob']):.3f}"
                )
                sc1 = plot_attention_map(
                    axes[1], a_coords, a_att,
                    f"Adapted {slide_id} y={y} p={float(a_out['bag_prob']):.3f}"
                )
                fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04)
                fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)
                fig.tight_layout()
                fig.savefig(Path(args.out_dir) / "attention_maps" / f"{slide_id}_attention_compare.png", dpi=200)
                plt.close(fig)

    if attn_rows:
        pd.DataFrame(attn_rows).to_csv(Path(args.out_dir) / "attention_summary.csv", index=False)
    if debug_rows:
        pd.DataFrame(debug_rows).to_csv(Path(args.out_dir) / "attention_debug_candidates.csv", index=False)

    summary = {
        "sampled_n_slides": int(len(df)),
        "patch_n_points": int(len(frozen_patch_X)),
        "slide_n_points": int(len(slide_frozen)),
        "umap_backend": "umap" if HAS_UMAP else "pca",
        "attention_note": "Attention maps from pre-extracted bag features reflect sampled patch subsets. For dense WSI heatmaps, re-extract more patches for selected slides.",
        "patch_clusters": int(args.patch_clusters),
        "slide_clusters": int(args.slide_clusters),
    }
    with open(Path(args.out_dir) / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved analysis to", args.out_dir)


if __name__ == "__main__":
    main()