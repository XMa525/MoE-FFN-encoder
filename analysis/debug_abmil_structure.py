#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from torchmil.models import ABMIL


class ABMILWrapper(nn.Module):
    def __init__(self, in_dim: int, device: str, att_dim: int = 128, gated: bool = False):
        super().__init__()
        self.model = ABMIL(
            in_shape=(in_dim,),
            att_dim=att_dim,
            att_act="tanh",
            gated=gated,
        )
        self.to(device)

    def forward(self, x):
        return self.model(x)


class MILWithRoleAux(nn.Module):
    def __init__(self, base_model: nn.Module, aux_dim: int = 3, hidden_dim: int = 16, device: str = "cuda"):
        super().__init__()
        self.base_model = base_model
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.to(device)

    def forward(self, bag_feats: torch.Tensor, role_aux: torch.Tensor = None):
        logits = self.base_model(bag_feats)
        if role_aux is None:
            return logits
        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)
        return logits + self.aux_mlp(role_aux)


def load_checkpoint(ckpt_path: str, device: torch.device):
    return torch.load(ckpt_path, map_location=device, weights_only=False)


def build_model_from_ckpt(ckpt: dict, feat_dim: int | None, device: torch.device):
    args = ckpt.get("args", {}) or {}
    feat_dim = int(feat_dim if feat_dim is not None else ckpt.get("feat_dim", args.get("feat_dim", 0)))
    if feat_dim <= 0:
        raise ValueError("Could not infer feat_dim")

    mil_model = args.get("mil_model", "abmil")
    if mil_model != "abmil":
        raise ValueError(f"Expected abmil, got {mil_model}")

    att_dim = int(args.get("att_dim", 128))
    abmil_gated = bool(args.get("abmil_gated", False))
    use_role_aux = bool(args.get("use_role_aux", False))
    role_aux_hidden_dim = int(args.get("role_aux_hidden_dim", 16))

    model = ABMILWrapper(
        in_dim=feat_dim,
        device=str(device),
        att_dim=att_dim,
        gated=abmil_gated,
    )

    if use_role_aux:
        model = MILWithRoleAux(
            base_model=model,
            aux_dim=3,
            hidden_dim=role_aux_hidden_dim,
            device=str(device),
        )

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, args, missing, unexpected


def describe_obj(name: str, obj: Any, depth: int = 0):
    prefix = "  " * depth
    if torch.is_tensor(obj):
        print(f"{prefix}{name}: Tensor shape={tuple(obj.shape)} dtype={obj.dtype}")
    elif isinstance(obj, dict):
        print(f"{prefix}{name}: dict keys={list(obj.keys())}")
        for k, v in obj.items():
            describe_obj(f"{k}", v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name}: {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj):
            describe_obj(f"[{i}]", v, depth + 1)
    else:
        print(f"{prefix}{name}: {type(obj)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--feature_pt", type=str, required=True)
    parser.add_argument("--feat_dim", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_instances", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.ckpt, device)
    model, train_args, missing, unexpected = build_model_from_ckpt(ckpt, args.feat_dim, device)

    print("\n=== missing / unexpected ===")
    print("missing:", list(missing))
    print("unexpected:", list(unexpected))

    print("\n=== train args ===")
    print(json.dumps(train_args, indent=2, ensure_ascii=False))

    print("\n=== named_modules ===")
    base = model.base_model.model if hasattr(model, "base_model") else model.model
    for name, module in base.named_modules():
        print(name, "->", module.__class__.__name__)

    obj = torch.load(args.feature_pt, map_location="cpu", weights_only=False)
    feats = obj["features"].float()
    feats = feats[:args.max_instances].to(device).unsqueeze(0)

    print("\n=== forward output (base model only) ===")
    out = base(feats)
    describe_obj("base_out", out)

    print("\n=== forward hooks: tensor-producing modules ===")
    hooks = []
    captured = []

    def mk_hook(name):
        def hook(module, inputs, output):
            if torch.is_tensor(output):
                captured.append((name, tuple(output.shape)))
            elif isinstance(output, (list, tuple)):
                for i, x in enumerate(output):
                    if torch.is_tensor(x):
                        captured.append((f"{name}[{i}]", tuple(x.shape)))
            elif isinstance(output, dict):
                for k, x in output.items():
                    if torch.is_tensor(x):
                        captured.append((f"{name}.{k}", tuple(x.shape)))
        return hook

    for name, module in base.named_modules():
        hooks.append(module.register_forward_hook(mk_hook(name)))

    _ = base(feats)

    for h in hooks:
        h.remove()

    for name, shape in captured:
        print(name, "->", shape)


if __name__ == "__main__":
    main()