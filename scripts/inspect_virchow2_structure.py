#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
import timm
from timm.layers import SwiGLUPacked


def load_virchow2_model(weight_path: str, device: str = "cuda") -> nn.Module:
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = timm.create_model(
        "vit_huge_patch14_224",
        pretrained=False,
        num_classes=0,
        reg_tokens=4,
        mlp_ratio=5.3375,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        init_values=1e-5,
    )

    state_dict = torch.load(weight_path, map_location="cpu")

    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "encoder" in state_dict:
            state_dict = state_dict["encoder"]

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("model.", "")
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v
    state_dict = new_state_dict

    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[Virchow2] strict load success: {weight_path}")
    except Exception as e:
        print(f"[Virchow2] strict load failed, fallback strict=False: {e}")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Virchow2] missing keys ({len(missing)}): {missing[:20]}")
        print(f"[Virchow2] unexpected keys ({len(unexpected)}): {unexpected[:20]}")

    if not hasattr(model, "pos_embed") or model.pos_embed is None:
        num_patches = model.patch_embed.num_patches + 1
        embed_dim = model.embed_dim
        model.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        print("[Virchow2] pos_embed was missing, created a fallback one.")

    model = model.to(device)
    model.eval()
    return model


def summarize_module_tree(module: nn.Module, max_depth: int = 2, prefix: str = ""):
    lines = []

    def _rec(m: nn.Module, depth: int, name: str):
        indent = "  " * depth
        lines.append(f"{indent}{name}: {m.__class__.__name__}")
        if depth >= max_depth:
            return
        for child_name, child in m.named_children():
            _rec(child, depth + 1, child_name)

    _rec(module, 0, prefix if prefix else module.__class__.__name__)
    return lines


def find_linear_modules(module: nn.Module):
    found = []
    for name, sub in module.named_modules():
        if isinstance(sub, nn.Linear):
            found.append((name, sub.in_features, sub.out_features))
    return found


def inspect_block(idx: int, block: nn.Module):
    print("=" * 100)
    print(f"[Block {idx}] class = {block.__class__.__name__}")

    print("[Direct children]")
    for name, sub in block.named_children():
        print(f"  - {name}: {sub.__class__.__name__}")

    print("\n[Block tree, depth<=2]")
    for line in summarize_module_tree(block, max_depth=2, prefix=f"block_{idx}"):
        print(line)

    print("\n[All Linear layers inside block]")
    linears = find_linear_modules(block)
    if len(linears) == 0:
        print("  (none)")
    else:
        for name, fin, fout in linears:
            print(f"  - {name}: {fin} -> {fout}")

    # 尝试标记最可能的 FFN / MLP 模块
    print("\n[FFN/MLP-like candidates]")
    found_ffn = False
    for name, sub in block.named_children():
        lname = name.lower()
        lcls = sub.__class__.__name__.lower()
        if any(k in lname for k in ["mlp", "ffn"]) or any(k in lcls for k in ["mlp", "swiglu"]):
            found_ffn = True
            print(f"  * candidate: {name} ({sub.__class__.__name__})")
            sub_linears = find_linear_modules(sub)
            if len(sub_linears) == 0:
                print("    no Linear found inside this candidate")
            else:
                for sname, fin, fout in sub_linears:
                    print(f"    - {sname}: {fin} -> {fout}")
    if not found_ffn:
        print("  (none found by name/class heuristic)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight_path",
        type=str,
        required=True,
        help="Local Virchow2 weight path, e.g. models/distill_teacher/Virchow2/pytorch_model.bin",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--inspect_last_n", type=int, default=4)
    args = parser.parse_args()

    print("[1] Loading Virchow2...")
    model = load_virchow2_model(args.weight_path, device=args.device)

    print("\n[2] Top-level summary]")
    print(f"model class: {model.__class__.__name__}")
    print(f"embed_dim: {getattr(model, 'embed_dim', 'N/A')}")
    print(f"num_features: {getattr(model, 'num_features', 'N/A')}")
    print(f"reg_tokens: {getattr(model, 'reg_tokens', 'N/A')}")

    for attr in ["patch_embed", "blocks", "norm", "fc_norm", "head"]:
        has_attr = hasattr(model, attr)
        print(f"has {attr}: {has_attr}")
        if has_attr:
            obj = getattr(model, attr)
            print(f"  -> type: {obj.__class__.__name__}")

    print("\n[3] Top-level named_children()]")
    for name, sub in model.named_children():
        print(f"  - {name}: {sub.__class__.__name__}")

    if not hasattr(model, "blocks"):
        print("\n[ERROR] model has no `blocks` attribute.")
        return

    blocks = model.blocks
    print("\n[4] Blocks summary]")
    print(f"blocks type: {blocks.__class__.__name__}")
    print(f"num blocks: {len(blocks)}")

    n = len(blocks)
    start = max(0, n - args.inspect_last_n)
    print(f"\n[5] Inspecting last {args.inspect_last_n} blocks: {list(range(start, n))}\n")

    for i in range(start, n):
        inspect_block(i, blocks[i])

    print("\n[6] patch_embed summary]")
    if hasattr(model, "patch_embed"):
        pe = model.patch_embed
        print(f"patch_embed class: {pe.__class__.__name__}")
        for name, sub in pe.named_children():
            print(f"  - {name}: {sub.__class__.__name__}")

        pe_linears = find_linear_modules(pe)
        if len(pe_linears) > 0:
            print("[Linear layers in patch_embed]")
            for name, fin, fout in pe_linears:
                print(f"  - {name}: {fin} -> {fout}")
        else:
            print("No Linear found in patch_embed (likely Conv2d-based).")


if __name__ == "__main__":
    main()