#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
from collections.abc import Mapping, Sequence
import torch


def short_type(x):
    if torch.is_tensor(x):
        return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
    return type(x).__name__


def summarize_obj(obj, prefix="", lines=None, max_depth=4, depth=0, max_items=30):
    if lines is None:
        lines = []

    if depth > max_depth:
        lines.append(f"{prefix}<max_depth_reached>")
        return lines

    if torch.is_tensor(obj):
        lines.append(f"{prefix}{short_type(obj)}")
        return lines

    if isinstance(obj, Mapping):
        lines.append(f"{prefix}dict(len={len(obj)})")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                lines.append(f"{prefix}  ... ({len(obj)-max_items} more keys)")
                break
            lines.append(f"{prefix}  [{repr(k)}] -> {short_type(v)}")
            if isinstance(v, (Mapping, list, tuple)) and depth < max_depth:
                summarize_obj(v, prefix + "    ", lines, max_depth, depth + 1, max_items)
        return lines

    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        lines.append(f"{prefix}{type(obj).__name__}(len={len(obj)})")
        for i, v in enumerate(obj[:max_items]):
            lines.append(f"{prefix}  [{i}] -> {short_type(v)}")
            if isinstance(v, (Mapping, list, tuple)) and depth < max_depth:
                summarize_obj(v, prefix + "    ", lines, max_depth, depth + 1, max_items)
        if len(obj) > max_items:
            lines.append(f"{prefix}  ... ({len(obj)-max_items} more items)")
        return lines

    lines.append(f"{prefix}{repr(obj)} ({type(obj).__name__})")
    return lines


def find_tensor_dicts(obj, path="root", results=None, max_depth=8):
    if results is None:
        results = []

    if max_depth < 0:
        return results

    if isinstance(obj, Mapping):
        tensor_keys = [k for k, v in obj.items() if torch.is_tensor(v)]
        if tensor_keys:
            results.append({
                "path": path,
                "num_tensor_items": len(tensor_keys),
                "sample_keys": tensor_keys[:20],
            })
        for k, v in obj.items():
            find_tensor_dicts(v, f"{path}.{k}", results, max_depth - 1)

    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        for i, v in enumerate(obj):
            find_tensor_dicts(v, f"{path}[{i}]", results, max_depth - 1)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_txt", type=str, default="")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")

    lines = []
    lines.append(f"[Checkpoint] {args.ckpt}")
    lines.append(f"[Top-level type] {type(ckpt).__name__}")
    lines.append("")

    lines.append("[Top-level summary]")
    summarize_obj(ckpt, lines=lines, max_depth=2, max_items=40)
    lines.append("")

    lines.append("[Tensor-dict candidates]")
    candidates = find_tensor_dicts(ckpt, path="root", max_depth=8)
    if not candidates:
        lines.append("No tensor-containing dict found.")
    else:
        for c in candidates:
            lines.append(f"- path={c['path']}, num_tensor_items={c['num_tensor_items']}")
            for k in c["sample_keys"]:
                lines.append(f"    sample_key: {k}")

    text = "\n".join(lines)
    print(text)

    if args.save_txt:
        Path(args.save_txt).write_text(text, encoding="utf-8")
        print(f"\n[Saved] {args.save_txt}")


if __name__ == "__main__":
    main()