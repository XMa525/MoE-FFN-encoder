#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
import torch

KEEP_WORDS = [
    "moe",
    "expert",
    "experts",
    "router",
    "gate",
    "gating",
    "shared",
    "routing",
    "dispatch",
    "cluster_bias",
    "routing_proj",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_txt", type=str, default="")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["student_state_dict"]

    lines = []
    lines.append(f"[CKPT] {args.ckpt}")
    lines.append(f"[student_state_dict tensor count] {len(state)}")
    lines.append("")

    matched = []
    for k, v in state.items():
        lk = k.lower()
        if any(w in lk for w in KEEP_WORDS):
            matched.append((k, tuple(v.shape), str(v.dtype)))

    lines.append(f"[Matched moe-like keys] {len(matched)}")
    lines.append("")
    for i, (k, shape, dtype) in enumerate(matched):
        lines.append(f"{i:04d} | {k} | shape={shape} | dtype={dtype}")

    # 顺便统计各层
    lines.append("")
    lines.append("[Layer-like summary]")
    counter = {}
    for k, _, _ in matched:
        parts = k.split(".")
        # 尝试截到 encoder.layer.N
        layer_tag = None
        for j in range(len(parts) - 2):
            if parts[j] == "layer" and j > 0:
                layer_tag = ".".join(parts[max(0, j-2):j+2])
                break
        if layer_tag is None:
            layer_tag = parts[0]
        counter[layer_tag] = counter.get(layer_tag, 0) + 1

    for k in sorted(counter.keys()):
        lines.append(f"{k}: {counter[k]}")

    text = "\n".join(lines)
    print(text)

    if args.save_txt:
        Path(args.save_txt).write_text(text, encoding="utf-8")
        print(f"\n[Saved] {args.save_txt}")

if __name__ == "__main__":
    main()