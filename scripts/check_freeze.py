# scripts/check_freeze.py

import torch
from collections import defaultdict


def check_trainable_parameters(model):
    """
    打印所有 trainable 参数
    """
    print("\n========== TRAINABLE PARAMETERS ==========\n")

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():

        num = param.numel()
        total_params += num

        if param.requires_grad:
            trainable_params += num
            print(f"[TRAINABLE] {name:60s} {num/1e6:.4f} M")

    print("\n==========================================")
    print(f"Total parameters:     {total_params/1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params/1e6:.2f} M")
    print("==========================================\n")


def check_layerwise_trainable(model):
    """
    按 layer 统计 trainable 参数
    """
    print("\n========== LAYERWISE TRAINABLE ==========\n")

    layer_stats = defaultdict(int)

    for name, param in model.named_parameters():

        if param.requires_grad:

            parts = name.split(".")
            layer = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]

            layer_stats[layer] += param.numel()

    for layer, count in layer_stats.items():
        print(f"{layer:20s} {count/1e6:.4f} M")

    print("\n=========================================\n")


def verify_freeze_strategy(model):
    """
    检查是否有错误解冻
    只允许：
        - LayerNorm
        - blocks.9
        - blocks.10
        - blocks.11
    """

    print("\n========== VERIFY FREEZE STRATEGY ==========\n")

    error_found = False

    for name, param in model.named_parameters():

        if param.requires_grad:

            if not (
                "norm" in name
                or "ln" in name
                or name.startswith("base_encoder.model.encoder.layer.9")
                or name.startswith("base_encoder.model.encoder.layer.10")
                or name.startswith("base_encoder.model.encoder.layer.11")
                or "proj_l9" in name
                or "proj_l12" in name
            ):
                print("⚠️ Unexpected trainable param:", name)
                error_found = True

    if not error_found:
        print("✅ Freeze strategy is correct.")

    print("\n===========================================\n")


def full_freeze_report(model):
    """
    一次性输出完整检查
    """

    check_trainable_parameters(model)

    check_layerwise_trainable(model)

    verify_freeze_strategy(model)