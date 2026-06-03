from __future__ import annotations

import yaml
import torch
import torch.nn as nn

from models.encoders.moe_encoder import MoEEncoder
from models.plugins.shared_role_prototype import SharedRolePrototype


def load_stage2_for_plugin(
    config_path: str,
    full_ckpt_path: str,
    role_proto_dir: str,
    device: str = "cuda",
    shared_proto_learnable: bool = False,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt = torch.load(full_ckpt_path, map_location="cpu")

    if "student_state_dict" not in ckpt:
        raise KeyError("student_state_dict not found in checkpoint")
    if "distiller_state_dict" not in ckpt:
        raise KeyError("distiller_state_dict not found in checkpoint")

    encoder = MoEEncoder(cfg["base_encoder"], cfg["moe_encoder"])
    missing, unexpected = encoder.load_state_dict(ckpt["student_state_dict"], strict=False)
    print(f"[load encoder] missing keys: {len(missing)}")
    print(f"[load encoder] unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("  first missing:", missing[:10])
    if len(unexpected) > 0:
        print("  first unexpected:", unexpected[:10])
    encoder = encoder.to(device)
    encoder.eval()

    distiller_sd = ckpt["distiller_state_dict"]
    if "proj_l12.weight" not in distiller_sd or "proj_l12.bias" not in distiller_sd:
        raise KeyError("proj_l12 not found in distiller_state_dict")

    proj_out_dim, proj_in_dim = distiller_sd["proj_l12.weight"].shape
    role_proj_head = nn.Linear(proj_in_dim, proj_out_dim)
    role_proj_head.load_state_dict(
        {
            "weight": distiller_sd["proj_l12.weight"],
            "bias": distiller_sd["proj_l12.bias"],
        }
    )
    role_proj_head = role_proj_head.to(device)
    role_proj_head.eval()

    shared_role_proto = SharedRolePrototype.from_files(
        role_proto_dir=role_proto_dir,
        normalize=True,
        learnable=shared_proto_learnable,
        device=device,
    )

    print("Loaded stage2 bundle for plugin:")
    print(f"  encoder moe_layers_idx = {encoder.moe_layers_idx}")
    print(f"  role proj shape       = {proj_in_dim} -> {proj_out_dim}")
    print(f"  num roles             = {shared_role_proto.num_roles}")
    print(f"  proto dim             = {shared_role_proto.proto_dim}")
    print(f"  shared proto learnable= {shared_proto_learnable}")

    return encoder, role_proj_head, shared_role_proto, cfg
