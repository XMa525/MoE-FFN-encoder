import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path


@torch.no_grad()
def extract_features(distiller, loader, device, save_path):
    distiller.eval()

    all_spec_feat = []
    all_dispatch_weight = []
    all_dispatch_mask = []
    all_cluster_ids = []
    all_labels = []
    all_paths = []

    for batch in tqdm(loader, desc="Extract Stage2 Features"):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        images = batch["image"]
        offline_cluster_ids = batch.get("offline_cluster_ids", None)

        # 手动 forward，便于拿中间结果
        distiller.tea_features.clear()
        _ = distiller.teacher(images)

        _, gate_info_list, feature_dict, moe_feature_list = distiller.student(
            images,
            return_gates=True,
            mask=None,
            is_eval=True,
            return_features=True,
            offline_cluster_ids=offline_cluster_ids,
        )

        if distiller.use_last_moe_output and len(moe_feature_list) > 0:
            spec_feat = moe_feature_list[-1]
        else:
            spec_feat = feature_dict["layer_12"]

        spec_patch = spec_feat[:, 1:, :]   # [B, N, D]
        B, N, D = spec_patch.shape

        dispatch_weight = distiller.get_last_dispatch_weight(gate_info_list, B, N)
        dispatch_mask = distiller.get_last_dispatch_mask(gate_info_list, B, N)

        all_spec_feat.append(spec_patch.cpu().numpy())
        all_dispatch_weight.append(dispatch_weight.cpu().numpy())
        all_dispatch_mask.append(dispatch_mask.cpu().numpy())

        if offline_cluster_ids is not None:
            all_cluster_ids.append(offline_cluster_ids.cpu().numpy())

        if "label" in batch:
            all_labels.append(batch["label"].cpu().numpy())

        if "path" in batch:
            all_paths.extend(batch["path"])

    save_dict = {
        "spec_feat": np.concatenate(all_spec_feat, axis=0),              # [B_all, N, D]
        "dispatch_weight": np.concatenate(all_dispatch_weight, axis=0),  # [B_all, N, E]
        "dispatch_mask": np.concatenate(all_dispatch_mask, axis=0),      # [B_all, N, E]
    }

    if len(all_cluster_ids) > 0:
        save_dict["cluster_ids"] = np.concatenate(all_cluster_ids, axis=0)
    if len(all_labels) > 0:
        save_dict["labels"] = np.concatenate(all_labels, axis=0)
    if len(all_paths) > 0:
        save_dict["paths"] = np.array(all_paths, dtype=object)

    np.savez_compressed(save_path, **save_dict)
    print(f"Saved to {save_path}")