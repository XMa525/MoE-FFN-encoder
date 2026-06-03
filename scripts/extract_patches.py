# scripts/extract_patches.py
import os
import openslide
import numpy as np
from PIL import Image

def extract_patches_from_wsi(wsi_path, output_dir, patch_size=256, level=0, stride=256):
    slide = openslide.OpenSlide(wsi_path)
    dims = slide.level_dimensions[level]
    w, h = dims

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
            patch_arr = np.array(patch)

            # 背景过滤：如果 patch 背景太多则跳过
            if patch_arr.mean() < 240:  # 简单阈值，你也可以用 Otsu
                patch.save(os.path.join(output_dir, f"{count:05d}.png"))
                count += 1

    slide.close()
    return count
