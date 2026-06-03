# import os
# import random
# from pathlib import Path

# import matplotlib.pyplot as plt
# from PIL import Image
# import openslide

# # =========================
# # Config
# # =========================
# root_dir = Path("../data/Parotid/sdpc_to_tif")

# benign_dir = root_dir / "Benign"
# malignant_dir = root_dir / "Malignant"

# save_dir = Path("outputs/parotid_preview")
# save_dir.mkdir(parents=True, exist_ok=True)
# save_path = save_dir / "parotid_benign_malignant_preview.png"

# num_each = 5
# seed = 42

# exts = {
#     ".tif", ".tiff", ".TIF", ".TIFF",
#     ".svs", ".SVS",
#     ".ndpi", ".NDPI",
#     ".mrxs", ".MRXS",
#     ".png", ".PNG",
#     ".jpg", ".jpeg", ".JPG", ".JPEG",
# }


# # =========================
# # Helpers
# # =========================
# def collect_images(folder: Path):
#     if not folder.exists():
#         print(f"[ERROR] Folder does not exist: {folder}")
#         return []

#     files = [
#         p for p in folder.rglob("*")
#         if p.is_file() and p.suffix in exts
#     ]
#     return sorted(files)


# def sample_files(files, n, seed=42):
#     if len(files) == 0:
#         raise ValueError("No images found. Please check root_dir, class folders, and file extensions.")

#     rng = random.Random(seed)
#     if len(files) < n:
#         print(f"[WARN] Only found {len(files)} images, using all of them.")
#         return files

#     return rng.sample(files, n)


# def load_wsi_preview_openslide(path: Path, max_size=1024):
#     """
#     Read a low-resolution WSI preview using OpenSlide.
#     This avoids loading the full-resolution WSI into memory.
#     """
#     path = str(path)

#     try:
#         slide = openslide.OpenSlide(path)

#         level_dims = slide.level_dimensions

#         # Choose the lowest-resolution level by default.
#         # This is usually enough for WSI overview preview.
#         level = len(level_dims) - 1
#         w, h = level_dims[level]

#         img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
#         slide.close()

#         img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
#         return img.copy()

#     except Exception as e:
#         print(f"[WARN] OpenSlide failed for {path}: {e}")
#         print("[WARN] Trying PIL fallback. This may be slow for very large images.")

#         # Fallback for normal png/jpg or small tif
#         img = Image.open(path).convert("RGB")
#         img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
#         return img.copy()


# # =========================
# # Main
# # =========================
# print(f"[INFO] Current working directory: {os.getcwd()}")
# print(f"[INFO] root_dir: {root_dir}")
# print(f"[INFO] benign_dir exists: {benign_dir.exists()} -> {benign_dir}")
# print(f"[INFO] malignant_dir exists: {malignant_dir.exists()} -> {malignant_dir}")

# benign_files = collect_images(benign_dir)
# malignant_files = collect_images(malignant_dir)

# print(f"[INFO] Found Benign: {len(benign_files)}")
# print(f"[INFO] Found Malignant: {len(malignant_files)}")

# print("\n[INFO] Example Benign files:")
# for p in benign_files[:5]:
#     print("  ", p)

# print("\n[INFO] Example Malignant files:")
# for p in malignant_files[:5]:
#     print("  ", p)

# benign_sample = sample_files(benign_files, num_each, seed=seed)
# malignant_sample = sample_files(malignant_files, num_each, seed=seed + 1)

# print("\n[INFO] Selected Benign:")
# for p in benign_sample:
#     print("  ", p.name)

# print("\n[INFO] Selected Malignant:")
# for p in malignant_sample:
#     print("  ", p.name)

# # =========================
# # Plot preview
# # =========================
# fig, axes = plt.subplots(
#     2,
#     num_each,
#     figsize=(num_each * 2.5, 5.2),
# )

# rows = [
#     ("Benign", benign_sample),
#     ("Malignant", malignant_sample),
# ]

# for row_idx, (label, paths) in enumerate(rows):
#     for col_idx in range(num_each):
#         ax = axes[row_idx, col_idx]
#         ax.set_xticks([])
#         ax.set_yticks([])

#         if col_idx >= len(paths):
#             ax.axis("off")
#             continue

#         path = paths[col_idx]
#         img = load_wsi_preview_openslide(path, max_size=1024)

#         ax.imshow(img)

#         short_name = path.stem
#         if len(short_name) > 22:
#             short_name = short_name[:22] + "..."

#         ax.set_title(f"{label}\n{short_name}", fontsize=8)

#         for spine in ax.spines.values():
#             spine.set_visible(False)

# plt.tight_layout()
# plt.savefig(save_path, dpi=300, bbox_inches="tight")
# plt.show()

# print(f"\n[INFO] Saved preview to: {save_path}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import openslide


# =========================
# Config
# =========================
wsi_path = Path(
    "../data/Parotid/sdpc_to_tif/Benign/F24-04890H01_H02.tif"
)

save_dir = Path(
    "outputs/parotid_preview"
)
save_dir.mkdir(parents=True, exist_ok=True)

case_name = "F24-04890H01_H02"
patch_save_dir = save_dir / f"{case_name}_patches"
patch_save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / f"{case_name}_patch_tiling_schematic.png"

patch_rows = 3
patch_cols = 4
num_patches = patch_rows * patch_cols

# 改成你真实 patch size；如果只是画图示意，512 或 1024 都可以
patch_size_level0 = 1024

thumb_max_size = 900
patch_vis_size = 512   # 单独保存 patch 的显示尺寸
dpi = 300


# =========================
# Helper functions
# =========================
def get_best_level_for_thumbnail(slide, max_size=900):
    dims = slide.level_dimensions
    for level in reversed(range(len(dims))):
        w, h = dims[level]
        if max(w, h) >= max_size:
            return level
    return len(dims) - 1


def read_thumbnail(slide, max_size=900):
    level = get_best_level_for_thumbnail(slide, max_size=max_size)
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img, level


def tissue_mask_from_thumbnail(img):
    arr = np.asarray(img).astype(np.uint8)
    mean = arr.mean(axis=2)
    diff = arr.max(axis=2) - arr.min(axis=2)

    # H&E tissue: not white background and has color variation
    mask = (mean < 235) & (diff > 8)
    return mask


def sample_tissue_patch_locations(
    slide,
    thumb_img,
    thumb_level,
    n=12,
    patch_size_level0=1024,
    min_tissue_ratio=0.30,
):
    mask = tissue_mask_from_thumbnail(thumb_img)
    th, tw = mask.shape

    level_downsample = slide.level_downsamples[thumb_level]
    patch_thumb = max(8, int(patch_size_level0 / level_downsample))
    stride = max(4, patch_thumb // 2)

    candidates = []

    for y in range(0, max(1, th - patch_thumb), stride):
        for x in range(0, max(1, tw - patch_thumb), stride):
            crop_mask = mask[y:y + patch_thumb, x:x + patch_thumb]
            tissue_ratio = crop_mask.mean()

            if tissue_ratio >= min_tissue_ratio:
                candidates.append((tissue_ratio, x, y))

    if len(candidates) == 0:
        raise RuntimeError(
            "No tissue-rich patch candidates found. "
            "Try lowering min_tissue_ratio or using a smaller patch_size_level0."
        )

    candidates = sorted(candidates, reverse=True)

    selected = []
    min_dist = patch_thumb * 1.5

    for score, x, y in candidates:
        keep = True
        for _, sx, sy in selected:
            if ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5 < min_dist:
                keep = False
                break
        if keep:
            selected.append((score, x, y))
        if len(selected) >= n:
            break

    if len(selected) < n:
        used = {(x, y) for _, x, y in selected}
        for score, x, y in candidates:
            if (x, y) not in used:
                selected.append((score, x, y))
            if len(selected) >= n:
                break

    locs_level0 = []
    for score, x, y in selected[:n]:
        x0 = int(x * level_downsample)
        y0 = int(y * level_downsample)
        locs_level0.append((x0, y0, score))

    return locs_level0


def read_patch(slide, x0, y0, patch_size_level0=1024):
    patch = slide.read_region(
        (int(x0), int(y0)),
        0,
        (patch_size_level0, patch_size_level0),
    ).convert("RGB")
    return patch


def draw_grid_on_thumbnail(ax, img, grid_rows=6, grid_cols=8):
    ax.imshow(img)
    ax.axis("off")

    w, h = img.size

    for i in range(1, grid_cols):
        x = i * w / grid_cols
        ax.plot([x, x], [0, h], color="white", lw=0.9, alpha=0.9)
        ax.plot([x, x], [0, h], color="black", lw=0.35, alpha=0.45)

    for j in range(1, grid_rows):
        y = j * h / grid_rows
        ax.plot([0, w], [y, y], color="white", lw=0.9, alpha=0.9)
        ax.plot([0, w], [y, y], color="black", lw=0.35, alpha=0.45)


# =========================
# Main
# =========================
if not wsi_path.exists():
    raise FileNotFoundError(f"WSI not found: {wsi_path}")

slide = openslide.OpenSlide(str(wsi_path))

thumb_img, thumb_level = read_thumbnail(slide, max_size=thumb_max_size)

locs = sample_tissue_patch_locations(
    slide,
    thumb_img=thumb_img,
    thumb_level=thumb_level,
    n=num_patches,
    patch_size_level0=patch_size_level0,
    min_tissue_ratio=0.30,
)

patches = []

for idx, (x0, y0, score) in enumerate(locs, start=1):
    patch = read_patch(
        slide,
        x0,
        y0,
        patch_size_level0=patch_size_level0,
    )

    # 保存原始大小 patch
    raw_patch_path = patch_save_dir / f"patch_{idx:03d}_x{x0}_y{y0}_raw.png"
    patch.save(raw_patch_path)

    # 保存 resize 后适合画图的 patch
    patch_vis = patch.resize(
        (patch_vis_size, patch_vis_size),
        Image.Resampling.LANCZOS,
    )
    vis_patch_path = patch_save_dir / f"patch_{idx:03d}_x{x0}_y{y0}_vis.png"
    patch_vis.save(vis_patch_path)

    patches.append(patch_vis)

slide.close()

print(f"[INFO] Saved individual patches to: {patch_save_dir}")


# =========================
# Plot schematic
# =========================
fig = plt.figure(figsize=(10.5, 4.2), dpi=dpi)

gs = fig.add_gridspec(
    nrows=1,
    ncols=3,
    width_ratios=[1.15, 0.20, 1.35],
    wspace=0.05,
)

# Left: WSI thumbnail with grid
ax_wsi = fig.add_subplot(gs[0, 0])
draw_grid_on_thumbnail(ax_wsi, thumb_img, grid_rows=6, grid_cols=8)
ax_wsi.set_title("WSI", fontsize=13, fontweight="bold", pad=6)

# Middle: arrow
ax_arrow = fig.add_subplot(gs[0, 1])
ax_arrow.axis("off")
ax_arrow.annotate(
    "",
    xy=(0.95, 0.5),
    xytext=(0.05, 0.5),
    xycoords="axes fraction",
    arrowprops=dict(
        arrowstyle="-|>",
        lw=2.0,
        color="black",
        mutation_scale=18,
    ),
)
ax_arrow.text(
    0.5,
    0.38,
    "tiling",
    ha="center",
    va="center",
    fontsize=10,
)

# Right: patch grid
right_gs = gs[0, 2].subgridspec(
    patch_rows,
    patch_cols,
    wspace=0.04,
    hspace=0.04,
)

for idx, patch in enumerate(patches):
    r = idx // patch_cols
    c = idx % patch_cols
    ax = fig.add_subplot(right_gs[r, c])
    ax.imshow(patch)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.4)
        spine.set_edgecolor("0.45")

fig.text(
    0.70,
    0.94,
    "Patch bag",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold",
)

plt.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.08)
plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
plt.show()

print(f"[INFO] Saved schematic to: {save_path}")