import os
import random
import math
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm


def simple_tissue_ratio(img, white_threshold=0.85):
    gray = ImageOps.grayscale(img)
    arr = np.asarray(gray).astype(np.float32) / 255.0
    tissue_mask = arr < white_threshold
    return tissue_mask.mean()


def collect_patch_paths(root):
    paths = []
    organs = sorted(os.listdir(root))
    for organ in organs:
        image_dir = os.path.join(root, organ, organ, "images")
        if not os.path.exists(image_dir):
            continue
        for fname in sorted(os.listdir(image_dir)):
            if fname.endswith((".png", ".jpg", ".jpeg")):
                paths.append(os.path.join(image_dir, fname))
    return paths


def make_montage(items, save_path, tile_size=128, ncols=8):
    n = len(items)
    nrows = math.ceil(n / ncols)
    from PIL import Image
    canvas = Image.new("RGB", (ncols * tile_size, nrows * tile_size), (255, 255, 255))

    for i, (img_path, ratio) in enumerate(items):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((tile_size, tile_size))
            draw = ImageDraw.Draw(img)
            draw.rectangle([(0, 0), (tile_size - 1, 16)], fill=(255, 255, 255))
            draw.text((3, 2), f"{ratio:.3f}", fill=(0, 0, 0))
        except Exception:
            img = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))

        x = (i % ncols) * tile_size
        y = (i // ncols) * tile_size
        canvas.paste(img, (x, y))

    canvas.save(save_path)
    print(f"Saved: {save_path}")


def main():
    root = "../data/raw"
    white_threshold = 0.90
    tissue_threshold = 0.10
    random_seed = 42

    # 关键：只抽一小部分
    sample_pool_size = 5000
    show_num = 64

    random.seed(random_seed)

    all_paths = collect_patch_paths(root)
    print(f"Total patches found: {len(all_paths)}")

    sample_paths = random.sample(all_paths, min(sample_pool_size, len(all_paths)))
    print(f"Randomly sampled: {len(sample_paths)}")

    kept = []
    filtered = []
    all_ratios = []

    for path in tqdm(sample_paths, desc="Checking sampled patches"):
        try:
            img = Image.open(path).convert("RGB")
            ratio = simple_tissue_ratio(img, white_threshold=white_threshold)
        except Exception:
            continue

        all_ratios.append(ratio)

        if ratio >= tissue_threshold:
            kept.append((path, ratio))
        else:
            filtered.append((path, ratio))

    print(f"Kept in sample: {len(kept)}")
    print(f"Filtered in sample: {len(filtered)}")

    os.makedirs("outputs/tissue_filter_quickcheck", exist_ok=True)

    if len(kept) > 0:
        kept_sample = random.sample(kept, min(show_num, len(kept)))
        make_montage(
            kept_sample,
            "outputs/tissue_filter_quickcheck/kept_random_64.png",
            tile_size=128,
            ncols=8
        )

    if len(filtered) > 0:
        filtered_sample = random.sample(filtered, min(show_num, len(filtered)))
        make_montage(
            filtered_sample,
            "outputs/tissue_filter_quickcheck/filtered_random_64.png",
            tile_size=128,
            ncols=8
        )

    # 重点看阈值附近
    near_filtered = [(p, r) for p, r in filtered if 0.05 <= r < 0.10]
    near_kept = [(p, r) for p, r in kept if 0.10 <= r < 0.15]

    if len(near_filtered) > 0:
        near_filtered_sample = random.sample(near_filtered, min(show_num, len(near_filtered)))
        make_montage(
            near_filtered_sample,
            "outputs/tissue_filter_quickcheck/near_filtered_0.05_0.10.png",
            tile_size=128,
            ncols=8
        )

    if len(near_kept) > 0:
        near_kept_sample = random.sample(near_kept, min(show_num, len(near_kept)))
        make_montage(
            near_kept_sample,
            "outputs/tissue_filter_quickcheck/near_kept_0.10_0.15.png",
            tile_size=128,
            ncols=8
        )


if __name__ == "__main__":
    main()