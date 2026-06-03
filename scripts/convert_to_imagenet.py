import os
import json
import argparse
import shutil
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool, cpu_count
import functools

# Constant: patches are always 224 pixels.
PATCH_SIZE = 224


def process_record(
    record,
    images_dir,
    output_dir,
    restore_context_size,
    lower_bound,
    upper_bound,
    center_index,
):
    """
    Process a single record:
      - If restore_context_size is 1, copy the center patch.
      - Otherwise, stitch the patches (selected from the grid) into one image and save it.
    """
    split = record.get("split", "train")
    cls = record["class"]
    slide = record["slide_id"]

    # Create destination directory: <output_dir>/<split>/<class>/<slide>/
    dest_dir = os.path.join(output_dir, split, cls, slide)
    os.makedirs(dest_dir, exist_ok=True)

    # For context size 1: simply copy the center patch.
    if restore_context_size == 1:
        src_path = os.path.join(images_dir, record["image_name"])
        dst_path = os.path.join(dest_dir, record["image_name"])
        try:
            shutil.copy2(src_path, dst_path)
        except Exception as e:
            print(f"Error copying {src_path} to {dst_path}: {e}")
        return

    # For context sizes > 1: stitch patches together.
    stitched_width = restore_context_size * PATCH_SIZE
    stitched_height = restore_context_size * PATCH_SIZE
    stitched_image = Image.new("RGB", (stitched_width, stitched_height))

    # Loop over the desired grid positions from lower_bound to upper_bound (inclusive)
    for i in range(lower_bound, upper_bound + 1):
        for j in range(lower_bound, upper_bound + 1):
            # For the center patch (always at the full grid's center (2,2) for a 5x5 grid),
            # use the image from "image_name". Otherwise, use "context_info".
            if i == center_index and j == center_index:
                patch_filename = record["image_name"]
            else:
                key = f"{i}_{j}"
                if key not in record["context_info"]:
                    print(f"Warning: Missing patch {key} for slide {slide}.")
                    continue
                patch_filename = record["context_info"][key]

            patch_path = os.path.join(images_dir, patch_filename)
            try:
                patch_img = Image.open(patch_path)
            except Exception as e:
                print(f"Error opening patch {patch_path}: {e}")
                continue

            # Compute paste position in the stitched image.
            paste_x = (j - lower_bound) * PATCH_SIZE
            paste_y = (i - lower_bound) * PATCH_SIZE
            stitched_image.paste(patch_img, (paste_x, paste_y))

    # Save the stitched image under the central patch's filename.
    output_filename = record["image_name"]
    output_path = os.path.join(dest_dir, output_filename)
    try:
        stitched_image.save(output_path)
    except Exception as e:
        print(f"Error saving stitched image {output_path}: {e}")


def restore_dataset(data_dir, output_dir, restore_context_size, num_workers):
    # Define paths based on data_dir.
    metadata_file = os.path.join(data_dir, "metadata.json")
    images_dir = os.path.join(data_dir, "images")

    # Create output directory if it does not exist.
    os.makedirs(output_dir, exist_ok=True)

    # Load metadata records.
    with open(metadata_file, "r") as f:
        records = json.load(f)

    full_grid_size = 5
    center_index = full_grid_size // 2
    lower_bound = (full_grid_size - restore_context_size) // 2
    upper_bound = lower_bound + restore_context_size - 1

    print(f"Processing {len(records)} records with {num_workers} worker(s)...")

    worker_func = functools.partial(
        process_record,
        images_dir=images_dir,
        output_dir=output_dir,
        restore_context_size=restore_context_size,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        center_index=center_index,
    )

    with Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(worker_func, records), total=len(records)):
            pass

    print("Restoration complete. Restored dataset is saved in:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert to ImageNet-style dataset by stitching patches using multiprocessing. "
        "Assumes metadata.json and images folder are in the same directory (--data_dir)."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing metadata.json and images folder",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to output the restored dataset structure",
    )
    parser.add_argument(
        "--context_size",
        type=int,
        choices=[1, 3, 5],
        default=5,
        help="Context size to restore: 5 (full grid), 3 (central 3x3), or 1 (only center).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count(),
        help="Number of worker processes (default: number of CPU cores)",
    )
    args = parser.parse_args()

    restore_dataset(args.data_dir, args.output_dir, args.context_size, args.num_workers)
