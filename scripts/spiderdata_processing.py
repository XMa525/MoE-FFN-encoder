import os
import glob
import tarfile
from .convert_to_imagenet import restore_dataset
from multiprocessing import cpu_count

# ----------------------------
# 配置参数
# ----------------------------
DATA_ROOT = "../data/raw"           # 每个 organ 的文件夹
OUTPUT_ROOT = "../data/HISTAI-spider"
CONTEXT_SIZE = 5
NUM_WORKERS = 8

ORGANS = ["SPIDER-skin", "SPIDER-colorectal", "SPIDER-thorax", "SPIDER-breast"]

OUTPUT_NAMES = {
    "SPIDER-skin": "Skin",
    "SPIDER-colorectal": "Colorectal",
    "SPIDER-thorax": "Thorax",
    "SPIDER-breast": "Breast"
}

# ----------------------------
# 1. 合并分卷 tar 并解压（带进度打印）
# ----------------------------
def merge_and_extract_tar(organ_dir):
    tar_parts = sorted(glob.glob(os.path.join(organ_dir, "*.tar.*")))
    if not tar_parts:
        raise FileNotFoundError(f"No tar.* files found in {organ_dir}")

    full_tar_path = os.path.join(organ_dir, "full.tar")
    print(f"  Merging {len(tar_parts)} tar parts into {full_tar_path} ...")
    with open(full_tar_path, "wb") as wfd:
        for i, part in enumerate(tar_parts):
            print(f"    Writing part {i+1}/{len(tar_parts)}: {os.path.basename(part)}")
            with open(part, "rb") as fd:
                wfd.write(fd.read())

    print(f"  Extracting {full_tar_path} ...")
    try:
        with tarfile.open(full_tar_path, "r") as tar:
            tar.extractall(path=organ_dir)
    except Exception as e:
        print(f"  [ERROR] Failed to extract {full_tar_path}: {e}")
        raise

    # 检查 metadata.json 是否存在
    metadata_path = os.path.join(organ_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"  [WARNING] metadata.json not found in {organ_dir}")
    else:
        print(f"  Found metadata.json in {organ_dir}")

    # 删除合并 tar 节省空间
    #os.remove(full_tar_path)
    return organ_dir

# ----------------------------
# 2. 处理单个 organ
# ----------------------------
def process_organ(organ_name):
    print(f"\n=== Processing organ: {organ_name} ===")
    organ_dir = os.path.join(DATA_ROOT, organ_name)

    try:
        extracted_dir = merge_and_extract_tar(organ_dir)
    except Exception as e:
        print(f"[ERROR] Failed to prepare {organ_name}: {e}")
        return

    output_dir = os.path.join(OUTPUT_ROOT, OUTPUT_NAMES[organ_name])
    print(f"  Restoring ImageNet-style dataset to {output_dir} ...")

    try:
        restore_dataset(
            data_dir=extracted_dir,
            output_dir=output_dir,
            restore_context_size=CONTEXT_SIZE,
            num_workers=NUM_WORKERS
        )
    except Exception as e:
        print(f"[ERROR] restore_dataset failed for {organ_name}: {e}")
        return

    print(f"=== Finished processing organ: {organ_name} ===")

# ----------------------------
# 3. 主程序
# ----------------------------
def main():
    for organ in ORGANS:
        process_organ(organ)
    print(f"\nAll organs processed. ImageNet-style dataset saved at {OUTPUT_ROOT}")


# notebook-friendly：直接调用 main()
if __name__ == "__main__":
    main()