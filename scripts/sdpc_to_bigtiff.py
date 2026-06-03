import os
import re
import math
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pyvips
import opensdpc
from tqdm import tqdm


# =========================
# 全局临时目录
# =========================
TMP_ROOT = "/home/maxinyu/tmp"
os.environ["TMPDIR"] = TMP_ROOT
os.makedirs(TMP_ROOT, exist_ok=True)


# =========================
# 路径配置
# =========================
SRC_ROOT = Path("/data/zhangsj/hzey_data/病理切片")
BENIGN_SRC = SRC_ROOT / "良性"
MALIGNANT_SRC = SRC_ROOT / "恶性"

OUT_ROOT = Path("/data/maxinyu/WSI_WORKSPACE/data/Parotid/sdpc_to_tif")
BENIGN_OUT = OUT_ROOT / "Benign"
MALIGNANT_OUT = OUT_ROOT / "Malignant"

START_IDX = 150
MAX_FILES_PER_CLASS = 70

# =========================
# 转换参数
# =========================
READ_TILE = 512
SAVE_TILE = 256

# 体积控制关键参数：从 90 改到 75
JPEG_Q = 75
COMPRESSION = "jpeg"


def sanitize_name(name: str) -> str:
    """
    清洗文件名，避免中文/空格/特殊符号影响后续 CLAM 处理
    """
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-\.]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem if stem else "slide"


def read_region_rgb(slide, x, y, w, h, level=0):
    region = slide.read_region((x, y), level, (w, h))

    if hasattr(region, "convert"):
        arr = np.array(region.convert("RGB"))
    else:
        arr = np.asarray(region)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
        elif arr.shape[-1] != 3:
            raise ValueError(f"Unexpected patch shape: {arr.shape}")

    return arr


def sdpc_to_base_tif(
    sdpc_path: str,
    out_tif_path: str,
    read_tile: int = READ_TILE,
    save_tile: int = SAVE_TILE,
    compression: str = COMPRESSION,
    jpeg_q: int = JPEG_Q,
):
    """
    第一步：用 opensdpc + pyvips 生成中间 TIFF
    """
    slide = opensdpc.OpenSdpc(sdpc_path)
    width, height = slide.level_dimensions[0]

    os.makedirs(os.path.dirname(out_tif_path), exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sdpc2tif_", dir=TMP_ROOT) as tmp_dir:
        mmap_path = os.path.join(tmp_dir, "level0_rgb.dat")

        print(f"[INFO] reading: {sdpc_path}")
        print(f"[INFO] size   : {width} x {height}")
        print(f"[INFO] temp   : {mmap_path}")
        print(f"[INFO] out    : {out_tif_path}")

        canvas = np.memmap(
            mmap_path,
            dtype=np.uint8,
            mode="w+",
            shape=(height, width, 3)
        )

        nx = math.ceil(width / read_tile)
        ny = math.ceil(height / read_tile)
        total = nx * ny

        with tqdm(total=total, desc=f"Reading {Path(sdpc_path).name}", unit="tile") as pbar:
            for iy in range(ny):
                for ix in range(nx):
                    x = ix * read_tile
                    y = iy * read_tile
                    w = min(read_tile, width - x)
                    h = min(read_tile, height - y)

                    arr = read_region_rgb(slide, x, y, w, h, level=0)
                    canvas[y:y+h, x:x+w, :] = arr
                    del arr
                    pbar.update(1)

        canvas.flush()

        vimg = pyvips.Image.new_from_memory(
            canvas,
            width=width,
            height=height,
            bands=3,
            format="uchar"
        )

        vimg.tiffsave(
            out_tif_path,
            tile=True,
            tile_width=save_tile,
            tile_height=save_tile,
            pyramid=True,
            bigtiff=True,
            compression=compression,
            Q=jpeg_q,
            properties=True
        )

        del vimg
        del canvas

    print(f"[DONE] base tif saved: {out_tif_path}")
    return out_tif_path


def repack_with_vips(src_tif: str, dst_tif: str):
    """
    第二步：用 vips 命令行重写成 OpenSlide 稳定识别的 pyramidal BigTIFF
    """
    cmd = [
        "vips",
        "tiffsave",
        src_tif,
        dst_tif,
        "--tile",
        "--pyramid",
        "--bigtiff",
        "--tile-width", str(SAVE_TILE),
        "--tile-height", str(SAVE_TILE),
        "--compression", COMPRESSION,
        "--Q", str(JPEG_Q),
        "--properties",
    ]

    print("[CMD]", " ".join(cmd))
    env = os.environ.copy()
    env["TMPDIR"] = TMP_ROOT
    subprocess.run(cmd, check=True, env=env)
    print(f"[DONE] pyramid tif saved: {dst_tif}")


def convert_one(sdpc_path: Path, out_dir: Path):
    safe_name = sanitize_name(sdpc_path.name)
    final_tif = out_dir / f"{safe_name}.tif"

    print(f"[OUT] {sdpc_path.name} -> {final_tif}")

    if final_tif.exists():
        print(f"[SKIP] already exists: {final_tif}")
        return

    os.makedirs(out_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sdpc_mid_", dir=TMP_ROOT) as tmpdir:
        mid_tif = Path(tmpdir) / f"{safe_name}_mid.tif"

        sdpc_to_base_tif(
            sdpc_path=str(sdpc_path),
            out_tif_path=str(mid_tif),
            read_tile=READ_TILE,
            save_tile=SAVE_TILE,
            compression=COMPRESSION,
            jpeg_q=JPEG_Q,
        )

        repack_with_vips(str(mid_tif), str(final_tif))


def batch_convert(src_dir: Path, out_dir: Path, label_name: str):
    all_files = sorted(src_dir.rglob("*.sdpc"))
    sdpc_files = all_files[START_IDX: START_IDX + MAX_FILES_PER_CLASS]
    print(f"\n===== {label_name}: converting {len(sdpc_files)} / {len(all_files)} .sdpc files =====")

    for sdpc_path in tqdm(sdpc_files, desc=f"{label_name} slides", unit="slide"):
        try:
            convert_one(sdpc_path, out_dir)
        except Exception as e:
            print(f"\n[ERROR] failed on {sdpc_path}: {e}")


def main():
    BENIGN_OUT.mkdir(parents=True, exist_ok=True)
    MALIGNANT_OUT.mkdir(parents=True, exist_ok=True)

    batch_convert(BENIGN_SRC, BENIGN_OUT, "Benign")
    batch_convert(MALIGNANT_SRC, MALIGNANT_OUT, "Malignant")

    print("\nAll done.")


if __name__ == "__main__":
    main()