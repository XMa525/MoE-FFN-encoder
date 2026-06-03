# scripts/extract_patches_clam.py
import os
import subprocess
from pathlib import Path

RAW_DIR = "data/raw"
PATCH_DIR = "data/patches"

def run_clam_patch_extraction():
    os.makedirs(PATCH_DIR, exist_ok=True)

    for cls in ["normal1", "tumor1"]:
        wsi_dir = Path(RAW_DIR) / cls
        out_dir = Path(PATCH_DIR) / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        for wsi_path in wsi_dir.glob("*.tif"):
            cmd = [
                "python", "clam/create_patches.py",
                "--source", str(wsi_path),
                "--save_dir", str(out_dir),
                "--patch_size", "256",
                "--step_size", "256",
                "--seg",
                "--patch"
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

if __name__ == "__main__":
    run_clam_patch_extraction()
