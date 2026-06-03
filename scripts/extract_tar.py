import os
import tarfile
from glob import glob

def safe_extract_tar(tar_dir, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    tar_files = sorted(glob(os.path.join(tar_dir, "*.tar.*")))
    
    # 只从 tar.00 开始解压
    tar0 = [f for f in tar_files if f.endswith(".tar.00")]
    if not tar0:
        raise FileNotFoundError("No .tar.00 found!")
    tar0_path = tar0[0]

    with tarfile.open(tar0_path, "r") as tar:
        members_to_extract = []
        for m in tar.getmembers():
            target_path = os.path.join(extract_dir, m.name)
            if not os.path.exists(target_path):
                members_to_extract.append(m)
        if members_to_extract:
            tar.extractall(path=extract_dir, members=members_to_extract)
            print(f"Extracted {len(members_to_extract)} new files from {tar0_path}")
        else:
            print(f"All files from {tar0_path} already exist. Skipping.")