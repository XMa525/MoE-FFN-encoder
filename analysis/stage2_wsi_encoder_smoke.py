import os
import torch
from tqdm import tqdm
from PIL import Image
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from analysis.build_camelyon_stage2_dataset import CamelyonWSIBagDataset

# ===== 你项目里已有的 encoder 构建函数，按你真实路径改 =====
from downstream.extract_bag_features import build_extractor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 先构建 dataset
    dataset = CamelyonWSIBagDataset(
        csv_path="../data/CAMELYON17/stage2_wsi_train.csv",
        raw_dir="../data/CAMELYON17/images",
        h5_dir="../data/CAMELYON17/patches/patches",
        patch_size=256,
        read_level=0,
        resize_to=224,
        max_patches=32,        # 先小一点做 smoke test
        sample_mode="random",
        seed=42,
        return_pil=True,
        transform=None,
        check_files=True,
    )

    item = dataset[0]
    print("===== Dataset Sample =====")
    print("slide_id    :", item["slide_id"])
    print("label       :", item["label"])
    print("num_patches :", item["num_patches"])

    # 2) 构建 encoder
    class Args:
        encoder_name = "moe_stage1"   # 你当前想测哪个 encoder 就改这里
        device = "cuda"
        dinov2_path = "./pretrained_models/dinov2-small"
        virchow2_weight = "models/distill_teacher/Virchow2/pytorch_model.bin"
        moe_config = "configs/phase2.yaml"
        moe_ckpt = "results/distilled_best_model/moe_encoder_best.pth"

    args = Args()
    extractor = build_extractor(args)

    # 3) 分 batch 过 encoder
    images = item["images"]   # List[PIL]
    batch_size = 8
    feat_list = []

    for start in tqdm(range(0, len(images), batch_size), desc="Encoder Forward"):
        end = min(start + batch_size, len(images))
        batch_images = images[start:end]

        with torch.no_grad():
            feats = extractor.extract_features(batch_images)   # [B, D]
            feats = feats.cpu()

        feat_list.append(feats)

    patch_feats = torch.cat(feat_list, dim=0)   # [N, D]
    bag_mean = patch_feats.mean(dim=0, keepdim=True)   # [1, D]

    print("\n===== Encoder Smoke Test =====")
    print("patch_feats shape:", tuple(patch_feats.shape))
    print("bag_mean shape   :", tuple(bag_mean.shape))
    print("feat_dim         :", patch_feats.shape[1])
    print("label            :", item["label"])


if __name__ == "__main__":
    main()