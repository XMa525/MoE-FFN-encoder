import torch
from torchvision import transforms

class FeatureExtractor:
    def __init__(self, encoder, device, train_transform=None,test_transform=None):
        self.encoder = encoder
        self.device = device

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        self.train_transform = train_transform or transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        self.test_transform = test_transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        #self.transform = transform if transform is not None else lambda x: x

    @torch.no_grad()
    def extract_features(self, slide, coords, use_chunk=True, chunk_size=128,train=True):
        """
        use_patch: 是否按 chunk_size 分块
        """
        patch_feats_list = []
        transform = self.train_transform if train else self.test_transform
        # ===== sample patch 初始化 gating_probs_list =====
        if use_chunk:
            sample_patch = coords[:min(len(coords), chunk_size)]
            sample_patches = []
            for x, y in sample_patch:
                patch = slide.read_region(location=(int(x), int(y)), level=0, size=(256, 256)).convert("RGB")
                patch = transform(patch)
                sample_patches.append(patch)
            sample_patches = torch.stack(sample_patches).to(self.device)
        else:
            # 全图输入
            if isinstance(coords[0], torch.Tensor):
                sample_patches = torch.stack(coords).to(self.device)
            else:
                sample_patches = []
                for x, y in coords:
                    patch = slide.read_region(location=(int(x), int(y)), level=0, size=(256, 256)).convert("RGB")
                    patch = transform(patch)
                    sample_patches.append(patch)
                sample_patches = torch.stack(sample_patches).to(self.device)

        # ===== 初始化 gating_probs_list =====
        _, sample_gate = self.encoder(sample_patches, return_gates=True)
        num_layers = len(sample_gate) if isinstance(sample_gate, list) else 1
        gating_probs_list = [[] for _ in range(num_layers)]
        del sample_patches, sample_gate
        torch.cuda.empty_cache()

        # ===== 分块或全图处理 =====
        if use_chunk:
            for i in range(0, len(coords), chunk_size):
                coords_chunk = coords[i:i+chunk_size]
                patches_chunk = []
                for x, y in coords_chunk:
                    patch = slide.read_region(location=(int(x), int(y)), level=0, size=(256, 256)).convert("RGB")
                    patch = transform(patch)
                    patches_chunk.append(patch)
                patches_chunk = torch.stack(patches_chunk).to(self.device)

                feats, gate_probs = self.encoder(patches_chunk, return_gates=True)
                feats_cls = feats[:, 0, :].cpu()
                patch_feats_list.append(feats_cls)

                if isinstance(gate_probs, list):
                    for l, layer_gate in enumerate(gate_probs):
                        gating_probs_list[l].append(layer_gate)
                else:
                    gating_probs_list[0].append(gate_probs)

                del patches_chunk, feats, feats_cls
                torch.cuda.empty_cache()
        else:
            feats, gate_probs = self.encoder(sample_patches, return_gates=True)
            feats_cls = feats[:, 0, :].cpu()
            patch_feats_list.append(feats_cls)

            if isinstance(gate_probs, list):
                for l, layer_gate in enumerate(gate_probs):
                    gating_probs_list[l].append(layer_gate)
            else:
                gating_probs_list[0].append(gate_probs)

            del feats, feats_cls, sample_patches
            torch.cuda.empty_cache()

        return patch_feats_list, gating_probs_list