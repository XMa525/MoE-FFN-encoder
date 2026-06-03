import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


class DINOv2Encoder(nn.Module):

    def __init__(
        self,
        model_name="facebook/dinov2-small",
        device="cuda",
        cache_dir="./pretrained_models"
    ):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        local_model_path = "./pretrained_models/dinov2-small"

        # ===== load pretrained backbone =====
        self.model = AutoModel.from_pretrained(
            local_model_path,
            local_files_only=True
        ).to(self.device)

        self.processor = AutoImageProcessor.from_pretrained(
            local_model_path,
            local_files_only=True
        )

        # ===== expose internal structure =====
        self.embeddings = self.model.embeddings
        self.blocks = self.model.encoder.layer   
        self.norm = self.model.layernorm

        self.embed_dim = self.model.config.hidden_size
        self.num_layers = len(self.blocks)

        print(f"[INFO] DINOv2 loaded.")
        print(f"[INFO] Layers: {self.num_layers}")
        print(f"[INFO] Embed dim: {self.embed_dim}")

    # ------------------------------------------------
    # patch embedding forward
    # ------------------------------------------------
    def patch_embed_forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        x = self.embeddings(pixel_values)
        return x

    # ------------------------------------------------
    # manual forward
    # ------------------------------------------------
    def forward(self, x,  return_tokens=False):

        # Patch + Pos embedding
        x = self.patch_embed_forward(x)   # [B, N, D]

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)[0]   # Dinov2Layer returns tuple

        # Final norm
        x = self.norm(x)

        if return_tokens:
            return x
        else:

            cls_token = x[:, 0, :]
            return cls_token
