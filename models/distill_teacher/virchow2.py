import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image



class Virchow2FeatureExtractor:
    def __init__(self,  device="cuda"):
        self.device = device
       

        self.model = timm.create_model(
            "vit_huge_patch14_224",
            pretrained=False,
            num_classes=0,
            reg_tokens=4,
            mlp_ratio=5.3375, 
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            init_values=1e-5
        )

        state_dict = torch.load(
            "models/distill_teacher/Virchow2/pytorch_model.bin",
            map_location="cpu"
        )

        # 有些HF权重会带 "model." 前缀
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        
        # Strict loading first
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print("⚠️ Strict loading failed, trying relaxed loading...")
            self.model.load_state_dict(state_dict, strict=False)

        # ----------------------------
        # Positional embedding safety patch
        # ----------------------------
        if not hasattr(self.model, "pos_embed") or self.model.pos_embed is None:
            num_patches = self.model.patch_embed.num_patches + 1
            embed_dim = self.model.embed_dim

            self.model.pos_embed = torch.nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )

        self.model = self.model.eval().to(self.device)
        # Transform 配置
        self.transforms = create_transform(
            input_size=(3, 224, 224),
            interpolation='bicubic',
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_pct=1.0
        )
        print("Model loaded successfully.")
    @torch.no_grad()
    def extract(self, images):
        """
        输入 PIL.Image
        输出 embedding: CLS token + patch mean
        """
        #x = self.transforms(pil_image).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        # 如果传入的是单张图，包装成列表
        if isinstance(images, Image.Image):
            images = [images]
            
        # 统一处理：List[PIL] -> Tensor[B, 3, 224, 224]
        x = torch.stack([self.transforms(img) for img in images]).to(self.device)
        output = self.model(x)  # [1, 261, 1280]
        class_token = output[:, 0]         # [1, 1280]
        #patch_tokens = output[:, 5:]       # [1, 256, 1280]
        reg_tokens = getattr(self.model, "reg_tokens", 0)
        patch_start = 1 + reg_tokens

        patch_tokens = output[:, patch_start:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)  # [1, 2560]
        return embedding