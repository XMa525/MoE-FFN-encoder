import torch
from torch.utils.data import DataLoader
from datasets.patch_dataset import PatchDataset
from models.encoders.dinov2_encoder import DINOv2Encoder
from models.mil.abmil import ABMIL
from models.heads.classifier import Classifier
from trainers.mil_trainer import MILTrainer

# ---------- Dataset ----------
dataset = PatchDataset(
    patch_root="data/patches",
    label_csv="data/raw/labels.csv",
)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# ---------- Models ----------
encoder = DINOv2Encoder()
mil_model = ABMIL(in_dim=encoder.feature_dim)
classifier = Classifier(in_dim=encoder.feature_dim, hidden_dim=256, num_classes=2)

# ---------- Trainer ----------
trainer = MILTrainer(
    encoder=encoder,
    mil_model=mil_model,
    classifier=classifier,
    dataloader=dataloader,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# ---------- Train Phase 0 ----------
trainer.train_epoch()
