# models/heads/classifier.py
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, num_classes=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)
