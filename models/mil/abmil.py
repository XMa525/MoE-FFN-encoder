import torch
import torch.nn as nn
import torch.nn.functional as F

class ABMIL(nn.Module):
    """
    Attention-based MIL
    """

    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, patch_feats):
        """
        patch_feats: [N, D]
        """
        attn_scores = self.attention(patch_feats)  # [N,1]
        attn_weights = torch.softmax(attn_scores, dim=0)  # [N,1]
        bag_feat = torch.sum(attn_weights * patch_feats, dim=0)  # [D]
        return bag_feat
