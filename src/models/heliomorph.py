import torch
import torch.nn as nn
from .tsve import TSVE
from .tpvml import TPVLM

class HelioMorphEdge(nn.Module):
    def __init__(self, num_classes=6, dim=128, k=64):
        super().__init__()
        self.backbone = TSVE(dim=dim)
        self.pruner = TPVLM(k=k)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        tokens = self.backbone(x)          # [B,N,C]
        tokens = self.pruner(tokens)       # [B,K,C]
        feat = tokens.transpose(1,2)       # [B,C,K]
        feat = self.pool(feat).squeeze(-1) # [B,C]
        logits = self.head(feat)
        return logits, tokens
