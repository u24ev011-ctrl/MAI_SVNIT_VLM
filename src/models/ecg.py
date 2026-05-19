import torch
import torch.nn as nn

class ECG(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, thermal_feat=None):
        if thermal_feat is None:
            return rgb_feat
        g = self.gate(thermal_feat)
        return rgb_feat + g * thermal_feat
