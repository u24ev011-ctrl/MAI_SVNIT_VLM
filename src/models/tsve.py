import torch
import torch.nn as nn

class TSVE(nn.Module):
    def __init__(self, in_ch=3, dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, dim, 3, 2, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.stem(x)
