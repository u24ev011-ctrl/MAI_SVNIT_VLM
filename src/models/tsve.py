import torch
import torch.nn as nn

class TinyTransformerBlock(nn.Module):
    def __init__(self, dim=128, heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TSVE(nn.Module):
    def __init__(self, in_ch=3, dim=128, depth=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 24, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, dim, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([TinyTransformerBlock(dim) for _ in range(depth)])

    def forward(self, x):
        x = self.stem(x)  # [B,C,H,W]
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)  # [B,N,C]
        for blk in self.blocks:
            x = blk(x)
        return x  # tokens
