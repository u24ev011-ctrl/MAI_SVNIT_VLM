import torch
import torch.nn as nn

class TPVLM(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.k = k

    def forward(self, tokens):
        # tokens: [B, C, H, W] -> flatten tokens
        B, C, H, W = tokens.shape
        flat = tokens.view(B, C, -1)  # [B, C, N]
        # simple magnitude-based pruning (placeholder)
        scores = flat.abs().mean(dim=1)  # [B, N]
        topk = torch.topk(scores, k=min(self.k, scores.size(-1)), dim=-1).indices
        pruned = torch.gather(flat, 2, topk.unsqueeze(1).expand(-1, C, -1))
        return pruned
