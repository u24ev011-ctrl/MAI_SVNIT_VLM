import torch
import torch.nn as nn

class TPVLM(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.k = k

    def forward(self, tokens):
        # tokens: [B, N, C]
        B, N, C = tokens.shape
        scores = tokens.abs().mean(dim=-1)  # [B,N]
        topk = torch.topk(scores, k=min(self.k, N), dim=-1).indices
        pruned = torch.gather(tokens, 1, topk.unsqueeze(-1).expand(-1, -1, C))
        return pruned
