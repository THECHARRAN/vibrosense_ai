import torch
import torch.nn as nn
class TemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h):
        # h: [B, T, D]
        w = self.score(h).squeeze(-1)
        a = torch.softmax(w, dim=1)
        context = (h * a.unsqueeze(-1)).sum(dim=1)
        return context, a
