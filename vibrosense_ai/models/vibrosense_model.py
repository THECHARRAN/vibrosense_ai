import torch
import torch.nn as nn
import torch.nn.functional as F
class TemporalEncoder(nn.Module):
    """
    Input: x [B, T, F]
    Output: h [B, T, D]
    Uses 1D convs with increasing dilation to capture temporal context.
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.project = nn.Conv1d(feature_dim, hidden_dim, kernel_size=1)
        self.block1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU()
        )
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.project(x)       
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        h = x.transpose(1, 2)    
        return h
class TemporalAttention(nn.Module):
    """
    Computes attention weights over time and returns a context vector.
    Input: h [B, T, D]
    Output: context [B, D], attn [B, T]
    """
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h):

        scores = self.score(h).squeeze(-1)
        attn = torch.softmax(scores, dim=1)

        context = (h * attn.unsqueeze(-1)).sum(dim=1)
        return context, attn
class ReconstructionHead(nn.Module):
    """
    Map per-time hidden features back to original feature_dim per time step.
    Input: h [B, T, D] -> output [B, T, F]
    """
    def __init__(self, hidden_dim, feature_dim):
        super().__init__()
        self.conv = nn.Conv1d(hidden_dim, feature_dim, kernel_size=1)
    def forward(self, h):
        x = h.transpose(1, 2)
        out = self.conv(x)
        return out.transpose(1, 2) 
class TrendHead(nn.Module):
    """
    Predict a short horizon of future risk scores / slopes from context vector.
    Input: context [B, D] -> output [B, horizon]
    """
    def __init__(self, hidden_dim, horizon=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//1),
            nn.ReLU(),
            nn.Linear(hidden_dim//1, horizon)
        )

    def forward(self, context):
        return self.net(context)
class VibroSenseV2(nn.Module):
    """
    Combines encoder, attention, recon head and trend head.
    Forward:
        inputs: x [B, T, F]
        outputs: recon [B, T, F], trend [B, horizon], attn [B, T]
    """
    def __init__(self, feature_dim, hidden_dim=64, horizon=5):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.encoder = TemporalEncoder(feature_dim, hidden_dim=hidden_dim)
        self.attn = TemporalAttention(hidden_dim)
        self.recon_head = ReconstructionHead(hidden_dim, feature_dim)
        self.trend_head = TrendHead(hidden_dim, horizon=horizon)

    def forward(self, x):
        """
        x: [B, T, F]
        recon: [B, T, F]
        trend: [B, horizon]
        attn: [B, T]
        """
        h = self.encoder(x)                 # [B, T, D]
        context, attn = self.attn(h)        # context: [B, D], attn: [B, T]
        recon = self.recon_head(h)          # [B, T, F]
        trend = self.trend_head(context)    # [B, horizon]
        return recon, trend, attn
