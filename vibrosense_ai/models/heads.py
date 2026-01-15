class ReconstructionHead(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(dim, out_dim)

    def forward(self, h):
        return self.fc(h)


class TrendHead(nn.Module):
    def __init__(self, dim, horizon=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )

    def forward(self, context):
        return self.fc(context)
