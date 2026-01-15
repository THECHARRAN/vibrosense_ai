import torch
from torch.utils.data import Dataset


class VibrosenseDataset(Dataset):
    def __init__(self, sequences, trend_targets=None):
        """
        sequences: np.ndarray or torch.Tensor [N, T, F]
        trend_targets: np.ndarray or torch.Tensor [N, horizon] or None
        """
        if isinstance(sequences, torch.Tensor):
            self.X = sequences.float()
        else:
            self.X = torch.tensor(sequences, dtype=torch.float32)

        if trend_targets is not None:
            if isinstance(trend_targets, torch.Tensor):
                self.y = trend_targets.float()
            else:
                self.y = torch.tensor(trend_targets, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]
