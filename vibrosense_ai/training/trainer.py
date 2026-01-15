import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device

        self.recon_loss_fn = nn.MSELoss()
        self.trend_loss_fn = nn.MSELoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
    
    def train_step(self, batch):
        # --- Robust batch unpacking ---
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            X, trend_target = batch
            X = torch.as_tensor(X, device=self.device)
            trend_target = torch.as_tensor(trend_target, device=self.device)
        else:
            X = torch.as_tensor(batch, device=self.device)
            trend_target = None


        X = X.to(self.device)

        recon, trend, attn = self.model(X)

        loss_recon = self.recon_loss_fn(recon, X)

        if trend_target is not None:
            loss_trend = self.trend_loss_fn(trend, trend_target)
        else:
            loss_trend = torch.tensor(0.0, device=self.device)

        loss = loss_recon + 0.4 * loss_trend

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item() ,
            "recon": loss_recon.item() ,
            "trend": loss_trend.item()
        }

    def train_epoch(self, epoch_idx=0):
        self.model.train()
        metrics = {"loss": 0.0, "recon": 0.0, "trend": 0.0}

        num_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch_idx}"):
            out = self.train_step(batch)
            for k in metrics:
                metrics[k] += out[k]
            num_batches += 1

        # Safe averaging (prevents ZeroDivisionError)
        if num_batches > 0:
            for k in metrics:
                metrics[k] /= num_batches

        return metrics
