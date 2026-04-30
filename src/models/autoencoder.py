"""
Autoencoder-based anomaly detector. Trained on normal traffic only;
reconstruction error above a learned threshold flags anomalies.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .deep_nn import get_device

logger = logging.getLogger(__name__)


class _AEModule(nn.Module):

    def __init__(self, input_dim: int, latent_dim: int = 16,
                 hidden_dims: Tuple[int, ...] = (128, 64)):
        super().__init__()
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class AutoencoderIDS:

    def __init__(self, input_dim: int, latent_dim: int = 16,
                 hidden_dims: Tuple[int, ...] = (128, 64),
                 lr: float = 1e-3, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = _AEModule(input_dim, latent_dim, hidden_dims).to(self.device)
        self.criterion = nn.MSELoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.threshold: Optional[float] = None
        self.input_dim = input_dim

    def train(self, X_normal: np.ndarray, epochs: int = 30,
              batch_size: int = 256, threshold_percentile: float = 95.0
              ) -> "AutoencoderIDS":
        """Fit on normal traffic, then set anomaly threshold at given percentile."""
        X_t = torch.tensor(X_normal, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=True)
        logger.info("Training Autoencoder on %s: %d normal samples, %d features",
                     self.device, X_normal.shape[0], X_normal.shape[1])

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            n = 0
            for (batch,) in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.criterion(recon, batch).mean()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.size(0)
                n += batch.size(0)

            if epoch % 5 == 0 or epoch == 1:
                logger.info("AE Epoch %3d/%d — MSE=%.6f", epoch, epochs, total_loss / n)

        errors = self.reconstruction_error(X_normal)
        self.threshold = float(np.percentile(errors, threshold_percentile))
        logger.info("AE threshold (%.0f-th percentile): %.6f",
                     threshold_percentile, self.threshold)
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            recon = self.model(X_t)
            errors = self.criterion(recon, X_t).mean(dim=1)
        return errors.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Autoencoder not trained — threshold not set.")
        errors = self.reconstruction_error(X)
        return (errors > self.threshold).astype(np.int64)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        from sklearn.metrics import classification_report
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        return {"accuracy": acc, "report": report}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "threshold": self.threshold,
            "input_dim": self.input_dim,
        }, path)
        logger.info("AE saved → %s", path)

    def load(self, path: str) -> "AutoencoderIDS":
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.threshold = ckpt["threshold"]
        logger.info("AE loaded ← %s (threshold=%.6f)", path, self.threshold)
        return self
