"""
DNN classifier for network intrusion detection (PyTorch).

Primary attack target — exposes differentiable gradients for FGSM/PGD/C&W.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _DNNModule(nn.Module):
    """4-layer DNN with batch norm, ReLU, dropout."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (256, 128, 64, 32),
                 dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 2))  # binary classification logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepNNIDS:
    """Training/inference wrapper with device management and checkpointing."""

    def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (256, 128, 64, 32),
                 dropout: float = 0.3, lr: float = 1e-3, weight_decay: float = 1e-4,
                 device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = _DNNModule(input_dim, hidden_dims, dropout).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=False
        )
        self.input_dim = input_dim

    def _make_loader(self, X: np.ndarray, y: np.ndarray, batch_size: int,
                     shuffle: bool = True) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                          shuffle=shuffle, drop_last=False)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              epochs: int = 30, batch_size: int = 256,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> "DeepNNIDS":
        loader = self._make_loader(X_train, y_train, batch_size)
        logger.info("Training DNN on %s: %d samples, %d features, %d epochs",
                     self.device, X_train.shape[0], X_train.shape[1], epochs)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * X_batch.size(0)
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += X_batch.size(0)

            avg_loss = total_loss / total
            train_acc = correct / total
            self.scheduler.step(avg_loss)

            if epoch % 5 == 0 or epoch == 1:
                msg = f"Epoch {epoch:3d}/{epochs} — loss={avg_loss:.4f}, acc={train_acc:.4f}"
                if X_val is not None and y_val is not None:
                    val_metrics = self.evaluate(X_val, y_val)
                    msg += f", val_acc={val_metrics['accuracy']:.4f}"
                logger.info(msg)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return torch.softmax(logits, dim=1).cpu().numpy()

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
            "optimizer_state": self.optimizer.state_dict(),
            "input_dim": self.input_dim,
        }, path)
        logger.info("DNN saved → %s", path)

    def load(self, path: str) -> "DeepNNIDS":
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        logger.info("DNN loaded ← %s", path)
        return self
