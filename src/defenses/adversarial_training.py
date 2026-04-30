"""
Adversarial Training — Madry et al., 2018.

Augments the training set with adversarial examples generated via PGD,
forcing the model to learn representations that are invariant to
bounded perturbations. This is the most principled defense with
provable guarantees under the threat model.

Training procedure:
    For each mini-batch:
        1. Generate PGD adversarial examples from the batch
        2. Combine clean + adversarial samples
        3. Update model on the mixed batch
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..attacks.pgd import pgd_attack
from ..models.deep_nn import DeepNNIDS, get_device

logger = logging.getLogger(__name__)


def adversarial_training(model: DeepNNIDS, X_train: np.ndarray, y_train: np.ndarray,
                         epsilon: float = 0.1, alpha: float = 0.01,
                         pgd_steps: int = 10, epochs: int = 20,
                         batch_size: int = 256, adv_ratio: float = 0.5,
                         X_val: Optional[np.ndarray] = None,
                         y_val: Optional[np.ndarray] = None) -> DeepNNIDS:
    """
    Retrain a DNN with adversarial training (PGD-AT).

    Args:
        model:      DeepNNIDS instance (will be modified in-place).
        X_train:    Clean training features.
        y_train:    Training labels.
        epsilon:    PGD perturbation budget.
        alpha:      PGD step size.
        pgd_steps:  PGD iterations per batch.
        epochs:     Number of adversarial training epochs.
        batch_size: Mini-batch size.
        adv_ratio:  Fraction of each batch replaced with adversarial examples.
        X_val:      Optional validation features.
        y_val:      Optional validation labels.

    Returns:
        The adversarially trained model.
    """
    device = model.device
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size,
                        shuffle=True, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    logger.info("Starting adversarial training: eps=%.3f, pgd_steps=%d, epochs=%d",
                epsilon, pgd_steps, epochs)

    for epoch in range(1, epochs + 1):
        model.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Generate adversarial examples for a subset of the batch
            n_adv = max(1, int(X_batch.size(0) * adv_ratio))
            X_clean_np = X_batch[:n_adv].cpu().numpy()
            y_clean_np = y_batch[:n_adv].cpu().numpy()

            # PGD attack on current model state
            model.model.eval()
            X_adv_np, _ = pgd_attack(
                model.model, X_clean_np, y_clean_np,
                epsilon=epsilon, alpha=alpha, num_steps=pgd_steps,
                random_start=True, device=device
            )
            model.model.train()

            X_adv_t = torch.tensor(X_adv_np, dtype=torch.float32, device=device)

            # Mix clean and adversarial
            X_mixed = torch.cat([X_adv_t, X_batch[n_adv:]], dim=0)
            y_mixed = torch.cat([y_batch[:n_adv], y_batch[n_adv:]], dim=0)

            model.optimizer.zero_grad()
            logits = model.model(X_mixed)
            loss = criterion(logits, y_mixed)
            loss.backward()
            model.optimizer.step()

            total_loss += loss.item() * X_mixed.size(0)
            correct += (logits.argmax(dim=1) == y_mixed).sum().item()
            total += X_mixed.size(0)

        avg_loss = total_loss / total
        train_acc = correct / total

        if epoch % 5 == 0 or epoch == 1:
            msg = f"AT Epoch {epoch:3d}/{epochs} — loss={avg_loss:.4f}, acc={train_acc:.4f}"
            if X_val is not None and y_val is not None:
                val_metrics = model.evaluate(X_val, y_val)
                msg += f", val_acc={val_metrics['accuracy']:.4f}"
            logger.info(msg)

    logger.info("Adversarial training complete.")
    return model
