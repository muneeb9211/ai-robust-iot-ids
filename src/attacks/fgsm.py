"""
Fast Gradient Sign Method (FGSM) — Goodfellow et al., 2015.

Single-step adversarial perturbation:

    x_adv = x + epsilon * sign(nabla_x L(theta, x, y))

FGSM is computationally cheap and serves as a lower bound on adversarial
vulnerability. If a model is not robust to FGSM, it will certainly fail
against stronger iterative attacks.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def fgsm_attack(model: nn.Module, X: np.ndarray, y: np.ndarray,
                epsilon: float = 0.1, device: torch.device = None
                ) -> Tuple[np.ndarray, dict]:
    """
    Generate FGSM adversarial examples.

    Args:
        model:   PyTorch classifier (logits output).
        X:       Clean input samples (N, D), float32.
        y:       True labels (N,), int64.
        epsilon: Perturbation magnitude (L-infinity).
        device:  Torch device.

    Returns:
        X_adv:  Adversarial samples (N, D).
        stats:  Dict with attack_success_rate, mean_perturbation, etc.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device, requires_grad=True)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    logits = model(X_t)
    loss = nn.CrossEntropyLoss()(logits, y_t)
    loss.backward()

    # Perturbation: step in the direction of the gradient sign
    perturbation = epsilon * X_t.grad.sign()
    X_adv = (X_t + perturbation).detach()

    # Evaluate attack
    with torch.no_grad():
        clean_preds = logits.argmax(dim=1)
        adv_preds = model(X_adv).argmax(dim=1)

    clean_correct = (clean_preds == y_t).cpu().numpy()
    adv_correct = (adv_preds == y_t).cpu().numpy()

    # Attack success = was correctly classified, now misclassified
    originally_correct = clean_correct.sum()
    fooled = (clean_correct & ~adv_correct).sum()
    attack_success_rate = fooled / max(originally_correct, 1)

    l2_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1)
    linf_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=float("inf"), dim=1)

    stats = {
        "attack": "FGSM",
        "epsilon": epsilon,
        "attack_success_rate": float(attack_success_rate),
        "clean_accuracy": float(clean_correct.mean()),
        "adversarial_accuracy": float(adv_correct.mean()),
        "mean_l2_perturbation": float(l2_norm.mean().item()),
        "mean_linf_perturbation": float(linf_norm.mean().item()),
        "samples_attacked": int(originally_correct),
        "samples_fooled": int(fooled),
    }

    logger.info("FGSM (eps=%.3f): success_rate=%.3f, clean_acc=%.3f → adv_acc=%.3f",
                epsilon, attack_success_rate, stats["clean_accuracy"],
                stats["adversarial_accuracy"])

    return X_adv.cpu().numpy(), stats
