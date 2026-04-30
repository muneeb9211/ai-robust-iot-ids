"""
Projected Gradient Descent (PGD) — Madry et al., 2018.

Iterative refinement of FGSM:

    x^{t+1} = Pi_{B(x, epsilon)} ( x^t + alpha * sign(nabla_x L(theta, x^t, y)) )

where Pi projects back onto the epsilon-ball around the original input.
PGD is considered the canonical first-order adversary and is the standard
benchmark for evaluating adversarial robustness.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def pgd_attack(model: nn.Module, X: np.ndarray, y: np.ndarray,
               epsilon: float = 0.1, alpha: float = 0.01,
               num_steps: int = 40, random_start: bool = True,
               device: torch.device = None) -> Tuple[np.ndarray, dict]:
    """
    Generate PGD adversarial examples.

    Args:
        model:        PyTorch classifier.
        X:            Clean inputs (N, D).
        y:            True labels (N,).
        epsilon:      L-infinity perturbation budget.
        alpha:        Step size per iteration.
        num_steps:    Number of PGD iterations.
        random_start: If True, initialize with uniform noise in [-epsilon, epsilon].
        device:       Torch device.

    Returns:
        X_adv:  Adversarial samples.
        stats:  Attack metrics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    # Clean predictions for reference
    with torch.no_grad():
        clean_preds = model(X_t).argmax(dim=1)

    # Initialize adversarial examples
    if random_start:
        delta = torch.zeros_like(X_t).uniform_(-epsilon, epsilon)
    else:
        delta = torch.zeros_like(X_t)
    delta = torch.clamp(delta, -epsilon, epsilon)

    for step in range(num_steps):
        delta.requires_grad_(True)
        logits = model(X_t + delta)
        loss = nn.CrossEntropyLoss()(logits, y_t)
        loss.backward()

        # Steepest ascent step
        grad_sign = delta.grad.sign()
        delta = (delta + alpha * grad_sign).detach()
        # Project back onto L-inf epsilon-ball
        delta = torch.clamp(delta, -epsilon, epsilon)

    X_adv = (X_t + delta).detach()

    # Evaluate
    with torch.no_grad():
        adv_preds = model(X_adv).argmax(dim=1)

    clean_correct = (clean_preds == y_t).cpu().numpy()
    adv_correct = (adv_preds == y_t).cpu().numpy()
    originally_correct = clean_correct.sum()
    fooled = (clean_correct & ~adv_correct).sum()
    attack_success_rate = fooled / max(originally_correct, 1)

    perturbation = delta.detach()
    l2_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=2, dim=1)
    linf_norm = torch.norm(perturbation.view(perturbation.size(0), -1), p=float("inf"), dim=1)

    stats = {
        "attack": "PGD",
        "epsilon": epsilon,
        "alpha": alpha,
        "num_steps": num_steps,
        "attack_success_rate": float(attack_success_rate),
        "clean_accuracy": float(clean_correct.mean()),
        "adversarial_accuracy": float(adv_correct.mean()),
        "mean_l2_perturbation": float(l2_norm.mean().item()),
        "mean_linf_perturbation": float(linf_norm.mean().item()),
        "samples_attacked": int(originally_correct),
        "samples_fooled": int(fooled),
    }

    logger.info("PGD (eps=%.3f, steps=%d): success_rate=%.3f, clean_acc=%.3f → adv_acc=%.3f",
                epsilon, num_steps, attack_success_rate, stats["clean_accuracy"],
                stats["adversarial_accuracy"])

    return X_adv.cpu().numpy(), stats
