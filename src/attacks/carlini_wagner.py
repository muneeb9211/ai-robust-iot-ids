"""
Carlini & Wagner L2 Attack — Carlini and Wagner, 2017.

Optimization-based attack that minimizes:

    min_delta  ||delta||_2  +  c * f(x + delta)

where f is a carefully chosen objective that encourages misclassification.
The C&W attack is significantly stronger than FGSM/PGD and serves as
the gold standard for evaluating adversarial robustness.

We use the formulation:
    f(x') = max( Z(x')_y - max_{i != y} Z(x')_i,  -kappa )

where Z(x') are logits and kappa is a confidence margin.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


def _cw_loss(logits: torch.Tensor, targets: torch.Tensor,
             kappa: float = 0.0) -> torch.Tensor:
    """
    C&W f-function: encourages the adversarial class logit to dominate.

    For targeted=False (untargeted): we want max_{i!=y} Z_i - Z_y to be large.
    """
    one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1.0)
    real = (one_hot * logits).sum(dim=1)              # Z_y
    other = ((1 - one_hot) * logits - 1e4 * one_hot).max(dim=1).values  # max_{i!=y} Z_i
    return torch.clamp(real - other + kappa, min=0.0)


def cw_l2_attack(model: nn.Module, X: np.ndarray, y: np.ndarray,
                 c: float = 1.0, kappa: float = 0.0,
                 num_steps: int = 200, lr: float = 0.01,
                 max_samples: int = 1000,
                 device: torch.device = None) -> Tuple[np.ndarray, dict]:
    """
    Generate C&W L2 adversarial examples (untargeted).

    Due to computational cost, processes at most max_samples. The remaining
    samples are returned unperturbed.

    Args:
        model:       PyTorch classifier.
        X:           Clean inputs (N, D).
        y:           True labels (N,).
        c:           Balance constant between perturbation norm and attack loss.
        kappa:       Confidence margin.
        num_steps:   Optimization steps.
        lr:          Learning rate for the perturbation optimizer.
        max_samples: Maximum samples to attack (C&W is expensive).
        device:      Torch device.

    Returns:
        X_adv:  Adversarial samples.
        stats:  Attack metrics.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    N = X.shape[0]
    n_attack = min(N, max_samples)

    X_full = torch.tensor(X, dtype=torch.float32, device=device)
    y_full = torch.tensor(y, dtype=torch.long, device=device)

    # Get clean predictions
    with torch.no_grad():
        clean_preds = model(X_full).argmax(dim=1)

    X_sub = X_full[:n_attack].clone()
    y_sub = y_full[:n_attack]

    # Optimize perturbation in unconstrained space (w), then tanh-map is
    # commonly used for image data. For tabular data, we directly optimize delta.
    delta = torch.zeros_like(X_sub, requires_grad=True)
    optimizer = optim.Adam([delta], lr=lr)

    best_adv = X_sub.clone()
    best_l2 = torch.full((n_attack,), float("inf"), device=device)

    for step in range(num_steps):
        optimizer.zero_grad()
        X_adv = X_sub + delta
        logits = model(X_adv)

        # L2 norm of perturbation
        l2_norms = torch.norm(delta.view(n_attack, -1), p=2, dim=1)

        # C&W loss
        f_loss = _cw_loss(logits, y_sub, kappa)
        loss = l2_norms.sum() + c * f_loss.sum()
        loss.backward()
        optimizer.step()

        # Track best adversarial examples (misclassified with smallest perturbation)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            misclassified = preds != y_sub
            improved = misclassified & (l2_norms < best_l2)
            best_adv[improved] = X_adv[improved].clone()
            best_l2[improved] = l2_norms[improved]

    # Build full adversarial array (attacked subset + unperturbed remainder)
    X_adv_full = X_full.clone()
    X_adv_full[:n_attack] = best_adv

    with torch.no_grad():
        adv_preds = model(X_adv_full).argmax(dim=1)

    clean_correct = (clean_preds == y_full).cpu().numpy()
    adv_correct = (adv_preds == y_full).cpu().numpy()
    originally_correct = clean_correct.sum()
    fooled = (clean_correct & ~adv_correct).sum()
    attack_success_rate = fooled / max(originally_correct, 1)

    # Stats on attacked subset only
    perturbation = (X_adv_full[:n_attack] - X_sub).detach()
    l2_norms_final = torch.norm(perturbation.view(n_attack, -1), p=2, dim=1)

    stats = {
        "attack": "C&W_L2",
        "c": c,
        "kappa": kappa,
        "num_steps": num_steps,
        "attack_success_rate": float(attack_success_rate),
        "clean_accuracy": float(clean_correct.mean()),
        "adversarial_accuracy": float(adv_correct.mean()),
        "mean_l2_perturbation": float(l2_norms_final.mean().item()),
        "samples_attacked": int(originally_correct),
        "samples_fooled": int(fooled),
        "subset_size": n_attack,
    }

    logger.info("C&W L2 (c=%.1f, steps=%d): success_rate=%.3f, clean_acc=%.3f → adv_acc=%.3f",
                c, num_steps, attack_success_rate, stats["clean_accuracy"],
                stats["adversarial_accuracy"])

    return X_adv_full.cpu().numpy(), stats
