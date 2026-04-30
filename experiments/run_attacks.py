#!/usr/bin/env python3
"""
Run adversarial attacks (FGSM, PGD, C&W) against the trained DNN.

Usage:
    python -m experiments.run_attacks [--config configs/default.yaml] [--synthetic]

Prerequisite: run train_baseline.py first.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_dataset
from src.models import DeepNNIDS
from src.attacks import fgsm_attack, pgd_attack, cw_l2_attack
from src.evaluation import evaluate_model, print_results_table, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run adversarial attacks on DNN")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    attack_cfg = cfg.get("attacks", {})
    dnn_cfg = cfg.get("dnn", {})

    # Load data
    X_train, X_test, y_train, y_test = get_dataset(
        data_dir=data_cfg.get("data_dir", "data/"),
        force_synthetic=args.synthetic or data_cfg.get("force_synthetic", False),
    )
    input_dim = X_train.shape[1]

    # Load trained DNN
    model_dir = Path(cfg.get("model_dir", "checkpoints/"))
    dnn = DeepNNIDS(
        input_dim=input_dim,
        hidden_dims=tuple(dnn_cfg.get("hidden_dims", [256, 128, 64, 32])),
        dropout=dnn_cfg.get("dropout", 0.3),
    )
    dnn.load(str(model_dir / "dnn_baseline.pt"))

    # Use a subset for attacks (speed)
    n_attack = min(len(X_test), attack_cfg.get("max_test_samples", 5000))
    X_atk = X_test[:n_attack]
    y_atk = y_test[:n_attack]

    results = []
    attack_stats = []

    # Baseline (clean)
    clean_preds = dnn.predict(X_atk)
    results.append(evaluate_model(y_atk, clean_preds, "DNN", "clean"))

    # --- FGSM ---
    fgsm_cfg = attack_cfg.get("fgsm", {})
    for eps in fgsm_cfg.get("epsilons", [0.05, 0.1, 0.2]):
        logger.info("Running FGSM attack (eps=%.3f)...", eps)
        X_adv, stats = fgsm_attack(dnn.model, X_atk, y_atk, epsilon=eps, device=dnn.device)
        adv_preds = dnn.predict(X_adv)
        results.append(evaluate_model(y_atk, adv_preds, "DNN", f"FGSM_eps{eps}"))
        attack_stats.append(stats)

    # --- PGD ---
    pgd_cfg = attack_cfg.get("pgd", {})
    for eps in pgd_cfg.get("epsilons", [0.05, 0.1]):
        logger.info("Running PGD attack (eps=%.3f)...", eps)
        X_adv, stats = pgd_attack(
            dnn.model, X_atk, y_atk,
            epsilon=eps,
            alpha=pgd_cfg.get("alpha", 0.01),
            num_steps=pgd_cfg.get("num_steps", 40),
            device=dnn.device,
        )
        adv_preds = dnn.predict(X_adv)
        results.append(evaluate_model(y_atk, adv_preds, "DNN", f"PGD_eps{eps}"))
        attack_stats.append(stats)

    # --- C&W L2 ---
    cw_cfg = attack_cfg.get("cw", {})
    logger.info("Running C&W L2 attack...")
    X_adv, stats = cw_l2_attack(
        dnn.model, X_atk, y_atk,
        c=cw_cfg.get("c", 1.0),
        kappa=cw_cfg.get("kappa", 0.0),
        num_steps=cw_cfg.get("num_steps", 200),
        lr=cw_cfg.get("lr", 0.01),
        max_samples=cw_cfg.get("max_samples", 1000),
        device=dnn.device,
    )
    adv_preds = dnn.predict(X_adv)
    results.append(evaluate_model(y_atk, adv_preds, "DNN", "CW_L2"))
    attack_stats.append(stats)

    # --- Summary ---
    print_results_table(results)
    save_results(results, output_dir="results/", filename="attack_results")

    # Save detailed attack statistics
    stats_path = Path("results/attack_stats.json")
    with open(stats_path, "w") as f:
        json.dump(attack_stats, f, indent=2)
    logger.info("Attack stats saved → %s", stats_path)
    logger.info("Attack evaluation complete.")


if __name__ == "__main__":
    main()
