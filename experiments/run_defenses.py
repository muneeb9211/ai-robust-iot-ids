#!/usr/bin/env python3
"""
Apply defenses and evaluate robustness under adversarial attacks.

Defenses tested:
1. Adversarial training (PGD-AT)
2. Input transformation (feature squeezing, spatial smoothing)
3. Ensemble (RF + DNN + AE)

Usage:
    python -m experiments.run_defenses [--config configs/default.yaml] [--synthetic]

Prerequisite: run train_baseline.py first.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_dataset
from src.models import RandomForestIDS, DeepNNIDS, AutoencoderIDS
from src.attacks import fgsm_attack, pgd_attack
from src.defenses import adversarial_training, feature_squeezing, spatial_smoothing, EnsembleIDS
from src.evaluation import evaluate_model, print_results_table, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run defenses and evaluate robustness")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    dnn_cfg = cfg.get("dnn", {})
    defense_cfg = cfg.get("defenses", {})
    attack_cfg = cfg.get("attacks", {})

    # Load data
    X_train, X_test, y_train, y_test = get_dataset(
        data_dir=data_cfg.get("data_dir", "data/"),
        force_synthetic=args.synthetic or data_cfg.get("force_synthetic", False),
    )
    input_dim = X_train.shape[1]

    model_dir = Path(cfg.get("model_dir", "checkpoints/"))
    n_attack = min(len(X_test), attack_cfg.get("max_test_samples", 5000))
    X_atk = X_test[:n_attack]
    y_atk = y_test[:n_attack]

    results = []
    eps_test = 0.1  # standard epsilon for robustness evaluation

    # ================================================================
    # Defense 1: Adversarial Training
    # ================================================================
    logger.info("=" * 60)
    logger.info("Defense 1: Adversarial Training (PGD-AT)")
    logger.info("=" * 60)

    at_cfg = defense_cfg.get("adversarial_training", {})
    dnn_at = DeepNNIDS(
        input_dim=input_dim,
        hidden_dims=tuple(dnn_cfg.get("hidden_dims", [256, 128, 64, 32])),
        dropout=dnn_cfg.get("dropout", 0.3),
        lr=dnn_cfg.get("lr", 1e-3),
    )
    # Initialize from baseline weights
    dnn_at.load(str(model_dir / "dnn_baseline.pt"))

    dnn_at = adversarial_training(
        dnn_at, X_train, y_train,
        epsilon=at_cfg.get("epsilon", 0.1),
        alpha=at_cfg.get("alpha", 0.01),
        pgd_steps=at_cfg.get("pgd_steps", 10),
        epochs=at_cfg.get("epochs", 15),
        batch_size=at_cfg.get("batch_size", 256),
    )
    dnn_at.save(str(model_dir / "dnn_adversarial_trained.pt"))

    # Clean accuracy
    at_preds = dnn_at.predict(X_atk)
    results.append(evaluate_model(y_atk, at_preds, "DNN-AT", "clean"))

    # Robustness under FGSM
    X_adv_fgsm, _ = fgsm_attack(dnn_at.model, X_atk, y_atk, epsilon=eps_test, device=dnn_at.device)
    at_adv_preds = dnn_at.predict(X_adv_fgsm)
    results.append(evaluate_model(y_atk, at_adv_preds, "DNN-AT", f"FGSM_eps{eps_test}"))

    # Robustness under PGD
    X_adv_pgd, _ = pgd_attack(dnn_at.model, X_atk, y_atk, epsilon=eps_test, device=dnn_at.device)
    at_pgd_preds = dnn_at.predict(X_adv_pgd)
    results.append(evaluate_model(y_atk, at_pgd_preds, "DNN-AT", f"PGD_eps{eps_test}"))

    # ================================================================
    # Defense 2: Input Transformations
    # ================================================================
    logger.info("=" * 60)
    logger.info("Defense 2: Input Transformations")
    logger.info("=" * 60)

    # Load baseline DNN
    dnn_base = DeepNNIDS(
        input_dim=input_dim,
        hidden_dims=tuple(dnn_cfg.get("hidden_dims", [256, 128, 64, 32])),
    )
    dnn_base.load(str(model_dir / "dnn_baseline.pt"))

    # Generate adversarial examples from baseline
    X_adv_fgsm_base, _ = fgsm_attack(dnn_base.model, X_atk, y_atk, epsilon=eps_test, device=dnn_base.device)

    # Feature squeezing defense
    sq_cfg = defense_cfg.get("feature_squeezing", {})
    X_squeezed = feature_squeezing(X_adv_fgsm_base, bit_depth=sq_cfg.get("bit_depth", 4))
    sq_preds = dnn_base.predict(X_squeezed)
    results.append(evaluate_model(y_atk, sq_preds, "DNN+Squeezing", f"FGSM_eps{eps_test}"))

    # Spatial smoothing defense
    sm_cfg = defense_cfg.get("spatial_smoothing", {})
    X_smoothed = spatial_smoothing(X_adv_fgsm_base, window_size=sm_cfg.get("window_size", 3))
    sm_preds = dnn_base.predict(X_smoothed)
    results.append(evaluate_model(y_atk, sm_preds, "DNN+Smoothing", f"FGSM_eps{eps_test}"))

    # ================================================================
    # Defense 3: Ensemble
    # ================================================================
    logger.info("=" * 60)
    logger.info("Defense 3: Ensemble (RF + DNN + AE)")
    logger.info("=" * 60)

    rf = RandomForestIDS()
    rf.load(str(model_dir / "rf_baseline.pkl"))

    ae = AutoencoderIDS(input_dim=input_dim)
    ae.load(str(model_dir / "ae_baseline.pt"))

    ensemble = EnsembleIDS(rf, dnn_base, ae)

    # Clean
    ens_preds = ensemble.predict(X_atk)
    results.append(evaluate_model(y_atk, ens_preds, "Ensemble", "clean"))

    # Under FGSM (attack crafted against DNN only)
    ens_adv_preds = ensemble.predict(X_adv_fgsm_base)
    results.append(evaluate_model(y_atk, ens_adv_preds, "Ensemble", f"FGSM_eps{eps_test}"))

    # --- Summary ---
    print_results_table(results)
    save_results(results, output_dir="results/", filename="defense_results")
    logger.info("Defense evaluation complete.")


if __name__ == "__main__":
    main()
