#!/usr/bin/env python3
"""
Train baseline IDS models: Random Forest, Deep Neural Network, Autoencoder.

Usage:
    python -m experiments.train_baseline [--config configs/default.yaml] [--synthetic]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_dataset
from src.models import RandomForestIDS, DeepNNIDS, AutoencoderIDS
from src.evaluation import evaluate_model, print_results_table, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train baseline IDS models")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic data")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    rf_cfg = cfg.get("random_forest", {})
    dnn_cfg = cfg.get("dnn", {})
    ae_cfg = cfg.get("autoencoder", {})

    # --- Load data ---
    logger.info("Loading dataset...")
    X_train, X_test, y_train, y_test = get_dataset(
        data_dir=data_cfg.get("data_dir", "data/"),
        force_synthetic=args.synthetic or data_cfg.get("force_synthetic", False),
    )
    input_dim = X_train.shape[1]
    logger.info("Input dimension: %d", input_dim)

    results = []
    model_dir = Path(cfg.get("model_dir", "checkpoints/"))
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Random Forest ---
    logger.info("=" * 60)
    logger.info("Training Random Forest")
    logger.info("=" * 60)
    rf = RandomForestIDS(
        n_estimators=rf_cfg.get("n_estimators", 200),
        max_depth=rf_cfg.get("max_depth", None),
        seed=cfg.get("seed", 42),
    )
    rf.train(X_train, y_train)
    rf.save(str(model_dir / "rf_baseline.pkl"))

    rf_preds = rf.predict(X_test)
    results.append(evaluate_model(y_test, rf_preds, "RandomForest", "clean"))

    # --- Deep Neural Network ---
    logger.info("=" * 60)
    logger.info("Training Deep Neural Network")
    logger.info("=" * 60)
    dnn = DeepNNIDS(
        input_dim=input_dim,
        hidden_dims=tuple(dnn_cfg.get("hidden_dims", [256, 128, 64, 32])),
        dropout=dnn_cfg.get("dropout", 0.3),
        lr=dnn_cfg.get("lr", 1e-3),
        weight_decay=dnn_cfg.get("weight_decay", 1e-4),
    )
    dnn.train(
        X_train, y_train,
        epochs=dnn_cfg.get("epochs", 30),
        batch_size=dnn_cfg.get("batch_size", 256),
        X_val=X_test, y_val=y_test,
    )
    dnn.save(str(model_dir / "dnn_baseline.pt"))

    dnn_preds = dnn.predict(X_test)
    results.append(evaluate_model(y_test, dnn_preds, "DNN", "clean"))

    # --- Autoencoder ---
    logger.info("=" * 60)
    logger.info("Training Autoencoder (normal traffic only)")
    logger.info("=" * 60)
    # Train on normal samples only
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]
    logger.info("Normal training samples: %d / %d", normal_mask.sum(), len(y_train))

    ae = AutoencoderIDS(
        input_dim=input_dim,
        latent_dim=ae_cfg.get("latent_dim", 16),
        hidden_dims=tuple(ae_cfg.get("hidden_dims", [128, 64])),
        lr=ae_cfg.get("lr", 1e-3),
    )
    ae.train(
        X_train_normal,
        epochs=ae_cfg.get("epochs", 30),
        batch_size=ae_cfg.get("batch_size", 256),
        threshold_percentile=ae_cfg.get("threshold_percentile", 95.0),
    )
    ae.save(str(model_dir / "ae_baseline.pt"))

    ae_preds = ae.predict(X_test)
    results.append(evaluate_model(y_test, ae_preds, "Autoencoder", "clean"))

    # --- Summary ---
    print_results_table(results)
    save_results(results, output_dir="results/", filename="baseline_results")
    logger.info("Baseline training complete.")


if __name__ == "__main__":
    main()
