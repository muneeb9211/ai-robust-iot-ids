#!/usr/bin/env python3
"""
Full end-to-end pipeline: Train → Attack → Defend → Compare.

Runs the complete adversarial robustness evaluation cycle:
1. Train baseline models (RF, DNN, Autoencoder)
2. Evaluate baselines on clean data
3. Generate adversarial examples (FGSM, PGD, C&W) against DNN
4. Measure accuracy degradation under each attack
5. Apply defenses (adversarial training, input transforms, ensemble)
6. Re-evaluate defended models under the same attacks
7. Print comprehensive comparison table

Usage:
    python -m experiments.full_pipeline [--config configs/default.yaml] [--synthetic]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_dataset
from src.models import RandomForestIDS, DeepNNIDS, AutoencoderIDS
from src.attacks import fgsm_attack, pgd_attack, cw_l2_attack
from src.defenses import adversarial_training, feature_squeezing, spatial_smoothing, EnsembleIDS
from src.evaluation import evaluate_model, print_results_table, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Full adversarial robustness pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic dataset (no download)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    t_start = time.time()
    seed = cfg.get("seed", 42)
    np.random.seed(seed)

    data_cfg = cfg.get("data", {})
    rf_cfg = cfg.get("random_forest", {})
    dnn_cfg = cfg.get("dnn", {})
    ae_cfg = cfg.get("autoencoder", {})
    attack_cfg = cfg.get("attacks", {})
    defense_cfg = cfg.get("defenses", {})

    # ================================================================
    # PHASE 1: DATA LOADING
    # ================================================================
    logger.info("=" * 70)
    logger.info("PHASE 1: Loading and preprocessing dataset")
    logger.info("=" * 70)

    X_train, X_test, y_train, y_test = get_dataset(
        data_dir=data_cfg.get("data_dir", "data/"),
        force_synthetic=args.synthetic or data_cfg.get("force_synthetic", False),
    )
    input_dim = X_train.shape[1]
    logger.info("Dataset: train=%d, test=%d, features=%d, attack_ratio=%.2f",
                len(X_train), len(X_test), input_dim, y_train.mean())

    model_dir = Path(cfg.get("model_dir", "checkpoints/"))
    model_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    n_attack = min(len(X_test), attack_cfg.get("max_test_samples", 5000))
    X_atk = X_test[:n_attack]
    y_atk = y_test[:n_attack]
    eps = attack_cfg.get("default_epsilon", 0.1)

    # ================================================================
    # PHASE 2: BASELINE TRAINING
    # ================================================================
    logger.info("=" * 70)
    logger.info("PHASE 2: Training baseline models")
    logger.info("=" * 70)

    # Random Forest
    logger.info("--- Random Forest ---")
    rf = RandomForestIDS(
        n_estimators=rf_cfg.get("n_estimators", 200),
        seed=seed,
    )
    rf.train(X_train, y_train)
    rf.save(str(model_dir / "rf_baseline.pkl"))
    rf_preds = rf.predict(X_atk)
    all_results.append(evaluate_model(y_atk, rf_preds, "RandomForest", "clean"))

    # DNN
    logger.info("--- Deep Neural Network ---")
    dnn = DeepNNIDS(
        input_dim=input_dim,
        hidden_dims=tuple(dnn_cfg.get("hidden_dims", [256, 128, 64, 32])),
        dropout=dnn_cfg.get("dropout", 0.3),
        lr=dnn_cfg.get("lr", 1e-3),
    )
    dnn.train(X_train, y_train,
              epochs=dnn_cfg.get("epochs", 30),
              batch_size=dnn_cfg.get("batch_size", 256))
    dnn.save(str(model_dir / "dnn_baseline.pt"))
    dnn_preds = dnn.predict(X_atk)
    all_results.append(evaluate_model(y_atk, dnn_preds, "DNN", "clean"))

    # Autoencoder
    logger.info("--- Autoencoder ---")
    normal_mask = y_train == 0
    ae = AutoencoderIDS(
        input_dim=input_dim,
        latent_dim=ae_cfg.get("latent_dim", 16),
        hidden_dims=tuple(ae_cfg.get("hidden_dims", [128, 64])),
    )
    ae.train(X_train[normal_mask],
             epochs=ae_cfg.get("epochs", 30),
             batch_size=ae_cfg.get("batch_size", 256))
    ae.save(str(model_dir / "ae_baseline.pt"))
    ae_preds = ae.predict(X_atk)
    all_results.append(evaluate_model(y_atk, ae_preds, "Autoencoder", "clean"))

    # ================================================================
    # PHASE 3: ADVERSARIAL ATTACKS
    # ================================================================
    logger.info("=" * 70)
    logger.info("PHASE 3: Adversarial attacks on DNN")
    logger.info("=" * 70)

    # FGSM
    logger.info("--- FGSM (eps=%.3f) ---", eps)
    X_adv_fgsm, fgsm_stats = fgsm_attack(dnn.model, X_atk, y_atk, epsilon=eps, device=dnn.device)
    fgsm_preds = dnn.predict(X_adv_fgsm)
    all_results.append(evaluate_model(y_atk, fgsm_preds, "DNN", f"FGSM_eps{eps}"))

    # PGD
    pgd_cfg = attack_cfg.get("pgd", {})
    logger.info("--- PGD (eps=%.3f) ---", eps)
    X_adv_pgd, pgd_stats = pgd_attack(
        dnn.model, X_atk, y_atk,
        epsilon=eps, alpha=pgd_cfg.get("alpha", 0.01),
        num_steps=pgd_cfg.get("num_steps", 40), device=dnn.device,
    )
    pgd_preds = dnn.predict(X_adv_pgd)
    all_results.append(evaluate_model(y_atk, pgd_preds, "DNN", f"PGD_eps{eps}"))

    # C&W L2
    cw_cfg = attack_cfg.get("cw", {})
    logger.info("--- Carlini & Wagner L2 ---")
    X_adv_cw, cw_stats = cw_l2_attack(
        dnn.model, X_atk, y_atk,
        c=cw_cfg.get("c", 1.0),
        num_steps=cw_cfg.get("num_steps", 200),
        max_samples=cw_cfg.get("max_samples", 1000),
        device=dnn.device,
    )
    cw_preds = dnn.predict(X_adv_cw)
    all_results.append(evaluate_model(y_atk, cw_preds, "DNN", "CW_L2"))

    # ================================================================
    # PHASE 4: DEFENSES
    # ================================================================
    logger.info("=" * 70)
    logger.info("PHASE 4: Applying defenses")
    logger.info("=" * 70)

    # --- Defense 1: Adversarial Training ---
    logger.info("--- Adversarial Training (PGD-AT) ---")
    at_cfg = defense_cfg.get("adversarial_training", {})
    dnn_at = DeepNNIDS(
        input_dim=input_dim,
        hidden_dims=tuple(dnn_cfg.get("hidden_dims", [256, 128, 64, 32])),
        dropout=dnn_cfg.get("dropout", 0.3),
        lr=at_cfg.get("lr", 5e-4),
    )
    dnn_at.load(str(model_dir / "dnn_baseline.pt"))
    dnn_at = adversarial_training(
        dnn_at, X_train, y_train,
        epsilon=at_cfg.get("epsilon", 0.1),
        alpha=at_cfg.get("alpha", 0.01),
        pgd_steps=at_cfg.get("pgd_steps", 10),
        epochs=at_cfg.get("epochs", 15),
    )
    dnn_at.save(str(model_dir / "dnn_adversarial_trained.pt"))

    at_clean_preds = dnn_at.predict(X_atk)
    all_results.append(evaluate_model(y_atk, at_clean_preds, "DNN-AT", "clean"))

    X_adv_at_fgsm, _ = fgsm_attack(dnn_at.model, X_atk, y_atk, epsilon=eps, device=dnn_at.device)
    at_fgsm_preds = dnn_at.predict(X_adv_at_fgsm)
    all_results.append(evaluate_model(y_atk, at_fgsm_preds, "DNN-AT", f"FGSM_eps{eps}"))

    X_adv_at_pgd, _ = pgd_attack(dnn_at.model, X_atk, y_atk, epsilon=eps, device=dnn_at.device)
    at_pgd_preds = dnn_at.predict(X_adv_at_pgd)
    all_results.append(evaluate_model(y_atk, at_pgd_preds, "DNN-AT", f"PGD_eps{eps}"))

    # --- Defense 2: Input Transformations ---
    logger.info("--- Input Transformation: Feature Squeezing ---")
    X_squeezed = feature_squeezing(X_adv_fgsm, bit_depth=4)
    sq_preds = dnn.predict(X_squeezed)
    all_results.append(evaluate_model(y_atk, sq_preds, "DNN+Squeezing", f"FGSM_eps{eps}"))

    logger.info("--- Input Transformation: Spatial Smoothing ---")
    X_smoothed = spatial_smoothing(X_adv_fgsm, window_size=3)
    sm_preds = dnn.predict(X_smoothed)
    all_results.append(evaluate_model(y_atk, sm_preds, "DNN+Smoothing", f"FGSM_eps{eps}"))

    # --- Defense 3: Ensemble ---
    logger.info("--- Ensemble (RF + DNN + AE) ---")
    ensemble = EnsembleIDS(rf, dnn, ae)
    ens_clean_preds = ensemble.predict(X_atk)
    all_results.append(evaluate_model(y_atk, ens_clean_preds, "Ensemble", "clean"))

    ens_fgsm_preds = ensemble.predict(X_adv_fgsm)
    all_results.append(evaluate_model(y_atk, ens_fgsm_preds, "Ensemble", f"FGSM_eps{eps}"))

    # ================================================================
    # PHASE 5: SUMMARY
    # ================================================================
    elapsed = time.time() - t_start
    logger.info("=" * 70)
    logger.info("PHASE 5: Summary (total time: %.1f seconds)", elapsed)
    logger.info("=" * 70)

    print_results_table(all_results)
    save_results(all_results, output_dir="results/", filename="full_pipeline_results")

    # Print key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find relevant results
    def _find(model, context):
        for r in all_results:
            if r["model"] == model and r["context"] == context:
                return r
        return None

    dnn_clean = _find("DNN", "clean")
    dnn_fgsm = _find("DNN", f"FGSM_eps{eps}")
    dnn_pgd = _find("DNN", f"PGD_eps{eps}")
    at_fgsm = _find("DNN-AT", f"FGSM_eps{eps}")
    at_pgd = _find("DNN-AT", f"PGD_eps{eps}")
    ens_fgsm = _find("Ensemble", f"FGSM_eps{eps}")

    if dnn_clean and dnn_fgsm:
        drop = dnn_clean["accuracy"] - dnn_fgsm["accuracy"]
        print(f"  FGSM accuracy drop:       {dnn_clean['accuracy']:.4f} → {dnn_fgsm['accuracy']:.4f} ({drop:+.4f})")
    if dnn_clean and dnn_pgd:
        drop = dnn_clean["accuracy"] - dnn_pgd["accuracy"]
        print(f"  PGD accuracy drop:        {dnn_clean['accuracy']:.4f} → {dnn_pgd['accuracy']:.4f} ({drop:+.4f})")
    if at_fgsm:
        print(f"  Adv. Training vs FGSM:    {at_fgsm['accuracy']:.4f}")
    if at_pgd:
        print(f"  Adv. Training vs PGD:     {at_pgd['accuracy']:.4f}")
    if ens_fgsm:
        print(f"  Ensemble vs FGSM:         {ens_fgsm['accuracy']:.4f}")

    print(f"\n  Total pipeline time: {elapsed:.1f}s")
    print("=" * 70)

    logger.info("Full pipeline complete. Results saved to results/")


if __name__ == "__main__":
    main()
