"""
Evaluation metrics for intrusion detection and adversarial robustness.

Provides:
- Standard classification metrics (accuracy, precision, recall, F1)
- Adversarial robustness metrics (attack success rate, accuracy degradation)
- Comparison tables across models and attack/defense configurations
- Result persistence to CSV and JSON
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

logger = logging.getLogger(__name__)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                   model_name: str = "model", context: str = "clean"
                   ) -> Dict:
    """
    Compute full classification metrics.

    Args:
        y_true:     Ground truth labels.
        y_pred:     Predicted labels.
        model_name: Identifier for this model.
        context:    Evaluation context (e.g., "clean", "fgsm_eps0.1").

    Returns:
        Dict with all metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Detection rate (recall for attack class = 1)
    tp = cm[1, 1] if cm.shape[0] > 1 else 0
    fn = cm[1, 0] if cm.shape[0] > 1 else 0
    detection_rate = tp / max(tp + fn, 1)

    # False positive rate
    fp = cm[0, 1] if cm.shape[0] > 1 else 0
    tn = cm[0, 0]
    false_positive_rate = fp / max(fp + tn, 1)

    result = {
        "model": model_name,
        "context": context,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "detection_rate": float(detection_rate),
        "false_positive_rate": float(false_positive_rate),
        "confusion_matrix": cm.tolist(),
    }

    logger.info("[%s | %s] acc=%.4f, prec=%.4f, rec=%.4f, F1=%.4f, DR=%.4f, FPR=%.4f",
                model_name, context, acc, prec, rec, f1, detection_rate, false_positive_rate)
    return result


def compare_results(results: List[Dict]) -> str:
    """
    Generate a formatted comparison table from a list of evaluation results.

    Returns:
        Formatted string table.
    """
    header = f"{'Model':<25} {'Context':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'DR':>7} {'FPR':>7}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in results:
        line = (f"{r['model']:<25} {r['context']:<20} "
                f"{r['accuracy']:>7.4f} {r['precision']:>7.4f} "
                f"{r['recall']:>7.4f} {r['f1_score']:>7.4f} "
                f"{r['detection_rate']:>7.4f} {r['false_positive_rate']:>7.4f}")
        lines.append(line)

    lines.append(sep)
    return "\n".join(lines)


def print_results_table(results: List[Dict]) -> None:
    """Print formatted comparison table to stdout."""
    table = compare_results(results)
    print("\n" + table + "\n")


def save_results(results: List[Dict], output_dir: str = "results/",
                 filename: str = "evaluation_results") -> None:
    """
    Save results to both JSON and CSV formats.

    Args:
        results:    List of evaluation result dicts.
        output_dir: Output directory.
        filename:   Base filename (without extension).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # JSON (full detail including confusion matrices)
    json_path = out / f"{filename}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved → %s", json_path)

    # CSV (flat metrics only)
    csv_path = out / f"{filename}.csv"
    csv_keys = ["model", "context", "accuracy", "precision", "recall",
                "f1_score", "detection_rate", "false_positive_rate"]
    with open(csv_path, "w") as f:
        f.write(",".join(csv_keys) + "\n")
        for r in results:
            vals = [str(r.get(k, "")) for k in csv_keys]
            f.write(",".join(vals) + "\n")
    logger.info("Results saved → %s", csv_path)

    # Human-readable table
    table_path = out / f"{filename}.txt"
    with open(table_path, "w") as f:
        f.write(compare_results(results) + "\n")
    logger.info("Results saved → %s", table_path)
