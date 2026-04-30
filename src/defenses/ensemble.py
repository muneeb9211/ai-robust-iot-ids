"""
Ensemble defense: majority voting across heterogeneous classifiers.

Combines Random Forest (non-differentiable), DNN (differentiable),
and Autoencoder (anomaly-based) predictions. The intuition is that
adversarial examples crafted against the DNN are unlikely to
simultaneously fool the RF and autoencoder, since they operate on
fundamentally different decision boundaries.

Ensemble voting strategies:
- Hard voting: majority class wins
- Soft voting: average predicted probabilities (RF + DNN only; AE contributes binary)
"""

import logging
from typing import Optional

import numpy as np

from ..models.random_forest import RandomForestIDS
from ..models.deep_nn import DeepNNIDS
from ..models.autoencoder import AutoencoderIDS

logger = logging.getLogger(__name__)


class EnsembleIDS:
    """Heterogeneous ensemble for robust intrusion detection."""

    def __init__(self, rf: RandomForestIDS, dnn: DeepNNIDS,
                 ae: Optional[AutoencoderIDS] = None,
                 weights: Optional[tuple] = None):
        """
        Args:
            rf:      Trained RandomForestIDS.
            dnn:     Trained DeepNNIDS.
            ae:      Trained AutoencoderIDS (optional).
            weights: Voting weights (rf_w, dnn_w, ae_w). Default: equal.
        """
        self.rf = rf
        self.dnn = dnn
        self.ae = ae

        n_models = 3 if ae is not None else 2
        if weights is None:
            self.weights = tuple([1.0] * n_models)
        else:
            self.weights = weights

    def predict(self, X: np.ndarray, voting: str = "hard") -> np.ndarray:
        """
        Ensemble prediction via majority voting.

        Args:
            X:      Input features (N, D).
            voting: "hard" (majority vote) or "soft" (weighted probability).

        Returns:
            Predicted labels (N,).
        """
        if voting == "hard":
            return self._hard_vote(X)
        elif voting == "soft":
            return self._soft_vote(X)
        else:
            raise ValueError(f"Unknown voting strategy: {voting}")

    def _hard_vote(self, X: np.ndarray) -> np.ndarray:
        """Weighted majority vote."""
        votes = np.zeros((X.shape[0], 2), dtype=np.float64)

        rf_pred = self.rf.predict(X)
        votes[np.arange(len(rf_pred)), rf_pred] += self.weights[0]

        dnn_pred = self.dnn.predict(X)
        votes[np.arange(len(dnn_pred)), dnn_pred] += self.weights[1]

        if self.ae is not None:
            ae_pred = self.ae.predict(X)
            votes[np.arange(len(ae_pred)), ae_pred] += self.weights[2]

        return votes.argmax(axis=1).astype(np.int64)

    def _soft_vote(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of predicted probabilities."""
        proba = np.zeros((X.shape[0], 2), dtype=np.float64)

        proba += self.weights[0] * self.rf.predict_proba(X)
        proba += self.weights[1] * self.dnn.predict_proba(X)

        if self.ae is not None:
            # AE only provides binary predictions; convert to pseudo-probabilities
            ae_pred = self.ae.predict(X)
            ae_proba = np.zeros((X.shape[0], 2))
            ae_proba[np.arange(len(ae_pred)), ae_pred] = 1.0
            proba += self.weights[2] * ae_proba

        return proba.argmax(axis=1).astype(np.int64)

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 voting: str = "hard") -> dict:
        from sklearn.metrics import classification_report
        y_pred = self.predict(X, voting=voting)
        acc = np.mean(y_pred == y)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        return {"accuracy": acc, "report": report, "voting": voting}
