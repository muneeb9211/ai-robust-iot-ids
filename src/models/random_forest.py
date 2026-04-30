"""
Random Forest baseline — non-differentiable, inherently robust to gradient attacks.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)


class RandomForestIDS:

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = None,
                 min_samples_split: int = 5, n_jobs: int = -1, seed: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=n_jobs,
            random_state=seed,
            class_weight="balanced",
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestIDS":
        logger.info("Training Random Forest: %d samples, %d features",
                     X_train.shape[0], X_train.shape[1])
        self.model.fit(X_train, y_train)
        self.is_trained = True

        train_acc = self.model.score(X_train, y_train)
        logger.info("RF train accuracy: %.4f", train_acc)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        return {"accuracy": acc, "report": report}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("RF model saved → %s", path)

    def load(self, path: str) -> "RandomForestIDS":
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
        logger.info("RF model loaded ← %s", path)
        return self
