"""
Acoustic & Prosodic Trainer - FULL IMPLEMENTATION
Author: Team Member A (Sanuthi)
Status: IMPLEMENTED
"""

import pandas as pd
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.utils.logger import get_logger

logger = get_logger(__name__)

class AcousticProsodicTrainer:
    """Trains multiple ML models on acoustic & prosodic features."""

    def __init__(self):
        logger.info("AcousticProsodicTrainer initialized.")

    def train_model(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        return {
            "model": model,
            "accuracy": accuracy,
            "f1": f1,
            "report": classification_report(y_test, preds)
        }

    def train_multiple_models(self, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        results = {}

        models = {
            "LogisticRegression": LogisticRegression(max_iter=200),
            "RandomForest": RandomForestClassifier(n_estimators=200)
        }

        logger.info("Training multiple models (LR, RF)...")

        for name, model in models.items():
            logger.info(f"Training {name}...")
            results[name] = self.train_model(model, X_train, y_train, X_test, y_test)

        return results
