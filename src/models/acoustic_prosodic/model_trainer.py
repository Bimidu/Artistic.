"""
Acoustic & Prosodic Trainer - BALANCED
Author: Team Member A (Sanuthi)
Status: IMPLEMENTED with BALANCED parameters
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AcousticProsodicTrainer:
    """
    Trains ML models on acoustic & prosodic features.
    COMPONENT-SPECIFIC: Only XGBoost and Random Forest (optimized for acoustic data).
    """

    # COMPONENT-SPECIFIC: Acoustic/Prosodic models
    # Only XGBoost and Random Forest supported for this component
    # These models work well with continuous acoustic features (pitch, energy, spectral)
    ALLOWED_MODEL_TYPES = ['xgboost', 'random_forest']
    
    # Acoustic-optimized hyperparameters (fine-tuned for prosodic/spectral features)
    MODEL_CONFIGS = {
        'xgboost': {
            'n_estimators': 120,             # Moderate for acoustic features
            'max_depth': 8,                  # Moderate depth
            'learning_rate': 0.08,           # Slower learning for generalization
            'subsample': 0.8,                # Use 80% of data
            'colsample_bytree': 0.8,         # Use 80% of features
            'min_child_weight': 3,           # Regularization for acoustic noise
            'gamma': 0.2,                    # Pruning for generalization
            'reg_alpha': 0.4,                # L1 regularization
            'reg_lambda': 1.8,               # L2 regularization
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
        },
        'random_forest': {
            'n_estimators': 150,             # Many trees for acoustic stability
            'max_depth': 12,                 # Moderate depth for acoustic patterns
            'min_samples_split': 8,          # Balance bias-variance
            'min_samples_leaf': 4,           # Prevent overfitting
            'max_features': 'sqrt',          # Use sqrt(n_features) for acoustic
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced',
        },
    }

    def __init__(self):
        """Initialize acoustic prosodic trainer."""
        self.models_ = {}
        logger.info(f"AcousticProsodicTrainer initialized - Component-specific models: {', '.join(self.ALLOWED_MODEL_TYPES)}")

    def train_model(
        self,
        model_name: str,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train a single model and optionally evaluate on test set.

        Args:
            model_name: Name of the model
            model: Model instance to train
            X_train: Training features
            y_train: Training labels
            X_test: Optional test features
            y_test: Optional test labels

        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training {model_name}...")

        # Fit model
        model.fit(X_train, y_train)

        result = {
            "model": model,
            "model_name": model_name
        }

        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            preds = model.predict(X_test)
            accuracy = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            result.update({
                "accuracy": accuracy,
                "f1": f1,
                "predictions": preds,
                "report": classification_report(y_test, preds, zero_division=0)
            })

            logger.info(f"{model_name} - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        else:
            logger.info(f"{model_name} - Training complete (no test evaluation)")

        # Store trained model
        self.models_[model_name] = model

        return result

    def train_multiple_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        custom_params: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Train multiple models on acoustic/prosodic features.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Optional test features
            y_test: Optional test labels
            custom_params: Optional dict of custom parameters per model type
                          Format: {'random_forest': {'n_estimators': 100}, ...}

        Returns:
            Dictionary of trained models and their results
        """
        logger.info("=" * 70)
        logger.info("Training Acoustic/Prosodic Models - COMPONENT-SPECIFIC")
        logger.info(f"Allowed models: {', '.join(self.ALLOWED_MODEL_TYPES)}")
        logger.info(f"Training samples: {len(X_train)}, Features: {X_train.shape[1]}")
        if X_test is not None:
            logger.info(f"Test samples: {len(X_test)}")
        logger.info("=" * 70)

        results = {}

        # Get parameters (use custom if provided, otherwise use acoustic-optimized defaults)
        params = custom_params if custom_params else self.MODEL_CONFIGS

        # Define COMPONENT-SPECIFIC models (only XGBoost and Random Forest)
        models = {
            "xgboost": XGBClassifier(
                **params.get('xgboost', self.MODEL_CONFIGS['xgboost'])
            ),
            "random_forest": RandomForestClassifier(
                **params.get('random_forest', self.MODEL_CONFIGS['random_forest'])
            ),
        }

        # Train each model
        for name, model in models.items():
            try:
                result = self.train_model(
                    name, model, X_train, y_train, X_test, y_test
                )
                results[name] = result
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {
                    "error": str(e),
                    "model_name": name
                }

        logger.info("=" * 70)
        logger.info(f"✓ Acoustic/Prosodic training complete: {len(results)} models")
        logger.info("=" * 70)

        return results

    def get_model(self, model_name: str):
        """Get a trained model by name."""
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models_.keys())}")
        return self.models_[model_name]

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with a trained model."""
        model = self.get_model(model_name)
        return model.predict(X)

    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities from a trained model."""
        model = self.get_model(model_name)
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model {model_name} does not support probability predictions")
        return model.predict_proba(X)