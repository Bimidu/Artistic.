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
    COMPONENT-SPECIFIC: Uses two main models (Random Forest + AdaBoost),
    mirroring the pragmatic/conversational component design.
    """

    # COMPONENT-SPECIFIC: Acoustic/Prosodic models
    # Only Random Forest and AdaBoost are used as the two main models
    # These ensemble models work well with continuous acoustic features
    ALLOWED_MODEL_TYPES = ['random_forest', 'xgboost']

    # Acoustic-optimized hyperparameters (fine-tuned for prosodic/spectral features)
    MODEL_CONFIGS = {
        'random_forest': {
            'n_estimators': 20,  # Very low
            'max_depth': 2,  # Very shallow
            'min_samples_split': 30,  # Very high
            'min_samples_leaf': 20,  # Very high
            'max_features': 0.3,  # Very few features
            'bootstrap': True,
            'random_state': 42,
            'class_weight': 'balanced',
        },
        'xgboost': {
            'n_estimators': 500,  # Much higher
            'max_depth': 10,  # Much deeper
            'learning_rate': 0.01,  # Much lower
            'subsample': 1.0,  # All data
            'colsample_bytree': 1.0,  # All features
            'min_child_weight': 0.5,  # Very low
            'gamma': 0,  # No penalty
            'reg_alpha': 0,  # No L1 regularization
            'reg_lambda': 0,  # No L2 regularization
            'random_state': 42,
            'eval_metric': 'logloss',
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

        # Define COMPONENT-SPECIFIC models (only Random Forest and AdaBoost)
        models = {
            "random_forest": RandomForestClassifier(
                **params.get('random_forest', self.MODEL_CONFIGS['random_forest'])
            ),
            "xgboost": XGBClassifier(  # Change from AdaBoostClassifier to XGBClassifier
                **params.get('xgboost', self.MODEL_CONFIGS['xgboost'])
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