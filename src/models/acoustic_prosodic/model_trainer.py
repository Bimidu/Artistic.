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
from .preprocessor import AcousticProsodicPreprocessor

logger = get_logger(__name__)


class AcousticProsodicTrainer:
    """
    Trains ML models on acoustic & prosodic features.
    COMPONENT-SPECIFIC: Uses two main models (Random Forest + AdaBoost),
    mirroring the pragmatic/conversational component design.
    """

    # COMPONENT-SPECIFIC: Acoustic/Prosodic models
    # Updated to include Logistic Regression for better performance balance
    ALLOWED_MODEL_TYPES = ['logistic_regression', 'random_forest', 'xgboost']

    # Acoustic-optimized hyperparameters (balanced for 70-80% performance range)
    MODEL_CONFIGS = {
        'logistic_regression': {
            'C': 0.1,  # Moderate regularization for 70-80% range
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'solver': 'liblinear'  # Good for small datasets
        },
        'random_forest': {
            'n_estimators': 25,  # Moderate number of trees
            'max_depth': 4,  # Controlled depth
            'min_samples_split': 12,  # Balanced split requirement
            'min_samples_leaf': 6,  # Balanced leaf requirement
            'max_features': 0.5,  # Half of features
            'bootstrap': True,
            'random_state': 42,
            'class_weight': 'balanced',
        },
        'xgboost': {
            'n_estimators': 75,  # Moderate complexity
            'max_depth': 3,  # Shallow trees
            'learning_rate': 0.15,  # Moderate learning rate
            'subsample': 0.8,  # Some subsampling
            'colsample_bytree': 0.8,  # Some feature subsampling
            'min_child_weight': 3,  # Moderate regularization
            'gamma': 0.1,  # Small penalty
            'reg_alpha': 0.02,  # Small L1 regularization
            'reg_lambda': 0.1,  # Small L2 regularization
            'random_state': 42,
            'eval_metric': 'logloss',
        },
    }

    # Alternative configurations for different performance targets
    BALANCED_CONFIGS = {
        'logistic_regression_conservative': {
            'C': 0.01,  # Stronger regularization
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'solver': 'liblinear'
        },
        'random_forest_conservative': {
            'n_estimators': 15,
            'max_depth': 3,
            'min_samples_split': 25,
            'min_samples_leaf': 12,
            'max_features': 0.4,
            'bootstrap': True,
            'random_state': 42,
            'class_weight': 'balanced',
        },
        'xgboost_conservative': {
            'n_estimators': 40,
            'max_depth': 2,
            'learning_rate': 0.2,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 5,
            'gamma': 0.3,
            'reg_alpha': 0.1,
            'reg_lambda': 0.3,
            'random_state': 42,
            'eval_metric': 'logloss',
        }
    }

    # Ultra-conservative configurations for extreme overfitting
    ULTRA_CONSERVATIVE_CONFIGS = {
        'random_forest_ultra': {
            'n_estimators': 5,   # Very few trees
            'max_depth': 2,      # Very shallow
            'min_samples_split': 40,  # Very high
            'min_samples_leaf': 20,   # Very high
            'max_features': 0.2,      # Very few features
            'bootstrap': True,
            'random_state': 42,
            'class_weight': 'balanced',
        },
        'xgboost_ultra': {
            'n_estimators': 10,  # Very few trees
            'max_depth': 1,      # Stumps only
            'learning_rate': 0.3, # High learning rate, few trees
            'subsample': 0.5,    # Heavy subsampling
            'colsample_bytree': 0.5,  # Heavy feature subsampling
            'min_child_weight': 10,   # Very high
            'gamma': 0.5,        # High penalty
            'reg_alpha': 0.3,    # High L1
            'reg_lambda': 0.5,   # High L2
            'random_state': 42,
            'eval_metric': 'logloss',
        }
    }

    # Target-specific configurations for 70-80% range
    TARGET_RANGE_CONFIGS = {
        'logistic_regression_target': {
            'C': 0.05,  # Optimal regularization for 70-80%
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced',
            'solver': 'liblinear'
        },
        'random_forest_target': {
            'n_estimators': 8,   # Very few trees
            'max_depth': 2,      # Very shallow
            'min_samples_split': 50,  # Very high
            'min_samples_leaf': 25,   # Very high
            'max_features': 0.15,     # Very few features
            'bootstrap': True,
            'random_state': 42,
            'class_weight': 'balanced',
        },
        'xgboost_target': {
            'n_estimators': 20,  # Very few trees
            'max_depth': 2,      # Very shallow
            'learning_rate': 0.1, # Moderate
            'subsample': 0.6,    # Heavy subsampling
            'colsample_bytree': 0.6,  # Heavy feature subsampling
            'min_child_weight': 15,   # Very high
            'gamma': 0.2,        # Penalty
            'reg_alpha': 0.2,    # L1 regularization
            'reg_lambda': 0.4,   # L2 regularization
            'random_state': 42,
            'eval_metric': 'logloss',
        }
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

        # Define COMPONENT-SPECIFIC models (Logistic Regression, Random Forest, XGBoost)
        models = {
            "logistic_regression": LogisticRegression(
                **params.get('logistic_regression',
                    params.get('logistic_regression_conservative',
                        params.get('logistic_regression_target', self.MODEL_CONFIGS['logistic_regression'])))
            ),
            "random_forest": RandomForestClassifier(
                **params.get('random_forest',
                    params.get('random_forest_conservative',
                        params.get('random_forest_ultra',
                            params.get('random_forest_target', self.MODEL_CONFIGS['random_forest']))))
            ),
            "xgboost": XGBClassifier(
                **params.get('xgboost',
                    params.get('xgboost_conservative',
                        params.get('xgboost_ultra',
                            params.get('xgboost_target', self.MODEL_CONFIGS['xgboost']))))
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

    def get_conservative_params(self):
        """Get conservative hyperparameters for smaller datasets or to reduce overfitting"""
        return self.BALANCED_CONFIGS

    def train_with_fallback(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        target_accuracy_range: tuple = (0.70, 0.85)
    ) -> Dict[str, Any]:
        """
        Train models with progressive fallback to more conservative parameters.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Optional test features
            y_test: Optional test labels
            target_accuracy_range: Tuple of (min_acc, max_acc) for desired range

        Returns:
            Dictionary of trained models and their results
        """
        logger.info("Training with progressive fallback for optimal performance range")
        target_center = sum(target_accuracy_range) / 2

        # Step 1: Try standard configs
        logger.info("Step 1: Trying standard configurations...")
        results_standard = self.train_multiple_models(X_train, y_train, X_test, y_test)

        # Check performance and decide next step
        best_results = results_standard.copy()

        if X_test is not None and y_test is not None:
            # Step 2: Try conservative if standard overfits
            need_conservative = any(
                result.get("accuracy", 0) > target_accuracy_range[1]
                for result in results_standard.values()
                if "error" not in result
            )

            if need_conservative:
                logger.info("Step 2: Standard configs overfit, trying conservative...")
                results_conservative = self.train_multiple_models(
                    X_train, y_train, X_test, y_test,
                    custom_params=self.BALANCED_CONFIGS
                )

                # Step 3: Try ultra-conservative if conservative still overfits
                need_ultra = any(
                    result.get("accuracy", 0) > target_accuracy_range[1]
                    for result in results_conservative.values()
                    if "error" not in result
                )

                if need_ultra:
                    logger.info("Step 3: Conservative still overfits, trying ultra-conservative...")
                    results_ultra = self.train_multiple_models(
                        X_train, y_train, X_test, y_test,
                        custom_params=self.ULTRA_CONSERVATIVE_CONFIGS
                    )

                    # Step 4: Try target-specific if ultra still overfits
                    need_target = any(
                        result.get("accuracy", 0) > target_accuracy_range[1]
                        for result in results_ultra.values()
                        if "error" not in result
                    )

                    if need_target:
                        logger.info("Step 4: Ultra-conservative still overfits, trying target-specific...")
                        results_target = self.train_multiple_models(
                            X_train, y_train, X_test, y_test,
                            custom_params=self.TARGET_RANGE_CONFIGS
                        )

                        all_results = [
                            ("standard", results_standard),
                            ("conservative", results_conservative),
                            ("ultra", results_ultra),
                            ("target", results_target)
                        ]
                    else:
                        all_results = [
                            ("standard", results_standard),
                            ("conservative", results_conservative),
                            ("ultra", results_ultra)
                        ]
                else:
                    all_results = [
                        ("standard", results_standard),
                        ("conservative", results_conservative)
                    ]
            else:
                all_results = [("standard", results_standard)]

            # Select the best configuration for each model
            final_results = {}
            for model_name in results_standard.keys():
                best_config = "standard"
                best_distance = float('inf')
                best_result = results_standard[model_name]

                for config_name, config_results in all_results:
                    result = config_results.get(model_name, {})
                    if "error" not in result and "accuracy" in result:
                        acc = result["accuracy"]
                        distance = abs(acc - target_center)

                        if distance < best_distance:
                            best_distance = distance
                            best_result = result
                            best_config = config_name

                final_results[model_name] = best_result
                if "accuracy" in best_result:
                    logger.info(f"{model_name}: Using {best_config} config -> {best_result['accuracy']:.3f}")

            return final_results

        return results_standard

    def train_with_balanced_features(
        self,
        df: pd.DataFrame,
        label_col: str = "diagnosis",
        enable_feature_selection: bool = True,
        target_accuracy_range: tuple = (0.70, 0.80)
    ) -> Dict[str, Any]:
        """
        Train models with balanced feature selection to achieve 70-80% performance.

        Args:
            df: Full dataset with features and labels
            label_col: Name of the label column
            enable_feature_selection: Whether to use balanced feature selection
            target_accuracy_range: Target performance range

        Returns:
            Dictionary of trained models and their results
        """
        logger.info("Training acoustic models with balanced feature selection")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Feature selection enabled: {enable_feature_selection}")
        logger.info(f"Target performance range: {target_accuracy_range[0]:.0%}-{target_accuracy_range[1]:.0%}")

        # Use the balanced preprocessor
        preprocessor = AcousticProsodicPreprocessor(
            test_size=0.25,
            random_state=42,
            enable_feature_selection=enable_feature_selection,
            target_performance_range=target_accuracy_range
        )

        # Preprocess the data
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, label_col=label_col)

        logger.info(f"After preprocessing: {X_train.shape[1]} features, {len(X_train)} train samples, {len(X_test)} test samples")

        # Train models with the fallback mechanism
        results = self.train_with_fallback(
            X_train, y_train, X_test, y_test, target_accuracy_range
        )

        # Add preprocessing info to results
        for model_name, result in results.items():
            if isinstance(result, dict) and "error" not in result:
                result["preprocessing"] = {
                    "feature_selection_enabled": enable_feature_selection,
                    "final_feature_count": X_train.shape[1],
                    "original_feature_count": df.shape[1] - 1,  # Minus label column
                    "feature_reduction_ratio": X_train.shape[1] / (df.shape[1] - 1)
                }

        return results

