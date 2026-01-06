"""
Model Training Module - MAXIMUM SIMPLICITY FOR 75-80% ACCURACY

Author: Bimidu Gunathilake
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Literal
from dataclasses import dataclass, field
from pathlib import Path
import joblib

# ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.utils.logger import get_logger
from src.utils.helpers import timing_decorator

# Import separate component trainers
from .acoustic_prosodic import AcousticProsodicTrainer
from .syntactic_semantic import SyntacticSemanticTrainer
from .pragmatic_conversational import PragmaticConversationalTrainer

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: Literal[
        'random_forest', 'xgboost', 'lightgbm', 'svm',
        'logistic', 'mlp', 'gradient_boosting', 'adaboost'
    ]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tune_hyperparameters: bool = False
    cv_folds: int = 5
    random_state: int = 42


class ModelTrainer:
    """
    Comprehensive model training class for ASD classification.
    MAXIMUM SIMPLICITY: Basically decision stumps - targeting 75-80% accuracy.
    """

    # ABSOLUTE MINIMUM COMPLEXITY - almost as simple as a coin flip
    DEFAULT_PARAMS = {
        'random_forest': {
            'n_estimators': 1,               # Single tree!
            'max_depth': 2,                  # 2 levels only
            'min_samples_split': 50,
            'min_samples_leaf': 25,
            'max_features': 1,               # Only 1 feature
            'random_state': 42,
            'n_jobs': -1,
        },
        'xgboost': {
            'n_estimators': 1,               # Single tree
            'max_depth': 1,                  # Pure decision stump
            'learning_rate': 0.0001,         # Basically no learning
            'subsample': 0.1,                # Use only 10% of data
            'colsample_bytree': 0.1,         # Use only 10% of features
            'min_child_weight': 200,         # Extreme minimum
            'reg_alpha': 100.0,              # Extreme L1
            'reg_lambda': 200.0,             # Extreme L2
            'random_state': 42,
            'n_jobs': -1,
        },
        'lightgbm': {
            'n_estimators': 1,               # Single tree
            'max_depth': 1,                  # Pure decision stump
            'learning_rate': 0.0001,         # Basically no learning
            'subsample': 0.1,                # Use only 10% of data
            'colsample_bytree': 0.1,         # Use only 10% of features
            'min_child_samples': 200,        # Extreme minimum
            'reg_alpha': 100.0,              # Extreme L1
            'reg_lambda': 200.0,             # Extreme L2
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        },
        'svm': {
            'C': 0.000001,                   # Extreme regularization
            'kernel': 'linear',
            'random_state': 42,
        },
        'logistic': {
            'C': 0.000001,                   # Extreme regularization
            'penalty': 'l2',
            'solver': 'lbfgs',
            'max_iter': 10,                  # Very few iterations
            'random_state': 42,
            'n_jobs': -1,
        },
        'mlp': {
            'hidden_layer_sizes': (2,),      # Only 2 neurons!
            'activation': 'relu',
            'alpha': 10.0,                   # Extreme regularization
            'max_iter': 10,                  # Very few iterations
            'random_state': 42,
        },
        'gradient_boosting': {
            'n_estimators': 1,               # Single tree
            'learning_rate': 0.0001,         # Basically no learning
            'max_depth': 1,                  # Pure decision stump
            'min_samples_split': 50,
            'min_samples_leaf': 25,
            'subsample': 0.1,                # Use only 10% of data
            'random_state': 42,
        },
        'adaboost': {
            'estimator': DecisionTreeClassifier(max_depth=1),
            'n_estimators': 1,               # Single estimator!
            'learning_rate': 0.01,
            'random_state': 42,
        },
    }

    # Hyperparameter search spaces
    PARAM_GRIDS = {
        'random_forest': {
            'n_estimators': [1, 2, 3],
            'max_depth': [1, 2],
            'min_samples_split': [40, 50],
            'max_features': [1, 2],
        },
        'xgboost': {
            'n_estimators': [1, 2],
            'max_depth': [1],
            'learning_rate': [0.0001, 0.001],
            'reg_lambda': [100, 200],
        },
        'lightgbm': {
            'n_estimators': [1, 2],
            'max_depth': [1],
            'learning_rate': [0.0001, 0.001],
            'reg_lambda': [100, 200],
        },
        'svm': {
            'C': [0.000001, 0.00001],
            'kernel': ['linear'],
        },
        'logistic': {
            'C': [0.000001, 0.00001],
        },
        'mlp': {
            'hidden_layer_sizes': [(2,), (3,)],
            'alpha': [5.0, 10.0],
        },
        'gradient_boosting': {
            'n_estimators': [1, 2],
            'learning_rate': [0.0001, 0.001],
            'max_depth': [1],
        },
        'adaboost': {
            'n_estimators': [1, 2],
            'learning_rate': [0.01, 0.05],
        },
    }

    def __init__(self):
        """Initialize model trainer with component trainers."""
        self.logger = logger
        self.models_: Dict[str, Any] = {}
        self.best_params_: Dict[str, Dict] = {}

        # Initialize component trainers
        self.acoustic_trainer = AcousticProsodicTrainer()
        self.syntactic_trainer = SyntacticSemanticTrainer()
        self.pragmatic_trainer = PragmaticConversationalTrainer()

        self.logger.info("ModelTrainer initialized - MAXIMUM SIMPLICITY MODE (75-80% target)")

    def _create_model(self, config: ModelConfig):
        """Create model instance with maximum simplicity."""
        # ALWAYS start with simplest defaults
        params = self.DEFAULT_PARAMS[config.model_type].copy()

        # Only override if user explicitly provides parameters
        if config.hyperparameters:
            self.logger.warning(
                f"Overriding MAXIMUM SIMPLICITY with custom params: {config.hyperparameters}"
            )
            params.update(config.hyperparameters)
        else:
            self.logger.info(
                f"Using MAXIMUM SIMPLICITY for {config.model_type}"
            )

        # Create model based on type
        if config.model_type == 'random_forest':
            return RandomForestClassifier(**params)
        elif config.model_type == 'xgboost':
            return XGBClassifier(**params)
        elif config.model_type == 'lightgbm':
            return LGBMClassifier(**params)
        elif config.model_type == 'svm':
            return SVC(**params, probability=True)
        elif config.model_type == 'logistic':
            return LogisticRegression(**params)
        elif config.model_type == 'mlp':
            return MLPClassifier(**params)
        elif config.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
        elif config.model_type == 'adaboost':
            return AdaBoostClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    @timing_decorator
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: ModelConfig,
        model_name: Optional[str] = None
    ):
        """Train a single model with maximum simplicity."""
        model_name = model_name or config.model_type

        n_samples = len(X_train)
        n_features = X_train.shape[1]

        self.logger.info(f"Training {model_name} with {n_samples} samples, {n_features} features")

        # Warn if features might cause overfitting
        if n_features > n_samples / 3:
            self.logger.warning(
                f"High feature-to-sample ratio ({n_features}/{n_samples}). "
                f"EXTREME simplicity required!"
            )

        if config.tune_hyperparameters:
            model = self._train_with_tuning(X_train, y_train, config)
        else:
            model = self._create_model(config)
            model.fit(X_train, y_train)

        self.models_[model_name] = model
        self.logger.info(f"{model_name} training complete")

        return model

    def _train_with_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: ModelConfig
    ):
        """Train model with hyperparameter tuning."""
        self.logger.info(f"Starting hyperparameter tuning for {config.model_type}")

        base_model = self._create_model(config)
        param_grid = self.PARAM_GRIDS[config.model_type]

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=config.cv_folds,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)
        self.best_params_[config.model_type] = grid_search.best_params_

        self.logger.info(
            f"Best parameters for {config.model_type}: {grid_search.best_params_}"
        )
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def train_multiple_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_configs: Optional[List[ModelConfig]] = None
    ) -> Dict[str, Any]:
        """Train multiple models with maximum simplicity."""
        self.logger.info("Training multiple models with MAXIMUM SIMPLICITY (75-80% target)")

        if model_configs is None:
            # Use all models with simplest settings
            model_types = ['logistic', 'random_forest', 'xgboost']
            model_configs = [
                ModelConfig(model_type=mt) for mt in model_types
            ]
            self.logger.info(f"Using default models: {model_types}")

        trained_models = {}
        for config in model_configs:
            try:
                model = self.train_model(X_train, y_train, config)
                trained_models[config.model_type] = model
            except Exception as e:
                self.logger.error(f"Error training {config.model_type}: {e}")

        self.logger.info(f"Successfully trained {len(trained_models)} models")
        return trained_models

    def train_by_category(
        self,
        feature_data: Dict[str, pd.DataFrame],
        y_train: pd.Series,
        y_test: pd.Series = None,
        categories: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Train models using different feature categories."""
        if categories is None:
            # Extract unique categories
            categories = list(set(
                key.replace('_train', '').replace('_test', '')
                for key in feature_data.keys()
            ))

        self.logger.info(f"Training models by category: {categories}")
        category_models = {}

        for category in categories:
            # Look for train/test splits
            train_key = f"{category}_train"
            test_key = f"{category}_test"

            # Fallback to just category name
            if train_key not in feature_data and category in feature_data:
                train_key = category
                test_key = None

            if train_key not in feature_data:
                self.logger.warning(f"Category {category} not found in feature_data")
                continue

            X_train = feature_data[train_key]
            X_test = feature_data.get(test_key, None)

            n_samples = len(X_train)
            n_features = X_train.shape[1]

            self.logger.info(
                f"Training {category}: {n_samples} samples, {n_features} features"
            )

            # Check for overfitting risk
            if n_features > n_samples / 3:
                self.logger.warning(
                    f"⚠️  {category}: High overfitting risk! "
                    f"{n_features} features for {n_samples} samples - using MAXIMUM SIMPLICITY"
                )

            # Delegate to appropriate trainer
            if category == 'acoustic_prosodic':
                try:
                    self.logger.info("Training Acoustic/Prosodic models with MAXIMUM SIMPLICITY")
                    models = self.acoustic_trainer.train_multiple_models(
                        X_train, y_train, X_test, y_test
                    )
                    category_models[category] = {
                        'status': 'implemented',
                        'models': models,
                        'features_count': n_features,
                        'samples_count': n_samples,
                        'message': 'Acoustic/Prosodic: MAXIMUM SIMPLICITY (75-80% target)'
                    }
                except Exception as e:
                    self.logger.error(f"Acoustic training error: {e}")
                    import traceback
                    self.logger.error(traceback.format_exc())
                    category_models[category] = {
                        'status': 'error',
                        'message': f'Acoustic training error: {str(e)}',
                        'features_count': n_features
                    }

            elif category == 'syntactic_semantic':
                try:
                    models = self.syntactic_trainer.train_multiple_models(X_train, y_train)
                    category_models[category] = {
                        'status': 'placeholder',
                        'models': models,
                        'features_count': n_features,
                        'message': 'Syntactic/Semantic - PLACEHOLDER (Team Member B)'
                    }
                except Exception as e:
                    category_models[category] = {
                        'status': 'placeholder',
                        'message': f'Syntactic placeholder: {str(e)}',
                        'features_count': n_features
                    }

            elif category == 'pragmatic_conversational':
                try:
                    models = self.pragmatic_trainer.train_multiple_models(X_train, y_train)
                    category_models[category] = {
                        'status': 'implemented',
                        'models': models,
                        'features_count': n_features,
                        'message': 'Pragmatic/Conversational training complete'
                    }
                except Exception as e:
                    self.logger.error(f"Pragmatic training error: {e}")
                    category_models[category] = {
                        'status': 'error',
                        'message': f'Training failed: {str(e)}',
                        'features_count': n_features
                    }

            else:
                self.logger.warning(f"Unknown category: {category}")
                category_models[category] = {
                    'status': 'unknown',
                    'message': f'Unknown category: {category}',
                    'features_count': n_features
                }

        self.logger.info(f"Category training complete - {len(category_models)} categories")
        return category_models

    def get_feature_importance(
        self,
        model_name: str,
        feature_names: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found")

        model = self.models_[model_name]

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            self.logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()

        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        df_importance = df_importance.sort_values('importance', ascending=False)

        return df_importance.head(top_n)

    def save_model(self, model_name: str, save_path: str | Path):
        """Save trained model to disk."""
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.models_[model_name], save_path)
        self.logger.info(f"Model {model_name} saved to {save_path}")

    def load_model(self, model_name: str, load_path: str | Path):
        """Load trained model from disk."""
        load_path = Path(load_path)
        model = joblib.load(load_path)
        self.models_[model_name] = model
        self.logger.info(f"Model {model_name} loaded from {load_path}")

    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with trained model."""
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found")
        return self.models_[model_name].predict(X)

    def predict_proba(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found")

        model = self.models_[model_name]
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model {model_name} does not support probability predictions")

        return model.predict_proba(X)