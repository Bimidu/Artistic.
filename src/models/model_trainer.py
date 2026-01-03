"""
Model Training Module

This module provides comprehensive model training for ASD classification
using features from all three categories:

1. Acoustic & Prosodic Features (Team Member A - Placeholder)
2. Syntactic & Semantic Features (Team Member B - Placeholder)  
3. Pragmatic & Conversational Features (Fully Implemented)

This is the main orchestrator that delegates to separate component trainers:
- src/models/acoustic_prosodic/ (Team Member A)
- src/models/syntactic_semantic/ (Team Member B)
- src/models/pragmatic_conversational/ (Fully Implemented)

Author: Bimidu Gunathilake
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Literal, Union
from dataclasses import dataclass, field
from pathlib import Path
import joblib

# ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
    """
    Configuration for model training.
    
    Attributes:
        model_type: Type of model to train
        hyperparameters: Model hyperparameters
        tune_hyperparameters: Whether to tune hyperparameters
        cv_folds: Number of cross-validation folds for tuning
        random_state: Random state for reproducibility
    """
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
    
    Supports multiple ML algorithms with hyperparameter tuning and
    cross-validation.
    """
    
    # Default hyperparameters for each model type
    DEFAULT_PARAMS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        },
        'svm': {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'random_state': 42,
        },
        'logistic': {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1,
        },
        'mlp': {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'max_iter': 500,
            'random_state': 42,
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'random_state': 42,
        },
        'adaboost': {
            'n_estimators': 100,
            'learning_rate': 1.0,
            'random_state': 42,
        },
    }
    
    # Hyperparameter search spaces for tuning
    PARAM_GRIDS = {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        'xgboost': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
        },
        'lightgbm': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
        },
        'svm': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
        },
        'logistic': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
        },
        'mlp': {
            'hidden_layer_sizes': [(50,), (100,), (100, 50), (150, 100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
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
        
        self.logger.info("ModelTrainer initialized with component trainers")
    
    def _create_model(self, config: ModelConfig):
        """
        Create model instance from configuration.
        
        Args:
            config: Model configuration
        
        Returns:
            Model instance
        """
        # Get default parameters and update with user params
        params = self.DEFAULT_PARAMS[config.model_type].copy()
        params.update(config.hyperparameters)
        
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
        """
        Train a single model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            config: Model configuration
            model_name: Name for the model (default: model_type)
        
        Returns:
            Trained model
        """
        model_name = model_name or config.model_type
        self.logger.info(f"Training {model_name} model...")
        
        if config.tune_hyperparameters:
            # Hyperparameter tuning
            model = self._train_with_tuning(X_train, y_train, config)
        else:
            # Train with default/provided parameters
            model = self._create_model(config)
            model.fit(X_train, y_train)
        
        # Store model
        self.models_[model_name] = model
        
        self.logger.info(f"{model_name} training complete")
        
        return model
    
    def _train_with_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config: ModelConfig
    ):
        """
        Train model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            config: Model configuration
        
        Returns:
            Best model from tuning
        """
        self.logger.info(f"Starting hyperparameter tuning for {config.model_type}")
        
        # Create base model
        base_model = self._create_model(config)
        
        # Get parameter grid
        param_grid = self.PARAM_GRIDS[config.model_type]
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=config.cv_folds,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best parameters
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
        """
        Train multiple models for comparison.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_configs: List of model configurations (None = train all)
        
        Returns:
            Dict of trained models
        """
        self.logger.info("Training multiple models for comparison")
        
        # If no configs provided, train all with defaults
        if model_configs is None:
            model_types = [
                'random_forest', 'xgboost', 'lightgbm',
                'logistic', 'svm'
            ]
            model_configs = [
                ModelConfig(model_type=mt) for mt in model_types
            ]
        
        # Train each model
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
        categories: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train models using different feature categories.
        
        Args:
            feature_data: Dict mapping category names to feature DataFrames
            y_train: Training labels
            categories: List of categories to train (None = all available)
        
        Returns:
            Dict mapping categories to trained models
        """
        if categories is None:
            categories = list(feature_data.keys())
        
        self.logger.info(f"Training models by category: {categories}")
        
        category_models = {}
        
        for category in categories:
            if category not in feature_data:
                self.logger.warning(f"Category {category} not found in feature_data")
                continue
            
            X_cat = feature_data[category]
            self.logger.info(f"Training models for {category} with {X_cat.shape[1]} features")
            
            # Delegate to appropriate component trainer
            if category == 'acoustic_prosodic':
                # Use acoustic trainer (placeholder)
                self.logger.info("Acoustic/Prosodic training - PLACEHOLDER")
                try:
                    models = self.acoustic_trainer.train_multiple_models(X_cat, y_train)
                    category_models[category] = {
                        'status': 'placeholder',
                        'models': models,
                        'features_count': X_cat.shape[1],
                        'message': 'Acoustic/Prosodic feature training - PLACEHOLDER (Team Member A)'
                    }
                except Exception as e:
                    category_models[category] = {
                        'status': 'placeholder',
                        'message': f'Acoustic training placeholder: {str(e)}',
                        'features_count': X_cat.shape[1]
                    }
                
            elif category == 'syntactic_semantic':
                # Use syntactic trainer (placeholder)
                self.logger.info("Syntactic/Semantic training - PLACEHOLDER")
                try:
                    models = self.syntactic_trainer.train_multiple_models(X_cat, y_train)
                    category_models[category] = {
                        'status': 'placeholder',
                        'models': models,
                        'features_count': X_cat.shape[1],
                        'message': 'Syntactic/Semantic feature training - PLACEHOLDER (Team Member B)'
                    }
                except Exception as e:
                    category_models[category] = {
                        'status': 'placeholder',
                        'message': f'Syntactic training placeholder: {str(e)}',
                        'features_count': X_cat.shape[1]
                    }
                
            elif category == 'pragmatic_conversational':
                # Use pragmatic trainer (fully implemented)
                self.logger.info("Pragmatic/Conversational training - IMPLEMENTED")
                try:
                    # Train models for pragmatic/conversational features
                    models = self.pragmatic_trainer.train_multiple_models(X_cat, y_train)
                    category_models[category] = {
                        'status': 'implemented',
                        'models': models,
                        'features_count': X_cat.shape[1],
                        'message': 'Successfully trained pragmatic/conversational models'
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error training pragmatic/conversational models: {e}")
                    category_models[category] = {
                        'status': 'error',
                        'message': f'Training failed: {str(e)}',
                        'features_count': X_cat.shape[1]
                    }
            
            else:
                # Unknown category
                self.logger.warning(f"Unknown category: {category}")
                category_models[category] = {
                    'status': 'unknown',
                    'message': f'Unknown category: {category}',
                    'features_count': X_cat.shape[1] if X_cat is not None else 0
                }
        
        self.logger.info(f"Category training complete - {len(category_models)} categories processed")
        
        return category_models
    
    def get_feature_importance(
        self,
        model_name: str,
        feature_names: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model_name: Name of trained model
            feature_names: List of feature names
            top_n: Number of top features to return
        
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models_[model_name]
        
        # Get feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            self.logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        df_importance = df_importance.sort_values('importance', ascending=False)
        
        return df_importance.head(top_n)
    
    def save_model(
        self,
        model_name: str,
        save_path: Union[str, Path]
    ):
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of model to save
            save_path: Path to save model
        """
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.models_[model_name], save_path)
        
        self.logger.info(f"Model {model_name} saved to {save_path}")
    
    def load_model(
        self,
        model_name: str,
        load_path: Union[str, Path]
    ):
        """
        Load trained model from disk.
        
        Args:
            model_name: Name to assign to loaded model
            load_path: Path to load model from
        """
        load_path = Path(load_path)
        
        # Load model
        model = joblib.load(load_path)
        self.models_[model_name] = model
        
        self.logger.info(f"Model {model_name} loaded from {load_path}")
    
    def predict(
        self,
        model_name: str,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            model_name: Name of model to use
            X: Features to predict
        
        Returns:
            np.ndarray: Predictions
        """
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found")
        
        return self.models_[model_name].predict(X)
    
    def predict_proba(
        self,
        model_name: str,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            model_name: Name of model to use
            X: Features to predict
        
        Returns:
            np.ndarray: Prediction probabilities
        """
        if model_name not in self.models_:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models_[model_name]
        
        if not hasattr(model, 'predict_proba'):
            raise ValueError(f"Model {model_name} does not support probability predictions")
        
        return model.predict_proba(X)

