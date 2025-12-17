"""
Complete Data Preprocessing Pipeline

This module provides an end-to-end preprocessing pipeline for all three
feature categories:

1. Acoustic & Prosodic Features (Team Member A - Placeholder)
2. Syntactic & Semantic Features (Team Member B - Placeholder)
3. Pragmatic & Conversational Features (Fully Implemented)

Combines validation, cleaning, feature selection, and train/test splitting.

Author: Bimidu Gunathilake
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal, Union
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib

from .data_validator import DataValidator, ValidationReport
from .data_cleaner import DataCleaner, FeatureScaler
from .feature_selector import FeatureSelector
from src.utils.logger import get_logger
from config import config

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Complete preprocessing pipeline for pragmatic/conversational features.
    
    Handles validation, cleaning, feature selection, scaling, and splitting.
    """
    
    def __init__(
        self,
        target_column: str = 'diagnosis',
        test_size: float = 0.2,
        random_state: int = 42,
        # Validation parameters
        min_samples: int = 10,
        max_missing_ratio: float = 0.3,
        # Cleaning parameters
        missing_strategy: Literal['mean', 'median', 'knn', 'drop'] = 'median',
        outlier_method: Literal['clip', 'remove', 'winsorize', 'none'] = 'clip',
        # Scaling parameters
        scaling_method: Literal['standard', 'minmax', 'robust'] = 'standard',
        # Feature selection parameters
        feature_selection: bool = True,
        n_features: int = 30,
    ):
        """
        Initialize preprocessing pipeline.
        
        Args:
            target_column: Name of target variable
            test_size: Fraction of data for test set
            random_state: Random state for reproducibility
            min_samples: Minimum samples required
            max_missing_ratio: Maximum allowed missing value ratio
            missing_strategy: Strategy for handling missing values
            outlier_method: Method for handling outliers
            scaling_method: Feature scaling method
            feature_selection: Whether to perform feature selection
            n_features: Number of features to select (if feature_selection=True)
        """
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.feature_selection = feature_selection
        self.n_features = n_features
        
        # Initialize components
        self.validator = DataValidator(
            min_samples=min_samples,
            max_missing_ratio=max_missing_ratio
        )
        self.cleaner = DataCleaner(
            missing_strategy=missing_strategy,
            outlier_method=outlier_method
        )
        self.scaler = FeatureScaler(method=scaling_method)
        self.selector = FeatureSelector() if feature_selection else None
        
        # Store fitted state
        self.feature_columns_: Optional[List[str]] = None
        self.selected_features_: Optional[List[str]] = None
        self.validation_report_: Optional[ValidationReport] = None
        
        self.logger = logger
        self.logger.info("DataPreprocessor initialized")
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Fit preprocessor and transform data into train/test sets.
        
        Args:
            df: Input DataFrame with features and target
            validate: Whether to validate data first
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info(f"Starting preprocessing pipeline for data with shape {df.shape}")
        
        # Step 1: Validation (optional)
        if validate:
            self.validation_report_ = self.validator.validate(
                df,
                target_column=self.target_column
            )
            
            if not self.validation_report_.is_valid:
                self.logger.error("Data validation failed")
                self.validation_report_.print_summary()
                raise ValueError("Data validation failed. Check validation report.")
            
            self.logger.info("Data validation passed")
        
        # Step 2: Identify feature columns
        self.feature_columns_ = [
            col for col in df.columns
            if col not in [self.target_column, 'participant_id', 'file_path']
        ]
        self.logger.info(f"Identified {len(self.feature_columns_)} feature columns")
        
        # Step 3: Clean data
        df_clean = self.cleaner.clean(
            df,
            target_column=self.target_column,
            feature_columns=self.feature_columns_
        )
        self.logger.info(f"Data cleaned - Shape: {df_clean.shape}")
        
        # Step 4: Split into features and target
        X = df_clean[self.feature_columns_]
        y = df_clean[self.target_column]
        
        # Step 5: Feature selection (on full dataset before split)
        if self.selector is not None:
            self.logger.info(f"Performing feature selection")
            self.selected_features_ = self.selector.select_from_model(
                X, y,
                max_features=self.n_features
            )
            X = X[self.selected_features_]
            self.logger.info(f"Selected {len(self.selected_features_)} features")
        else:
            self.selected_features_ = self.feature_columns_
        
        # Step 6: Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        self.logger.info(
            f"Split data - Train: {X_train.shape}, Test: {X_test.shape}"
        )
        
        # Step 7: Scale features (fit on train, transform both)
        X_train = self.scaler.fit_transform(X_train, self.selected_features_)
        X_test = self.scaler.transform(X_test, self.selected_features_)
        self.logger.info("Features scaled")
        
        self.logger.info("Preprocessing pipeline complete")
        
        return X_train, X_test, y_train, y_test
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
        
        Returns:
            pd.DataFrame: Transformed features
        """
        if self.selected_features_ is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Clean data
        df_clean = self.cleaner.clean(
            df,
            target_column=self.target_column,
            feature_columns=self.feature_columns_
        )
        
        # Select features
        X = df_clean[self.selected_features_]
        
        # Scale
        X_scaled = self.scaler.transform(X, self.selected_features_)
        
        return X_scaled
    
    def prepare_k_fold_splits(
        self,
        df: pd.DataFrame,
        n_splits: int = 5
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Prepare K-fold cross-validation splits.
        
        Args:
            df: Input DataFrame
            n_splits: Number of folds
        
        Returns:
            List of (X_train, X_val, y_train, y_val) tuples
        """
        self.logger.info(f"Preparing {n_splits}-fold cross-validation splits")
        
        # Clean and select features
        df_clean = self.cleaner.clean(
            df,
            target_column=self.target_column,
            feature_columns=self.feature_columns_
        )
        
        X = df_clean[self.feature_columns_]
        y = df_clean[self.target_column]
        
        # Feature selection
        if self.selector is not None:
            self.selected_features_ = self.selector.select_from_model(
                X, y,
                max_features=self.n_features
            )
            X = X[self.selected_features_]
        else:
            self.selected_features_ = self.feature_columns_
        
        # Create stratified folds
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        splits = []
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Scale each fold independently
            scaler = FeatureScaler(method=self.scaler.method)
            X_train_fold = scaler.fit_transform(X_train_fold, self.selected_features_)
            X_val_fold = scaler.transform(X_val_fold, self.selected_features_)
            
            splits.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))
        
        self.logger.info(f"Created {n_splits} cross-validation splits")
        
        return splits
    
    def preprocess_by_category(
        self,
        feature_data: Dict[str, pd.DataFrame],
        target_column: str = 'diagnosis'
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Preprocess data by feature category.
        
        Args:
            feature_data: Dict mapping category names to feature DataFrames
            target_column: Name of target variable
        
        Returns:
            Dict mapping categories to (X_train, X_test, y_train, y_test) tuples
        """
        self.logger.info(f"Preprocessing data by category: {list(feature_data.keys())}")
        
        category_splits = {}
        
        for category, X_cat in feature_data.items():
            self.logger.info(f"Preprocessing {category} with {X_cat.shape[1]} features")
            
            # Add target column if not present
            if target_column not in X_cat.columns:
                self.logger.warning(f"Target column {target_column} not found in {category}")
                continue
            
            # Create temporary DataFrame for preprocessing
            df_cat = X_cat.copy()
            
            try:
                if category == 'acoustic_prosodic':
                    # Placeholder for Team Member A
                    self.logger.info("Acoustic/Prosodic preprocessing - PLACEHOLDER")
                    category_splits[category] = {
                        'status': 'placeholder',
                        'message': 'Acoustic/Prosodic preprocessing not implemented yet (Team Member A)',
                        'features_count': df_cat.shape[1] - 1  # Exclude target
                    }
                    
                elif category == 'syntactic_semantic':
                    # Placeholder for Team Member B
                    self.logger.info("Syntactic/Semantic preprocessing - PLACEHOLDER")
                    category_splits[category] = {
                        'status': 'placeholder',
                        'message': 'Syntactic/Semantic preprocessing not implemented yet (Team Member B)',
                        'features_count': df_cat.shape[1] - 1  # Exclude target
                    }
                    
                elif category == 'pragmatic_conversational':
                    # Fully implemented
                    self.logger.info("Pragmatic/Conversational preprocessing - IMPLEMENTED")
                    
                    # Use the main preprocessing pipeline
                    X_train, X_test, y_train, y_test = self.fit_transform(df_cat, validate=True)
                    
                    category_splits[category] = {
                        'status': 'implemented',
                        'data': (X_train, X_test, y_train, y_test),
                        'features_count': len(self.selected_features_) if self.selected_features_ else X_train.shape[1],
                        'message': 'Successfully preprocessed pragmatic/conversational features'
                    }
                    
                else:
                    # Unknown category
                    self.logger.warning(f"Unknown category: {category}")
                    category_splits[category] = {
                        'status': 'unknown',
                        'message': f'Unknown category: {category}',
                        'features_count': df_cat.shape[1] - 1 if target_column in df_cat.columns else df_cat.shape[1]
                    }
                    
            except Exception as e:
                self.logger.error(f"Error preprocessing {category}: {e}")
                category_splits[category] = {
                    'status': 'error',
                    'message': f'Preprocessing failed: {str(e)}',
                    'features_count': df_cat.shape[1] - 1 if target_column in df_cat.columns else df_cat.shape[1]
                }
        
        self.logger.info(f"Category preprocessing complete - {len(category_splits)} categories processed")
        
        return category_splits
    
    def save(self, save_path: Union[str, Path]):
        """
        Save fitted preprocessor to disk.
        
        Args:
            save_path: Path to save preprocessor
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessor state
        state = {
            'feature_columns': self.feature_columns_,
            'selected_features': self.selected_features_,
            'scaler': self.scaler,
            'cleaner': self.cleaner,
            'target_column': self.target_column,
        }
        
        joblib.dump(state, save_path)
        self.logger.info(f"Preprocessor saved to {save_path}")
    
    @classmethod
    def load(cls, load_path: Union[str, Path]) -> 'DataPreprocessor':
        """
        Load fitted preprocessor from disk.
        
        Args:
            load_path: Path to load preprocessor from
        
        Returns:
            DataPreprocessor: Loaded preprocessor
        """
        load_path = Path(load_path)
        state = joblib.load(load_path)
        
        # Create new instance
        preprocessor = cls()
        preprocessor.feature_columns_ = state['feature_columns']
        preprocessor.selected_features_ = state['selected_features']
        preprocessor.scaler = state['scaler']
        preprocessor.cleaner = state['cleaner']
        preprocessor.target_column = state['target_column']
        
        logger.info(f"Preprocessor loaded from {load_path}")
        
        return preprocessor
    
    def get_preprocessing_summary(self) -> Dict:
        """
        Get summary of preprocessing pipeline.
        
        Returns:
            Dict: Summary information
        """
        summary = {
            'total_features': len(self.feature_columns_) if self.feature_columns_ else 0,
            'selected_features': len(self.selected_features_) if self.selected_features_ else 0,
            'feature_selection_enabled': self.selector is not None,
            'scaling_method': self.scaler.method,
            'target_column': self.target_column,
        }
        
        if self.validation_report_ is not None:
            summary['validation'] = {
                'is_valid': self.validation_report_.is_valid,
                'total_samples': self.validation_report_.total_samples,
                'warnings': len(self.validation_report_.warnings),
                'errors': len(self.validation_report_.errors),
            }
        
        if self.selected_features_ is not None:
            summary['selected_feature_names'] = self.selected_features_
        
        return summary
    
    def print_summary(self):
        """Print preprocessing pipeline summary."""
        summary = self.get_preprocessing_summary()
        
        print("\n" + "="*70)
        print("PREPROCESSING PIPELINE SUMMARY")
        print("="*70)
        
        print(f"\nFeatures:")
        print(f"  Total features: {summary['total_features']}")
        print(f"  Selected features: {summary['selected_features']}")
        print(f"  Feature selection: {'Enabled' if summary['feature_selection_enabled'] else 'Disabled'}")
        
        print(f"\nScaling:")
        print(f"  Method: {summary['scaling_method']}")
        
        print(f"\nTarget:")
        print(f"  Column: {summary['target_column']}")
        
        if 'validation' in summary:
            print(f"\nValidation:")
            print(f"  Status: {'✓ Valid' if summary['validation']['is_valid'] else '✗ Invalid'}")
            print(f"  Samples: {summary['validation']['total_samples']}")
            print(f"  Warnings: {summary['validation']['warnings']}")
            print(f"  Errors: {summary['validation']['errors']}")
        
        print("\n" + "="*70 + "\n")

