"""
Data Cleaning Module

This module provides data cleaning functionality for pragmatic and conversational
feature datasets, handling missing values, outliers, and data normalization.

Key functionalities:
- Missing value imputation
- Outlier handling
- Feature scaling and normalization
- Data transformation

Author: Bimidu Gunathilake
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """
    Data cleaning and preprocessing class for pragmatic/conversational features.
    
    Handles missing values, outliers, and prepares data for ML training.
    """
    
    def __init__(
        self,
        missing_strategy: Literal['mean', 'median', 'knn', 'drop'] = 'median',
        outlier_method: Literal['clip', 'remove', 'winsorize', 'none'] = 'clip',
        outlier_std_threshold: float = 3.0,
    ):
        """
        Initialize data cleaner.
        
        Args:
            missing_strategy: Strategy for handling missing values
                - 'mean': Replace with mean
                - 'median': Replace with median
                - 'knn': KNN imputation
                - 'drop': Drop rows with missing values
            outlier_method: Method for handling outliers
                - 'clip': Clip to threshold
                - 'remove': Remove outlier rows
                - 'winsorize': Winsorize outliers
                - 'none': Keep outliers
            outlier_std_threshold: Standard deviations for outlier detection
        """
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.outlier_std_threshold = outlier_std_threshold
        self.logger = logger
        
        # Initialize imputer
        if missing_strategy == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        elif missing_strategy == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif missing_strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            self.imputer = None
        
        self.logger.info(
            f"DataCleaner initialized - missing_strategy={missing_strategy}, "
            f"outlier_method={outlier_method}"
        )
    
    def clean(
        self,
        df: pd.DataFrame,
        target_column: str = 'diagnosis',
        feature_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Clean the dataset.
        
        Args:
            df: Input DataFrame
            target_column: Name of target variable
            feature_columns: List of feature columns (None = auto-detect)
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        self.logger.info(f"Starting data cleaning for dataset with shape {df.shape}")
        
        df_clean = df.copy()
        
        # Auto-detect feature columns
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean, feature_columns, target_column)
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean, feature_columns)
        
        # Remove any remaining NaN rows (only in feature columns)
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=feature_columns)
        removed_rows = initial_rows - len(df_clean)
        
        if removed_rows > 0:
            self.logger.warning(f"Removed {removed_rows} rows with NaN in feature columns")
        
        self.logger.info(f"Cleaning complete - Final shape: {df_clean.shape}")
        
        return df_clean
    
    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Handle missing values in features."""
        df_clean = df.copy()

        # Filter to only numeric feature columns for imputation
        numeric_feature_columns = [
            col for col in feature_columns
            if col in df_clean.select_dtypes(include=[np.number]).columns
        ]

        # Check for missing values in numeric columns
        missing_counts = df_clean[numeric_feature_columns].isna().sum()
        features_with_missing = missing_counts[missing_counts > 0]

        if len(features_with_missing) == 0:
            self.logger.info("No missing values found")
            return df_clean

        self.logger.info(
            f"Found missing values in {len(features_with_missing)} features"
        )

        # Handle based on strategy
        if self.missing_strategy == 'drop':
            # Drop rows with any missing values
            df_clean = df_clean.dropna(subset=numeric_feature_columns)
            self.logger.info(f"Dropped rows with missing values - New shape: {df_clean.shape}")

        elif self.imputer is not None:
            # Impute missing values (only on numeric columns)
            df_clean[numeric_feature_columns] = self.imputer.fit_transform(
                df_clean[numeric_feature_columns]
            )
            self.logger.info(f"Imputed missing values using {self.missing_strategy}")

        return df_clean
    
    def _handle_outliers(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """Handle outliers in features."""
        if self.outlier_method == 'none':
            return df
        
        df_clean = df.copy()
        outliers_found = 0
        
        for col in feature_columns:
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                continue
            
            # Calculate z-scores
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            
            if std == 0:
                continue
            
            z_scores = np.abs((df_clean[col] - mean) / std)
            outlier_mask = z_scores > self.outlier_std_threshold
            
            if outlier_mask.sum() == 0:
                continue
            
            outliers_found += outlier_mask.sum()
            
            if self.outlier_method == 'clip':
                # Clip to threshold
                lower_bound = mean - (self.outlier_std_threshold * std)
                upper_bound = mean + (self.outlier_std_threshold * std)
                
                # Skip if bounds are invalid
                if np.isnan(lower_bound) or np.isnan(upper_bound):
                    continue
                if np.isinf(lower_bound) or np.isinf(upper_bound):
                    continue
                    
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
            
            elif self.outlier_method == 'winsorize':
                # Winsorize (replace with percentile values)
                lower_percentile = df_clean[col].quantile(0.01)
                upper_percentile = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(lower_percentile, upper_percentile)
            
            elif self.outlier_method == 'remove':
                # Mark for removal
                df_clean = df_clean[~outlier_mask]
        
        if outliers_found > 0:
            self.logger.info(
                f"Handled {outliers_found} outliers using method: {self.outlier_method}"
            )
        
        return df_clean
    
    def remove_low_variance_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        variance_threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with variance below threshold.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns
            variance_threshold: Minimum variance threshold
        
        Returns:
            Tuple of (cleaned DataFrame, list of removed features)
        """
        df_clean = df.copy()
        removed_features = []
        
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                variance = df_clean[col].var()
                if variance < variance_threshold:
                    removed_features.append(col)
                    df_clean = df_clean.drop(columns=[col])
        
        if removed_features:
            self.logger.info(f"Removed {len(removed_features)} low-variance features")
        
        return df_clean, removed_features
    
    def remove_highly_correlated_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        correlation_threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
        """
        Remove one feature from each highly correlated pair.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns
            correlation_threshold: Correlation threshold for removal
        
        Returns:
            Tuple of (cleaned DataFrame, list of removed pairs)
        """
        df_clean = df.copy()
        removed_pairs = []
        features_to_remove = set()
        
        # Calculate correlation matrix
        numeric_features = [
            col for col in feature_columns
            if pd.api.types.is_numeric_dtype(df_clean[col]) and col in df_clean.columns
        ]
        
        if len(numeric_features) < 2:
            return df_clean, removed_pairs
        
        corr_matrix = df_clean[numeric_features].corr().abs()
        
        # Find highly correlated pairs
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    feat1 = corr_matrix.columns[i]
                    feat2 = corr_matrix.columns[j]
                    
                    # Remove the second feature
                    if feat2 not in features_to_remove:
                        features_to_remove.add(feat2)
                        removed_pairs.append((feat1, feat2))
        
        if features_to_remove:
            df_clean = df_clean.drop(columns=list(features_to_remove))
            self.logger.info(
                f"Removed {len(features_to_remove)} highly correlated features"
            )
        
        return df_clean, removed_pairs


class FeatureScaler:
    """
    Feature scaling class for normalization.
    
    Supports multiple scaling methods for pragmatic/conversational features.
    """
    
    def __init__(
        self,
        method: Literal['standard', 'minmax', 'robust'] = 'standard'
    ):
        """
        Initialize feature scaler.
        
        Args:
            method: Scaling method
                - 'standard': StandardScaler (z-score normalization)
                - 'minmax': MinMaxScaler (0-1 normalization)
                - 'robust': RobustScaler (median and IQR)
        """
        self.method = method
        self.logger = logger
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        self.logger.info(f"FeatureScaler initialized with method: {method}")
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Fit scaler and transform features.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns to scale
        
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        # Only scale numeric features that exist
        numeric_features = [
            col for col in feature_columns
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        if not numeric_features:
            self.logger.warning("No numeric features to scale")
            return df_scaled
        
        # Fit and transform
        df_scaled[numeric_features] = self.scaler.fit_transform(
            df_scaled[numeric_features]
        )
        
        self.logger.info(f"Scaled {len(numeric_features)} features using {self.method}")
        
        return df_scaled
    
    def transform(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Transform features using fitted scaler.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns to scale
        
        Returns:
            pd.DataFrame: DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        numeric_features = [
            col for col in feature_columns
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        if not numeric_features:
            return df_scaled
        
        df_scaled[numeric_features] = self.scaler.transform(
            df_scaled[numeric_features]
        )
        
        return df_scaled

