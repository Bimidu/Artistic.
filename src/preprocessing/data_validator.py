"""
Data Validation Module

This module provides comprehensive data validation and quality checks
for feature datasets before machine learning model training.

Key functionalities:
- Missing value detection
- Data type validation
- Range checks
- Outlier detection
- Class balance checks
- Feature correlation analysis

Author: Bimidu Gunathilake
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """
    Data validation report containing all quality checks.
    
    Attributes:
        is_valid: Overall validation status
        total_samples: Total number of samples
        total_features: Total number of features
        missing_values: Dict of features with missing values
        invalid_types: Dict of features with invalid types
        outliers: Dict of features with outlier counts
        class_distribution: Distribution of target classes
        high_correlation_pairs: Pairs of highly correlated features
        warnings: List of validation warnings
        errors: List of validation errors
    """
    is_valid: bool
    total_samples: int
    total_features: int
    missing_values: Dict[str, int] = field(default_factory=dict)
    invalid_types: Dict[str, str] = field(default_factory=dict)
    outliers: Dict[str, int] = field(default_factory=dict)
    class_distribution: Dict[str, int] = field(default_factory=dict)
    high_correlation_pairs: List[Tuple[str, str, float]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'is_valid': self.is_valid,
            'total_samples': self.total_samples,
            'total_features': self.total_features,
            'missing_values': self.missing_values,
            'invalid_types': self.invalid_types,
            'outliers': self.outliers,
            'class_distribution': self.class_distribution,
            'high_correlation_pairs': [
                {'feature1': f1, 'feature2': f2, 'correlation': corr}
                for f1, f2, corr in self.high_correlation_pairs
            ],
            'warnings': self.warnings,
            'errors': self.errors,
        }
    
    def print_summary(self):
        """Print a formatted summary of the validation report."""
        print("\n" + "="*70)
        print("DATA VALIDATION REPORT")
        print("="*70)
        
        status = "[OK] VALID" if self.is_valid else "[X] INVALID"
        print(f"\nStatus: {status}")
        print(f"Total Samples: {self.total_samples}")
        print(f"Total Features: {self.total_features}")
        
        if self.class_distribution:
            print(f"\nClass Distribution:")
            for cls, count in self.class_distribution.items():
                percentage = (count / self.total_samples) * 100
                print(f"  {cls}: {count} ({percentage:.1f}%)")
        
        if self.missing_values:
            print(f"\nMissing Values:")
            for feature, count in list(self.missing_values.items())[:10]:
                print(f"  {feature}: {count}")
            if len(self.missing_values) > 10:
                print(f"  ... and {len(self.missing_values) - 10} more")
        
        if self.outliers:
            print(f"\nOutliers Detected:")
            for feature, count in list(self.outliers.items())[:10]:
                print(f"  {feature}: {count}")
            if len(self.outliers) > 10:
                print(f"  ... and {len(self.outliers) - 10} more")
        
        if self.high_correlation_pairs:
            print(f"\nHigh Correlation Pairs (>{0.9}):")
            for f1, f2, corr in self.high_correlation_pairs[:5]:
                print(f"  {f1} <-> {f2}: {corr:.3f}")
            if len(self.high_correlation_pairs) > 5:
                print(f"  ... and {len(self.high_correlation_pairs) - 5} more")
        
        if self.warnings:
            print(f"\n⚠ Warnings ({len(self.warnings)}):")
            for warning in self.warnings[:5]:
                print(f"  - {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")
        
        if self.errors:
            print(f"\n✗ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        print("\n" + "="*70 + "\n")


class DataValidator:
    """
    Comprehensive data validation class for ML pipeline.
    
    Validates data quality, detects issues, and generates reports
    for feature datasets before model training.
    """
    
    def __init__(
        self,
        min_samples: int = 10,
        max_missing_ratio: float = 0.3,
        correlation_threshold: float = 0.95,
        outlier_std_threshold: float = 3.0,
    ):
        """
        Initialize data validator.
        
        Args:
            min_samples: Minimum number of samples required
            max_missing_ratio: Maximum allowed ratio of missing values per feature
            correlation_threshold: Threshold for detecting high correlation
            outlier_std_threshold: Standard deviations for outlier detection
        """
        self.min_samples = min_samples
        self.max_missing_ratio = max_missing_ratio
        self.correlation_threshold = correlation_threshold
        self.outlier_std_threshold = outlier_std_threshold
        self.logger = logger
        
        self.logger.info(
            f"DataValidator initialized - min_samples={min_samples}, "
            f"max_missing={max_missing_ratio}, corr_threshold={correlation_threshold}"
        )
    
    def validate(
        self,
        df: pd.DataFrame,
        target_column: str = 'diagnosis',
        feature_columns: Optional[List[str]] = None,
    ) -> ValidationReport:
        """
        Perform comprehensive validation on dataset.
        
        Args:
            df: Input DataFrame
            target_column: Name of target variable column
            feature_columns: List of feature columns (None = auto-detect)
        
        Returns:
            ValidationReport: Comprehensive validation report
        """
        self.logger.info(f"Starting validation for dataset with shape {df.shape}")
        
        # Initialize report
        report = ValidationReport(
            is_valid=True,
            total_samples=len(df),
            total_features=len(df.columns) - 1 if target_column in df.columns else len(df.columns)
        )
        
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        
        # Run validation checks
        self._check_sample_size(df, report)
        self._check_target_column(df, target_column, report)
        self._check_missing_values(df, feature_columns, report)
        self._check_data_types(df, feature_columns, report)
        self._check_outliers(df, feature_columns, report)
        self._check_class_balance(df, target_column, report)
        self._check_feature_correlation(df, feature_columns, report)
        self._check_feature_variance(df, feature_columns, report)
        
        # Determine overall validity
        report.is_valid = len(report.errors) == 0
        
        self.logger.info(
            f"Validation complete - Valid: {report.is_valid}, "
            f"Warnings: {len(report.warnings)}, Errors: {len(report.errors)}"
        )
        
        return report
    
    def _check_sample_size(self, df: pd.DataFrame, report: ValidationReport):
        """Check if dataset has minimum required samples."""
        if len(df) < self.min_samples:
            report.errors.append(
                f"Insufficient samples: {len(df)} < {self.min_samples} required"
            )
            self.logger.error(f"Insufficient samples: {len(df)}")
    
    def _check_target_column(
        self,
        df: pd.DataFrame,
        target_column: str,
        report: ValidationReport
    ):
        """Check if target column exists and is valid."""
        if target_column not in df.columns:
            report.errors.append(f"Target column '{target_column}' not found")
            self.logger.error(f"Target column '{target_column}' not found")
            return
        
        # Check for missing values in target
        missing_target = df[target_column].isna().sum()
        if missing_target > 0:
            report.errors.append(
                f"Target column has {missing_target} missing values"
            )
            self.logger.error(f"Target has {missing_target} missing values")
        
        # Get class distribution
        report.class_distribution = df[target_column].value_counts().to_dict()
    
    def _check_missing_values(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        report: ValidationReport
    ):
        """Check for missing values in features."""
        for col in feature_columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                missing_ratio = missing_count / len(df)
                report.missing_values[col] = missing_count
                
                if missing_ratio > self.max_missing_ratio:
                    report.warnings.append(
                        f"Feature '{col}' has {missing_ratio:.1%} missing values"
                    )
        
        if report.missing_values:
            self.logger.warning(
                f"Found {len(report.missing_values)} features with missing values"
            )
    
    def _check_data_types(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        report: ValidationReport
    ):
        """Check data types of features."""
        for col in feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                report.invalid_types[col] = str(df[col].dtype)
                report.warnings.append(
                    f"Feature '{col}' has non-numeric type: {df[col].dtype}"
                )
        
        if report.invalid_types:
            self.logger.warning(
                f"Found {len(report.invalid_types)} features with non-numeric types"
            )
    
    def _check_outliers(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        report: ValidationReport
    ):
        """Detect outliers using standard deviation method."""
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                mean = df[col].mean()
                std = df[col].std()
                
                if std > 0:
                    z_scores = np.abs((df[col] - mean) / std)
                    outlier_count = (z_scores > self.outlier_std_threshold).sum()
                    
                    if outlier_count > 0:
                        report.outliers[col] = outlier_count
        
        if report.outliers:
            total_outliers = sum(report.outliers.values())
            self.logger.info(f"Detected {total_outliers} outliers across {len(report.outliers)} features")
    
    def _check_class_balance(
        self,
        df: pd.DataFrame,
        target_column: str,
        report: ValidationReport
    ):
        """Check for class imbalance in target variable."""
        if target_column not in df.columns:
            return
        
        class_counts = df[target_column].value_counts()
        if len(class_counts) < 2:
            report.errors.append("Dataset must have at least 2 classes")
            return
        
        # Check imbalance ratio
        min_count = class_counts.min()
        max_count = class_counts.max()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 3:
            report.warnings.append(
                f"Significant class imbalance detected (ratio: {imbalance_ratio:.2f})"
            )
            self.logger.warning(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    
    def _check_feature_correlation(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        report: ValidationReport
    ):
        """Check for highly correlated features."""
        numeric_features = [
            col for col in feature_columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        
        if len(numeric_features) < 2:
            return
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_features].corr().abs()
        
        # Find high correlations (excluding diagonal)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if corr_value > self.correlation_threshold:
                    feat1 = corr_matrix.columns[i]
                    feat2 = corr_matrix.columns[j]
                    report.high_correlation_pairs.append((feat1, feat2, corr_value))
        
        if report.high_correlation_pairs:
            report.warnings.append(
                f"Found {len(report.high_correlation_pairs)} highly correlated feature pairs"
            )
            self.logger.warning(
                f"Found {len(report.high_correlation_pairs)} highly correlated pairs"
            )
    
    def _check_feature_variance(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        report: ValidationReport
    ):
        """Check for zero or near-zero variance features."""
        low_variance_features = []
        
        for col in feature_columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                variance = df[col].var()
                if variance < 1e-10:  # Near-zero variance
                    low_variance_features.append(col)
        
        if low_variance_features:
            report.warnings.append(
                f"Found {len(low_variance_features)} features with near-zero variance"
            )
            self.logger.warning(
                f"Found {len(low_variance_features)} low-variance features"
            )
    
    def validate_file(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> ValidationReport:
        """
        Validate data from a CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments passed to validate()
        
        Returns:
            ValidationReport: Validation report
        """
        file_path = Path(file_path)
        self.logger.info(f"Loading data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            return self.validate(df, **kwargs)
        except Exception as e:
            self.logger.error(f"Error loading file: {e}")
            raise

