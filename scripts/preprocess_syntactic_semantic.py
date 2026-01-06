"""
Syntactic Semantic Feature Preprocessing Script

This script applies comprehensive data cleaning and preprocessing techniques
specifically designed for syntactic and semantic linguistic features.

Preprocessing steps:
1. Diagnosis validation and standardization
2. Missing value handling
3. Zero-heavy row removal
4. Outlier detection and handling
5. Feature normalization
6. Duplicate removal
7. Age-based filtering

Author: Randil Haturusinghe
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SyntacticSemanticPreprocessor:
    """
    Preprocessor for syntactic and semantic linguistic features.
    
    Handles data quality issues specific to NLP features extracted from
    conversational transcripts.
    """
    
    # Valid diagnosis codes
    VALID_DIAGNOSES = ['ASD', 'TD', 'TYP', 'DD']
    
    # Diagnosis normalization mapping
    DIAGNOSIS_MAPPING = {
        'TYP': 'TD',  # Normalize TYP to TD
        'TYPICAL': 'TD',
        'TYPICALLY DEVELOPING': 'TD',
    }
    
    # Age range for valid transcripts (in months)
    MIN_AGE_MONTHS = 24  # 2 years
    MAX_AGE_MONTHS = 120  # 10 years
    
    # Feature quality thresholds
    MIN_NONZERO_FEATURES = 10  # Minimum non-zero features per row
    MAX_ZERO_RATIO = 0.7  # Maximum ratio of zero features allowed
    OUTLIER_STD_THRESHOLD = 4  # Standard deviations for outlier detection
    
    def __init__(self, input_path: str = 'output/syntactic_semantic_features.csv'):
        """
        Initialize preprocessor.
        
        Args:
            input_path: Path to raw feature CSV file
        """
        self.input_path = Path(input_path)
        self.df = None
        self.feature_cols = None
        self.metadata_cols = ['participant_id', 'file_path', 'diagnosis', 'age_months']
        
        logger.info(f"Initialized preprocessor for {input_path}")
    
    def load_data(self) -> pd.DataFrame:
        """Load raw feature data."""
        logger.info(f"Loading data from {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        
        # Identify feature columns (everything after metadata)
        self.feature_cols = [col for col in self.df.columns if col not in self.metadata_cols + ['dataset']]
        
        logger.info(f"Loaded {len(self.df)} rows, {len(self.feature_cols)} features")
        return self.df
    
    def clean_diagnosis(self) -> pd.DataFrame:
        """
        Clean and standardize diagnosis labels.
        
        Steps:
        1. Remove rows with missing diagnosis
        2. Normalize diagnosis labels (TYP -> TD)
        3. Remove invalid diagnosis codes (e.g., age values)
        4. Filter to valid diagnosis groups
        """
        logger.info("Step 1: Cleaning diagnosis labels")
        
        initial_count = len(self.df)
        
        # Remove missing diagnoses
        self.df = self.df.dropna(subset=['diagnosis'])
        logger.info(f"  Removed {initial_count - len(self.df)} rows with missing diagnosis")
        
        # Normalize diagnosis labels
        self.df['diagnosis'] = self.df['diagnosis'].str.upper().str.strip()
        self.df['diagnosis'] = self.df['diagnosis'].replace(self.DIAGNOSIS_MAPPING)
        
        # Remove invalid diagnosis codes (e.g., "7;00.", "5;10." which are age values)
        invalid_mask = ~self.df['diagnosis'].isin(self.VALID_DIAGNOSES)
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logger.info(f"  Removing {invalid_count} rows with invalid diagnosis codes")
            logger.info(f"    Invalid codes: {self.df[invalid_mask]['diagnosis'].unique()}")
            self.df = self.df[~invalid_mask]
        
        # Log final distribution
        logger.info(f"  Final diagnosis distribution:")
        for diag, count in self.df['diagnosis'].value_counts().items():
            logger.info(f"    {diag}: {count}")
        
        return self.df
    
    def handle_missing_age(self) -> pd.DataFrame:
        """
        Handle missing age values.
        
        Strategy: Remove rows with missing age since age is important for
        developmental analysis.
        """
        logger.info("Step 2: Handling missing age values")
        
        initial_count = len(self.df)
        missing_age = self.df['age_months'].isnull().sum()
        
        if missing_age > 0:
            logger.info(f"  Found {missing_age} rows with missing age")
            self.df = self.df.dropna(subset=['age_months'])
            logger.info(f"  Removed {initial_count - len(self.df)} rows")
        
        # Filter by age range
        age_filtered = self.df[
            (self.df['age_months'] >= self.MIN_AGE_MONTHS) &
            (self.df['age_months'] <= self.MAX_AGE_MONTHS)
        ]
        
        removed = len(self.df) - len(age_filtered)
        if removed > 0:
            logger.info(f"  Removed {removed} rows outside age range [{self.MIN_AGE_MONTHS}, {self.MAX_AGE_MONTHS}] months")
            self.df = age_filtered
        
        logger.info(f"  Age range: {self.df['age_months'].min():.0f} - {self.df['age_months'].max():.0f} months")
        logger.info(f"  Mean age: {self.df['age_months'].mean():.1f} ± {self.df['age_months'].std():.1f} months")
        
        return self.df
    
    def remove_zero_heavy_rows(self) -> pd.DataFrame:
        """
        Remove rows with too many zero features.
        
        Rows with mostly zeros indicate:
        - Very sparse transcripts (few utterances)
        - Parsing failures
        - Invalid data
        """
        logger.info("Step 3: Removing zero-heavy rows")
        
        initial_count = len(self.df)
        
        # Count non-zero features per row
        feature_data = self.df[self.feature_cols]
        nonzero_counts = (feature_data != 0).sum(axis=1)
        zero_ratio = (feature_data == 0).sum(axis=1) / len(self.feature_cols)
        
        # Filter by minimum non-zero features
        valid_mask = (nonzero_counts >= self.MIN_NONZERO_FEATURES) & (zero_ratio <= self.MAX_ZERO_RATIO)
        
        removed = (~valid_mask).sum()
        if removed > 0:
            logger.info(f"  Removing {removed} rows with < {self.MIN_NONZERO_FEATURES} non-zero features")
            logger.info(f"    or > {self.MAX_ZERO_RATIO*100:.0f}% zero features")
            self.df = self.df[valid_mask]
        
        logger.info(f"  Retained {len(self.df)} rows")
        
        return self.df
    
    def handle_outliers(self, method='clip') -> pd.DataFrame:
        """
        Handle outliers in feature values.
        
        Args:
            method: 'clip' (winsorize), 'remove', or 'keep'
        
        Outliers can be:
        - Legitimate extreme values (very complex/simple speech)
        - Parsing errors
        - Data quality issues
        
        Default: Clip to 4 standard deviations (conservative)
        """
        logger.info(f"Step 4: Handling outliers (method={method})")
        
        if method == 'keep':
            logger.info("  Keeping all outliers")
            return self.df
        
        feature_data = self.df[self.feature_cols].copy()
        outlier_counts = {}
        
        for col in self.feature_cols:
            if feature_data[col].std() == 0:
                continue
            
            mean = feature_data[col].mean()
            std = feature_data[col].std()
            
            # Identify outliers
            lower_bound = mean - self.OUTLIER_STD_THRESHOLD * std
            upper_bound = mean + self.OUTLIER_STD_THRESHOLD * std
            
            outliers = (feature_data[col] < lower_bound) | (feature_data[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_counts[col] = outlier_count
                
                if method == 'clip':
                    # Winsorize: clip to bounds
                    feature_data.loc[feature_data[col] < lower_bound, col] = lower_bound
                    feature_data.loc[feature_data[col] > upper_bound, col] = upper_bound
                elif method == 'remove':
                    # Mark for removal
                    self.df = self.df[~outliers]
        
        if method == 'clip':
            self.df[self.feature_cols] = feature_data
            logger.info(f"  Clipped outliers in {len(outlier_counts)} features")
            if outlier_counts:
                top_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                for feat, count in top_outliers:
                    logger.info(f"    {feat}: {count} outliers clipped")
        
        return self.df
    
    def remove_duplicate_features(self) -> pd.DataFrame:
        """
        Remove duplicate feature columns.
        
        Some features appear multiple times in the output (e.g., lexical_diversity_semantic).
        Keep only the first occurrence.
        """
        logger.info("Step 5: Removing duplicate feature columns")
        
        initial_cols = len(self.df.columns)
        
        # Find duplicate column names
        duplicates = self.df.columns[self.df.columns.duplicated()].unique()
        
        if len(duplicates) > 0:
            logger.info(f"  Found {len(duplicates)} duplicate column names:")
            for dup in duplicates:
                logger.info(f"    {dup}")
            
            # Keep first occurrence of each column
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]
            logger.info(f"  Removed {initial_cols - len(self.df.columns)} duplicate columns")
            
            # Update feature_cols
            self.feature_cols = [col for col in self.df.columns if col not in self.metadata_cols + ['dataset']]
        else:
            logger.info("  No duplicate columns found")
        
        return self.df
    
    def handle_missing_features(self) -> pd.DataFrame:
        """
        Handle missing values in feature columns.
        
        Strategy: Fill with 0 (missing features indicate absence of that linguistic pattern)
        """
        logger.info("Step 6: Handling missing feature values")
        
        missing_counts = self.df[self.feature_cols].isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            logger.info(f"  Found {total_missing} missing feature values")
            features_with_missing = missing_counts[missing_counts > 0]
            for feat, count in features_with_missing.items():
                logger.info(f"    {feat}: {count} missing")
            
            # Fill with 0
            self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)
            logger.info("  Filled missing values with 0")
        else:
            logger.info("  No missing feature values")
        
        return self.df
    
    def normalize_diagnosis_groups(self) -> pd.DataFrame:
        """
        Ensure balanced representation or handle class imbalance.
        
        For now, just log the distribution. Advanced techniques like
        SMOTE can be applied during model training.
        """
        logger.info("Step 7: Checking diagnosis group balance")
        
        counts = self.df['diagnosis'].value_counts()
        total = len(self.df)
        
        logger.info("  Diagnosis distribution:")
        for diag, count in counts.items():
            pct = count / total * 100
            logger.info(f"    {diag}: {count} ({pct:.1f}%)")
        
        # Check for severe imbalance
        min_count = counts.min()
        max_count = counts.max()
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 3:
            logger.warning(f"  ⚠️  Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            logger.warning("     Consider using stratified sampling or SMOTE during training")
        
        return self.df
    
    def generate_summary_statistics(self) -> dict:
        """Generate summary statistics for the cleaned dataset."""
        logger.info("Generating summary statistics")
        
        stats = {
            'total_samples': len(self.df),
            'total_features': len(self.feature_cols),
            'diagnosis_distribution': self.df['diagnosis'].value_counts().to_dict(),
            'age_range': (self.df['age_months'].min(), self.df['age_months'].max()),
            'age_mean': self.df['age_months'].mean(),
            'age_std': self.df['age_months'].std(),
            'nonzero_feature_ratio': (self.df[self.feature_cols] != 0).sum().sum() / (len(self.df) * len(self.feature_cols)),
        }
        
        return stats
    
    def save_cleaned_data(self, output_path: str = 'output/syntactic_semantic_cleaned.csv') -> None:
        """
        Save cleaned dataset to CSV.
        
        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")
        logger.info(f"  Shape: {self.df.shape}")
    
    def run_full_pipeline(
        self,
        output_path: str = 'output/syntactic_semantic_cleaned.csv',
        outlier_method: str = 'clip'
    ) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            output_path: Path to save cleaned data
            outlier_method: How to handle outliers ('clip', 'remove', 'keep')
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("="*80)
        logger.info("SYNTACTIC SEMANTIC FEATURE PREPROCESSING PIPELINE")
        logger.info("="*80)
        
        # Load data
        self.load_data()
        initial_count = len(self.df)
        
        # Run preprocessing steps
        self.clean_diagnosis()
        self.handle_missing_age()
        self.remove_zero_heavy_rows()
        self.handle_outliers(method=outlier_method)
        self.remove_duplicate_features()
        self.handle_missing_features()
        self.normalize_diagnosis_groups()
        
        # Generate summary
        stats = self.generate_summary_statistics()
        
        # Save cleaned data
        self.save_cleaned_data(output_path)
        
        # Final summary
        logger.info("="*80)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("="*80)
        logger.info(f"Initial samples: {initial_count}")
        logger.info(f"Final samples: {stats['total_samples']}")
        logger.info(f"Removed: {initial_count - stats['total_samples']} ({(initial_count - stats['total_samples'])/initial_count*100:.1f}%)")
        logger.info(f"Features: {stats['total_features']}")
        logger.info(f"Non-zero feature ratio: {stats['nonzero_feature_ratio']:.2%}")
        logger.info("="*80)
        
        return self.df


def main():
    """Main execution function."""
    # Initialize preprocessor
    preprocessor = SyntacticSemanticPreprocessor(
        input_path='output/syntactic_semantic_features.csv'
    )
    
    # Run full pipeline
    cleaned_df = preprocessor.run_full_pipeline(
        output_path='output/syntactic_semantic_cleaned.csv',
        outlier_method='clip'  # Options: 'clip', 'remove', 'keep'
    )
    
    # Display sample
    print("\n" + "="*80)
    print("SAMPLE OF CLEANED DATA")
    print("="*80)
    print(cleaned_df.head())
    
    print("\n" + "="*80)
    print("FEATURE STATISTICS")
    print("="*80)
    feature_cols = [col for col in cleaned_df.columns 
                   if col not in ['participant_id', 'file_path', 'diagnosis', 'age_months', 'dataset']]
    print(cleaned_df[feature_cols].describe().T[['mean', 'std', 'min', 'max']])


if __name__ == "__main__":
    main()
