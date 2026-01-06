"""
Feature Selection for Syntactic Semantic Features

This script implements multiple feature selection methodologies specifically
designed for linguistic features in ASD detection.

Methods implemented:
1. Variance Threshold (remove low-variance features)
2. Correlation Analysis (remove redundant features)
3. Univariate Selection (ANOVA F-test, Mutual Information)
4. Recursive Feature Elimination (RFE)
5. L1-based Selection (Lasso)
6. Tree-based Importance (Random Forest)
7. Sequential Feature Selection
8. Ensemble Selection (voting across methods)

Author: Randil Haturusinghe
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE,
    SelectFromModel,
    SequentialFeatureSelector
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SyntacticSemanticFeatureSelector:
    """
    Comprehensive feature selection for syntactic and semantic features.
    
    Uses multiple selection methods and ensemble voting to identify
    the most discriminative features for ASD classification.
    """
    
    def __init__(self, input_path: str = 'output/syntactic_semantic_cleaned.csv'):
        """
        Initialize feature selector.
        
        Args:
            input_path: Path to cleaned feature CSV
        """
        self.input_path = Path(input_path)
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Store selected features from each method
        self.selected_features = {}
        
        logger.info(f"Initialized feature selector for {input_path}")
    
    def load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load and prepare data for feature selection."""
        logger.info(f"Loading data from {self.input_path}")
        
        self.df = pd.read_csv(self.input_path)
        
        # Separate features and target
        metadata_cols = ['participant_id', 'file_path', 'diagnosis', 'age_months', 'dataset']
        self.feature_names = [col for col in self.df.columns if col not in metadata_cols]
        
        self.X = self.df[self.feature_names].values
        self.y = self.label_encoder.fit_transform(self.df['diagnosis'])
        
        logger.info(f"Loaded {len(self.df)} samples with {len(self.feature_names)} features")
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        
        return self.df, self.X, self.y
    
    def variance_threshold_selection(self, threshold: float = 0.01) -> List[str]:
        """
        Method 1: Remove low-variance features.
        
        Low variance indicates features that don't vary much across samples,
        providing little discriminative power.
        
        Args:
            threshold: Minimum variance threshold
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Method 1: Variance Threshold (threshold={threshold})")
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.X)
        
        selected_mask = selector.get_support()
        selected = [self.feature_names[i] for i, selected in enumerate(selected_mask) if selected]
        removed = [self.feature_names[i] for i, selected in enumerate(selected_mask) if not selected]
        
        logger.info(f"  Selected: {len(selected)}/{len(self.feature_names)} features")
        if removed:
            logger.info(f"  Removed low-variance: {', '.join(removed)}")
        
        self.selected_features['variance_threshold'] = selected
        return selected
    
    def correlation_based_selection(self, threshold: float = 0.85) -> List[str]:
        """
        Method 2: Remove highly correlated features.
        
        When two features are highly correlated, they provide redundant
        information. Keep the one with higher variance.
        
        Args:
            threshold: Correlation threshold for removal
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Method 2: Correlation-based Selection (threshold={threshold})")
        
        # Calculate correlation matrix
        corr_matrix = pd.DataFrame(self.X, columns=self.feature_names).corr().abs()
        
        # Find highly correlated pairs
        to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    # Remove feature with lower variance
                    var_i = np.var(self.X[:, i])
                    var_j = np.var(self.X[:, j])
                    
                    if var_i < var_j:
                        to_remove.add(corr_matrix.columns[i])
                    else:
                        to_remove.add(corr_matrix.columns[j])
        
        selected = [f for f in self.feature_names if f not in to_remove]
        
        logger.info(f"  Selected: {len(selected)}/{len(self.feature_names)} features")
        if to_remove:
            logger.info(f"  Removed correlated: {', '.join(sorted(to_remove))}")
        
        self.selected_features['correlation'] = selected
        return selected
    
    def univariate_selection(self, k: int = 15, method: str = 'f_classif') -> List[str]:
        """
        Method 3: Univariate statistical tests.
        
        Select features based on univariate statistical tests between
        each feature and the target.
        
        Args:
            k: Number of top features to select
            method: 'f_classif' (ANOVA F-test) or 'mutual_info' (Mutual Information)
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Method 3: Univariate Selection (k={k}, method={method})")
        
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown method: {method}")
        
        selector = SelectKBest(score_func=score_func, k=min(k, len(self.feature_names)))
        selector.fit(self.X, self.y)
        
        # Get feature scores
        scores = selector.scores_
        feature_scores = pd.DataFrame({
            'feature': self.feature_names,
            'score': scores
        }).sort_values('score', ascending=False)
        
        selected_mask = selector.get_support()
        selected = [self.feature_names[i] for i, sel in enumerate(selected_mask) if sel]
        
        logger.info(f"  Selected: {len(selected)} features")
        logger.info(f"  Top 5: {', '.join(feature_scores['feature'].head(5).tolist())}")
        
        self.selected_features[f'univariate_{method}'] = selected
        return selected
    
    def rfe_selection(self, n_features: int = 15) -> List[str]:
        """
        Method 4: Recursive Feature Elimination.
        
        Recursively removes features and builds a model on remaining features
        until the specified number is reached.
        
        Args:
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Method 4: Recursive Feature Elimination (n={n_features})")
        
        # Use Random Forest as the estimator
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        selector = RFE(
            estimator=estimator,
            n_features_to_select=min(n_features, len(self.feature_names)),
            step=1
        )
        
        # Scale features for better performance
        X_scaled = self.scaler.fit_transform(self.X)
        selector.fit(X_scaled, self.y)
        
        selected_mask = selector.get_support()
        selected = [self.feature_names[i] for i, sel in enumerate(selected_mask) if sel]
        
        logger.info(f"  Selected: {len(selected)} features")
        
        self.selected_features['rfe'] = selected
        return selected
    
    def lasso_selection(self, alpha: float = 0.01) -> List[str]:
        """
        Method 5: L1-based feature selection (Lasso).
        
        L1 regularization drives coefficients of less important features to zero.
        
        Args:
            alpha: Regularization strength
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Method 5: L1-based Selection (Lasso, alpha={alpha})")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Use saga solver which supports multiclass L1 regularization
        lasso = LogisticRegression(
            penalty='l1',
            solver='saga',  # saga supports multiclass + L1
            C=1/alpha,
            random_state=42,
            max_iter=1000
        )
        
        selector = SelectFromModel(lasso, prefit=False, threshold='median')
        selector.fit(X_scaled, self.y)
        
        selected_mask = selector.get_support()
        selected = [self.feature_names[i] for i, sel in enumerate(selected_mask) if sel]
        
        logger.info(f"  Selected: {len(selected)} features")
        
        self.selected_features['lasso'] = selected
        return selected
    
    def tree_based_selection(self, threshold: float = 0.01) -> List[str]:
        """
        Method 6: Tree-based feature importance.
        
        Random Forest provides feature importance scores based on
        how much each feature decreases impurity.
        
        Args:
            threshold: Minimum importance threshold
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Method 6: Tree-based Importance (threshold={threshold})")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10
        )
        
        rf.fit(self.X, self.y)
        
        # Get feature importances
        importances = rf.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        selected = feature_importance[feature_importance['importance'] >= threshold]['feature'].tolist()
        
        logger.info(f"  Selected: {len(selected)} features")
        logger.info(f"  Top 5: {', '.join(feature_importance['feature'].head(5).tolist())}")
        
        self.selected_features['tree_importance'] = selected
        return selected
    
    def sequential_selection(self, n_features: int = 15, direction: str = 'forward') -> List[str]:
        """
        Method 7: Sequential Feature Selection.
        
        Greedily adds (forward) or removes (backward) features based on
        cross-validated model performance.
        
        Args:
            n_features: Number of features to select
            direction: 'forward' or 'backward'
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Method 7: Sequential Selection ({direction}, n={n_features})")
        
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        
        selector = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=min(n_features, len(self.feature_names)),
            direction=direction,
            cv=3,
            n_jobs=-1
        )
        
        X_scaled = self.scaler.fit_transform(self.X)
        selector.fit(X_scaled, self.y)
        
        selected_mask = selector.get_support()
        selected = [self.feature_names[i] for i, sel in enumerate(selected_mask) if sel]
        
        logger.info(f"  Selected: {len(selected)} features")
        
        self.selected_features[f'sequential_{direction}'] = selected
        return selected
    
    def ensemble_selection(self, min_votes: int = 3) -> List[str]:
        """
        Method 8: Ensemble voting across all methods.
        
        Select features that are chosen by at least min_votes methods.
        This provides a robust set of features agreed upon by multiple methods.
        
        Args:
            min_votes: Minimum number of methods that must select a feature
            
        Returns:
            List of selected feature names
        """
        logger.info(f"Method 8: Ensemble Voting (min_votes={min_votes})")
        
        # Count votes for each feature
        vote_counts = {feature: 0 for feature in self.feature_names}
        
        for method, features in self.selected_features.items():
            for feature in features:
                vote_counts[feature] += 1
        
        # Select features with enough votes
        selected = [f for f, votes in vote_counts.items() if votes >= min_votes]
        
        # Sort by vote count
        selected_sorted = sorted(selected, key=lambda f: vote_counts[f], reverse=True)
        
        logger.info(f"  Selected: {len(selected_sorted)} features")
        logger.info(f"  Vote distribution:")
        
        vote_dist = {}
        for votes in sorted(set(vote_counts.values()), reverse=True):
            features_with_votes = [f for f, v in vote_counts.items() if v == votes]
            vote_dist[votes] = len(features_with_votes)
            logger.info(f"    {votes} votes: {len(features_with_votes)} features")
        
        self.selected_features['ensemble'] = selected_sorted
        return selected_sorted
    
    def evaluate_feature_set(self, features: List[str], method_name: str) -> Dict[str, float]:
        """
        Evaluate a feature set using cross-validation.
        
        Args:
            features: List of feature names
            method_name: Name of the selection method
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not features:
            return {'accuracy': 0.0, 'std': 0.0}
        
        # Get feature indices
        feature_indices = [self.feature_names.index(f) for f in features]
        X_subset = self.X[:, feature_indices]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_subset)
        
        # Cross-validation with Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores = cross_val_score(rf, X_scaled, self.y, cv=5, scoring='accuracy')
        
        return {
            'accuracy': scores.mean(),
            'std': scores.std(),
            'n_features': len(features)
        }
    
    def generate_selection_report(self) -> pd.DataFrame:
        """
        Generate comprehensive report of all selection methods.
        
        Returns:
            DataFrame with method comparison
        """
        logger.info("Generating selection report")
        
        report_data = []
        
        for method, features in self.selected_features.items():
            metrics = self.evaluate_feature_set(features, method)
            
            report_data.append({
                'method': method,
                'n_features': metrics['n_features'],
                'cv_accuracy': metrics['accuracy'],
                'cv_std': metrics['std'],
                'features': ', '.join(features[:10]) + ('...' if len(features) > 10 else '')
            })
        
        report_df = pd.DataFrame(report_data).sort_values('cv_accuracy', ascending=False)
        
        return report_df
    
    def save_selected_features(
        self,
        output_dir: str = 'output',
        selected_method: str = 'ensemble'
    ) -> None:
        """
        Save selected features and create filtered dataset.
        
        Args:
            output_dir: Output directory
            selected_method: Which method's features to use
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature selection report
        report = self.generate_selection_report()
        report_path = output_dir / 'feature_selection_report.csv'
        report.to_csv(report_path, index=False)
        logger.info(f"Saved selection report to {report_path}")
        
        # Save selected features list
        if selected_method in self.selected_features:
            selected = self.selected_features[selected_method]
            
            features_path = output_dir / f'selected_features_{selected_method}.txt'
            with open(features_path, 'w') as f:
                for feature in selected:
                    f.write(f"{feature}\n")
            logger.info(f"Saved {len(selected)} selected features to {features_path}")
            
            # Create filtered dataset
            metadata_cols = ['participant_id', 'file_path', 'diagnosis', 'age_months', 'dataset']
            filtered_df = self.df[metadata_cols + selected]
            
            filtered_path = output_dir / f'syntactic_semantic_selected_{selected_method}.csv'
            filtered_df.to_csv(filtered_path, index=False)
            logger.info(f"Saved filtered dataset to {filtered_path}")
        else:
            logger.warning(f"Method '{selected_method}' not found in selected_features")
    
    def run_full_pipeline(
        self,
        output_dir: str = 'output',
        selected_method: str = 'ensemble'
    ) -> Dict[str, List[str]]:
        """
        Run complete feature selection pipeline.
        
        Args:
            output_dir: Output directory for results
            selected_method: Which method to use for final selection
            
        Returns:
            Dictionary of selected features from all methods
        """
        logger.info("="*80)
        logger.info("FEATURE SELECTION PIPELINE")
        logger.info("="*80)
        
        # Load data
        self.load_data()
        
        # Run all selection methods
        self.variance_threshold_selection(threshold=0.01)
        self.correlation_based_selection(threshold=0.85)
        self.univariate_selection(k=15, method='f_classif')
        self.univariate_selection(k=15, method='mutual_info')
        self.rfe_selection(n_features=15)
        self.lasso_selection(alpha=0.01)
        self.tree_based_selection(threshold=0.01)
        self.sequential_selection(n_features=15, direction='forward')
        
        # Ensemble voting
        self.ensemble_selection(min_votes=3)
        
        # Generate and display report
        report = self.generate_selection_report()
        
        logger.info("="*80)
        logger.info("SELECTION RESULTS")
        logger.info("="*80)
        print("\n" + report.to_string(index=False))
        
        # Save results
        self.save_selected_features(output_dir, selected_method)
        
        logger.info("="*80)
        logger.info("FEATURE SELECTION COMPLETE")
        logger.info("="*80)
        
        return self.selected_features


def main():
    """Main execution function."""
    selector = SyntacticSemanticFeatureSelector(
        input_path='output/syntactic_semantic_cleaned.csv'
    )
    
    selected_features = selector.run_full_pipeline(
        output_dir='output',
        selected_method='ensemble'  # Use ensemble voting
    )
    
    # Display ensemble features
    print("\n" + "="*80)
    print("ENSEMBLE SELECTED FEATURES (3+ votes)")
    print("="*80)
    for i, feature in enumerate(selected_features['ensemble'], 1):
        print(f"{i:2d}. {feature}")


if __name__ == "__main__":
    main()
