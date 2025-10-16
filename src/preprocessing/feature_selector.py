"""
Feature Selection Module

This module provides feature selection methods for pragmatic and conversational
features to identify the most relevant features for ASD detection.

Key functionalities:
- Statistical feature selection
- Model-based feature selection
- Recursive feature elimination
- Feature importance analysis

Author: Bimidu Gunathilake
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass, field

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureImportance:
    """
    Feature importance scores from selection methods.
    
    Attributes:
        feature_name: Name of the feature
        importance_score: Importance score
        rank: Rank of feature (1 = most important)
        method: Selection method used
    """
    feature_name: str
    importance_score: float
    rank: int
    method: str


class FeatureSelector:
    """
    Feature selection class for pragmatic/conversational features.
    
    Provides multiple methods to select the most relevant features
    for ASD classification.
    """
    
    def __init__(self):
        """Initialize feature selector."""
        self.logger = logger
        self.selected_features_: Optional[List[str]] = None
        self.feature_importance_: Optional[List[FeatureImportance]] = None
        
        self.logger.info("FeatureSelector initialized")
    
    def select_k_best(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 30,
        score_func: str = 'f_classif'
    ) -> List[str]:
        """
        Select K best features using statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select
            score_func: Scoring function ('f_classif' or 'mutual_info')
        
        Returns:
            List[str]: Selected feature names
        """
        self.logger.info(f"Selecting {k} best features using {score_func}")
        
        # Choose scoring function
        if score_func == 'f_classif':
            scorer = f_classif
        elif score_func == 'mutual_info':
            scorer = mutual_info_classif
        else:
            raise ValueError(f"Unknown score function: {score_func}")
        
        # Select features
        selector = SelectKBest(score_func=scorer, k=min(k, X.shape[1]))
        selector.fit(X, y)
        
        # Get selected features
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Store feature importance
        scores = selector.scores_
        self.feature_importance_ = [
            FeatureImportance(
                feature_name=feat,
                importance_score=score,
                rank=rank + 1,
                method='statistical'
            )
            for rank, (feat, score) in enumerate(
                sorted(zip(X.columns, scores), key=lambda x: x[1], reverse=True)
            )
        ]
        
        self.selected_features_ = selected_features
        self.logger.info(f"Selected {len(selected_features)} features")
        
        return selected_features
    
    def select_from_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: str = 'median',
        max_features: Optional[int] = None
    ) -> List[str]:
        """
        Select features based on Random Forest feature importance.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            threshold: Importance threshold ('mean', 'median', or float)
            max_features: Maximum number of features to select
        
        Returns:
            List[str]: Selected feature names
        """
        self.logger.info(f"Selecting features from model with threshold={threshold}")
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        
        # Select features
        selector = SelectFromModel(
            rf,
            threshold=threshold,
            max_features=max_features,
            prefit=True
        )
        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Store feature importance
        self.feature_importance_ = [
            FeatureImportance(
                feature_name=feat,
                importance_score=imp,
                rank=rank + 1,
                method='random_forest'
            )
            for rank, (feat, imp) in enumerate(
                sorted(zip(X.columns, importance), key=lambda x: x[1], reverse=True)
            )
        ]
        
        self.selected_features_ = selected_features
        self.logger.info(f"Selected {len(selected_features)} features")
        
        return selected_features
    
    def select_rfe(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 30
    ) -> List[str]:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
        
        Returns:
            List[str]: Selected feature names
        """
        self.logger.info(f"Selecting {n_features} features using RFE")
        
        # Use Random Forest as base estimator
        rf = RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )
        
        # Recursive feature elimination
        rfe = RFE(
            estimator=rf,
            n_features_to_select=min(n_features, X.shape[1]),
            step=1
        )
        rfe.fit(X, y)
        
        # Get selected features
        selected_mask = rfe.get_support()
        selected_features = X.columns[selected_mask].tolist()
        
        # Get rankings
        rankings = rfe.ranking_
        self.feature_importance_ = [
            FeatureImportance(
                feature_name=feat,
                importance_score=1.0 / rank,  # Inverse rank as importance
                rank=rank,
                method='rfe'
            )
            for feat, rank in sorted(
                zip(X.columns, rankings),
                key=lambda x: x[1]
            )
        ]
        
        self.selected_features_ = selected_features
        self.logger.info(f"Selected {len(selected_features)} features")
        
        return selected_features
    
    def select_by_category(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        categories: Dict[str, List[str]],
        top_per_category: int = 10
    ) -> List[str]:
        """
        Select top features from each category.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            categories: Dict mapping category names to feature lists
            top_per_category: Number of top features per category
        
        Returns:
            List[str]: Selected feature names
        """
        self.logger.info(
            f"Selecting top {top_per_category} features per category"
        )
        
        selected_features = []
        all_importance = []
        
        # Train Random Forest on all features
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        importance = dict(zip(X.columns, rf.feature_importances_))
        
        # Select top features from each category
        for category_name, category_features in categories.items():
            # Filter to features that exist in X
            available_features = [f for f in category_features if f in X.columns]
            
            if not available_features:
                continue
            
            # Sort by importance
            category_importance = [
                (feat, importance.get(feat, 0))
                for feat in available_features
            ]
            category_importance.sort(key=lambda x: x[1], reverse=True)
            
            # Select top features
            top_features = [
                feat for feat, _ in category_importance[:top_per_category]
            ]
            selected_features.extend(top_features)
            
            # Store importance
            for rank, (feat, imp) in enumerate(category_importance):
                all_importance.append(
                    FeatureImportance(
                        feature_name=feat,
                        importance_score=imp,
                        rank=rank + 1,
                        method=f'category_{category_name}'
                    )
                )
        
        self.feature_importance_ = all_importance
        self.selected_features_ = list(set(selected_features))  # Remove duplicates
        
        self.logger.info(f"Selected {len(self.selected_features_)} features total")
        
        return self.selected_features_
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.
        
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        if self.feature_importance_ is None:
            raise ValueError("No feature importance available. Run selection first.")
        
        return pd.DataFrame([
            {
                'feature': fi.feature_name,
                'importance': fi.importance_score,
                'rank': fi.rank,
                'method': fi.method
            }
            for fi in self.feature_importance_
        ])
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save plot (None = display)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
            return
        
        if self.feature_importance_ is None:
            raise ValueError("No feature importance available. Run selection first.")
        
        # Get top features
        df = self.get_feature_importance_df()
        df_top = df.nlargest(top_n, 'importance')
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(df_top['feature'], df_top['importance'])
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

