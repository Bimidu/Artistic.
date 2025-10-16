"""
Model Evaluation Module

This module provides comprehensive evaluation metrics and analysis
for ASD classification models.

Key metrics:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Confusion Matrix
- Classification Report
- Cross-validation scores

Author: Bimidu Gunathilake
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, matthews_corrcoef
)
from sklearn.model_selection import cross_val_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """
    Comprehensive evaluation report for a model.
    
    Attributes:
        model_name: Name of the model
        accuracy: Accuracy score
        precision: Precision score
        recall: Recall score
        f1_score: F1 score
        roc_auc: ROC-AUC score (if binary classification)
        matthews_corr: Matthews correlation coefficient
        confusion_matrix: Confusion matrix
        classification_report: Detailed classification report
        cv_scores: Cross-validation scores (if available)
        feature_importance: Feature importance scores (if available)
    """
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    matthews_corr: float = 0.0
    confusion_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    classification_report: Dict[str, Any] = field(default_factory=dict)
    cv_scores: Optional[List[float]] = None
    feature_importance: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'matthews_corr': self.matthews_corr,
            'confusion_matrix': self.confusion_matrix.tolist() if len(self.confusion_matrix) > 0 else [],
            'classification_report': self.classification_report,
            'cv_scores': self.cv_scores,
            'cv_mean': np.mean(self.cv_scores) if self.cv_scores else None,
            'cv_std': np.std(self.cv_scores) if self.cv_scores else None,
        }
    
    def print_summary(self):
        """Print formatted evaluation summary."""
        print("\n" + "="*70)
        print(f"EVALUATION REPORT: {self.model_name}")
        print("="*70)
        
        print(f"\nMain Metrics:")
        print(f"  Accuracy:  {self.accuracy:.4f}")
        print(f"  Precision: {self.precision:.4f}")
        print(f"  Recall:    {self.recall:.4f}")
        print(f"  F1-Score:  {self.f1_score:.4f}")
        
        if self.roc_auc is not None:
            print(f"  ROC-AUC:   {self.roc_auc:.4f}")
        
        print(f"  Matthews:  {self.matthews_corr:.4f}")
        
        if self.cv_scores:
            print(f"\nCross-Validation:")
            print(f"  Mean Score: {np.mean(self.cv_scores):.4f}")
            print(f"  Std Dev:    {np.std(self.cv_scores):.4f}")
            print(f"  Min Score:  {np.min(self.cv_scores):.4f}")
            print(f"  Max Score:  {np.max(self.cv_scores):.4f}")
        
        if len(self.confusion_matrix) > 0:
            print(f"\nConfusion Matrix:")
            print(self.confusion_matrix)
        
        print("\n" + "="*70 + "\n")


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    
    Provides metrics, visualizations, and analysis for trained models.
    """
    
    def __init__(self):
        """Initialize model evaluator."""
        self.logger = logger
        self.reports_: Dict[str, EvaluationReport] = {}
        
        self.logger.info("ModelEvaluator initialized")
    
    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        cv_folds: int = 5
    ) -> EvaluationReport:
        """
        Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of model
            X_train: Training features (for CV evaluation)
            y_train: Training labels (for CV evaluation)
            cv_folds: Number of CV folds
        
        Returns:
            EvaluationReport: Comprehensive evaluation report
        """
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        matthews = matthews_corrcoef(y_test, y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_test, y_pred,
            output_dict=True,
            zero_division=0
        )
        
        # ROC-AUC (for binary classification)
        roc_auc = None
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        # Cross-validation scores (if training data provided)
        cv_scores = None
        if X_train is not None and y_train is not None:
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv_folds,
                    scoring='f1_weighted'
                ).tolist()
            except Exception as e:
                self.logger.warning(f"Could not perform cross-validation: {e}")
        
        # Create report
        report = EvaluationReport(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            matthews_corr=matthews,
            confusion_matrix=cm,
            classification_report=class_report,
            cv_scores=cv_scores
        )
        
        # Store report
        self.reports_[model_name] = report
        
        self.logger.info(
            f"Evaluation complete - Accuracy: {accuracy:.4f}, F1: {f1:.4f}"
        )
        
        return report
    
    def evaluate_multiple(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None
    ) -> Dict[str, EvaluationReport]:
        """
        Evaluate multiple models.
        
        Args:
            models: Dict of trained models
            X_test: Test features
            y_test: Test labels
            X_train: Training features (for CV)
            y_train: Training labels (for CV)
        
        Returns:
            Dict of evaluation reports
        """
        self.logger.info(f"Evaluating {len(models)} models")
        
        reports = {}
        for model_name, model in models.items():
            try:
                report = self.evaluate(
                    model, X_test, y_test,
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train
                )
                reports[model_name] = report
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
        
        return reports
    
    def compare_models(
        self,
        reports: Optional[Dict[str, EvaluationReport]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models side-by-side.
        
        Args:
            reports: Dict of evaluation reports (None = use stored reports)
        
        Returns:
            pd.DataFrame: Comparison DataFrame
        """
        if reports is None:
            reports = self.reports_
        
        if not reports:
            raise ValueError("No reports available for comparison")
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, report in reports.items():
            data = {
                'Model': model_name,
                'Accuracy': report.accuracy,
                'Precision': report.precision,
                'Recall': report.recall,
                'F1-Score': report.f1_score,
                'Matthews': report.matthews_corr,
            }
            
            if report.roc_auc is not None:
                data['ROC-AUC'] = report.roc_auc
            
            if report.cv_scores:
                data['CV Mean'] = np.mean(report.cv_scores)
                data['CV Std'] = np.std(report.cv_scores)
            
            comparison_data.append(data)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1-Score', ascending=False)
        
        return df
    
    def plot_confusion_matrix(
        self,
        model_name: str,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            model_name: Name of model
            class_names: Class names for labels
            save_path: Path to save plot (None = display)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.warning("matplotlib/seaborn not available for plotting")
            return
        
        if model_name not in self.reports_:
            raise ValueError(f"No report found for model: {model_name}")
        
        cm = self.reports_[model_name].confusion_matrix
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve (for binary classification).
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of model
            save_path: Path to save plot (None = display)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
            return
        
        if not hasattr(model, 'predict_proba'):
            self.logger.warning("Model does not support probability predictions")
            return
        
        # Get predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_model_comparison(
        self,
        reports: Optional[Dict[str, EvaluationReport]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of multiple models.
        
        Args:
            reports: Dict of evaluation reports (None = use stored)
            save_path: Path to save plot (None = display)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
            return
        
        df = self.compare_models(reports)
        
        # Plot metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax.bar(x + i * width, df[metric], width, label=metric)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_reports(self, save_path: str | Path):
        """
        Save all evaluation reports to file.
        
        Args:
            save_path: Path to save reports
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert reports to dict format
        reports_dict = {
            name: report.to_dict()
            for name, report in self.reports_.items()
        }
        
        # Save as JSON
        import json
        with open(save_path, 'w') as f:
            json.dump(reports_dict, f, indent=2)
        
        self.logger.info(f"Reports saved to {save_path}")

