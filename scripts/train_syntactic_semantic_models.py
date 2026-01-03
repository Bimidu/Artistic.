"""
Syntactic Semantic Model Training Pipeline

Comprehensive training script for ASD classification using syntactic and semantic features.
Trains 7 different models and provides detailed evaluation and comparison.

Models:
1. Random Forest
2. XGBoost
3. LightGBM
4. Gradient Boosting
5. AdaBoost
6. SVM
7. Logistic Regression

Author: AI Assistant
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML models
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SyntacticSemanticModelPipeline:
    """
    Comprehensive model training pipeline for syntactic semantic features.
    
    Trains multiple models, evaluates performance, and provides detailed
    comparison and analysis.
    """
    
    def __init__(
        self,
        data_path: str = 'output/syntactic_semantic_cleaned.csv',
        selected_features_path: str = 'output/selected_features_ensemble.txt',
        output_dir: str = 'output/models/syntactic_semantic'
    ):
        """
        Initialize model pipeline.
        
        Args:
            data_path: Path to cleaned feature data
            selected_features_path: Path to selected features list (optional)
            output_dir: Directory to save models and results
        """
        self.data_path = Path(data_path)
        self.selected_features_path = Path(selected_features_path) if selected_features_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Models
        self.models = {}
        self.trained_models = {}
        self.results = {}
        
        logger.info(f"Initialized SyntacticSemanticModelPipeline")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_data(self, use_selected_features: bool = True) -> None:
        """
        Load and prepare data for training.
        
        Args:
            use_selected_features: Whether to use selected features or all features
        """
        logger.info(f"Loading data from {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        # Separate metadata and features
        metadata_cols = ['participant_id', 'file_path', 'diagnosis', 'age_months', 'dataset']
        
        # Get feature columns
        if use_selected_features and self.selected_features_path and self.selected_features_path.exists():
            logger.info(f"Using selected features from {self.selected_features_path}")
            with open(self.selected_features_path) as f:
                self.feature_names = [line.strip() for line in f if line.strip()]
            
            # Verify features exist in data
            missing_features = set(self.feature_names) - set(self.df.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                self.feature_names = [f for f in self.feature_names if f in self.df.columns]
        else:
            logger.info("Using all available features")
            self.feature_names = [col for col in self.df.columns if col not in metadata_cols]
        
        # Prepare features and target
        self.X = self.df[self.feature_names]
        self.y = self.label_encoder.fit_transform(self.df['diagnosis'])
        
        logger.info(f"Loaded {len(self.df)} samples with {len(self.feature_names)} features")
        logger.info(f"Classes: {list(self.label_encoder.classes_)}")
        logger.info(f"Class distribution: {dict(zip(*np.unique(self.y, return_counts=True)))}")
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> None:
        """
        Split data into train and test sets with stratification.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        logger.info(f"Splitting data (test_size={test_size}, stratified)")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
    
    def initialize_models(self) -> None:
        """Initialize all models with optimized hyperparameters."""
        logger.info("Initializing models")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'XGBoost': XGBClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'AdaBoost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            ),
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                random_state=42,
                probability=True,
                class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
                solver='lbfgs'
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def train_model(
        self,
        model_name: str,
        model,
        use_scaled: bool = True
    ) -> Dict:
        """
        Train a single model and evaluate performance.
        
        Args:
            model_name: Name of the model
            model: Model instance
            use_scaled: Whether to use scaled features
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training {model_name}...")
        
        # Select features
        X_train = self.X_train_scaled if use_scaled else self.X_train.values
        X_test = self.X_test_scaled if use_scaled else self.X_test.values
        
        # Train model
        model.fit(X_train, self.y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Probabilities (if available)
        y_pred_proba_test = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba_test = model.predict_proba(X_test)
        
        # Calculate metrics
        results = {
            'model': model,
            'model_name': model_name,
            'train_accuracy': accuracy_score(self.y_train, y_pred_train),
            'test_accuracy': accuracy_score(self.y_test, y_pred_test),
            'precision': precision_score(self.y_test, y_pred_test, average='weighted', zero_division=0),
            'recall': recall_score(self.y_test, y_pred_test, average='weighted', zero_division=0),
            'f1_score': f1_score(self.y_test, y_pred_test, average='weighted', zero_division=0),
            'predictions': y_pred_test,
            'probabilities': y_pred_proba_test,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred_test),
            'classification_report': classification_report(
                self.y_test, y_pred_test,
                target_names=self.label_encoder.classes_,
                zero_division=0
            )
        }
        
        # ROC AUC for multiclass (if probabilities available)
        if y_pred_proba_test is not None and len(self.label_encoder.classes_) > 2:
            try:
                results['roc_auc'] = roc_auc_score(
                    self.y_test, y_pred_proba_test,
                    multi_class='ovr',
                    average='weighted'
                )
            except:
                results['roc_auc'] = None
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, X_train, self.y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy',
            n_jobs=-1
        )
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        logger.info(f"{model_name} - Test Accuracy: {results['test_accuracy']:.4f}, "
                   f"F1: {results['f1_score']:.4f}, CV: {results['cv_mean']:.4f}±{results['cv_std']:.4f}")
        
        return results
    
    def train_all_models(self) -> Dict:
        """
        Train all models and collect results.
        
        Returns:
            Dictionary of all training results
        """
        logger.info("="*80)
        logger.info("TRAINING ALL MODELS")
        logger.info("="*80)
        
        for model_name, model in self.models.items():
            try:
                # Determine if scaling is needed
                use_scaled = model_name in ['SVM', 'Logistic Regression']
                
                results = self.train_model(model_name, model, use_scaled=use_scaled)
                self.results[model_name] = results
                self.trained_models[model_name] = model
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        
        return self.results
    
    def get_results_summary(self) -> pd.DataFrame:
        """
        Generate summary table of all model results.
        
        Returns:
            DataFrame with model comparison
        """
        summary_data = []
        
        for model_name, results in self.results.items():
            if 'error' in results:
                continue
            
            summary_data.append({
                'Model': model_name,
                'Train Acc': results['train_accuracy'],
                'Test Acc': results['test_accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score'],
                'ROC AUC': results.get('roc_auc', np.nan),
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std']
            })
        
        summary_df = pd.DataFrame(summary_data).sort_values('F1 Score', ascending=False)
        return summary_df
    
    def get_feature_importance(self, model_name: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from a trained model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.results[model_name]['model']
        
        # Get importance values
        if hasattr(model, 'feature_importances_'):
            importance_values = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            if len(model.coef_.shape) > 1:
                importance_values = np.abs(model.coef_).mean(axis=0)
            else:
                importance_values = np.abs(model.coef_)
        else:
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_values
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_models(self) -> None:
        """Save all trained models to disk."""
        logger.info("Saving models...")
        
        models_dir = self.output_dir / 'trained_models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_path = models_dir / f"{model_name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scaler and label encoder
        joblib.dump(self.scaler, models_dir / 'scaler.pkl')
        joblib.dump(self.label_encoder, models_dir / 'label_encoder.pkl')
        
        # Save feature names
        with open(models_dir / 'feature_names.txt', 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        
        logger.info(f"Saved {len(self.trained_models)} models to {models_dir}")
    
    def save_results(self) -> None:
        """Save training results and summary."""
        logger.info("Saving results...")
        
        # Save summary table
        summary_df = self.get_results_summary()
        summary_path = self.output_dir / 'model_comparison.csv'
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved model comparison to {summary_path}")
        
        # Save detailed results
        for model_name, results in self.results.items():
            if 'error' in results:
                continue
            
            model_dir = self.output_dir / 'detailed_results' / model_name.lower().replace(' ', '_')
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save classification report
            with open(model_dir / 'classification_report.txt', 'w') as f:
                f.write(results['classification_report'])
            
            # Save confusion matrix
            cm_df = pd.DataFrame(
                results['confusion_matrix'],
                index=self.label_encoder.classes_,
                columns=self.label_encoder.classes_
            )
            cm_df.to_csv(model_dir / 'confusion_matrix.csv')
    
    def plot_results(self) -> None:
        """Generate visualization plots for model comparison."""
        logger.info("Generating plots...")
        
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        summary_df = self.get_results_summary()
        
        # 1. Model comparison bar plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['Test Acc', 'Precision', 'Recall', 'F1 Score']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            summary_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(f'{metric} by Model', fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrices for best models
        top_models = summary_df.head(3)['Model'].tolist()
        
        fig, axes = plt.subplots(1, min(3, len(top_models)), figsize=(15, 5))
        if len(top_models) == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(top_models):
            cm = self.results[model_name]['confusion_matrix']
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.label_encoder.classes_,
                yticklabels=self.label_encoder.classes_,
                ax=axes[idx]
            )
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature importance for best model
        best_model = summary_df.iloc[0]['Model']
        importance_df = self.get_feature_importance(best_model, top_n=15)
        
        if not importance_df.empty:
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top 15 Feature Importance - {best_model}', fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Saved plots to {plots_dir}")
    
    def run_full_pipeline(
        self,
        use_selected_features: bool = True,
        test_size: float = 0.2
    ) -> pd.DataFrame:
        """
        Run the complete training pipeline.
        
        Args:
            use_selected_features: Whether to use selected features
            test_size: Test set proportion
            
        Returns:
            Summary DataFrame with model comparison
        """
        logger.info("="*80)
        logger.info("SYNTACTIC SEMANTIC MODEL TRAINING PIPELINE")
        logger.info("="*80)
        
        # Load and prepare data
        self.load_data(use_selected_features=use_selected_features)
        self.split_data(test_size=test_size)
        
        # Initialize and train models
        self.initialize_models()
        self.train_all_models()
        
        # Generate summary
        summary_df = self.get_results_summary()
        
        # Save everything
        self.save_models()
        self.save_results()
        self.plot_results()
        
        # Print summary
        logger.info("="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        print("\n" + summary_df.to_string(index=False))
        
        # Best model
        best_model = summary_df.iloc[0]
        logger.info("="*80)
        logger.info(f"BEST MODEL: {best_model['Model']}")
        logger.info(f"  Test Accuracy: {best_model['Test Acc']:.4f}")
        logger.info(f"  F1 Score: {best_model['F1 Score']:.4f}")
        logger.info(f"  CV Score: {best_model['CV Mean']:.4f} ± {best_model['CV Std']:.4f}")
        logger.info("="*80)
        
        return summary_df


def main():
    """Main execution function."""
    # Initialize pipeline
    pipeline = SyntacticSemanticModelPipeline(
        data_path='output/syntactic_semantic_cleaned.csv',
        selected_features_path='output/selected_features_ensemble.txt',
        output_dir='output/models/syntactic_semantic'
    )
    
    # Run pipeline
    summary = pipeline.run_full_pipeline(
        use_selected_features=True,  # Use ensemble-selected features
        test_size=0.2
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {pipeline.output_dir}")
    print(f"  - Models: {pipeline.output_dir}/trained_models/")
    print(f"  - Comparison: {pipeline.output_dir}/model_comparison.csv")
    print(f"  - Plots: {pipeline.output_dir}/plots/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
