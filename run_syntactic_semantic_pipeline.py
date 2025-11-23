"""
Complete Syntactic & Semantic Pipeline with Comprehensive Visualizations

This script runs the full pipeline from feature extraction to model training
and generates extensive plots for all metrics.

Author: Randil Haturusinghe
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parsers.chat_parser import CHATParser
from src.features.feature_extractor import FeatureExtractor
from src.models.syntactic_semantic.preprocessor import SyntacticSemanticPreprocessor
from src.models.syntactic_semantic.model_trainer import (
    SyntacticSemanticTrainer,
    SyntacticSemanticModelConfig
)
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
from scipy import stats

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def create_output_directories():
    """Create output directories for models and plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = Path("/Users/user/PycharmProjects/Artistic./src/models/syntactic_semantic")
    plots_dir = base_dir / "plots" / f"run_{timestamp}"
    models_dir = base_dir / "trained_models"

    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    return plots_dir, models_dir, timestamp

def extract_features_from_data():
    """Extract syntactic/semantic features from all CHAT files."""
    print("\n" + "="*80)
    print("STEP 1: FEATURE EXTRACTION")
    print("="*80 + "\n")

    # Find all CHAT files
    data_dirs = [
        Path("/Users/user/PycharmProjects/Artistic./data/asdbank_eigsti"),
        Path("/Users/user/PycharmProjects/Artistic./data/asdbank_flusberg"),
    ]

    chat_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            chat_files.extend(list(data_dir.rglob("*.cha")))

    print(f"Found {len(chat_files)} CHAT files")

    # Initialize feature extractor for syntactic/semantic features
    extractor = FeatureExtractor(categories=['syntactic_semantic'])

    # Extract features
    print("\nExtracting syntactic/semantic features...")
    df = extractor.extract_from_files(chat_files[:100], show_progress=True)  # Limit to 100 for speed

    print(f"\nExtracted features from {len(df)} transcripts")
    print(f"Total features: {len([c for c in df.columns if c not in ['participant_id', 'file_path', 'diagnosis', 'age_months']])}")

    return df

def preprocess_data(df):
    """Preprocess the extracted features."""
    print("\n" + "="*80)
    print("STEP 2: DATA PREPROCESSING")
    print("="*80 + "\n")

    # Remove rows with missing diagnosis
    df_clean = df.dropna(subset=['diagnosis'])
    print(f"Samples with diagnosis: {len(df_clean)}")
    print(f"Diagnosis distribution:\n{df_clean['diagnosis'].value_counts()}")

    # Initialize preprocessor
    preprocessor = SyntacticSemanticPreprocessor(
        target_column='diagnosis',
        test_size=0.2,
        random_state=42,
        feature_selection=True,
        n_features=25
    )

    # Fit and transform
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df_clean)

    print(f"\nPreprocessing complete:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Features selected: {len(X_train.columns)}")
    print(f"  Selected features: {list(X_train.columns)[:10]}...")

    return X_train, X_test, y_train, y_test, preprocessor

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and evaluate."""
    print("\n" + "="*80)
    print("STEP 3: MODEL TRAINING")
    print("="*80 + "\n")

    # Initialize trainer
    trainer = SyntacticSemanticTrainer()

    # Train multiple models
    print("Training models...")
    results = trainer.train_multiple_models(
        X_train, y_train, X_test, y_test
    )

    print(f"\nTraining complete!")
    print(f"Models trained: {len(results['models'])}")
    print(f"Best model: {results['best_model']}")

    # Print evaluation summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80 + "\n")

    for model_name, metrics in results['evaluation_summary'].items():
        print(f"{model_name.upper()}:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall:    {metrics.get('recall', 0):.4f}")
        print(f"  F1 Score:  {metrics.get('f1_score', 0):.4f}")
        print(f"  AUC-ROC:   {metrics.get('auc_roc', 0):.4f}")
        print()

    return trainer, results

def generate_all_plots(X_train, X_test, y_train, y_test, trainer, results, preprocessor, plots_dir):
    """Generate comprehensive visualizations."""
    print("\n" + "="*80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*80 + "\n")

    # Create subdirectories for different plot types
    (plots_dir / "basic").mkdir(exist_ok=True)
    (plots_dir / "model_performance").mkdir(exist_ok=True)
    (plots_dir / "feature_analysis").mkdir(exist_ok=True)
    (plots_dir / "advanced").mkdir(exist_ok=True)
    (plots_dir / "statistical").mkdir(exist_ok=True)

    print("Generating plots...")

    # === BASIC PLOTS ===
    print("\n1. Basic Plots...")

    # 1.1 Class distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    y_train.value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Diagnosis')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)

    y_test.value_counts().plot(kind='bar', ax=axes[1], color='coral')
    axes[1].set_title('Test Set - Class Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Diagnosis')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(plots_dir / "basic" / "01_class_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 1.2 Feature distributions (top 16 features)
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for idx, col in enumerate(X_train.columns[:16]):
        axes[idx].hist(X_train[col], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[idx].set_title(col, fontsize=10)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')

    plt.suptitle('Feature Distributions (Top 16 Features)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(plots_dir / "basic" / "02_feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 1.3 Box plots by class
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    X_train_with_labels = X_train.copy()
    X_train_with_labels['diagnosis'] = y_train.values

    for idx, col in enumerate(X_train.columns[:16]):
        X_train_with_labels.boxplot(column=col, by='diagnosis', ax=axes[idx])
        axes[idx].set_title(col, fontsize=10)
        axes[idx].set_xlabel('Diagnosis')
        axes[idx].set_ylabel('Value')
        plt.setp(axes[idx].xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle('Feature Distributions by Diagnosis (Top 16 Features)', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(plots_dir / "basic" / "03_boxplots_by_class.png", dpi=300, bbox_inches='tight')
    plt.close()

    # === MODEL PERFORMANCE PLOTS ===
    print("2. Model Performance Plots...")

    # 2.1 Model comparison bar chart
    models = list(results['evaluation_summary'].keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [results['evaluation_summary'][m].get(metric, 0) for m in models]
        axes[idx].bar(models, values, color=sns.color_palette("husl", len(models)))
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim([0, 1])
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # Overall comparison
    axes[5].axis('off')
    comparison_data = []
    for model in models:
        comparison_data.append([
            model,
            results['evaluation_summary'][model].get('accuracy', 0),
            results['evaluation_summary'][model].get('f1_score', 0),
            results['evaluation_summary'][model].get('auc_roc', 0)
        ])

    comparison_df = pd.DataFrame(comparison_data, columns=['Model', 'Accuracy', 'F1', 'AUC-ROC'])
    table = axes[5].table(cellText=comparison_df.values, colLabels=comparison_df.columns,
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "model_performance" / "01_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2.2 Confusion matrices for all models
    n_models = len(models)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, model_name in enumerate(models):
        model = results['models'][model_name]
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)

        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[idx],
                    xticklabels=model.classes_, yticklabels=model.classes_)
        axes[idx].set_title(f'{model_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    # Hide extra subplot
    axes[5].axis('off')

    plt.suptitle('Confusion Matrices (Normalized)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "model_performance" / "02_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2.3 ROC Curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    colors = sns.color_palette("husl", len(models))

    for idx, model_name in enumerate(models):
        model = results['models'][model_name]

        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)

            # For multiclass, compute ROC for each class
            if len(model.classes_) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=model.classes_[1])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[idx], lw=2,
                       label=f'{model_name} (AUC = {roc_auc:.3f})')
            else:
                # Use one-vs-rest for multiclass
                from sklearn.preprocessing import label_binarize
                y_test_bin = label_binarize(y_test, classes=model.classes_)

                for i, class_name in enumerate(model.classes_):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, lw=2, alpha=0.7,
                           label=f'{model_name} - {class_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "model_performance" / "03_roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2.4 Precision-Recall Curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for idx, model_name in enumerate(models):
        model = results['models'][model_name]

        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)

            if len(model.classes_) == 2:
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1],
                                                               pos_label=model.classes_[1])
                avg_precision = average_precision_score(y_test == model.classes_[1], y_pred_proba[:, 1])
                ax.plot(recall, precision, color=colors[idx], lw=2,
                       label=f'{model_name} (AP = {avg_precision:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "model_performance" / "04_precision_recall_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # === FEATURE ANALYSIS PLOTS ===
    print("3. Feature Analysis Plots...")

    # 3.1 Feature importance for tree-based models
    for model_name in ['random_forest', 'xgboost', 'lightgbm']:
        if model_name in results['models'] and results['models'][model_name] is not None:
            importance_df = trainer.get_syntactic_semantic_feature_importance(
                model_name, list(X_train.columns), top_n=20
            )

            fig, ax = plt.subplots(1, 1, figsize=(12, 10))

            colors_map = {
                'syntactic_complexity': 'steelblue',
                'grammatical': 'coral',
                'semantic': 'lightgreen',
                'vocabulary': 'gold',
                'other': 'gray'
            }

            bar_colors = [colors_map.get(cat, 'gray') for cat in importance_df['category']]

            ax.barh(range(len(importance_df)), importance_df['importance'], color=bar_colors)
            ax.set_yticks(range(len(importance_df)))
            ax.set_yticklabels(importance_df['feature'])
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title(f'Feature Importance - {model_name.replace("_", " ").title()}',
                        fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors_map[cat], label=cat.replace('_', ' ').title())
                             for cat in colors_map.keys()]
            ax.legend(handles=legend_elements, loc='lower right')

            plt.tight_layout()
            plt.savefig(plots_dir / "feature_analysis" / f"01_feature_importance_{model_name}.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

    # 3.2 Feature correlation heatmap
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    corr_matrix = X_train.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot=False, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(plots_dir / "feature_analysis" / "02_correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 3.3 Pairplot of top features
    top_features = list(X_train.columns[:5])
    pairplot_data = X_train[top_features].copy()
    pairplot_data['diagnosis'] = y_train.values

    g = sns.pairplot(pairplot_data, hue='diagnosis', diag_kind='kde',
                     plot_kws={'alpha': 0.6}, height=3)
    g.fig.suptitle('Pairplot of Top 5 Features', fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(plots_dir / "feature_analysis" / "03_pairplot_top_features.png", dpi=300, bbox_inches='tight')
    plt.close()

    # === ADVANCED PLOTS ===
    print("4. Advanced Plots...")

    # 4.1 Learning curves for best model
    best_model_name = results['best_model']
    if best_model_name and results['models'][best_model_name] is not None:
        from sklearn.model_selection import learning_curve as sk_learning_curve

        model = results['models'][best_model_name]

        train_sizes, train_scores, val_scores = sk_learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Training score', linewidth=2)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color='steelblue')

        ax.plot(train_sizes, val_mean, 'o-', color='coral', label='Cross-validation score', linewidth=2)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color='coral')

        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Accuracy Score', fontsize=12)
        ax.set_title(f'Learning Curves - {best_model_name.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "advanced" / "01_learning_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 4.2 T-SNE visualization
    from sklearn.manifold import TSNE

    print("   Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_train_tsne = tsne.fit_transform(X_train)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    scatter = ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1],
                        c=pd.Categorical(y_train).codes, cmap='viridis',
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title('t-SNE Visualization of Training Data', fontsize=14, fontweight='bold')

    # Add colorbar with class labels
    unique_labels = y_train.unique()
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(unique_labels)))
    cbar.set_label('Diagnosis', fontsize=12)
    cbar.ax.set_yticklabels(unique_labels)

    plt.tight_layout()
    plt.savefig(plots_dir / "advanced" / "02_tsne_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4.3 PCA visualization
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                        c=pd.Categorical(y_train).codes, cmap='viridis',
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title('PCA Visualization of Training Data', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(unique_labels)))
    cbar.set_label('Diagnosis', fontsize=12)
    cbar.ax.set_yticklabels(unique_labels)

    plt.tight_layout()
    plt.savefig(plots_dir / "advanced" / "03_pca_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4.4 PCA explained variance
    pca_full = PCA()
    pca_full.fit(X_train)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scree plot
    axes[0].plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
                pca_full.explained_variance_ratio_, 'o-', linewidth=2, color='steelblue')
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title('Scree Plot', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Cumulative explained variance
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'o-', linewidth=2, color='coral')
    axes[1].axhline(y=0.95, color='red', linestyle='--', label='95% Variance')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "advanced" / "04_pca_variance_explained.png", dpi=300, bbox_inches='tight')
    plt.close()

    # === STATISTICAL PLOTS ===
    print("5. Statistical Analysis Plots...")

    # 5.1 Feature statistics summary
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Mean values
    feature_means = X_train.mean().sort_values(ascending=False)[:20]
    axes[0, 0].barh(range(len(feature_means)), feature_means.values, color='steelblue')
    axes[0, 0].set_yticks(range(len(feature_means)))
    axes[0, 0].set_yticklabels(feature_means.index)
    axes[0, 0].set_xlabel('Mean Value')
    axes[0, 0].set_title('Top 20 Features by Mean Value', fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)

    # Standard deviation
    feature_std = X_train.std().sort_values(ascending=False)[:20]
    axes[0, 1].barh(range(len(feature_std)), feature_std.values, color='coral')
    axes[0, 1].set_yticks(range(len(feature_std)))
    axes[0, 1].set_yticklabels(feature_std.index)
    axes[0, 1].set_xlabel('Standard Deviation')
    axes[0, 1].set_title('Top 20 Features by Std Dev', fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)

    # Skewness
    feature_skew = X_train.apply(lambda x: stats.skew(x)).sort_values(ascending=False)[:20]
    axes[1, 0].barh(range(len(feature_skew)), feature_skew.values, color='lightgreen')
    axes[1, 0].set_yticks(range(len(feature_skew)))
    axes[1, 0].set_yticklabels(feature_skew.index)
    axes[1, 0].set_xlabel('Skewness')
    axes[1, 0].set_title('Top 20 Features by Skewness', fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis='x', alpha=0.3)

    # Kurtosis
    feature_kurt = X_train.apply(lambda x: stats.kurtosis(x)).sort_values(ascending=False)[:20]
    axes[1, 1].barh(range(len(feature_kurt)), feature_kurt.values, color='gold')
    axes[1, 1].set_yticks(range(len(feature_kurt)))
    axes[1, 1].set_yticklabels(feature_kurt.index)
    axes[1, 1].set_xlabel('Kurtosis')
    axes[1, 1].set_title('Top 20 Features by Kurtosis', fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(axis='x', alpha=0.3)

    plt.suptitle('Feature Statistical Properties', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "statistical" / "01_feature_statistics.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5.2 Q-Q plots for normality check
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    for idx, col in enumerate(X_train.columns[:16]):
        stats.probplot(X_train[col], dist="norm", plot=axes[idx])
        axes[idx].set_title(f'{col}', fontsize=10)
        axes[idx].grid(alpha=0.3)

    plt.suptitle('Q-Q Plots for Normality Check (Top 16 Features)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "statistical" / "02_qq_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 5.3 Violin plots by diagnosis
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()

    X_train_with_labels = X_train.copy()
    X_train_with_labels['diagnosis'] = y_train.values

    for idx, col in enumerate(X_train.columns[:16]):
        sns.violinplot(data=X_train_with_labels, x='diagnosis', y=col, ax=axes[idx], palette='Set2')
        axes[idx].set_title(col, fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.suptitle('Violin Plots by Diagnosis (Top 16 Features)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "statistical" / "03_violin_plots.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nAll plots saved to: {plots_dir}")
    print(f"\nPlot categories:")
    print(f"  - Basic: {len(list((plots_dir / 'basic').glob('*.png')))} plots")
    print(f"  - Model Performance: {len(list((plots_dir / 'model_performance').glob('*.png')))} plots")
    print(f"  - Feature Analysis: {len(list((plots_dir / 'feature_analysis').glob('*.png')))} plots")
    print(f"  - Advanced: {len(list((plots_dir / 'advanced').glob('*.png')))} plots")
    print(f"  - Statistical: {len(list((plots_dir / 'statistical').glob('*.png')))} plots")

def save_results(trainer, results, models_dir, preprocessor, timestamp):
    """Save trained models and preprocessor."""
    print("\n" + "="*80)
    print("STEP 5: SAVING MODELS")
    print("="*80 + "\n")

    # Save each model
    for model_name, model in results['models'].items():
        if model is not None:
            model_path = models_dir / f"{model_name}_{timestamp}.joblib"
            trainer.save_model(model_name, model_path)
            print(f"Saved {model_name} to {model_path}")

    # Save preprocessor
    preprocessor_path = models_dir / f"preprocessor_{timestamp}.joblib"
    preprocessor.save(preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")

    # Save evaluation summary
    summary_path = models_dir / f"evaluation_summary_{timestamp}.csv"
    summary_df = pd.DataFrame(results['evaluation_summary']).T
    summary_df.to_csv(summary_path)
    print(f"Saved evaluation summary to {summary_path}")

def main():
    """Run the complete pipeline."""
    print("\n" + "="*80)
    print("SYNTACTIC & SEMANTIC COMPLETE PIPELINE")
    print("="*80)
    print("This will:")
    print("  1. Extract syntactic/semantic features from CHAT files")
    print("  2. Preprocess the data")
    print("  3. Train multiple models")
    print("  4. Generate comprehensive visualizations")
    print("  5. Save trained models")
    print("="*80 + "\n")

    # Create output directories
    plots_dir, models_dir, timestamp = create_output_directories()

    # Step 1: Extract features
    df = extract_features_from_data()

    # Step 2: Preprocess
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # Step 3: Train models
    trainer, results = train_models(X_train, y_train, X_test, y_test)

    # Step 4: Generate plots
    generate_all_plots(X_train, X_test, y_train, y_test, trainer, results, preprocessor, plots_dir)

    # Step 5: Save results
    save_results(trainer, results, models_dir, preprocessor, timestamp)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults saved:")
    print(f"  - Plots: {plots_dir}")
    print(f"  - Models: {models_dir}")
    print(f"\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
