"""
Simplified Syntactic & Semantic Pipeline with Comprehensive Visualizations
Compatible with Python 3.8

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
import joblib
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.parsers.chat_parser import CHATParser
from src.features.feature_extractor import FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
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
    df = extractor.extract_from_files(chat_files[:100], show_progress=True)

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

    # Get feature columns
    feature_cols = [c for c in df_clean.columns if c not in ['participant_id', 'file_path', 'diagnosis', 'age_months']]

    X = df_clean[feature_cols]
    y = df_clean['diagnosis']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    print(f"\nPreprocessing complete:")
    print(f"  Training samples: {len(X_train_scaled)}")
    print(f"  Test samples: {len(X_test_scaled)}")
    print(f"  Features: {len(X_train_scaled.columns)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and evaluate."""
    print("\n" + "="*80)
    print("STEP 3: MODEL TRAINING")
    print("="*80 + "\n")

    models = {
        'random_forest': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=150, max_depth=8, random_state=42),
        'svm': SVC(C=1.0, kernel='rbf', probability=True, random_state=42),
        'logistic': LogisticRegression(C=1.0, max_iter=2000, random_state=42, n_jobs=-1),
    }

    trained_models = {}
    evaluation_summary = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        evaluation_summary[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }

        print(f"  Accuracy: {evaluation_summary[name]['accuracy']:.4f}")
        print(f"  F1 Score: {evaluation_summary[name]['f1_score']:.4f}")

    # Find best model
    best_model = max(evaluation_summary.items(), key=lambda x: x[1]['f1_score'])[0]
    print(f"\nBest model: {best_model}")

    return trained_models, evaluation_summary, best_model

def generate_all_plots(X_train, X_test, y_train, y_test, models, evaluation_summary, plots_dir):
    """Generate comprehensive visualizations."""
    print("\n" + "="*80)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("="*80 + "\n")

    # Create subdirectories
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

    # 1.2 Feature distributions
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

    # === MODEL PERFORMANCE PLOTS ===
    print("2. Model Performance Plots...")

    # 2.1 Model comparison
    model_names = list(evaluation_summary.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        values = [evaluation_summary[m][metric] for m in model_names]
        axes[idx].bar(model_names, values, color=sns.color_palette("husl", len(model_names)))
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim([0, 1])
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)

        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "model_performance" / "01_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2.2 Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[idx],
                    xticklabels=model.classes_, yticklabels=model.classes_)
        axes[idx].set_title(f'{name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.suptitle('Confusion Matrices (Normalized)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "model_performance" / "02_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()

    # === FEATURE ANALYSIS PLOTS ===
    print("3. Feature Analysis Plots...")

    # 3.1 Feature importance (Random Forest)
    rf_model = models['random_forest']
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.barh(range(len(importance_df)), importance_df['importance'], color='steelblue')
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance - Random Forest (Top 20)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "feature_analysis" / "01_feature_importance_rf.png", dpi=300, bbox_inches='tight')
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

    # === ADVANCED PLOTS ===
    print("4. Advanced Plots...")

    # 4.1 PCA visualization
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

    unique_labels = y_train.unique()
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(unique_labels)))
    cbar.set_label('Diagnosis', fontsize=12)
    cbar.ax.set_yticklabels(unique_labels)

    plt.tight_layout()
    plt.savefig(plots_dir / "advanced" / "01_pca_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4.2 PCA explained variance
    pca_full = PCA()
    pca_full.fit(X_train)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
                pca_full.explained_variance_ratio_, 'o-', linewidth=2, color='steelblue')
    axes[0].set_xlabel('Principal Component', fontsize=12)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
    axes[0].set_title('Scree Plot', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)

    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    axes[1].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'o-', linewidth=2, color='coral')
    axes[1].axhline(y=0.95, color='red', linestyle='--', label='95% Variance')
    axes[1].set_xlabel('Number of Components', fontsize=12)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
    axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "advanced" / "02_pca_variance_explained.png", dpi=300, bbox_inches='tight')
    plt.close()

    # === STATISTICAL PLOTS ===
    print("5. Statistical Analysis Plots...")

    # 5.1 Feature statistics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    feature_means = X_train.mean().sort_values(ascending=False)[:20]
    axes[0, 0].barh(range(len(feature_means)), feature_means.values, color='steelblue')
    axes[0, 0].set_yticks(range(len(feature_means)))
    axes[0, 0].set_yticklabels(feature_means.index)
    axes[0, 0].set_xlabel('Mean Value')
    axes[0, 0].set_title('Top 20 Features by Mean', fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)

    feature_std = X_train.std().sort_values(ascending=False)[:20]
    axes[0, 1].barh(range(len(feature_std)), feature_std.values, color='coral')
    axes[0, 1].set_yticks(range(len(feature_std)))
    axes[0, 1].set_yticklabels(feature_std.index)
    axes[0, 1].set_xlabel('Standard Deviation')
    axes[0, 1].set_title('Top 20 Features by Std Dev', fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)

    feature_skew = X_train.apply(lambda x: stats.skew(x)).sort_values(ascending=False)[:20]
    axes[1, 0].barh(range(len(feature_skew)), feature_skew.values, color='lightgreen')
    axes[1, 0].set_yticks(range(len(feature_skew)))
    axes[1, 0].set_yticklabels(feature_skew.index)
    axes[1, 0].set_xlabel('Skewness')
    axes[1, 0].set_title('Top 20 Features by Skewness', fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis='x', alpha=0.3)

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

    print(f"\nAll plots saved to: {plots_dir}")
    print(f"\nPlot categories:")
    print(f"  - Basic: {len(list((plots_dir / 'basic').glob('*.png')))} plots")
    print(f"  - Model Performance: {len(list((plots_dir / 'model_performance').glob('*.png')))} plots")
    print(f"  - Feature Analysis: {len(list((plots_dir / 'feature_analysis').glob('*.png')))} plots")
    print(f"  - Advanced: {len(list((plots_dir / 'advanced').glob('*.png')))} plots")
    print(f"  - Statistical: {len(list((plots_dir / 'statistical').glob('*.png')))} plots")

def save_results(models, evaluation_summary, models_dir, scaler, timestamp):
    """Save trained models and results."""
    print("\n" + "="*80)
    print("STEP 5: SAVING MODELS")
    print("="*80 + "\n")

    for name, model in models.items():
        model_path = models_dir / f"{name}_{timestamp}.joblib"
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")

    scaler_path = models_dir / f"scaler_{timestamp}.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")

    summary_path = models_dir / f"evaluation_summary_{timestamp}.csv"
    summary_df = pd.DataFrame(evaluation_summary).T
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

    plots_dir, models_dir, timestamp = create_output_directories()

    df = extract_features_from_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    models, evaluation_summary, best_model = train_models(X_train, y_train, X_test, y_test)
    generate_all_plots(X_train, X_test, y_train, y_test, models, evaluation_summary, plots_dir)
    save_results(models, evaluation_summary, models_dir, scaler, timestamp)

    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nResults saved:")
    print(f"  - Plots: {plots_dir}")
    print(f"  - Models: {models_dir}")
    print(f"  - Best model: {best_model}")
    print(f"\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
