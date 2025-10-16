"""
How to Use Extracted Features

This script demonstrates how to load and use the extracted features
from the output directory for machine learning and analysis.

Author: Bimidu Gunathilake
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.pragmatic_conversational.model_trainer import PragmaticConversationalTrainer
from src.preprocessing.preprocessor import DataPreprocessor
from config import config


def load_combined_features():
    """
    Load the combined features from all datasets
    """
    print("[CHART] Loading Combined Features...")
    
    features_file = config.paths.output_dir / "all_asdbank_features.csv"
    
    if not features_file.exists():
        print(f"[X] Features file not found: {features_file}")
        print("Run example_usage.py first to extract features")
        return None
    
    df = pd.read_csv(features_file)
    print(f"[CHECK] Loaded {len(df)} samples with {len(df.columns)} features")
    
    return df


def analyze_feature_distribution(df):
    """
    Analyze the distribution of features across diagnosis groups
    """
    print("\n[GRAPH] Feature Distribution Analysis")
    print("="*50)
    
    # Diagnosis distribution
    print("\n[DIAGNOSIS] Diagnosis Distribution:")
    diagnosis_counts = df['diagnosis'].value_counts()
    for diagnosis, count in diagnosis_counts.items():
        print(f"   {diagnosis}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # Dataset distribution
    print("\n[DATASET] Dataset Distribution:")
    dataset_counts = df['dataset'].value_counts()
    for dataset, count in dataset_counts.items():
        print(f"   {dataset}: {count} samples")
    
    # Feature statistics
    print("\n[CHART] Feature Statistics:")
    feature_cols = [col for col in df.columns if col not in ['participant_id', 'file_path', 'diagnosis', 'dataset']]
    
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Features with non-zero values: {len([col for col in feature_cols if df[col].sum() > 0])}")
    print(f"   Features with all zeros: {len([col for col in feature_cols if df[col].sum() == 0])}")


def prepare_ml_data(df):
    """
    Prepare data for machine learning
    """
    print("\n[ML] Preparing Data for Machine Learning...")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['participant_id', 'file_path', 'diagnosis', 'dataset']]
    X = df[feature_cols]
    y = df['diagnosis']
    
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Unique labels: {y.unique()}")
    
    return X, y


def train_models_with_features():
    """
    Train machine learning models using extracted features
    """
    print("\n[RUN] Training Models with Extracted Features...")
    
    # Load features
    df = load_combined_features()
    if df is None:
        return
    
    # Analyze distribution
    analyze_feature_distribution(df)
    
    # Prepare ML data
    X, y = prepare_ml_data(df)
    
    # Filter out rows with all-zero features (no valid utterances)
    valid_mask = X.sum(axis=1) > 0
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    
    print(f"\n[CHART] Valid samples for training: {len(X_valid)} (filtered from {len(X)})")
    
    if len(X_valid) == 0:
        print("[X] No valid samples found for training")
        return
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Preprocess features
    print("\n[BROOM] Preprocessing features...")
    X_processed = preprocessor.preprocess(X_valid)
    
    print(f"   Processed features shape: {X_processed.shape}")
    
    # Initialize trainer
    trainer = PragmaticConversationalTrainer()
    
    # Train models
    print("\n[TARGET] Training models...")
    models = trainer.train_multiple_models(X_processed, y_valid)
    
    print(f"[CHECK] Trained {len(models)} models successfully!")
    
    return models, X_processed, y_valid


def load_individual_dataset_features():
    """
    Load features from individual datasets for specific analysis
    """
    print("\n[DIR] Loading Individual Dataset Features...")
    
    datasets = [
        ("eigsti_asd", "Eigsti ASD Features"),
        ("eigsti_td", "Eigsti TD Features"),
        ("nadig", "Nadig Features"),
        ("aac", "AAC Features"),
        ("rollins", "Rollins Features")
    ]
    
    for dataset_name, description in datasets:
        file_path = config.paths.output_dir / f"{dataset_name}_features.csv"
        
        if file_path.exists():
            df = pd.read_csv(file_path)
            print(f"   [CHECK] {description}: {len(df)} samples")
        else:
            print(f"   [X] {description}: File not found")


def compare_datasets():
    """
    Compare features across different datasets
    """
    print("\n[SEARCH] Comparing Datasets...")
    
    df = load_combined_features()
    if df is None:
        return
    
    # Compare key features across datasets
    key_features = ['mlu_words', 'type_token_ratio', 'turns_per_minute', 'echolalia_ratio']
    
    print(f"\n[CHART] Key Feature Comparison by Dataset:")
    print(f"{'Dataset':<20} {'Samples':<10} {'MLU':<10} {'TTR':<10} {'Turns/min':<12} {'Echolalia':<12}")
    print("-" * 80)
    
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        mlu_mean = subset['mlu_words'].mean() if 'mlu_words' in subset.columns else 0
        ttr_mean = subset['type_token_ratio'].mean() if 'type_token_ratio' in subset.columns else 0
        turns_mean = subset['turns_per_minute'].mean() if 'turns_per_minute' in subset.columns else 0
        echo_mean = subset['echolalia_ratio'].mean() if 'echolalia_ratio' in subset.columns else 0
        
        print(f"{dataset:<20} {len(subset):<10} {mlu_mean:<10.3f} {ttr_mean:<10.3f} {turns_mean:<12.3f} {echo_mean:<12.3f}")


def export_features_for_external_analysis():
    """
    Export features in different formats for external analysis tools
    """
    print("\n[DISK] Exporting Features for External Analysis...")
    
    df = load_combined_features()
    if df is None:
        return
    
    # Export to different formats
    export_dir = config.paths.output_dir / "exports"
    export_dir.mkdir(exist_ok=True)
    
    # CSV for Excel/R
    csv_path = export_dir / "features_for_analysis.csv"
    df.to_csv(csv_path, index=False)
    print(f"   [CHECK] CSV exported: {csv_path}")
    
    # JSON for web applications
    json_path = export_dir / "features_for_analysis.json"
    df.to_json(json_path, orient='records', indent=2)
    print(f"   [CHECK] JSON exported: {json_path}")
    
    # Separate files by diagnosis
    for diagnosis in df['diagnosis'].unique():
        subset = df[df['diagnosis'] == diagnosis]
        diagnosis_path = export_dir / f"features_{diagnosis.lower()}.csv"
        subset.to_csv(diagnosis_path, index=False)
        print(f"   [CHECK] {diagnosis} features exported: {diagnosis_path}")


def main():
    """Main function demonstrating feature usage"""
    
    print("[TARGET] EXTRACTED FEATURES USAGE EXAMPLES")
    print("="*50)
    
    # 1. Load and analyze features
    df = load_combined_features()
    if df is None:
        return
    
    # 2. Analyze distribution
    analyze_feature_distribution(df)
    
    # 3. Load individual datasets
    load_individual_dataset_features()
    
    # 4. Compare datasets
    compare_datasets()
    
    # 5. Export for external analysis
    export_features_for_external_analysis()
    
    # 6. Train models (optional - uncomment to run)
    # train_models_with_features()
    
    print("\n[CHECK] Feature usage examples completed!")
    print(f"[DIR] All features stored in: {config.paths.output_dir}")
    print(f"[CHART] Main combined file: all_asdbank_features.csv")
    print(f"[ML] Models will be stored in: {config.paths.models_dir}")


if __name__ == "__main__":
    main()
