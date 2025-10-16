"""
Complete Model Training Example

This script demonstrates the complete pipeline from feature extraction
through model training to evaluation and saving.

Author: Bimidu Gunathilake
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_extractor import FeatureExtractor
from src.preprocessing import DataPreprocessor, DataValidator
from src.models import ModelTrainer, ModelEvaluator, ModelRegistry, ModelConfig
from src.models.model_registry import ModelMetadata
from config import config


def main():
    """Complete training pipeline example."""
    
    print("\n" + "="*70)
    print("ASD DETECTION - MODEL TRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Extract features (or load existing)
    print("Step 1: Loading feature data...")
    
    # Option A: Load pre-extracted features
    features_file = config.paths.output_dir / "asd_vs_td_features.csv"
    
    if features_file.exists():
        print(f"Loading features from: {features_file}")
        import pandas as pd
        df = pd.read_csv(features_file)
    else:
        print("No features file found. Please run feature extraction first:")
        print("  python examples/example_usage.py")
        print("  Or manually extract features from your dataset")
        return
    
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns\n")
    
    # Step 2: Data Preprocessing
    print("Step 2: Preprocessing data...")
    
    preprocessor = DataPreprocessor(
        target_column='diagnosis',
        test_size=0.2,
        random_state=42,
        missing_strategy='median',
        outlier_method='clip',
        scaling_method='standard',
        feature_selection=True,
        n_features=30  # Select top 30 features
    )
    
    # Validate and preprocess
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df, validate=True)
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    preprocessor.print_summary()
    
    # Save preprocessor
    preprocessor_path = config.paths.models_dir / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)
    print(f"Preprocessor saved to: {preprocessor_path}\n")
    
    # Step 3: Train multiple models
    print("Step 3: Training models...")
    
    trainer = ModelTrainer()
    
    # Define models to train
    model_configs = [
        ModelConfig(model_type='random_forest', tune_hyperparameters=False),
        ModelConfig(model_type='xgboost', tune_hyperparameters=False),
        ModelConfig(model_type='lightgbm', tune_hyperparameters=False),
        ModelConfig(model_type='logistic', tune_hyperparameters=False),
    ]
    
    # Train all models
    models = trainer.train_multiple_models(X_train, y_train, model_configs)
    
    print(f"\nTrained {len(models)} models successfully\n")
    
    # Step 4: Evaluate models
    print("Step 4: Evaluating models...")
    
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    reports = evaluator.evaluate_multiple(
        models,
        X_test, y_test,
        X_train, y_train
    )
    
    # Print individual reports
    for name, report in reports.items():
        report.print_summary()
    
    # Compare models
    comparison_df = evaluator.compare_models()
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    print()
    
    # Save comparison
    comparison_path = config.paths.output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison saved to: {comparison_path}\n")
    
    # Step 5: Save best models to registry
    print("Step 5: Saving models to registry...")
    
    registry = ModelRegistry()
    
    # Save each model with metadata
    for model_name, model in models.items():
        report = reports[model_name]
        
        metadata = ModelMetadata(
            model_name=model_name,
            model_type=model_name,
            version="1.0.0",
            accuracy=report.accuracy,
            f1_score=report.f1_score,
            feature_names=preprocessor.selected_features_,
            n_features=len(preprocessor.selected_features_),
            training_samples=len(X_train),
            description=f"ASD detection model trained on pragmatic/conversational features"
        )
        
        registry.register_model(
            model=model,
            metadata=metadata,
            preprocessor=preprocessor
        )
        
        print(f"  ✓ {model_name} registered")
    
    print("\nAll models saved to registry\n")
    
    # Show registry summary
    registry.print_summary()
    
    # Step 6: Show best model
    print("Step 6: Best Model Selection...")
    
    best_name, best_metadata = registry.get_best_model()
    
    print(f"\nBest Model: {best_name}")
    print(f"  Type: {best_metadata.model_type}")
    print(f"  Accuracy: {best_metadata.accuracy:.4f}")
    print(f"  F1-Score: {best_metadata.f1_score:.4f}")
    print(f"  Features: {best_metadata.n_features}")
    
    # Step 7: Test prediction with best model
    print("\n" + "="*70)
    print("Testing Prediction with Best Model")
    print("="*70 + "\n")
    
    best_model, best_preprocessor = registry.load_model(best_name, load_preprocessor=True)
    
    # Take first few test samples
    sample_X = X_test.head(5)
    sample_y = y_test.head(5)
    
    predictions = best_model.predict(sample_X)
    probabilities = best_model.predict_proba(sample_X) if hasattr(best_model, 'predict_proba') else None
    
    print("Sample Predictions:")
    for i, (true_label, pred_label) in enumerate(zip(sample_y, predictions)):
        match = "✓" if true_label == pred_label else "✗"
        prob_str = f" (conf: {probabilities[i].max():.2f})" if probabilities is not None else ""
        print(f"  {i+1}. True: {true_label}, Predicted: {pred_label}{prob_str} {match}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Start the API: python -m uvicorn src.api.app:app --reload")
    print("  2. Access documentation: http://localhost:8000/docs")
    print("  3. Make predictions via API")
    print("\n")


if __name__ == "__main__":
    main()

