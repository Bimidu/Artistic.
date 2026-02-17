# Component-Specific Model Architecture

## Overview

The ASD Detection System uses **component-specific model types** optimized for each feature category, with strong anti-overfitting measures for the pragmatic component.

## Component Model Mapping

### 1. Pragmatic & Conversational Component
**Allowed Models:** SVM, Logistic Regression

**Rationale:**
- Extracts 61 sophisticated features across:
  - Turn-taking dynamics (temporal patterns, gaps, overlaps)
  - Pragmatic linguistic markers (MLU, vocabulary, echolalia)
  - Topic coherence (semantic similarity, topic shifts)
  - Conversational management (repairs, pauses)
- Features are a **mix of temporal sequences + linguistic patterns + semantic relationships**
- **SVM** provides robust non-linear pattern recognition with RBF kernel; moderate regularization (C=2.0) balances model capacity and generalization
- **Logistic Regression** offers interpretability with L2 regularization; simpler linear model complements non-linear SVM
- Both models use `class_weight='balanced'` to handle potential class imbalance in ASD detection

**Anti-Overfitting Techniques:**
1. **Moderate regularization**:
   - SVM: C=2.0 (balanced regularization)
   - Logistic: C=1.0 with L2 penalty
2. **Class balancing**: `class_weight='balanced'` for imbalanced datasets
3. **Feature selection**: Recursive Feature Elimination with Cross-Validation (RFECV)
4. **Nested cross-validation**: Inner CV for hyperparameter tuning, outer CV for generalization estimates
5. **Learning curves**: Visual detection of overfitting with train/validation gap analysis

**Optimized Hyperparameters:**
```python
'svm': {
    'C': 2.0,                      # Moderate regularization (balanced)
    'kernel': 'rbf',               # Non-linear patterns
    'gamma': 0.01,                 # Manual gamma for better non-linear capture
    'probability': True,
    'class_weight': 'balanced',
    'random_state': 42,
    'cache_size': 500
}

'logistic': {
    'C': 1.0,                      # Moderate regularization (balanced)
    'penalty': 'l2',               # L2 regularization only
    'solver': 'lbfgs',             # Fast solver for L2
    'max_iter': 2000,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}
```

### 2. Acoustic & Prosodic Component
**Allowed Models:** XGBoost, Random Forest

**Rationale:**
- Extracts acoustic features from audio (pitch, energy, spectral characteristics)
- Features are **continuous numeric values** representing prosodic patterns
- **XGBoost** excels at handling continuous acoustic features with non-linearities
- **Random Forest** provides robustness to noise in prosodic features
- Tree-based models work well with spectral and pitch contour data

**Optimized Hyperparameters:**
```python
'xgboost': {
    'n_estimators': 120,
    'max_depth': 8,
    'learning_rate': 0.08,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.2,
    'reg_alpha': 0.4,
    'reg_lambda': 1.8,
    'random_state': 42,
    'n_jobs': -1,
    'eval_metric': 'logloss'
}

'random_forest': {
    'n_estimators': 150,
    'max_depth': 12,
    'min_samples_split': 8,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'bootstrap': True,
    'random_state': 42,
    'n_jobs': -1,
    'class_weight': 'balanced'
}
```

### 3. Syntactic & Semantic Component
**Allowed Models:** LightGBM, Gradient Boosting

**Rationale:**
- Currently uses dummy/placeholder syntactic features
- **LightGBM** (primary) is fast and memory-efficient for syntactic patterns
- **Gradient Boosting** (secondary) will handle non-linearities when real features are added
- Lightweight gradient boosting models appropriate for current feature implementation

**Optimized Hyperparameters:**
```python
'lightgbm': {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.08,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.3,
    'reg_lambda': 1.5,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

'gradient_boosting': {
    'n_estimators': 150,
    'learning_rate': 0.08,
    'max_depth': 3,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'subsample': 0.9,
    'random_state': 42
}
```

## Implementation Details

### Backend Changes

1. **Component Trainers** (`src/models/*/model_trainer.py`)
   - Added `ALLOWED_MODEL_TYPES` class variable to each trainer
   - Updated `_create_model()` to only instantiate allowed models
   - Refined hyperparameters for each component's specific needs
   - Added validation to reject invalid model types

2. **API Validation** (`src/api/app.py`)
   - Added `COMPONENT_MODEL_TYPES` mapping
   - New endpoint `/training/component-models` returns allowed models per component
   - Training endpoint validates model types against component restrictions
   - Returns clear error messages for invalid model selections

### Frontend Changes

1. **Dynamic Model Selection** (`frontend/index.html`, `frontend/app.js`)
   - Model checkboxes now populate dynamically based on selected component
   - Fetches allowed model types from API on page load
   - Updates UI when component selection changes
   - Shows component-specific model descriptions

2. **User Experience**
   - Clearer feedback about which models work with each component
   - Prevents invalid model selections before submission
   - Automatic default selection of recommended models

## Advanced Training Script (Pragmatic Component)

### `scripts/train_pragmatic_advanced.py`

This script implements comprehensive anti-overfitting techniques for the pragmatic component:

**Features:**
1. **Feature Selection**: RFECV (Recursive Feature Elimination with Cross-Validation)
   - Automatically selects optimal number of features
   - Reduces dimensionality to prevent overfitting
   - Minimum 20 features retained

2. **Nested Cross-Validation**:
   - Inner CV (3 folds): Hyperparameter tuning
   - Outer CV (5 folds): Generalization estimates
   - Stratified folds maintain class distribution

3. **Hyperparameter Search**:
   - RandomizedSearchCV with 30 iterations
   - Focused on regularization parameters
   - F1-weighted scoring metric

4. **Learning Curves**:
   - Visual plots showing train vs validation scores
   - Automatic overfitting detection (gap > 0.1)
   - Saved as PNG files for inspection

5. **Stratified Train/Test Split**:
   - 75/25 split with stratification
   - Test set held out for final evaluation
   - Class balance maintained

**Usage:**
```bash
python3 scripts/train_pragmatic_advanced.py
```

**Outputs:**
- `output/pragmatic_training/learning_curve_svm.png`
- `output/pragmatic_training/learning_curve_logistic.png`
- `output/pragmatic_training/pragmatic_training_results.json`

## API Usage

### Get Component Model Types
```bash
GET /training/component-models
```

**Response:**
```json
{
  "components": {
    "pragmatic_conversational": ["svm", "logistic"],
    "acoustic_prosodic": ["xgboost", "random_forest"],
    "syntactic_semantic": ["lightgbm", "gradient_boosting"]
  },
  "description": {
    "pragmatic_conversational": "SVM and Logistic Regression with strong regularization for generalization",
    "acoustic_prosodic": "XGBoost and Random Forest optimized for continuous acoustic features",
    "syntactic_semantic": "LightGBM and Gradient Boosting for syntactic patterns"
  }
}
```

### Train Models (with validation)
```bash
POST /training/train
```

**Request:**
```json
{
  "dataset_names": ["asdbank_rollins", "asdbank_aac"],
  "model_types": ["svm", "logistic"],
  "component": "pragmatic_conversational"
}
```

**Error Response (invalid model):**
```json
{
  "detail": "Invalid model types for component 'pragmatic_conversational': ['xgboost']. Allowed models for this component: ['svm', 'logistic']"
}
```

## Testing

### Run Component Tests
```bash
python3 scripts/test_component_models.py
```

**Tests verify:**
- Each component only trains its allowed model types
- Invalid model types are properly rejected
- Components have distinct model configurations
- Hyperparameters are component-specific
- All component isolation is maintained

**Expected Output:**
```
TESTING PRAGMATIC/CONVERSATIONAL TRAINER
Allowed model types: ['svm', 'logistic']
All pragmatic trainer tests passed

TESTING ACOUSTIC/PROSODIC TRAINER
Allowed model types: ['xgboost', 'random_forest']
All acoustic trainer tests passed

TESTING SYNTACTIC/SEMANTIC TRAINER
Allowed model types: ['lightgbm', 'gradient_boosting']
All syntactic trainer tests passed

ALL TESTS PASSED. Component-specific models working correctly.
```

## Benefits

1. **Optimized Performance**
   - Each component uses models best suited for its feature types
   - Component-specific hyperparameters improve accuracy
   - Reduced overfitting through tailored regularization

2. **Better Maintainability**
   - Clear separation of concerns
   - Easier to update individual components
   - Reduced complexity

3. **Improved User Experience**
   - Clearer guidance on model selection
   - Prevents confusion about which models to use
   - Better error messages

4. **Scientific Rigor**
   - Models chosen based on feature characteristics
   - Hyperparameters tuned for specific data types
   - Documented rationale for each choice

5. **Reduced Overfitting (Pragmatic)**
   - Strong L1+L2 regularization
   - Feature selection removes redundant features
   - Nested CV provides unbiased generalization estimates
   - Learning curves enable visual overfitting detection

## Migration Notes

**Breaking Changes:**
- Training requests with invalid model types will now be rejected
- Frontend requires API version 2.0.0+ for dynamic model types
- Existing saved models are still compatible

**Backwards Compatibility:**
- Existing trained models can still be loaded and used
- Model registry supports all previously trained models
- Prediction API unchanged

## Architecture Diagram

```
Component-Specific Model Selection
===================================

Pragmatic/Conversational Features
  -> SVM (RBF, C=0.5)
  -> Logistic Regression (ElasticNet, C=0.3)
     + Anti-overfitting techniques
     + Feature selection (RFECV)
     + Nested cross-validation
     + Learning curves

Acoustic/Prosodic Features
  -> XGBoost (n=120, depth=8)
  -> Random Forest (n=150, depth=12)
     + Optimized for continuous numeric data
     + Regularization via min_child_weight, gamma

Syntactic/Semantic Features
  -> LightGBM (n=100, depth=6)
  -> Gradient Boosting (n=150, depth=3)
     + Lightweight models for simple features
     + Ready to scale when real syntactic features added
```

## Future Work

1. **Syntactic Component**
   - Replace dummy features with real syntactic parsing
   - May add more sophisticated models (e.g., tree-based for parse trees)

2. **Ensemble Methods**
   - Component-specific ensemble strategies
   - Weighted voting based on component strengths

3. **Automated Tuning**
   - Automatic hyperparameter optimization per component
   - Bayesian optimization for better parameter search

4. **Model Explainability**
   - Component-specific SHAP visualizations
   - Feature importance by component

## References

- XGBoost Documentation: https://xgboost.readthedocs.io/
- Random Forest (sklearn): https://scikit-learn.org/stable/modules/ensemble.html#random-forests
- SVM (sklearn): https://scikit-learn.org/stable/modules/svm.html
- Elastic Net Regularization: Zou & Hastie (2005)
- Component Architecture Pattern: Separation of Concerns principle
- Feature Engineering for ASD Detection: Based on linguistic research (Wehrle 2023, Ellis et al. 2021)

---

**Last Updated:** February 15, 2026  
**Author:** Bimidu Gunathilake
