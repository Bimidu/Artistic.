# Component-Specific Model Update Summary

## Changes Implemented

### 1. New Component Model Mapping

All three components now use distinct, optimized model types:

| Component | Old Models | New Models | Rationale |
|-----------|-----------|------------|-----------|
| **Pragmatic/Conversational** | XGBoost, Random Forest | **SVM, Logistic Regression** | Strong regularization to prevent overfitting on mixed temporal/linguistic features |
| **Acoustic/Prosodic** | SVM, LightGBM | **XGBoost, Random Forest** | Tree-based models excel at continuous acoustic features |
| **Syntactic/Semantic** | Logistic, Gradient Boosting | **LightGBM, Gradient Boosting** | Fast gradient boosting for syntactic patterns |

### 2. Anti-Overfitting Measures (Pragmatic Component)

The pragmatic component now includes comprehensive techniques to reduce overfitting:

- **Strong L1+L2 Regularization**:
  - SVM: C=0.5 (vs previous C=1.0 in other components)
  - Logistic: C=0.3 with ElasticNet penalty (l1_ratio=0.5)

- **Feature Selection**: 
  - RFECV (Recursive Feature Elimination with Cross-Validation)
  - Automatically selects optimal number of features

- **Nested Cross-Validation**:
  - Inner 3-fold CV for hyperparameter tuning
  - Outer 5-fold CV for unbiased generalization estimates

- **Learning Curves**:
  - Visual plots showing train vs validation performance
  - Automatic overfitting detection (gap > 0.1)

- **Class Balancing**:
  - `class_weight='balanced'` for both models

### 3. Files Modified

**Backend:**
- `src/models/pragmatic_conversational/model_trainer.py`
  - Changed `ALLOWED_MODEL_TYPES` to `['svm', 'logistic']`
  - Updated hyperparameters with strong regularization
  - Removed XGBoost and Random Forest

- `src/models/acoustic_prosodic/model_trainer.py`
  - Changed `ALLOWED_MODEL_TYPES` to `['xgboost', 'random_forest']`
  - Updated hyperparameters for acoustic features
  - Removed SVM and LightGBM

- `src/models/syntactic_semantic/model_trainer.py`
  - Changed `ALLOWED_MODEL_TYPES` to `['lightgbm', 'gradient_boosting']`
  - Updated hyperparameters
  - Removed Logistic Regression

**API:**
- `src/api/app.py`
  - Updated `COMPONENT_MODEL_TYPES` mapping
  - Updated `/training/component-models` endpoint descriptions

**Frontend:**
- `frontend/index.html`
  - Updated default model checkboxes to SVM and Logistic
  
- `frontend/app.js`
  - Updated model label for SVM (removed "RBF" suffix)

**Scripts:**
- `scripts/test_component_models.py`
  - Updated all tests to reflect new model types
  - Verified component isolation

- **NEW**: `scripts/train_pragmatic_advanced.py`
  - Comprehensive anti-overfitting training script
  - Nested CV, feature selection, learning curves

**Documentation:**
- `COMPONENT_SPECIFIC_MODELS.md`
  - Complete rewrite with new model types
  - Added anti-overfitting techniques section
  - Added advanced training script documentation

### 4. Test Results

All tests pass successfully:

```
TESTING PRAGMATIC/CONVERSATIONAL TRAINER
Allowed model types: ['svm', 'logistic']
✓ SVM training successful
✓ Logistic Regression training successful
✓ Correctly rejected invalid model (xgboost)
All pragmatic trainer tests passed

TESTING ACOUSTIC/PROSODIC TRAINER
Allowed model types: ['xgboost', 'random_forest']
✓ XGBoost training successful
✓ Random Forest training successful
All acoustic trainer tests passed

TESTING SYNTACTIC/SEMANTIC TRAINER
Allowed model types: ['lightgbm', 'gradient_boosting']
✓ LightGBM training successful
✓ Gradient Boosting training successful
✓ Correctly rejected invalid model (xgboost)
All syntactic trainer tests passed

TESTING COMPONENT ISOLATION
✓ Component model sets are distinct
All component isolation tests passed

ALL TESTS PASSED. Component-specific models working correctly.
```

### 5. How to Use

**Standard Training (via API):**
```bash
# Train pragmatic models
curl -X POST "http://localhost:8000/training/train" \
  -H "Content-Type: application/json" \
  -d '{
    "component": "pragmatic_conversational",
    "model_types": ["svm", "logistic"],
    "datasets": ["aac", "nadig", "rollins"]
  }'
```

**Advanced Training (pragmatic only, with anti-overfitting):**
```bash
python3 scripts/train_pragmatic_advanced.py
```

This will:
- Perform feature selection (RFECV)
- Run nested cross-validation
- Generate learning curves
- Detect overfitting
- Save results to `output/pragmatic_training/`

**Test Component Models:**
```bash
python3 scripts/test_component_models.py
```

### 6. Key Benefits

1. **Reduced Overfitting**: Strong regularization and feature selection for pragmatic component
2. **Better Generalization**: Nested CV provides unbiased performance estimates
3. **Distinct Model Types**: Each component uses models suited to its feature characteristics
4. **Visual Feedback**: Learning curves show train/validation gap
5. **Maintainability**: Clear separation of concerns across components

### 7. Breaking Changes

- Pragmatic component no longer supports XGBoost or Random Forest
- Acoustic component no longer supports SVM or LightGBM
- Syntactic component no longer supports Logistic Regression
- API will reject invalid model types with clear error messages

### 8. Next Steps (Optional)

1. Run advanced training script on real pragmatic features:
   ```bash
   python3 scripts/train_pragmatic_advanced.py
   ```

2. Inspect learning curves in `output/pragmatic_training/`

3. Review overfitting metrics in `pragmatic_training_results.json`

4. If overfitting is still high:
   - Increase regularization (lower C values)
   - Select fewer features
   - Collect more training data

---

**Status**: All changes implemented and tested successfully
**Date**: February 15, 2026
