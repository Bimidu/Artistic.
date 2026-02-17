# Pragmatic Model Hyperparameter Update

## Issue
Both SVM and Logistic Regression models achieved identical F1 scores (65.4%), suggesting:
1. Over-regularization preventing models from learning effectively (underfitting)
2. Both models hitting the same performance ceiling due to too-restrictive constraints

## Root Cause Analysis

**Previous hyperparameters were too conservative:**

| Model | Previous C | Previous Penalty | Issue |
|-------|-----------|------------------|-------|
| SVM | 0.5 | N/A | Very strong regularization → underfitting |
| Logistic | 0.3 | ElasticNet (L1+L2) | Extremely strong regularization + feature elimination → underfitting |

**Why identical F1 scores?**
- Both models were so heavily regularized they couldn't capture meaningful patterns
- Both converged to similar decision boundaries (mostly linear/simple)
- The aggressive regularization effectively "flattened" the differences between SVM and Logistic Regression

## Changes Made

### 1. Updated Hyperparameters in `model_trainer.py`

**SVM:**
```python
# BEFORE
'svm': {
    'C': 0.5,           # Too strong regularization
    'gamma': 'scale',   # Auto-scaling may not be optimal
}

# AFTER
'svm': {
    'C': 2.0,           # Moderate regularization (4x increase)
    'gamma': 0.01,      # Manual gamma for better non-linear capture
}
```

**Logistic Regression:**
```python
# BEFORE
'logistic': {
    'C': 0.3,              # Too strong regularization
    'penalty': 'elasticnet',  # L1+L2 removes features aggressively
    'solver': 'saga',      # Required for ElasticNet
    'l1_ratio': 0.5,       # 50% L1 penalty
}

# AFTER
'logistic': {
    'C': 1.0,              # Moderate regularization (3.3x increase)
    'penalty': 'l2',       # L2 only (better convergence)
    'solver': 'lbfgs',     # Faster for L2
    # No l1_ratio needed
}
```

### 2. Key Improvements

**Increased Model Capacity:**
- SVM C increased from 0.5 → 2.0 (400% increase in model capacity)
- Logistic C increased from 0.3 → 1.0 (333% increase in model capacity)
- Both models can now learn more complex patterns

**Better Non-Linear Capture (SVM):**
- Changed gamma from `'scale'` to `0.01`
- Manual gamma gives more control over RBF kernel width
- Should improve SVM's ability to capture non-linear conversational patterns

**Simplified Logistic Regression:**
- Removed ElasticNet penalty (was too aggressive with feature selection)
- Pure L2 regularization is more stable and converges faster
- Changed solver from `saga` to `lbfgs` (faster for L2)

**Models Now Properly Distinct:**
- SVM: Non-linear RBF kernel with moderate regularization → captures complex patterns
- Logistic: Linear model with L2 regularization → simpler, more interpretable baseline
- Expected: SVM should now outperform Logistic on non-linear patterns

### 3. Updated Advanced Training Script

`scripts/train_pragmatic_advanced.py` now includes:

**SVM parameter search:**
```python
'C': [0.5, 1.0, 1.5, 2.0, 3.0, 5.0],        # Wider range
'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 'scale']  # More gamma options
```

**Logistic parameter search:**
```python
'C': [0.3, 0.5, 1.0, 2.0, 3.0, 5.0],   # Wider range
'penalty': ['l2'],                      # L2 only
'solver': ['lbfgs', 'saga'],           # Both solvers
```

### 4. Expected Performance Improvements

**F1 Score targets:**
- Previous: ~65.4% (both models)
- Expected: 70-78% (with proper model differentiation)

**Model differentiation:**
- SVM should excel at capturing non-linear turn-taking patterns
- Logistic should provide interpretable baseline (likely 2-5% lower F1 than SVM)

**Why this should work:**
1. Models now have capacity to learn meaningful patterns
2. SVM's RBF kernel can capture non-linear conversational dynamics
3. Regularization still present (C=2.0 and C=1.0) but not overly restrictive
4. Class balancing maintained to handle ASD/TD imbalance

## How to Test

### Quick Test (Use Updated Defaults)
```bash
# Train with new defaults via API or frontend
# Expected: Different F1 scores, likely 70-78% range
```

### Advanced Test (Hyperparameter Search)
```bash
# Run advanced training script for optimal parameters
python3 scripts/train_pragmatic_advanced.py

# This will:
# - Search over expanded parameter ranges
# - Use nested cross-validation
# - Generate learning curves
# - Save best parameters to JSON
```

## Trade-offs

**Pros:**
- Better model capacity → higher F1 scores expected
- Proper differentiation between SVM and Logistic
- Still maintains class balancing for imbalance handling
- Faster training (lbfgs faster than saga for logistic)

**Cons:**
- Slightly increased risk of overfitting (mitigated by still-moderate regularization)
- May need more training data to reach full potential
- SVM training may take slightly longer with manual gamma

**Mitigation:**
- Still using moderate regularization (not aggressive)
- Cross-validation still recommended
- Learning curves in advanced script detect overfitting
- Can always retune if overfitting occurs

## Next Steps

1. **Retrain models** with new hyperparameters
2. **Compare F1 scores** - should now be different and higher
3. **If F1 still < 70%**, run advanced tuning script:
   ```bash
   python3 scripts/train_pragmatic_advanced.py
   ```
4. **If overfitting detected**, slightly reduce C values (e.g., SVM: 1.5, Logistic: 0.7)

## Summary

Changed from **overly conservative** (underfitting) to **balanced** hyperparameters:
- SVM: More capacity to learn non-linear patterns
- Logistic: Simpler, faster, more stable
- Expected: 5-12% F1 improvement, models now properly distinct

---

**Date:** February 15, 2026  
**Status:** Ready for testing
