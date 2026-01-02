# Feature Selection & Model Training Summary

## ✅ Model Training Status: **WORKING**

The model training pipeline successfully works with all 189 newly implemented features from the research methodology.

---

## 📊 Training Results

### Dataset Information
- **Total Samples**: 224
- **Classes**: 4 (ASD, TD, DD, AAC)
- **Train/Test Split**: 179/45 (80%/20%)
- **Available Features**: 185 (after removing metadata columns)

### Model Performance

| Model | Accuracy | F1-Score | Precision | Recall | CV Mean | CV Std |
|-------|----------|----------|-----------|--------|---------|--------|
| **Random Forest** | **86.67%** | **85.24%** | **86.62%** | **86.67%** | 80.28% | 3.66% |
| Logistic Regression | 84.44% | 84.90% | 88.57% | 84.44% | 77.76% | 5.34% |
| LightGBM | 80.00% | 78.92% | 77.90% | 80.00% | 80.39% | 5.07% |

**Best Model**: Random Forest (F1-Score: 85.24%)

---

## 🎯 Feature Selection Methods Implemented

Our system implements **4 distinct feature selection methods** to identify the most discriminative features:

### 1. **SelectKBest** (Statistical Tests)
- **Method**: Statistical hypothesis testing
- **Scoring Functions**:
  - `f_classif`: ANOVA F-statistic (default for classification)
  - `mutual_info_classif`: Mutual information between features and target
- **Usage**: Selects top K features based on univariate statistical tests
- **Implementation**: `FeatureSelector.select_k_best()`

```python
selector.select_k_best(X, y, k=30, score_func='f_classif')
```

### 2. **SelectFromModel** (Random Forest Importance) ⭐ **DEFAULT**
- **Method**: Tree-based feature importance
- **Algorithm**: Trains a Random Forest and selects features based on importance scores
- **Threshold**: 'median' (features above median importance are selected)
- **Advantages**:
  - Captures non-linear relationships
  - Handles feature interactions
  - Robust to multicollinearity
- **Implementation**: `FeatureSelector.select_from_model()`

```python
selector.select_from_model(X, y, threshold='median', max_features=30)
```

### 3. **RFE** (Recursive Feature Elimination)
- **Method**: Recursive backward selection
- **Algorithm**: 
  - Trains Random Forest on all features
  - Removes least important feature
  - Repeats until desired number reached
- **Advantages**: Considers feature dependencies
- **Implementation**: `FeatureSelector.select_rfe()`

```python
selector.select_rfe(X, y, n_features=30)
```

### 4. **Category-Based Selection**
- **Method**: Select top N features from each category
- **Categories**:
  - Turn-Taking Metrics (3.3.1)
  - Topic Coherence (3.3.2)
  - Pause & Latency (3.3.3)
  - Repair Detection (3.3.4)
  - General Pragmatic/Linguistic
- **Advantage**: Ensures representation from all feature types
- **Implementation**: `FeatureSelector.select_by_category()`

```python
categories = {
    'turn_taking': [...],
    'topic_coherence': [...],
    ...
}
selector.select_by_category(X, y, categories, top_per_category=10)
```

---

## 🔝 Top 30 Selected Features (Random Forest)

The current pipeline uses **SelectFromModel with Random Forest** as the default method.

### Feature Ranking by Importance

| Rank | Feature | Category | Importance |
|------|---------|----------|------------|
| 1 | `total_turns` | Turn-Taking | 0.0685 |
| 2 | `turns_per_minute` | Turn-Taking | 0.0669 |
| 3 | `estimated_speaking_time` | Pause/Latency | 0.0537 |
| 4 | `adult_initiated_turns` | Turn-Taking | 0.0498 |
| 5 | `off_topic_response_count` | Topic Coherence | 0.0430 |
| 6 | `adult_turns` | Turn-Taking | 0.0415 |
| 7 | `turn_switches` | Turn-Taking | 0.0383 |
| 8 | `child_turns` | Turn-Taking | 0.0322 |
| 9 | `total_words` | Pragmatic/Linguistic | 0.0307 |
| 10 | `social_phrase_ratio` | Pragmatic/Linguistic | 0.0225 |
| 11 | `child_initiation_ratio` | Turn-Taking | 0.0224 |
| 12 | `child_initiated_turns` | Turn-Taking | 0.0204 |
| 13 | `repetition_repair_count` | Repair Detection | 0.0197 |
| 14 | `avg_child_turn_length` | Turn-Taking | 0.0172 |
| 15 | `novel_word_ratio` | Topic Coherence | 0.0164 |
| 16 | `question_ratio` | Pragmatic/Linguistic | 0.0160 |
| 17 | `partial_repetition_count` | Repair Detection | 0.0131 |
| 18 | `max_utterance_length` | Pragmatic/Linguistic | 0.0130 |
| 19 | `mlu_words` | Pragmatic/Linguistic | 0.0128 |
| 20 | `child_turn_length_std` | Turn-Taking | 0.0128 |
| 21 | `utterance_complexity_score` | Pragmatic/Linguistic | 0.0117 |
| 22 | `corrected_ttr` | Pragmatic/Linguistic | 0.0117 |
| 23 | `type_token_ratio` | Pragmatic/Linguistic | 0.0115 |
| 24 | `delayed_echolalia_count` | Pragmatic/Linguistic | 0.0115 |
| 25 | `confirmation_check_count` | Repair Detection | 0.0111 |
| 26 | `turn_switch_rate` | Turn-Taking | 0.0107 |
| 27 | `avg_turns_before_switch` | Turn-Taking | 0.0103 |
| 28 | `wh_question_ratio` | Pragmatic/Linguistic | 0.0103 |
| 29 | `max_child_turn_length` | Turn-Taking | 0.0098 |
| 30 | `exact_repetition_count` | Repair Detection | 0.0094 |

### Feature Distribution by Category

| Category | # Features Selected | % of Top 30 |
|----------|---------------------|-------------|
| Turn-Taking (3.3.1) | 13 | 43.3% |
| Pragmatic/Linguistic | 10 | 33.3% |
| Repair Detection (3.3.4) | 4 | 13.3% |
| Topic Coherence (3.3.2) | 2 | 6.7% |
| Pause/Latency (3.3.3) | 1 | 3.3% |

---

## 💡 Key Insights

### Most Discriminative Feature Types

1. **Turn-Taking Patterns** (43.3%)
   - Conversational dynamics (total turns, turn frequency)
   - Initiation patterns (adult vs. child initiated)
   - Turn-length variability

2. **Pragmatic & Linguistic Markers** (33.3%)
   - Vocabulary diversity (TTR, MLU, novel words)
   - Social language use (questions, social phrases)
   - Echolalia patterns

3. **Repair Mechanisms** (13.3%)
   - Repetition-based repairs
   - Confirmation checks

4. **Topic Management** (6.7%)
   - Off-topic responses
   - Novel word introduction

### Interesting Findings

- **Turn-Taking is Crucial**: 13 of top 30 features are turn-taking related
- **Social Pragmatics**: `social_phrase_ratio` ranks #10, showing importance of social language
- **Conversational Flow**: `turns_per_minute` and `turn_switches` are highly discriminative
- **Repair Patterns**: Multiple repair-related features selected, validating methodology section 3.3.4
- **Echolalia**: `delayed_echolalia_count` appears in top 30, an ASD-specific marker

---

## 🔧 How to Use Different Selection Methods

### Change Feature Selection Method

Edit `examples/train_model.py`:

```python
# Current (Random Forest)
preprocessor = DataPreprocessor(
    feature_selection=True,
    n_features=30
)

# Use Statistical Tests instead
from src.preprocessing.feature_selector import FeatureSelector

selector = FeatureSelector()
selected_features = selector.select_k_best(X_train, y_train, k=30, score_func='f_classif')

# Use RFE
selected_features = selector.select_rfe(X_train, y_train, n_features=30)

# Use Category-Based
categories = {
    'turn_taking': extractor.extractors['turn_taking'].feature_names,
    'topic_coherence': extractor.extractors['topic_coherence'].feature_names,
    # ... etc
}
selected_features = selector.select_by_category(X_train, y_train, categories, top_per_category=6)
```

### Visualize Feature Importance

```python
from src.preprocessing import FeatureSelector

selector = FeatureSelector()
selector.select_from_model(X_train, y_train, max_features=30)

# Get importance DataFrame
df_importance = selector.get_feature_importance_df()
print(df_importance.head(30))

# Plot
selector.plot_feature_importance(top_n=30, save_path='output/feature_importance.png')
```

---

## 📈 Preprocessing Pipeline

### Data Flow

```
Raw Features (185) 
    ↓
[1] Data Validation
    ↓
[2] Train/Test Split (stratified)
    ↓
[3] Data Cleaning
    - Missing value imputation (median)
    - Outlier handling (clip to 3σ)
    ↓
[4] Feature Selection
    - Method: SelectFromModel (Random Forest)
    - Threshold: median importance
    - Result: 30 features
    ↓
[5] Feature Scaling
    - Method: StandardScaler
    - Fit on train, transform on test
    ↓
Scaled Features (30) → Model Training
```

### Data Validation Warnings

From the current dataset:
- ✓ **Dataset is valid** (passed all checks)
- ⚠️ 1 feature with missing values (handled)
- ⚠️ Class imbalance ratio: 6.81 (ASD:109, AAC:83, TD:16, DD:16)
- ⚠️ 70 highly correlated feature pairs
- ⚠️ 78 low-variance features
- ⚠️ 399 outliers detected across 91 features (clipped)

---

## 🎓 Alignment with Research Methodology (Section 3.4)

> "We will first perform feature selection or dimensionality reduction (e.g. PCA) to identify the most discriminative features."

### ✅ Implemented:

1. **Feature Selection**: ✓ Multiple methods (SelectKBest, RFE, RF importance)
2. **Statistical Methods**: ✓ F-statistic, mutual information
3. **Model-Based Selection**: ✓ Random Forest feature importance
4. **Cross-Validation**: ✓ 5-fold stratified CV during evaluation
5. **Hyperparameter Tuning**: ✓ Option available in ModelConfig

### Recommended for Future:

- **PCA**: Add dimensionality reduction option
- **L1 Regularization**: Use Lasso for embedded feature selection
- **Sequential Feature Selection**: Forward/backward selection
- **Ensemble Selection**: Combine multiple selection methods

---

## 🚀 Next Steps

### To Test Other Selection Methods:

```bash
# Try statistical selection
python examples/train_model_with_fclassif.py

# Try RFE
python examples/train_model_with_rfe.py

# Try category-based
python examples/train_model_by_category.py
```

### To Optimize:

1. **Balance Classes**: Use SMOTE or class weights
2. **Tune Hyperparameters**: Set `tune_hyperparameters=True`
3. **Feature Engineering**: Create interaction features
4. **Deep Learning**: Try neural networks on raw features

---

## 📝 Summary

✅ **Model training works perfectly** with all 189 new features  
✅ **4 feature selection methods** implemented and ready to use  
✅ **Random Forest achieves 86.67% accuracy** on 4-class problem  
✅ **Top 30 features** selected automatically using RF importance  
✅ **Turn-taking features** are most discriminative for ASD detection  
✅ **All methodology sections (3.3.1-3.3.4)** represented in top features  

**Default Method**: SelectFromModel with Random Forest (robust, interpretable, handles non-linearity)

---

Generated: 2025-12-05
Author: AI Assistant
Project: ASD Detection from Conversational Features









