# Syntactic & Semantic Features - Usage Guide

**Status:** ✅ Fully Implemented
**Version:** 2.0.0
**Author:** Randil Haturusinghe
**Features:** 27 total

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Feature Extraction](#feature-extraction)
3. [Clinical Interpretation](#clinical-interpretation)
4. [Model Training](#model-training)
5. [Complete Examples](#complete-examples)
6. [Feature Reference](#feature-reference)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

Ensure you have the required dependencies:

```bash
pip install spacy nltk textstat
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### Basic Usage

```python
from src.features.syntactic_semantic import SyntacticSemanticFeatures
from src.parsers.chat_parser import parse_chat_file

# Extract features
extractor = SyntacticSemanticFeatures()
transcript = parse_chat_file("path/to/file.cha")
result = extractor.extract(transcript)

print(f"Extracted {len(result.features)} features")
print(result.features)
```

---

## Feature Extraction

### Step 1: Parse Transcript

```python
from src.parsers.chat_parser import parse_chat_file

# Parse CHAT format transcript
transcript = parse_chat_file("data/transcripts/sample.cha")

print(f"Total utterances: {len(transcript.utterances)}")
print(f"Child utterances: {len(transcript.child_utterances)}")
```

### Step 2: Initialize Extractor

```python
from src.features.syntactic_semantic import SyntacticSemanticFeatures

extractor = SyntacticSemanticFeatures()

# View all feature names
print(f"Feature count: {len(extractor.feature_names)}")
print("Features:")
for i, feature in enumerate(extractor.feature_names, 1):
    print(f"  {i}. {feature}")
```

### Step 3: Extract Features

```python
# Extract all 27 features
result = extractor.extract(transcript)

# Access features
features = result.features
metadata = result.metadata

print(f"\nFeature values:")
for name, value in features.items():
    print(f"  {name}: {value:.4f}")

print(f"\nMetadata:")
print(f"  Utterances analyzed: {metadata['num_child_utterances']}")
print(f"  Tokens analyzed: {metadata['num_tokens_analyzed']}")
```

### Step 4: Process Multiple Files

```python
from pathlib import Path
import pandas as pd

def extract_features_batch(transcript_dir):
    """Extract features from all transcripts in a directory."""
    extractor = SyntacticSemanticFeatures()
    all_features = []

    for cha_file in Path(transcript_dir).rglob("*.cha"):
        try:
            transcript = parse_chat_file(str(cha_file))
            result = extractor.extract(transcript)

            # Add filename to features
            result.features['filename'] = cha_file.name
            all_features.append(result.features)

            print(f"✓ Processed {cha_file.name}")
        except Exception as e:
            print(f"✗ Error processing {cha_file.name}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    return df

# Usage
features_df = extract_features_batch("data/transcripts/")
print(f"\nExtracted features from {len(features_df)} files")
features_df.to_csv("output/syntactic_semantic_features.csv", index=False)
```

---

## Clinical Interpretation

### Basic Interpretation

```python
from src.features.syntactic_semantic import ClinicalInterpreter

# Create interpreter
interpreter = ClinicalInterpreter()

# Interpret a single feature
interpretation = interpreter.interpret_feature(
    'avg_dependency_depth',
    2.1
)
print(interpretation)
```

**Output:**
```
Feature: Average Dependency Depth
Value: 2.100

What it measures:
Measures the average structural complexity of sentences through dependency tree depth

Typical patterns:
- ASD: Often lower (1.5-2.5) - simpler sentence structures with less embedding
- TD: Typically higher (2.5-4.0) - more complex, embedded sentence structures

Interpretation:
Lower values suggest simpler syntax. Values <2.0 may indicate reduced syntactic complexity

Clinical relevance:
Syntactic complexity is a marker of language development and cognitive-linguistic abilities
```

### Profile Interpretation

```python
# Extract features first
result = extractor.extract(transcript)

# Generate comprehensive interpretation
profile_report = interpreter.interpret_profile(result.features)
print(profile_report)
```

### Risk Assessment

```python
# Identify ASD risk indicators
risk_indicators = interpreter.get_asd_risk_indicators(result.features)

if risk_indicators:
    print(f"Found {len(risk_indicators)} potential ASD indicators:")
    for feature, reason, value in risk_indicators:
        print(f"\n  Feature: {feature}")
        print(f"  Value: {value:.3f}")
        print(f"  Concern: {reason}")
else:
    print("No strong ASD risk indicators detected.")

# Generate summary
summary = interpreter.generate_clinical_summary(result.features)
print(summary)
```

---

## Model Training

### Prepare Training Data

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load features with labels
df = pd.read_csv("output/syntactic_semantic_features.csv")

# Assume 'diagnosis' column exists (ASD or TD)
X = df.drop(['diagnosis', 'filename'], axis=1)
y = df['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")
```

### Train Single Model

```python
from src.models.syntactic_semantic import (
    SyntacticSemanticTrainer,
    SyntacticSemanticModelConfig
)

# Create trainer
trainer = SyntacticSemanticTrainer()

# Configure model
config = SyntacticSemanticModelConfig(
    model_type='lightgbm',
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.08
    }
)

# Train model
result = trainer.train_model(
    X_train, y_train,
    X_test, y_test,
    config=config,
    model_name="syntactic_semantic_lightgbm"
)

print(f"\nModel: {result['config'].model_type}")
print(f"Features: {result['feature_count']}")
print(f"Accuracy: {result['evaluation']['accuracy']:.3f}")
print(f"F1 Score: {result['evaluation']['f1_score']:.3f}")
```

### Train Multiple Models

```python
# Train all supported models (LightGBM and Gradient Boosting)
results = trainer.train_multiple_models(
    X_train, y_train,
    X_test, y_test
)

print("\nModel Comparison:")
print("-" * 60)
for model_name, eval_results in results['evaluation_summary'].items():
    print(f"{model_name}:")
    print(f"  Accuracy: {eval_results['accuracy']:.3f}")
    print(f"  F1 Score: {eval_results['f1_score']:.3f}")
    print(f"  Precision: {eval_results['precision']:.3f}")
    print(f"  Recall: {eval_results['recall']:.3f}")
    print()

print(f"Best Model: {results['best_model']}")
```

### Feature Importance

```python
# Get feature importance
importance_df = trainer.get_syntactic_semantic_feature_importance(
    model_name="syntactic_semantic_lightgbm",
    feature_names=X_train.columns.tolist(),
    top_n=15
)

print("\nTop 15 Most Important Features:")
print(importance_df.to_string(index=False))
```

### Save Model

```python
from pathlib import Path

# Save trained model
save_path = Path("models/syntactic_semantic_lightgbm/model.joblib")
trainer.save_model("syntactic_semantic_lightgbm", save_path)

print(f"Model saved to: {save_path}")
```

---

## Complete Examples

### Example 1: End-to-End Analysis

```python
from pathlib import Path
from src.parsers.chat_parser import parse_chat_file
from src.features.syntactic_semantic import (
    SyntacticSemanticFeatures,
    ClinicalInterpreter
)

def analyze_transcript(cha_file_path):
    """Complete analysis of a single transcript."""

    # 1. Parse transcript
    print(f"Parsing {cha_file_path}...")
    transcript = parse_chat_file(cha_file_path)

    # 2. Extract features
    print("Extracting features...")
    extractor = SyntacticSemanticFeatures()
    result = extractor.extract(transcript)

    # 3. Display features
    print("\n" + "="*70)
    print("FEATURE EXTRACTION RESULTS")
    print("="*70)
    print(f"File: {Path(cha_file_path).name}")
    print(f"Child utterances: {result.metadata['num_child_utterances']}")
    print(f"Tokens analyzed: {result.metadata['num_tokens_analyzed']}")

    print("\nKey Features:")
    key_features = [
        'avg_dependency_depth',
        'grammatical_error_rate',
        'semantic_coherence',
        'lexical_diversity_semantic'
    ]
    for feature in key_features:
        print(f"  {feature}: {result.features[feature]:.3f}")

    # 4. Clinical interpretation
    print("\n" + "="*70)
    print("CLINICAL INTERPRETATION")
    print("="*70)
    interpreter = ClinicalInterpreter()
    summary = interpreter.generate_clinical_summary(result.features)
    print(summary)

    return result

# Usage
result = analyze_transcript("data/transcripts/child_001.cha")
```

### Example 2: Batch Processing with Progress

```python
from tqdm import tqdm
import pandas as pd

def batch_extract_with_progress(transcript_dir, output_csv):
    """Extract features from all files with progress bar."""

    extractor = SyntacticSemanticFeatures()
    all_features = []

    # Get all .cha files
    cha_files = list(Path(transcript_dir).rglob("*.cha"))

    print(f"Found {len(cha_files)} transcript files")

    # Process with progress bar
    for cha_file in tqdm(cha_files, desc="Extracting features"):
        try:
            transcript = parse_chat_file(str(cha_file))
            result = extractor.extract(transcript)

            # Add metadata
            result.features['filename'] = cha_file.name
            result.features['file_path'] = str(cha_file)

            # Infer diagnosis from path (if available)
            if '/ASD/' in str(cha_file) or '_ASD_' in cha_file.name:
                result.features['diagnosis'] = 'ASD'
            elif '/TD/' in str(cha_file) or '_TD_' in cha_file.name:
                result.features['diagnosis'] = 'TD'

            all_features.append(result.features)

        except Exception as e:
            tqdm.write(f"Error processing {cha_file.name}: {e}")

    # Save to CSV
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)

    print(f"\n✓ Saved features to {output_csv}")
    print(f"  Total files: {len(df)}")
    print(f"  Total features: {len(df.columns)}")

    return df

# Usage
features_df = batch_extract_with_progress(
    "data/transcripts/",
    "output/features.csv"
)
```

### Example 3: Feature Validation

```python
def validate_features(features_df):
    """Validate extracted features for data quality."""

    print("="*70)
    print("FEATURE VALIDATION REPORT")
    print("="*70)

    # Check for missing values
    missing = features_df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️ Missing Values:")
        print(missing[missing > 0])
    else:
        print("\n✓ No missing values")

    # Check for infinite values
    numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
    infinite = features_df[numeric_cols].apply(lambda x: np.isinf(x).sum())
    if infinite.sum() > 0:
        print("\n⚠️ Infinite Values:")
        print(infinite[infinite > 0])
    else:
        print("✓ No infinite values")

    # Check value ranges
    print("\n📊 Feature Statistics:")
    print(features_df[numeric_cols].describe())

    # Check for zero variance
    zero_var = features_df[numeric_cols].var() == 0
    if zero_var.sum() > 0:
        print("\n⚠️ Zero Variance Features:")
        print(zero_var[zero_var].index.tolist())
    else:
        print("✓ All features have variance")

    return True

# Usage
validate_features(features_df)
```

---

## Feature Reference

### Syntactic Complexity (6 features)

| Feature | Range | ASD Pattern | TD Pattern |
|---------|-------|-------------|------------|
| avg_dependency_depth | 1-6 | 1.5-2.5 (lower) | 2.5-4.0 (higher) |
| max_dependency_depth | 1-15 | 3-6 (lower) | 6-10+ (higher) |
| avg_dependency_distance | 0-5 | 1.0-1.5 (lower) | 1.5-2.5 (higher) |
| clause_complexity | 0-2 | 0.1-0.4 (lower) | 0.4-1.0 (higher) |
| subordination_index | 0-1 | 0.1-0.3 (lower) | 0.3-0.6 (higher) |
| coordination_index | 0-1 | Variable | 0.2-0.5 (moderate) |

### Grammatical Accuracy (5 features)

| Feature | Range | ASD Pattern | TD Pattern |
|---------|-------|-------------|------------|
| grammatical_error_rate | 0-1 | 0.15-0.40 (higher) | 0.05-0.15 (lower) |
| tense_consistency_score | 0-1 | Variable | 0.6-0.8 (balanced) |
| tense_variety | 0-1 | 0.3-0.5 (lower) | 0.5-0.8 (higher) |
| structure_diversity | 0-1 | 0.3-0.6 (lower) | 0.6-0.9 (higher) |
| pos_tag_diversity | 0-0.2 | Lower | Higher |

### Semantic Features (4 features)

| Feature | Range | ASD Pattern | TD Pattern |
|---------|-------|-------------|------------|
| semantic_coherence | 0-1 | 0.2-0.5 (lower) | 0.5-0.8 (higher) |
| semantic_density | 0-10 | Variable | 3-6 (moderate) |
| lexical_diversity_semantic | 0-1 | 0.4-0.6 (lower) | 0.6-0.8 (higher) |
| thematic_consistency | 0-1 | Extreme (very high or low) | 0.3-0.6 (balanced) |

---

## Troubleshooting

### spaCy Model Not Found

**Error:** `OSError: [E050] Can't find model 'en_core_web_sm'`

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### NLTK Data Not Found

**Error:** `LookupError: Resource wordnet not found`

**Solution:**
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### Empty Feature Values

**Issue:** All features are 0.0

**Causes:**
- No valid child utterances in transcript
- Transcript parsing failed
- All utterances filtered out

**Solution:**
```python
# Check transcript before extraction
print(f"Total utterances: {len(transcript.utterances)}")
print(f"Valid child utterances: {len(transcript.child_utterances)}")
print(f"Valid utterances: {len(transcript.valid_utterances)}")

# Check first few utterances
for i, utt in enumerate(transcript.child_utterances[:5]):
    print(f"{i+1}. {utt.speaker}: {utt.text}")
```

### Memory Issues with Large Datasets

**Issue:** Out of memory when processing many files

**Solution:**
```python
# Process in batches
def process_in_batches(files, batch_size=100):
    extractor = SyntacticSemanticFeatures()

    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        batch_features = []

        for file in batch:
            result = extractor.extract(parse_chat_file(file))
            batch_features.append(result.features)

        # Save batch
        pd.DataFrame(batch_features).to_csv(
            f"output/batch_{i//batch_size}.csv",
            index=False
        )

        print(f"Processed batch {i//batch_size + 1}")
```

---

## Best Practices

### 1. Always Validate Input

```python
def safe_extract(cha_file):
    """Extract features with validation."""
    try:
        transcript = parse_chat_file(cha_file)

        # Validate transcript
        if len(transcript.child_utterances) == 0:
            print(f"⚠️ No child utterances in {cha_file}")
            return None

        if len(transcript.child_utterances) < 10:
            print(f"⚠️ Only {len(transcript.child_utterances)} utterances in {cha_file}")

        # Extract features
        extractor = SyntacticSemanticFeatures()
        result = extractor.extract(transcript)

        return result

    except Exception as e:
        print(f"✗ Error: {e}")
        return None
```

### 2. Save Intermediate Results

```python
# Save extracted features immediately
result = extractor.extract(transcript)
pd.DataFrame([result.features]).to_csv(
    f"output/features_{transcript_id}.csv",
    index=False
)
```

### 3. Use Logging

```python
import logging

logging.basicConfig(
    filename='feature_extraction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info(f"Processing {cha_file}")
logging.info(f"Extracted {len(result.features)} features")
```

---

## Additional Resources

- **Implementation Report:** `docs/SYNTACTIC_SEMANTIC_IMPLEMENTATION_REPORT.md`
- **Source Code:** `src/features/syntactic_semantic/`
- **Model Trainer:** `src/models/syntactic_semantic/`
- **Example Notebooks:** `examples/syntactic_semantic_examples.ipynb` (if available)

---

**Version:** 2.0.0
**Last Updated:** February 16, 2026
**Status:** ✅ Production Ready
