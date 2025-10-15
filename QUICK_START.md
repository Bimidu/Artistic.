# Quick Start Guide

Get started with ASD detection feature extraction in 5 minutes!

## 🚀 Setup (2 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# No configuration needed! Default settings work out of the box.
# Data directory is automatically set to ./data
# Output directory is automatically set to ./output
```

## 📝 Basic Usage (3 minutes)

### Option A: Run Examples (Easiest)

```bash
# Run the example script
python examples/example_usage.py
```

This will:
- Parse a sample CHAT file
- Extract all pragmatic & conversational features
- Display results

### Option B: Python Script

Create `test.py`:

```python
from src.parsers.chat_parser import CHATParser
from src.features.feature_extractor import FeatureExtractor

# 1. Parse a transcript
parser = CHATParser()
transcript = parser.parse_file("data/asdbank_eigsti/Eigsti/ASD/1010.cha")

# 2. Extract features
extractor = FeatureExtractor(categories='pragmatic_conversational')
features = extractor.extract_from_transcript(transcript)

# 3. Display results
print(f"\n{'='*60}")
print(f"Participant: {features.participant_id}")
print(f"Diagnosis: {features.diagnosis}")
print(f"Age: {features.age_months} months")
print(f"\nKey Features:")
print(f"  MLU (words): {features.features['mlu_words']:.2f}")
print(f"  Vocabulary (TTR): {features.features['type_token_ratio']:.2f}")
print(f"  Echolalia ratio: {features.features['echolalia_ratio']:.2%}")
print(f"  Question ratio: {features.features['question_ratio']:.2%}")
print(f"\nTotal features extracted: {len(features.features)}")
print(f"{'='*60}\n")
```

Run it:
```bash
python test.py
```

## 🎯 Common Tasks

### Task 1: Extract Features from One Dataset

```python
from src.features.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()

# Extract and save
df = extractor.extract_from_directory(
    directory='data/asdbank_eigsti/Eigsti/ASD',
    output_file='output/asd_features.csv'
)

print(f"Extracted features from {len(df)} transcripts")
print(f"Saved to: output/asd_features.csv")
```

### Task 2: Compare ASD vs TD

```python
from src.features.feature_extractor import FeatureExtractor
import pandas as pd

extractor = FeatureExtractor()

# Extract from both groups
asd_df = extractor.extract_from_directory('data/asdbank_eigsti/Eigsti/ASD')
td_df = extractor.extract_from_directory('data/asdbank_eigsti/Eigsti/TD')

# Add labels
asd_df['diagnosis'] = 'ASD'
td_df['diagnosis'] = 'TD'

# Combine
df = pd.concat([asd_df, td_df], ignore_index=True)

# Compare
print("\nComparison:")
print(df.groupby('diagnosis')[['mlu_words', 'echolalia_ratio', 'question_ratio']].mean())

# Save
df.to_csv('output/asd_vs_td_comparison.csv', index=False)
```

### Task 3: Build Complete Inventory

```python
from src.parsers.dataset_inventory import DatasetInventory

inventory = DatasetInventory()

# Build inventory for all datasets
inventory.build_inventory()

# Get summary
summary = inventory.get_dataset_summary()
for dataset, counts in summary.items():
    print(f"\n{dataset}:")
    for diagnosis, count in counts.items():
        print(f"  {diagnosis}: {count}")

# Export
inventory.export_to_csv('output/complete_inventory.csv')
```

## 📊 What Features Are Extracted?

### **61 Pragmatic & Conversational Features**

1. **Turn-Taking (15 features)**
   - `turns_per_minute`, `child_turn_ratio`
   - `avg_response_latency`, `child_initiation_ratio`

2. **Linguistic (14 features)**
   - `mlu_words`, `type_token_ratio`
   - `lexical_density`, `utterance_complexity_score`

3. **Pragmatic (16 features)**
   - `echolalia_ratio`, `pronoun_reversal_count`
   - `question_ratio`, `social_phrase_ratio`

4. **Conversational (16 features)**
   - `topic_shift_ratio`, `discourse_marker_ratio`
   - `self_repair_count`, `topic_relevance_score`

## 🔍 Check Feature Categories

```python
from src.features.feature_extractor import FeatureExtractor

extractor = FeatureExtractor()
extractor.print_category_info()
```

Output:
```
======================================================================
FEATURE EXTRACTION CATEGORIES
======================================================================

○ ACOUSTIC & PROSODIC
   Status: ○ PLACEHOLDER
   Team: Team Member A
   Description: Acoustic and prosodic features from audio

○ SYNTACTIC & SEMANTIC
   Status: ○ PLACEHOLDER
   Team: Team Member B
   Description: Syntactic and semantic features from text

● PRAGMATIC & CONVERSATIONAL
   Status: ✓ IMPLEMENTED
   Team: Current Implementation
   Description: Pragmatic and conversational features
   Features: 61
======================================================================
```

## 🐛 Troubleshooting

### Issue: "File not found"
```python
# Check if file exists
from pathlib import Path
file_path = Path("data/asdbank_eigsti/Eigsti/ASD/1010.cha")
print(f"File exists: {file_path.exists()}")
```

### Issue: "No module named 'src'"
```bash
# Make sure you're in the project root directory
cd /Users/bimidugunathilake/Documents/SE/Projects/Artistic.
```

### Issue: "All features are zero"
```python
# Check if transcript has valid child utterances
transcript = parser.parse_file("your_file.cha")
print(f"Total utterances: {transcript.total_utterances}")
print(f"Child utterances: {len(transcript.child_utterances)}")
print(f"Valid utterances: {len(transcript.valid_utterances)}")
```

## 📈 Next Steps

1. **Explore Examples**: Check `examples/example_usage.py` for more examples

2. **Read Documentation**: See `README.md` for comprehensive guide

3. **Integration**: See `IMPLEMENTATION_SUMMARY.md` for team integration

4. **Phase 3**: Ready for machine learning implementation!

## 💡 Pro Tips

### Tip 1: Use Progress Bars
```python
# Progress bars are enabled by default
df = extractor.extract_from_directory(
    directory='data/asdbank_eigsti',
    # show_progress=True  # default
)
```

### Tip 2: Normalize Features
```python
# Normalize before ML training
normalized_df = extractor.normalize_features(df, method='zscore')
# Options: 'zscore', 'minmax', 'robust'
```

### Tip 3: Get Summary Statistics
```python
summary = extractor.get_feature_summary(df)
print(f"Total samples: {summary['total_samples']}")
print(f"Feature count: {summary['feature_count']}")
print(f"Diagnosis distribution: {summary['diagnosis_counts']}")
```

### Tip 4: Cache Inventory
```python
# First run builds and caches
inventory.build_inventory()  # Takes time

# Subsequent runs load from cache
inventory.build_inventory()  # Fast!

# Force rebuild
inventory.build_inventory(force_rebuild=True)
```

## ✅ Verification

Run this to verify everything works:

```python
from src.parsers.chat_parser import CHATParser
from src.features.feature_extractor import FeatureExtractor

# Test parsing
parser = CHATParser()
print("✓ Parser initialized")

# Test feature extraction
extractor = FeatureExtractor()
print("✓ Feature extractor initialized")

# List features
print(f"✓ {len(extractor.all_feature_names)} features available")

print("\n✅ All systems operational!")
```

---

**You're all set!** 🎉

For detailed documentation, see:
- `README.md` - Complete guide
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `examples/example_usage.py` - Code examples

