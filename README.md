# ASD Detection from Conversational Features

A comprehensive system for detecting Autism Spectrum Disorder (ASD) in children through analysis of conversational and pragmatic features extracted from CHAT-formatted transcripts.

## 📋 Project Overview

This project implements **Phase 1 & 2** of an ASD detection system that analyzes conversational patterns in children. The system extracts pragmatic and conversational features from TalkBank ASDBank datasets to identify patterns that distinguish ASD from typically developing (TD) children.

### Feature Categories

The complete system is designed for **three categories of features**:

1. **🎵 Acoustic & Prosodic Features** (Team Member A - Placeholder)
   - Pitch variations, speech rate, pause patterns
   - *Status: To be implemented*

2. **📝 Syntactic & Semantic Features** (Team Member B - Placeholder)
   - Grammar structures, semantic relationships
   - *Status: To be implemented*

3. **💬 Pragmatic & Conversational Features** (✅ **FULLY IMPLEMENTED**)
   - Turn-taking patterns
   - Linguistic complexity (MLU, vocabulary diversity)
   - Pragmatic language (echolalia, questions, pronouns)
   - Conversational management (topic, discourse, repairs)
   - Behavioral/non-verbal markers

## 🗂️ Dataset Information

### Available Datasets
- **asdbank_aac**: 18 minimally speaking autistic children (Canada)
- **asdbank_eigsti**: 16 ASD + 16 TD + 16 DD children (Rochester)
- **asdbank_flusberg**: 6 autistic children, longitudinal (Amherst)
- **asdbank_nadig**: 20 ASD + 18 TYP children (Montreal, bilingual)
- **asdbank_quigley_mcnalley**: 105 HR + 98 LR children
- **asdbank_rollins**: 5 autistic children

### Key Features Extracted

#### Turn-Taking Features (15 features)
- Turn frequency and distribution
- Response latency
- Turn initiation patterns
- Turn switching behavior

#### Linguistic Features (14 features)
- MLU (Mean Length of Utterance) in words & morphemes
- Vocabulary diversity (Type-Token Ratio)
- Grammatical complexity
- Lexical density

#### Pragmatic Features (16 features)
- Echolalia detection (immediate & delayed)
- Question usage patterns
- Pronoun usage and reversal
- Social language markers
- Response appropriateness

#### Conversational Features (16 features)
- Topic management and shifts
- Discourse marker usage
- Conversational repair strategies
- Non-verbal behavioral markers
- Topic relevance

**Total: 61+ pragmatic & conversational features**

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   cd /Users/bimidugunathilake/Documents/SE/Projects/Artistic.
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## 📖 Usage

### Example 1: Parse a Single CHAT File

```python
from src.parsers.chat_parser import CHATParser

# Initialize parser
parser = CHATParser()

# Parse a transcript
transcript = parser.parse_file("data/asdbank_eigsti/Eigsti/ASD/1010.cha")

print(f"Participant: {transcript.participant_id}")
print(f"Diagnosis: {transcript.diagnosis}")
print(f"Total Utterances: {transcript.total_utterances}")
```

### Example 2: Build Dataset Inventory

```python
from src.parsers.dataset_inventory import DatasetInventory

# Initialize inventory
inventory = DatasetInventory()

# Build complete inventory
inventory.build_inventory(datasets=['asdbank_eigsti'])

# Get summary
summary = inventory.get_dataset_summary()
df = inventory.to_dataframe()

# Export
inventory.export_to_csv("output/inventory.csv")
```

### Example 3: Extract Features from Single Transcript

```python
from src.features.feature_extractor import FeatureExtractor
from src.parsers.chat_parser import CHATParser

# Parse transcript
parser = CHATParser()
transcript = parser.parse_file("data/asdbank_eigsti/Eigsti/ASD/1010.cha")

# Extract features (pragmatic & conversational only)
extractor = FeatureExtractor(categories='pragmatic_conversational')
features = extractor.extract_from_transcript(transcript)

print(f"MLU: {features.features['mlu_words']:.2f}")
print(f"Echolalia Ratio: {features.features['echolalia_ratio']:.2f}")
```

### Example 4: Batch Feature Extraction

```python
from src.features.feature_extractor import FeatureExtractor

# Initialize extractor
extractor = FeatureExtractor(categories='pragmatic_conversational')

# Extract from entire directory
df = extractor.extract_from_directory(
    directory='data/asdbank_eigsti/Eigsti/ASD',
    output_file='output/asd_features.csv'
)

print(f"Extracted {df.shape[1]} features from {df.shape[0]} transcripts")
```

### Example 5: Compare ASD vs TD

```python
# Extract features from both groups
asd_df = extractor.extract_from_directory('data/asdbank_eigsti/Eigsti/ASD')
td_df = extractor.extract_from_directory('data/asdbank_eigsti/Eigsti/TD')

# Compare key features
print(f"ASD MLU: {asd_df['mlu_words'].mean():.2f}")
print(f"TD MLU: {td_df['mlu_words'].mean():.2f}")
```

## 📁 Project Structure

```
Artistic./
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
├── data/                    # Dataset directory
│   ├── asdbank_aac/
│   ├── asdbank_eigsti/
│   ├── asdbank_flusberg/
│   ├── asdbank_nadig/
│   └── ...
│
├── src/                     # Source code
│   ├── parsers/            # Phase 1: CHAT parsing
│   │   ├── chat_parser.py
│   │   └── dataset_inventory.py
│   │
│   ├── features/           # Phase 2: Feature extraction
│   │   ├── base_features.py
│   │   ├── feature_extractor.py
│   │   │
│   │   ├── acoustic_prosodic/        # Category 1 (Team Member A)
│   │   │   ├── acoustic_prosodic.py
│   │   │   └── __init__.py
│   │   │
│   │   ├── syntactic_semantic/       # Category 2 (Team Member B)
│   │   │   ├── syntactic_semantic.py
│   │   │   └── __init__.py
│   │   │
│   │   └── pragmatic_conversational/ # Category 3 (Implemented)
│   │       ├── turn_taking.py
│   │       ├── linguistic.py
│   │       ├── pragmatic.py
│   │       ├── conversational.py
│   │       └── __init__.py
│   │
│   └── utils/              # Utilities
│       ├── logger.py
│       └── helpers.py
│
├── examples/               # Usage examples
│   └── example_usage.py
│
├── output/                # Output directory
├── models/                # ML models (Phase 3)
├── logs/                  # Log files
└── cache/                 # Cached data
```

## 🔬 Feature Details

### Turn-Taking Features
- `turns_per_minute`: Conversation engagement
- `child_turn_ratio`: Child participation level
- `avg_response_latency`: Processing speed indicator
- `child_initiation_ratio`: Social initiation ability

### Linguistic Features
- `mlu_words`: Language development marker
- `type_token_ratio`: Vocabulary diversity
- `lexical_density`: Language complexity
- `utterance_complexity_score`: Composite metric

### Pragmatic Features
- `echolalia_ratio`: Repetition patterns (common in ASD)
- `pronoun_reversal_count`: Pronoun confusion
- `question_ratio`: Question usage
- `social_phrase_ratio`: Social language use

### Conversational Features
- `topic_shift_ratio`: Topic management
- `discourse_marker_ratio`: Conversation structuring
- `self_repair_count`: Communication awareness
- `topic_relevance_score`: Conversational coherence

## 🔧 Configuration

Edit `config.py` or `.env` to customize:

```python
# Data paths
DATA_DIR=./data
OUTPUT_DIR=./output

# Feature extraction
MIN_UTTERANCES=10
MIN_WORDS=5
WINDOW_SIZE=300

# Processing
N_JOBS=4
BATCH_SIZE=32
```

## 📊 Output

The system generates:

1. **CSV Files**: Feature matrices for ML training
2. **JSON Cache**: Processed inventory for quick access
3. **Logs**: Detailed processing information
4. **Summary Statistics**: Feature distributions and comparisons

Example output (`output/features.csv`):
```csv
participant_id,diagnosis,age_months,mlu_words,echolalia_ratio,question_ratio,...
1010,ASD,63,2.45,0.23,0.08,...
1011,TD,48,3.12,0.01,0.15,...
```

## 🧪 Testing

Run the example script:
```bash
python examples/example_usage.py
```

## 📚 Key Dependencies

- **pylangacq** (0.20.0): CHAT file parsing
- **pandas** (2.1.3): Data manipulation
- **numpy** (1.26.2): Numerical computing
- **tqdm**: Progress bars
- **loguru**: Advanced logging

## 🤝 Integration for Other Team Members

### For Team Member A (Acoustic/Prosodic Features)

1. Implement `src/features/acoustic_prosodic/acoustic_prosodic.py`
2. Extract features from audio files
3. Use libraries: librosa, praat-parselmouth
4. Access audio via `transcript.metadata.get('media')`

```python
# Your implementation in acoustic_prosodic/acoustic_prosodic.py
class AcousticProsodicFeatures(BaseFeatureExtractor):
    def extract(self, transcript):
        audio_path = self._get_audio_path(transcript)
        # Extract pitch, speech rate, etc.
        return FeatureResult(features={...})
```

### For Team Member B (Syntactic/Semantic Features)

1. Implement `src/features/syntactic_semantic/syntactic_semantic.py`
2. Extract grammar and semantic features
3. Use libraries: spaCy, NLTK
4. Access morphology via `utterance.morphology`

```python
# Your implementation in syntactic_semantic/syntactic_semantic.py
class SyntacticSemanticFeatures(BaseFeatureExtractor):
    def extract(self, transcript):
        # Extract dependency depth, semantic roles, etc.
        return FeatureResult(features={...})
```

## 📈 Next Steps (Phase 3+)

- [ ] Machine Learning model training
- [ ] Cross-validation and evaluation
- [ ] Backend API development (FastAPI/Flask)
- [ ] Integration of all three feature categories
- [ ] Web interface for predictions

## 📝 Citation

If using the ASDBank dataset, please cite:
- [TalkBank ASDBank](https://asd.talkbank.org/)
- Individual corpus citations in dataset README files

## 📄 License

[Your License Here]

## 👥 Authors

ASD Detection Team - 2024

## 🐛 Troubleshooting

**Issue**: `FileNotFoundError` when parsing
- **Solution**: Ensure data files are in the correct directory structure

**Issue**: `ImportError` for pylangacq
- **Solution**: Run `pip install pylangacq`

**Issue**: Features return all zeros
- **Solution**: Check transcript has valid child utterances

## 📞 Support

For questions or issues:
- Check examples in `examples/example_usage.py`
- Review log files in `logs/`
- Contact team members for integration

---

**Status**: ✅ Phase 1 & 2 Complete | 🔄 Phase 3 In Progress

