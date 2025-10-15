# Project Structure

Complete file structure and organization for the ASD Detection system.

## 📁 Directory Tree

```
Artistic./
│
├── 📄 config.py                          # Central configuration management
├── 📄 requirements.txt                   # Python dependencies
├── 📄 README.md                         # Main documentation
├── 📄 QUICK_START.md                    # Quick start guide
├── 📄 IMPLEMENTATION_SUMMARY.md         # Technical implementation details
├── 📄 PROJECT_STRUCTURE.md              # This file
│
├── 📂 data/                             # Dataset directory (user provided)
│   ├── asdbank_aac/                     # 18 ASD children (minimally speaking)
│   ├── asdbank_eigsti/                  # 16 ASD + 16 TD + 16 DD
│   ├── asdbank_flusberg/                # 6 ASD (longitudinal)
│   ├── asdbank_nadig/                   # 20 ASD + 18 TYP
│   ├── asdbank_quigley_mcnalley/       # 105 HR + 98 LR
│   └── asdbank_rollins/                 # 5 ASD
│
├── 📂 src/                              # Source code
│   ├── 📄 __init__.py
│   │
│   ├── 📂 parsers/                      # PHASE 1: Data Parsing
│   │   ├── 📄 __init__.py
│   │   ├── 📄 chat_parser.py           # CHAT file parsing (470 lines)
│   │   └── 📄 dataset_inventory.py     # Dataset management (424 lines)
│   │
│   ├── 📂 features/                     # PHASE 2: Feature Extraction
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base_features.py         # Base classes & utilities
│   │   ├── 📄 feature_extractor.py     # Main orchestrator
│   │   │
│   │   ├── 🔵 CATEGORY 1: ACOUSTIC & PROSODIC (Team Member A)
│   │   ├── 📂 acoustic_prosodic/
│   │   │   ├── 📄 __init__.py
│   │   │   └── 📄 acoustic_prosodic.py # Acoustic features (12)
│   │   │
│   │   ├── 🔵 CATEGORY 2: SYNTACTIC & SEMANTIC (Team Member B)
│   │   ├── 📂 syntactic_semantic/
│   │   │   ├── 📄 __init__.py
│   │   │   └── 📄 syntactic_semantic.py # Syntactic features (12)
│   │   │
│   │   └── 🟢 CATEGORY 3: PRAGMATIC & CONVERSATIONAL (Implemented)
│   │       ├── 📂 pragmatic_conversational/
│   │       │   ├── 📄 __init__.py
│   │       │   ├── 📄 turn_taking.py   # Turn-taking features (15)
│   │       │   ├── 📄 linguistic.py    # Linguistic features (14)
│   │       │   ├── 📄 pragmatic.py     # Pragmatic features (16)
│   │       │   └── 📄 conversational.py # Conversational features (16)
│   │
│   └── 📂 utils/                        # Utility functions
│       ├── 📄 __init__.py
│       ├── 📄 logger.py                # Logging configuration
│       └── 📄 helpers.py               # Helper functions
│
├── 📂 examples/                         # Usage examples
│   └── 📄 example_usage.py             # Comprehensive examples
│
├── 📂 output/                          # Generated outputs
│   ├── inventory.csv                   # Dataset inventory
│   ├── features.csv                    # Extracted features
│   └── *.csv                           # Analysis results
│
├── 📂 models/                          # ML models (Phase 3)
│   └── (to be implemented)
│
├── 📂 logs/                            # Log files
│   └── asd_detection.log              # Application logs
│
└── 📂 cache/                           # Cached data
    └── inventory.json                  # Cached inventory
```

## 🔧 Core Modules

### 1. Configuration (`config.py`)

```python
PathConfig          # File paths and directories
FeatureConfig       # Feature extraction parameters
ProcessingConfig    # System settings
LoggingConfig       # Logging configuration
DatasetConfig       # Dataset mappings
```

### 2. Parsers (`src/parsers/`)

#### `chat_parser.py`
```python
CHATParser          # Main parser class
  ├── parse_file()                    # Parse single .cha file
  ├── parse_directory()               # Batch parsing
  └── _extract_metadata()             # Metadata extraction

TranscriptData      # Parsed transcript container
  ├── participant_id                  # Participant identifier
  ├── diagnosis                       # Clinical diagnosis
  ├── age_months                      # Age in months
  ├── utterances                      # All utterances
  └── speakers                        # Speaker information

Utterance          # Single utterance representation
  ├── speaker                         # Speaker code (CHI, MOT, INV)
  ├── text                           # Utterance text
  ├── tokens                         # Word tokens
  ├── morphology                     # POS/morphology (%mor)
  ├── grammar                        # Grammar relations (%gra)
  ├── timing                         # Timestamp
  ├── actions                        # Actions (%act)
  └── comments                       # Comments (%com)
```

#### `dataset_inventory.py`
```python
DatasetInventory    # Dataset management
  ├── build_inventory()              # Build complete inventory
  ├── get_participants_by_diagnosis() # Filter by diagnosis
  ├── get_dataset_summary()          # Get statistics
  ├── to_dataframe()                 # Export to DataFrame
  └── export_to_csv()                # Save to CSV

ParticipantInfo     # Participant metadata
  ├── participant_id
  ├── dataset
  ├── diagnosis
  ├── age_months
  ├── num_sessions
  └── total_utterances
```

### 3. Features (`src/features/`)

#### Architecture
```
BaseFeatureExtractor (Abstract)
  ├── extract()                      # Main extraction method
  ├── feature_names                  # List of features
  └── utility methods

Category 1: Acoustic & Prosodic (Placeholder)
  └── AcousticProsodicFeatures
      └── 12 features (pitch, rate, prosody, pauses)

Category 2: Syntactic & Semantic (Placeholder)
  └── SyntacticSemanticFeatures
      └── 12 features (syntax, grammar, semantics)

Category 3: Pragmatic & Conversational (✅ Implemented)
  ├── TurnTakingFeatures
  │   └── 15 features
  ├── LinguisticFeatures
  │   └── 14 features
  ├── PragmaticFeatures
  │   └── 16 features
  └── ConversationalFeatures
      └── 16 features
```

#### Main Orchestrator
```python
FeatureExtractor    # Coordinates all extractors
  ├── extract_from_transcript()      # Single transcript
  ├── extract_from_files()           # Batch processing
  ├── extract_from_directory()       # Directory processing
  ├── normalize_features()           # Feature normalization
  ├── get_feature_summary()          # Statistics
  └── print_category_info()          # Display info

FeatureSet         # Feature container
  ├── participant_id
  ├── diagnosis
  ├── features                       # Dict of features
  └── metadata
```

### 4. Utilities (`src/utils/`)

```python
# logger.py
setup_logger()      # Configure logging
get_logger()        # Get logger instance

# helpers.py
timing_decorator    # Time function execution
safe_divide()       # Safe division
calculate_ratio()   # Calculate ratios
normalize_text()    # Text normalization
is_valid_utterance() # Utterance validation
extract_timing_info() # Parse timing
get_age_in_months() # Parse age
```

## 📊 Feature Categories

### Category 3: Pragmatic & Conversational (61 features) ✅

#### Turn-Taking (15 features)
```
total_turns
child_turns
adult_turns
turns_per_minute
child_turn_ratio
avg_turn_length_words
avg_child_turn_length
avg_adult_turn_length
avg_response_latency
median_response_latency
child_initiated_turns
adult_initiated_turns
child_initiation_ratio
turn_switches
avg_turns_before_switch
```

#### Linguistic (14 features)
```
mlu_words
mlu_morphemes
avg_word_length
max_utterance_length
total_words
unique_words
type_token_ratio
corrected_ttr
noun_ratio
verb_ratio
adjective_ratio
pronoun_ratio
function_word_ratio
lexical_density
utterance_complexity_score
```

#### Pragmatic (16 features)
```
echolalia_ratio
immediate_echolalia_count
delayed_echolalia_count
partial_repetition_ratio
question_ratio
question_diversity
yes_no_question_ratio
wh_question_ratio
pronoun_usage_ratio
first_person_pronoun_ratio
pronoun_error_ratio
pronoun_reversal_count
social_phrase_ratio
greeting_count
politeness_marker_count
appropriate_response_ratio
unintelligible_ratio
```

#### Conversational (16 features)
```
topic_shift_ratio
topic_maintenance_score
topic_intro_marker_ratio
avg_topic_duration
discourse_marker_ratio
continuation_marker_ratio
repair_marker_ratio
acknowledgment_ratio
self_repair_count
other_repair_count
clarification_request_ratio
nonverbal_behavior_ratio
laughter_ratio
vocal_behavior_diversity
topic_relevance_score
off_topic_ratio
```

### Category 1: Acoustic & Prosodic (12 features) 🔵
```
mean_pitch
pitch_std
pitch_range
pitch_slope
speaking_rate
articulation_rate
pause_rate
intonation_variability
stress_pattern_score
rhythm_score
mean_pause_duration
filled_pause_ratio
```
**Status**: Placeholder for Team Member A

### Category 2: Syntactic & Semantic (12 features) 🔵
```
avg_dependency_depth
max_dependency_depth
clause_complexity
subordination_index
grammatical_error_rate
tense_consistency_score
agreement_error_rate
structure_diversity
semantic_coherence
semantic_density
thematic_consistency
vocabulary_abstractness
semantic_role_diversity
word_sense_accuracy
```
**Status**: Placeholder for Team Member B

## 📈 Data Flow

```
1. DATA INPUT
   └─> .cha files (CHAT format)

2. PHASE 1: PARSING
   └─> CHATParser
       ├─> Extract metadata
       ├─> Parse utterances
       ├─> Extract morphology/grammar
       └─> TranscriptData object

3. DATASET INVENTORY
   └─> DatasetInventory
       ├─> Scan all files
       ├─> Aggregate by participant
       ├─> Cache results
       └─> Export CSV/JSON

4. PHASE 2: FEATURE EXTRACTION
   └─> FeatureExtractor
       ├─> Turn-taking features
       ├─> Linguistic features
       ├─> Pragmatic features
       ├─> Conversational features
       └─> FeatureSet object

5. OUTPUT
   ├─> CSV files (features.csv)
   ├─> Summary statistics
   └─> Normalized features

6. PHASE 3 (Future)
   └─> Machine Learning
       ├─> Model training
       ├─> Evaluation
       └─> Prediction API
```

## 🔄 Workflow Examples

### Workflow 1: Single File Analysis
```
.cha file → CHATParser → TranscriptData → FeatureExtractor → FeatureSet → Analysis
```

### Workflow 2: Batch Processing
```
Directory → CHATParser (batch) → Multiple TranscriptData → FeatureExtractor → DataFrame → CSV
```

### Workflow 3: Complete Pipeline
```
Raw Data → Inventory → Filter → Parse → Extract Features → Normalize → ML Ready
```

## 📝 File Relationships

```
config.py
  └─> Used by: All modules

src/utils/
  ├─> logger.py
  │   └─> Used by: All modules
  └─> helpers.py
      └─> Used by: parsers/, features/

src/parsers/
  ├─> chat_parser.py
  │   └─> Uses: utils/helpers, utils/logger
  └─> dataset_inventory.py
      └─> Uses: chat_parser, utils/logger

src/features/
  ├─> base_features.py
  │   └─> Uses: parsers/chat_parser, utils/
  ├─> turn_taking.py
  │   └─> Extends: base_features
  ├─> linguistic.py
  │   └─> Extends: base_features
  ├─> pragmatic.py
  │   └─> Extends: base_features
  ├─> conversational.py
  │   └─> Extends: base_features
  └─> feature_extractor.py
      └─> Coordinates: All feature extractors
```

## 🎯 Entry Points

### For Users
```python
# Main entry point
examples/example_usage.py

# Quick test
python -c "from src.parsers.chat_parser import CHATParser; print('✓ System ready')"
```

### For Developers
```python
# Parser development
src/parsers/chat_parser.py

# Feature development
src/features/[module].py

# Configuration
config.py
```

### For Team Integration
```python
# Team Member A (Acoustic/Prosodic)
src/features/acoustic_prosodic.py

# Team Member B (Syntactic/Semantic)
src/features/syntactic_semantic.py
```

## 📦 Dependencies

### Core
- pylangacq (CHAT parsing)
- pandas (data manipulation)
- numpy (numerical computing)

### Utilities
- tqdm (progress bars)
- loguru (logging)
- python-dotenv (configuration)

### Future (Team Integration)
- librosa (Team A - audio)
- spacy (Team B - NLP)

## 🔐 Configuration Files

```
config.py          # Main configuration
.env              # Environment variables (optional)
.env.example      # Example configuration
```

## 📊 Output Files

```
output/
  ├── inventory.csv              # Dataset inventory
  ├── features.csv               # Extracted features
  ├── asd_features.csv          # ASD group features
  ├── td_features.csv           # TD group features
  └── asd_vs_td_comparison.csv  # Comparison results

cache/
  └── inventory.json            # Cached inventory

logs/
  └── asd_detection.log         # Application logs
```

## 🎓 Learning Path

1. **Start Here**: `QUICK_START.md`
2. **Understand System**: `README.md`
3. **Technical Details**: `IMPLEMENTATION_SUMMARY.md`
4. **Code Structure**: `PROJECT_STRUCTURE.md` (this file)
5. **Try Examples**: `examples/example_usage.py`
6. **Integrate**: Follow integration guides

---

**Total Lines of Code**: ~3,500+
**Total Features**: 61 (implemented) + 24 (placeholders)
**Documentation**: 5 comprehensive guides
**Examples**: Multiple usage scenarios

