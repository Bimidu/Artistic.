# ASD Detection System

A multimodal machine learning system for Autism Spectrum Disorder (ASD) detection from speech analysis. The system analyzes both audio recordings and text transcripts to extract linguistic, pragmatic, and conversational features that may indicate ASD patterns.

## Overview

This system implements a research-based approach to ASD detection using three independent feature extraction components:

1. **Pragmatic & Conversational** (Implemented) - Turn-taking patterns, topic coherence, pause analysis, repair detection
2. **Acoustic & Prosodic** (Placeholder) - Pitch, prosody, voice quality features from audio
3. **Syntactic & Semantic** (Placeholder) - POS analysis, dependency parsing, semantic coherence

Each component independently extracts features and trains models. A fusion layer combines predictions from all components into a final ASD probability score.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            Input Handler                 │
                    │   (Audio or Text/CHAT File)             │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │         Audio Preprocessing              │
                    │   (Transcription if audio input)         │
                    └─────────────────┬───────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│   Pragmatic &       │   │   Acoustic &        │   │   Syntactic &       │
│   Conversational    │   │   Prosodic          │   │   Semantic          │
│                     │   │                     │   │                     │
│ • Turn-taking       │   │ • Pitch analysis    │   │ • POS tagging       │
│ • Topic coherence   │   │ • Prosody features  │   │ • Dependencies      │
│ • Pause analysis    │   │ • Voice quality     │   │ • Semantic analysis │
│ • Repair detection  │   │ • Spectral features │   │ • Clause structure  │
│ • Audio features    │   │                     │   │                     │
│                     │   │ [PLACEHOLDER]       │   │ [PLACEHOLDER]       │
│ [IMPLEMENTED]       │   │ (Team Member A)     │   │ (Team Member B)     │
└─────────┬───────────┘   └─────────┬───────────┘   └─────────┬───────────┘
          │                         │                         │
          │    ┌────────────────────┴─────────────────────┐   │
          └────►           Model Fusion                   ◄───┘
               │                                          │
               │  • Weighted averaging                    │
               │  • Voting                                │
               │  • Stacking (meta-learner)               │
               └────────────────────┬─────────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────────┐
                    │        Final ASD Prediction          │
                    │   + Annotated Transcript             │
                    └─────────────────────────────────────┘
```

## Features

### Pragmatic & Conversational Features (207 features)

| Category | Count | Description |
|----------|-------|-------------|
| Turn-Taking | 45 | Turn frequency, length, inter-turn gaps, overlaps, interruptions |
| Topic Coherence | 28 | LDA topic modeling, semantic similarity, topic shifts |
| Pause & Latency | 34 | Response latency, filled/unfilled pauses, speaking ratio |
| Repair Detection | 35 | Self-repair, other-repair, clarification requests |
| Pragmatic Linguistic | 35 | MLU, vocabulary diversity, echolalia, pronouns |
| Audio-Derived | 30 | Pause patterns from audio, response timing, speaking rate |

### Annotated Transcripts

The system generates visual annotations showing where features were extracted:

- `[TURN]` - Speaker turn boundaries
- `[PAUSE]` - Long pauses (> 1 second)
- `[LATENCY]` - Delayed responses
- `[REPAIR]` - Self-corrections
- `[TOPIC]` - Topic changes
- `[MARKER]` - Discourse markers

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Artistic

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For audio support, you may need ffmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

### Requirements

- Python 3.10+
- Key packages: `pylangacq`, `librosa`, `scikit-learn`, `xgboost`, `fastapi`
- For audio transcription: `openai-whisper` (optional, for Whisper backend)

## Quick Start

### 1. Run Feature Extraction

```python
from src.features.feature_extractor import FeatureExtractor
from src.parsers.chat_parser import CHATParser
from pathlib import Path

# Initialize
parser = CHATParser()
extractor = FeatureExtractor(categories='pragmatic_conversational')

# Extract features from a transcript
transcript = parser.parse_file(Path("data/asdbank_eigsti/Eigsti/ASD/1010.cha"))
feature_set = extractor.extract_from_transcript(transcript)

print(f"Extracted {len(feature_set.features)} features")
```

### 2. Process Audio Files

```python
from src.audio import AudioProcessor
from src.features.feature_extractor import FeatureExtractor

# Initialize
processor = AudioProcessor(whisper_model_size='base')
extractor = FeatureExtractor()

# Process audio
result = processor.process("audio.wav")

# Extract features with audio support
feature_set = extractor.extract_with_audio(
    result.transcript_data,
    audio_path=result.audio_path,
    transcription_result=result.transcription
)
```

### 3. Start the API

```bash
# Run the API server
python run_api.py

# Or with uvicorn
uvicorn src.api.app:app --reload --port 8000
```

### 4. Use the Frontend

Open `frontend.html` in a browser and:
- **User Mode**: Upload audio/text for ASD prediction
- **Training Mode**: Manage datasets and model training

## API Endpoints

### User Mode Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/audio` | POST | Predict ASD from audio file |
| `/predict/text` | POST | Predict ASD from text input |
| `/predict/transcript` | POST | Predict ASD from CHAT file |

### Training Mode Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/training/datasets` | GET | List available datasets |
| `/training/extract-features` | POST | Extract features for training |
| `/training/train` | POST | Start model training |
| `/training/inspect-features` | POST | Inspect feature extraction |

### Information Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/features` | GET | List all features |
| `/models` | GET | List trained models |
| `/components` | GET | List component status |

## Project Structure

```
Artistic/
├── src/
│   ├── api/                      # FastAPI application
│   │   └── app.py                # API endpoints
│   ├── audio/                    # Audio preprocessing (common)
│   │   ├── transcriber.py        # Speech-to-text
│   │   └── audio_processor.py    # Audio pipeline
│   ├── features/                 # Feature extraction
│   │   ├── feature_extractor.py  # Main orchestrator
│   │   ├── base_features.py      # Base classes
│   │   ├── pragmatic_conversational/  # IMPLEMENTED
│   │   │   ├── turn_taking.py
│   │   │   ├── topic_coherence.py
│   │   │   ├── pause_latency.py
│   │   │   ├── repair_detection.py
│   │   │   ├── pragmatic_linguistic.py
│   │   │   └── audio_features.py     # Audio-derived features
│   │   ├── acoustic_prosodic/    # PLACEHOLDER (Team A)
│   │   │   └── audio_features.py
│   │   └── syntactic_semantic/   # PLACEHOLDER (Team B)
│   │       └── audio_features.py
│   ├── models/                   # Model training
│   │   ├── model_trainer.py      # Training orchestrator
│   │   └── model_registry.py     # Model management
│   ├── parsers/                  # Input parsing
│   │   └── chat_parser.py        # CHAT format parser
│   ├── pipeline/                 # Processing pipeline
│   │   ├── input_handler.py      # Unified input handling
│   │   ├── annotated_transcript.py  # Annotation generation
│   │   └── model_fusion.py       # Multi-model fusion
│   └── utils/                    # Utilities
│       ├── logger.py
│       └── helpers.py
├── data/                         # Datasets (not in repo)
│   ├── asdbank_aac/
│   ├── asdbank_eigsti/
│   └── ...
├── models/                       # Trained models
├── output/                       # Feature CSVs
├── config.py                     # Configuration
├── frontend.html                 # Web interface
├── run_api.py                    # API runner
└── requirements.txt
```

## For Team Members

### Implementing Your Component

Each component should implement:

1. **Feature Extraction** in `src/features/<component>/`
2. **Audio Features** in `src/features/<component>/audio_features.py`
3. **Model Training** in `src/models/<component>/`

#### Example: Acoustic & Prosodic (Team Member A)

```python
# src/features/acoustic_prosodic/audio_features.py

from src.features.base_features import BaseFeatureExtractor, FeatureResult

class AcousticAudioFeatures(BaseFeatureExtractor):
    @property
    def feature_names(self) -> list:
        return [
            'audio_pitch_mean',
            'audio_pitch_std',
            # ... your features
        ]
    
    def extract(self, transcript, audio_path=None, **kwargs):
        features = {}
        
        if audio_path:
            # Your audio analysis here
            pass
        
        return FeatureResult(
            features=features,
            feature_type='acoustic_audio',
            metadata={}
        )
```

### Integration Points

1. Register in `src/features/<component>/__init__.py`
2. Update `FeatureExtractor` to include your extractor
3. Implement component trainer in `src/models/<component>/`
4. Update `ModelFusion` weights if needed

## Model Fusion

The system supports multiple fusion strategies:

```python
from src.pipeline.model_fusion import ModelFusion, ComponentPrediction

fusion = ModelFusion(
    method='weighted',  # 'voting', 'averaging', 'stacking'
    component_weights={
        'pragmatic_conversational': 0.5,
        'acoustic_prosodic': 0.25,
        'syntactic_semantic': 0.25,
    }
)

# Fuse component predictions
result = fusion.fuse([
    ComponentPrediction(component='pragmatic_conversational', prediction='ASD', probability=0.7, ...),
    ComponentPrediction(component='acoustic_prosodic', prediction='TD', probability=0.4, ...),
])

print(f"Final: {result.final_prediction} ({result.confidence:.2%})")
```

## Configuration

Edit `config.py` to customize:

```python
# Feature extraction parameters
MIN_UTTERANCES = 5
MIN_WORDS = 10

# Audio processing
WHISPER_MODEL_SIZE = 'base'
SAMPLE_RATE = 16000

# Training
N_JOBS = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

## Data Format

### CHAT Format (.cha)

The system uses CHILDES CHAT format transcripts:

```
@Begin
@Participants: CHI Child, MOT Mother
@ID: eng|corpus|CHI|3;6|male|ASD|||
*CHI: I want cookie .
*MOT: what kind of cookie ?
*CHI: um chocolate cookie .
@End
```

### Audio Files

Supported formats: WAV, MP3, FLAC
- Recommended: 16kHz sample rate, mono
- System will automatically convert if needed

## Citation

If you use this system in your research, please cite:

```
@software{asd_detection_system,
  title = {ASD Detection System: Multimodal Speech Analysis},
  author = {Bimidu Gunathilake},
  year = {2024},
  url = {repository-url}
}
```

## License

[Add your license here]

## Contact

For questions or collaboration:
- Pragmatic & Conversational: Bimidu Gunathilake
- Acoustic & Prosodic: Team Member A
- Syntactic & Semantic: Team Member B
