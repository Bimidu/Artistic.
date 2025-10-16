# ASD Detection System

A comprehensive machine learning system for detecting Autism Spectrum Disorder (ASD) in children using conversational patterns from TalkBank ASDBank datasets.

## 🎯 Overview

This system extracts **pragmatic and conversational features** from speech transcripts and trains machine learning models to classify children as either ASD or Typically Developing (TD). The architecture supports three feature categories with separate training pipelines:

- **Acoustic & Prosodic** - Team Member A (Placeholder)
- **Syntactic & Semantic** - Team Member B (Placeholder)  
- **Pragmatic & Conversational** - Fully Implemented ✅

## 🏗️ System Architecture

### Complete System Diagram

```mermaid
graph TB
    %% Data Input Layer
    subgraph "📊 Data Sources"
        A[TalkBank ASDBank<br/>CHAT Files (.cha)]
        B[Metadata Files<br/>(.cdc, .xlsx)]
    end
    
    %% Phase 1: Data Parsing
    subgraph "🔍 Phase 1: Data Parsing"
        C[CHAT Parser<br/>chat_parser.py]
        D[Dataset Inventory<br/>dataset_inventory.py]
    end
    
    %% Phase 2: Feature Extraction
    subgraph "⚙️ Phase 2: Feature Extraction"
        subgraph "Team Member A (Placeholder)"
            E1[Acoustic & Prosodic<br/>Features (12)]
        end
        subgraph "Team Member B (Placeholder)"
            E2[Syntactic & Semantic<br/>Features (12)]
        end
        subgraph "Implemented ✅"
            E3[Pragmatic & Conversational<br/>Features (61)]
            E4[Turn-Taking Features (15)]
            E5[Linguistic Features (14)]
            E6[Pragmatic Features (16)]
            E7[Conversational Features (16)]
        end
    end
    
    %% Phase 3: Preprocessing
    subgraph "🧹 Phase 3: Data Preprocessing"
        F1[Data Validator<br/>data_validator.py]
        F2[Data Cleaner<br/>data_cleaner.py]
        F3[Feature Selector<br/>feature_selector.py]
        F4[Data Preprocessor<br/>preprocessor.py]
    end
    
    %% Phase 4: Model Training
    subgraph "🤖 Phase 4: Machine Learning"
        subgraph "Component Trainers"
            G1[Acoustic Trainer<br/>(Placeholder)]
            G2[Syntactic Trainer<br/>(Placeholder)]
            G3[Pragmatic Trainer<br/>(Implemented ✅)]
        end
        subgraph "ML Models"
            H1[Random Forest]
            H2[XGBoost]
            H3[LightGBM]
            H4[SVM]
            H5[Logistic Regression]
            H6[Neural Network]
        end
        G4[Model Evaluator<br/>model_evaluator.py]
        G5[Model Registry<br/>model_registry.py]
    end
    
    %% Phase 5: API Layer
    subgraph "🌐 Phase 5: FastAPI Backend"
        I1[REST API<br/>app.py]
        I2[Prediction Endpoint<br/>/predict]
        I3[Feature Endpoint<br/>/features]
        I4[Model Endpoint<br/>/models]
        I5[Health Check<br/>/health]
    end
    
    %% Output Layer
    subgraph "📈 Output"
        J1[ASD/TD Predictions]
        J2[Feature Importance]
        J3[Model Metrics]
        J4[API Documentation]
    end
    
    %% Data Flow Connections
    A --> C
    B --> D
    C --> E1
    C --> E2
    C --> E3
    C --> E4
    C --> E5
    C --> E6
    C --> E7
    
    E1 --> F1
    E2 --> F1
    E3 --> F1
    E4 --> F1
    E5 --> F1
    E6 --> F1
    E7 --> F1
    
    F1 --> F2
    F2 --> F3
    F3 --> F4
    
    F4 --> G1
    F4 --> G2
    F4 --> G3
    
    G1 --> H1
    G2 --> H2
    G3 --> H1
    G3 --> H2
    G3 --> H3
    G3 --> H4
    G3 --> H5
    G3 --> H6
    
    G3 --> G4
    G4 --> G5
    
    G5 --> I1
    I1 --> I2
    I1 --> I3
    I1 --> I4
    I1 --> I5
    
    I2 --> J1
    I3 --> J2
    I4 --> J3
    I5 --> J4
    
    %% Styling
    classDef implemented fill:#90EE90,stroke:#333,stroke-width:2px
    classDef placeholder fill:#FFE4B5,stroke:#333,stroke-width:2px
    classDef data fill:#E6F3FF,stroke:#333,stroke-width:2px
    classDef api fill:#F0E68C,stroke:#333,stroke-width:2px
    classDef output fill:#FFB6C1,stroke:#333,stroke-width:2px
    
    class E3,E4,E5,E6,E7,G3 implemented
    class E1,E2,G1,G2 placeholder
    class A,B,C,D,F1,F2,F3,F4 data
    class I1,I2,I3,I4,I5 api
    class J1,J2,J3,J4 output
```

### Component Status Legend
- 🟢 **Green (Implemented)**: Pragmatic & Conversational features - Production ready
- 🟡 **Yellow (Placeholder)**: Acoustic & Syntactic features - Ready for team implementation
- 🔵 **Blue (Data)**: Parsing and preprocessing components
- 🟡 **Yellow (API)**: FastAPI backend services
- 🟣 **Pink (Output)**: Results and documentation

### Directory Structure

```
src/
├── parsers/                    # CHAT file parsing
├── features/                   # Feature extraction
│   ├── acoustic_prosodic/     # Team Member A (placeholder)
│   ├── syntactic_semantic/    # Team Member B (placeholder)
│   └── pragmatic_conversational/ # Fully implemented (61 features)
├── preprocessing/              # Data preprocessing pipeline
├── models/                     # ML training & evaluation
│   ├── acoustic_prosodic/     # Team Member A (placeholder)
│   ├── syntactic_semantic/    # Team Member B (placeholder)
│   └── pragmatic_conversational/ # Fully implemented
└── api/                        # FastAPI backend
```

## 🚀 Quick Start

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Extract Features
```bash
python examples/example_usage.py
```

### 3. Train Models
```bash
python examples/train_model.py
```

### 4. Start API
```bash
python run_api.py
# Visit: http://localhost:8000/docs
```

## 📊 Implemented Features

### ✅ Pragmatic & Conversational (61 Features)

**Turn-Taking (15 features)**
- Turn length statistics, overlap patterns, pause analysis

**Linguistic Complexity (14 features)**  
- MLU (Mean Length of Utterance), TTR (Type-Token Ratio), sentence complexity

**Pragmatic Markers (16 features)**
- Echolalia detection, pronoun reversal, social language use

**Conversational Management (16 features)**
- Topic shifts, conversational repairs, coherence measures

## 🤖 Machine Learning Pipeline

### Models Supported
- Random Forest
- XGBoost  
- LightGBM
- Support Vector Machine
- Logistic Regression
- Neural Network (MLP)

### Preprocessing
- Data validation & cleaning
- Feature scaling & selection
- Train/test splitting with stratification

### Evaluation
- Accuracy, Precision, Recall, F1-Score
- ROC AUC, Confusion Matrix
- Feature importance analysis

## 🔧 Component Development

### For Team Member A (Acoustic/Prosodic)
```python
# Implement in: src/models/acoustic_prosodic/
class AcousticProsodicTrainer:
    def train_multiple_models(self, X_train, y_train):
        # Your implementation here
        pass
```

**Focus Areas:**
- Pitch features (mean, std, range, slope)
- Spectral features (MFCC, spectral centroid)  
- Temporal features (speaking rate, pause patterns)
- Prosodic features (intonation, stress, rhythm)

**Required Libraries:** `librosa`, `praat-parselmouth`, `scikit-learn`

### For Team Member B (Syntactic/Semantic)
```python
# Implement in: src/models/syntactic_semantic/
class SyntacticSemanticTrainer:
    def train_multiple_models(self, X_train, y_train):
        # Your implementation here
        pass
```

**Focus Areas:**
- Syntactic features (dependency depth, clause complexity)
- Semantic features (coherence, density, consistency)
- Grammatical features (error rates, agreement)
- Complexity features (structure diversity, abstractness)

**Required Libraries:** `spacy`, `nltk`, `scikit-learn`

### Current Implementation (Pragmatic/Conversational)
```python
# Already implemented in: src/models/pragmatic_conversational/
from src.models.pragmatic_conversational import PragmaticConversationalTrainer

trainer = PragmaticConversationalTrainer()
models = trainer.train_multiple_models(X_train, y_train, X_test, y_test)
```

## 🌐 API Endpoints

### Core Endpoints
- `POST /predict` - Make predictions on new data
- `GET /features` - List all supported features
- `GET /categories` - Show implementation status by category
- `GET /models` - List available trained models

### Example Usage
```python
import requests

# Make prediction
response = requests.post("http://localhost:8000/predict", 
                        json={"features": feature_dict})
prediction = response.json()

# Check feature status
response = requests.get("http://localhost:8000/categories")
status = response.json()
```

## 📁 Data Format

### Input Data
```python
# CHAT format files (.cha) from TalkBank ASDBank
data = {
    'participant_id': 'ASD_001',
    'transcript': '*CHI: hello there .',
    'diagnosis': 'ASD'  # or 'TD'
}
```

### Feature Output
```python
# 61 pragmatic/conversational features
features = {
    'mlu_words': 4.2,
    'type_token_ratio': 0.65,
    'echolalia_ratio': 0.1,
    'turn_taking_patterns': 0.3,
    # ... 57 more features
}
```

## 🧪 Testing

### Run Feature Extraction
```bash
python examples/example_usage.py
```

### Run Model Training
```bash
python examples/train_model.py
```

### Test API
```bash
python run_api.py
# Visit http://localhost:8000/docs for interactive testing
```

## 📈 Performance

### Current Implementation (Pragmatic/Conversational)
- **Features:** 61 pragmatic/conversational features
- **Models:** 7 ML algorithms with hyperparameter tuning
- **Evaluation:** Comprehensive metrics and feature importance
- **Status:** Production-ready

### Placeholder Components
- **Acoustic/Prosodic:** Ready for Team Member A implementation
- **Syntactic/Semantic:** Ready for Team Member B implementation

## 🔄 Integration Workflow

1. **Team Member A** implements acoustic/prosodic trainer
2. **Team Member B** implements syntactic/semantic trainer  
3. **Main orchestrator** automatically integrates all components
4. **API** serves predictions from all feature categories

```python
# Main orchestrator handles all components
from src.models.model_trainer import ModelTrainer

trainer = ModelTrainer()  # Has all component trainers
results = trainer.train_by_category(feature_data, y_train)

# Results show status for each component:
# - acoustic_prosodic: "placeholder" or "implemented"
# - syntactic_semantic: "placeholder" or "implemented"  
# - pragmatic_conversational: "implemented"
```

## 📋 Requirements

### Core Dependencies
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning
- `xgboost` - Gradient boosting
- `lightgbm` - Light gradient boosting
- `fastapi` - API framework
- `uvicorn` - ASGI server

### Feature-Specific Dependencies
- `pylangacq` - CHAT file parsing
- `loguru` - Enhanced logging
- `joblib` - Model serialization

## 🎯 Next Steps

1. **Team Member A**: Implement acoustic/prosodic features
2. **Team Member B**: Implement syntactic/semantic features
3. **Integration**: All components work together via orchestrator
4. **Production**: Deploy complete system with all feature categories

## 📞 Support

- **Current Implementation**: Pragmatic/conversational features fully working
- **Team Development**: Each component can be developed independently
- **Integration**: Main orchestrator handles component coordination
- **Documentation**: All APIs and interfaces clearly defined

---

**Status**: Pragmatic/conversational component production-ready. Acoustic/syntactic components ready for team implementation.