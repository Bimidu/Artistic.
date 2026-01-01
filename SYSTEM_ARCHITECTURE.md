# ASD Detection System - Detailed Architecture

## Table of Contents
1. [Overall System Architecture](#1-overall-system-architecture)
2. [Input Processing Flow](#2-input-processing-flow)
3. [Feature Extraction Pipeline](#3-feature-extraction-pipeline)
4. [Model Training Flow](#4-model-training-flow)
5. [Prediction Flow](#5-prediction-flow)
6. [API Architecture](#6-api-architecture)
7. [Component Details](#7-component-details)
8. [Data Flow](#8-data-flow)

---

## 1. Overall System Architecture

```mermaid
graph TB
    subgraph UI["User Interface Layer"]
        FRONTEND[Frontend HTML/JS]
        USER_MODE["User Mode"]
        TRAIN_MODE["Training Mode"]
    end
    
    subgraph API["API Layer - FastAPI"]
        GATEWAY["API Gateway: app.py"]
        PREDICT_API["predict endpoints"]
        TRAINING_API["training endpoints"]
        INFO_API["info endpoints"]
    end
    
    subgraph PROCESS["Processing Pipeline Layer"]
        INPUT_H["Input Handler"]
        AUDIO_PROC["Audio Processor"]
        TRANSCRIBE["Transcriber"]
    end
    
    subgraph EXTRACT["Feature Extraction Layer"]
        EXTRACTOR["Feature Extractor"]
        
        subgraph PRAG_COMP["Pragmatic Component - IMPLEMENTED"]
            PRAG_TEXT["Text Features"]
            PRAG_AUDIO["Audio Features"]
        end
        
        subgraph ACOU_COMP["Acoustic Component - PLACEHOLDER"]
            ACOU_AUDIO["Audio Features"]
        end
        
        subgraph SYNT_COMP["Syntactic Component - PLACEHOLDER"]
            SYNT_TEXT["Text Features"]
        end
    end
    
    subgraph MODEL["Model Layer"]
        TRAINER["Model Trainer"]
        REGISTRY["Model Registry"]
        FUSION["Model Fusion"]
    end
    
    subgraph OUTPUT["Output Layer"]
        ANNOT["Annotated Transcript"]
        PRED["Prediction Results"]
        FEATURES["Feature CSVs"]
    end
    
    subgraph STORAGE["Data Storage"]
        DB_DATA[("Dataset Files")]
        DB_MODELS[("Trained Models")]
        DB_OUTPUT[("Output Files")]
    end
    
    FRONTEND --> GATEWAY
    USER_MODE --> PREDICT_API
    TRAIN_MODE --> TRAINING_API
    
    GATEWAY --> INPUT_H
    INPUT_H --> AUDIO_PROC
    AUDIO_PROC --> TRANSCRIBE
    INPUT_H --> EXTRACTOR
    
    EXTRACTOR --> PRAG_TEXT
    EXTRACTOR --> PRAG_AUDIO
    EXTRACTOR --> ACOU_AUDIO
    EXTRACTOR --> SYNT_TEXT
    
    PRAG_TEXT --> TRAINER
    PRAG_AUDIO --> TRAINER
    ACOU_AUDIO --> TRAINER
    SYNT_TEXT --> TRAINER
    
    TRAINER --> REGISTRY
    REGISTRY --> FUSION
    
    FUSION --> PRED
    EXTRACTOR --> ANNOT
    EXTRACTOR --> FEATURES
    
    DB_DATA --> INPUT_H
    REGISTRY --> DB_MODELS
    FEATURES --> DB_OUTPUT
    
    style PRAG_TEXT fill:#90EE90
    style PRAG_AUDIO fill:#90EE90
    style ACOU_AUDIO fill:#FFB6C1
    style SYNT_TEXT fill:#FFB6C1
```

---

## 2. Input Processing Flow

```mermaid
flowchart TD
    START([Input File]) --> TYPE{Input Type?}
    
    TYPE -->|Audio: WAV, MP3| AUDIO[Audio Processor]
    TYPE -->|CHAT File| CHAT[CHAT Parser]
    TYPE -->|Plain Text| TEXT[Text Handler]
    
    AUDIO --> LOAD[Load Audio with librosa]
    LOAD --> STT[Speech-to-Text: Whisper or Google]
    STT --> TRANS_RESULT[TranscriptionResult Object]
    
    TRANS_RESULT --> CONV1[Convert to TranscriptData]
    CHAT --> PARSE[Parse with pylangacq]
    PARSE --> CONV2[Convert to TranscriptData]
    TEXT --> CONV3[Convert to TranscriptData]
    
    CONV1 --> PROCESSED[ProcessedInput Object]
    CONV2 --> PROCESSED
    CONV3 --> PROCESSED
    
    PROCESSED --> HAS_AUDIO{Has Audio?}
    HAS_AUDIO -->|Yes| AUDIO_PATH[Store audio_path and transcription_result]
    HAS_AUDIO -->|No| NO_AUDIO[Set audio_path = None]
    
    AUDIO_PATH --> OUTPUT[Complete ProcessedInput]
    NO_AUDIO --> OUTPUT
    
    OUTPUT --> NEXT([To Feature Extraction])
    
    style AUDIO fill:#87CEEB
    style STT fill:#87CEEB
    style TRANS_RESULT fill:#87CEEB
```

---

## 3. Feature Extraction Pipeline

```mermaid
flowchart TD
    START([ProcessedInput]) --> CHECK{Has Audio?}
    
    CHECK -->|Yes| EXTRACT_AUDIO[extract_with_audio method]
    CHECK -->|No| EXTRACT_TEXT[extract_from_transcript method]
    
    subgraph ORCHESTRATOR["Feature Extractor Orchestrator"]
        EXTRACT_AUDIO --> INIT[Initialize Active Extractors]
        EXTRACT_TEXT --> INIT
        
        INIT --> CATEGORIES{For Each Category}
    end
    
    subgraph PRAGMATIC["Pragmatic and Conversational - IMPLEMENTED"]
        PRAG[Pragmatic Component]
        
        PRAG --> TURN["Turn Taking: 45 features"]
        PRAG --> TOPIC["Topic Coherence: 28 features"]
        PRAG --> PAUSE["Pause and Latency: 34 features"]
        PRAG --> REPAIR["Repair Detection: 35 features"]
        PRAG --> LING["Pragmatic Linguistic: 35 features"]
        PRAG --> AUDIO_P["Audio Features: 30 features"]
        
        TURN --> TURN_F["turn frequency, turn length, turn gaps, overlaps"]
        TOPIC --> TOPIC_F["topic coherence, topic shifts, semantic similarity"]
        PAUSE --> PAUSE_F["response latency, filled pauses, unfilled pauses"]
        REPAIR --> REPAIR_F["self repairs, other repairs, clarifications"]
        LING --> LING_F["MLU, vocabulary diversity, echolalia"]
        AUDIO_P --> AUDIO_F["audio pause duration, speaking rate, timing"]
    end
    
    subgraph ACOUSTIC["Acoustic and Prosodic - PLACEHOLDER"]
        ACOU[Acoustic Component]
        ACOU --> ACOU_HOLD[Team Member A Area]
        ACOU_HOLD --> ACOU_F["pitch, prosody, spectral features"]
    end
    
    subgraph SYNTACTIC["Syntactic and Semantic - PLACEHOLDER"]
        SYNT[Syntactic Component]
        SYNT --> SYNT_HOLD[Team Member B Area]
        SYNT_HOLD --> SYNT_F["POS tags, dependencies, clause structure"]
    end
    
    CATEGORIES --> PRAG
    CATEGORIES --> ACOU
    CATEGORIES --> SYNT
    
    TURN_F --> MERGE[Merge All Features]
    TOPIC_F --> MERGE
    PAUSE_F --> MERGE
    REPAIR_F --> MERGE
    LING_F --> MERGE
    AUDIO_F --> MERGE
    ACOU_F --> MERGE
    SYNT_F --> MERGE
    
    MERGE --> FEATURE_SET["FeatureSet Object: 207+ features"]
    
    FEATURE_SET --> OUTPUT([Output])
    
    style PRAG fill:#90EE90
    style TURN fill:#90EE90
    style TOPIC fill:#90EE90
    style PAUSE fill:#90EE90
    style REPAIR fill:#90EE90
    style LING fill:#90EE90
    style AUDIO_P fill:#90EE90
    
    style ACOU fill:#FFB6C1
    style SYNT fill:#FFB6C1
```

---

## 4. Model Training Flow

```mermaid
flowchart TD
    START([Training Initiated]) --> SELECT[Select Datasets via Frontend]
    
    SELECT --> PATHS[Dataset Paths List]
    
    PATHS --> EXTRACT_PHASE[Extract Features from Each Dataset]
    
    subgraph EXTRACTION["Feature Extraction Phase"]
        EXTRACT_PHASE --> DIR[For Each Directory]
        DIR --> FILES{For Each File}
        FILES -->|.cha| CHAT_EX[Extract from CHAT]
        FILES -->|.wav| AUDIO_EX[Extract from Audio]
        
        CHAT_EX --> DF1[DataFrame Row]
        AUDIO_EX --> DF2[DataFrame Row]
        
        DF1 --> COMBINE[Combine All Rows]
        DF2 --> COMBINE
    end
    
    COMBINE --> CSV[Save to training_features.csv]
    
    CSV --> LOAD[Load Feature CSV]
    
    subgraph COMPONENT_TRAIN["Component Training Phase"]
        LOAD --> COMP{Train Each Component}
        
        COMP --> TRAIN_PRAG[Train Pragmatic Component]
        COMP --> TRAIN_ACOU[Train Acoustic Component]
        COMP --> TRAIN_SYNT[Train Syntactic Component]
        
        subgraph INDIVIDUAL["Individual Component Training"]
            TRAIN_PRAG --> PREP_P[Prepare Data: X, y split]
            PREP_P --> MODELS_P[Train Multiple Models]
            
            MODELS_P --> RF_P[Random Forest]
            MODELS_P --> XGB_P[XGBoost]
            MODELS_P --> LGB_P[LightGBM]
            MODELS_P --> SVM_P[SVM]
            
            RF_P --> EVAL_P[Evaluate Models with Cross-validation]
            XGB_P --> EVAL_P
            LGB_P --> EVAL_P
            SVM_P --> EVAL_P
            
            EVAL_P --> BEST_P[Select Best Model based on F1/Accuracy]
        end
        
        TRAIN_ACOU --> PREP_A[Similar Training Process]
        TRAIN_SYNT --> PREP_S[Similar Training Process]
        
        BEST_P --> SAVE_P[Save Model and Metadata to Registry]
        PREP_A --> SAVE_A[Save Model and Metadata]
        PREP_S --> SAVE_S[Save Model and Metadata]
    end
    
    SAVE_P --> REGISTRY[("Model Registry")]
    SAVE_A --> REGISTRY
    SAVE_S --> REGISTRY
    
    REGISTRY --> FUSION_PREP[Prepare Fusion Layer]
    
    FUSION_PREP --> WEIGHTS[Configure Component Weights]
    
    WEIGHTS --> COMPLETE([Training Complete])
    
    style TRAIN_PRAG fill:#90EE90
    style RF_P fill:#90EE90
    style XGB_P fill:#90EE90
    style LGB_P fill:#90EE90
    style SVM_P fill:#90EE90
    
    style TRAIN_ACOU fill:#FFB6C1
    style TRAIN_SYNT fill:#FFB6C1
```

---

## 5. Prediction Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant InputHandler
    participant FeatureExtractor
    participant ModelRegistry
    participant ComponentModels
    participant ModelFusion
    participant Annotator
    
    User->>Frontend: Upload Audio/Text
    Frontend->>API: POST /predict/audio
    
    API->>InputHandler: process(input)
    
    alt Audio Input
        InputHandler->>InputHandler: Load Audio
        InputHandler->>InputHandler: Transcribe with Whisper/Google
        InputHandler->>InputHandler: Create TranscriptData
    else Text/CHAT Input
        InputHandler->>InputHandler: Parse Input
        InputHandler->>InputHandler: Create TranscriptData
    end
    
    InputHandler-->>API: ProcessedInput
    
    API->>FeatureExtractor: extract_with_audio() or extract_from_transcript()
    
    FeatureExtractor->>FeatureExtractor: Initialize extractors
    
    par Parallel Feature Extraction
        FeatureExtractor->>FeatureExtractor: Extract Pragmatic Features
        FeatureExtractor->>FeatureExtractor: Extract Acoustic Features
        FeatureExtractor->>FeatureExtractor: Extract Syntactic Features
    end
    
    FeatureExtractor-->>API: FeatureSet with 207+ features
    
    API->>ModelRegistry: get_best_model()
    ModelRegistry-->>API: model and preprocessor
    
    API->>API: Preprocess features
    
    alt Single Model Prediction (Current)
        API->>ComponentModels: model.predict(features)
        ComponentModels-->>API: prediction and probabilities
    else Multi-Model Fusion (Future)
        API->>ComponentModels: Predict with each component
        ComponentModels-->>ModelFusion: Component predictions
        ModelFusion->>ModelFusion: Weighted averaging/Voting/Stacking
        ModelFusion-->>API: Fused prediction
    end
    
    API->>Annotator: annotate(transcript, features)
    
    Annotator->>Annotator: Identify feature locations
    Annotator->>Annotator: Add markers for TURN, PAUSE, etc
    Annotator->>Annotator: Generate HTML with colors
    
    Annotator-->>API: AnnotatedTranscript
    
    API-->>Frontend: prediction, confidence, probabilities, annotated_html
    Frontend-->>User: Display Results and Annotated Transcript
```

---

## 6. API Architecture

```mermaid
graph TB
    subgraph ENDPOINTS["API Endpoints Structure"]
        ROOT["/ - Root Info"]
        HEALTH["/health - Health Check"]
        
        subgraph USER["User Mode Endpoints"]
            PRED_AUDIO["/predict/audio"]
            PRED_TEXT["/predict/text"]
            PRED_TRANS["/predict/transcript"]
        end
        
        subgraph TRAINING["Training Mode Endpoints"]
            TRAIN_DATASETS["/training/datasets"]
            TRAIN_EXTRACT["/training/extract-features"]
            TRAIN_TRAIN["/training/train"]
            TRAIN_STATUS["/training/status"]
            TRAIN_INSPECT["/training/inspect-features"]
        end
        
        subgraph INFO["Information Endpoints"]
            INFO_FEATURES["/features"]
            INFO_MODELS["/models"]
            INFO_COMPONENTS["/components"]
        end
    end
    
    subgraph FLOW["Request Flow"]
        CLIENT[Client Request]
        MIDDLEWARE[CORS Middleware]
        ROUTER[API Router]
        HANDLER[Endpoint Handler]
        ERROR[Error Handler]
    end
    
    subgraph RESPONSE["Response Types"]
        JSON_RESP[JSON Response]
        HTML_RESP[HTML Response]
        FILE_RESP[File Response]
    end
    
    CLIENT --> MIDDLEWARE
    MIDDLEWARE --> ROUTER
    ROUTER --> HANDLER
    HANDLER -->|Success| JSON_RESP
    HANDLER -->|Error| ERROR
    ERROR --> JSON_RESP
    HANDLER --> HTML_RESP
    HANDLER --> FILE_RESP
```

### API Request/Response Models

```mermaid
classDiagram
    class PredictionRequest {
        +Dict features
        +Optional~str~ model_name
    }
    
    class TextPredictionRequest {
        +str text
        +Optional~str~ participant_id
    }
    
    class PredictionResponse {
        +str prediction
        +float confidence
        +Dict probabilities
        +str model_used
    }
    
    class AnnotatedPredictionResponse {
        +str prediction
        +float confidence
        +Dict probabilities
        +int features_extracted
        +str annotated_transcript_html
        +Dict annotation_summary
    }
    
    class FeatureExtractionRequest {
        +List~str~ dataset_paths
        +str output_filename
    }
    
    class TrainingRequest {
        +List~str~ dataset_paths
        +List~str~ model_types
        +str component
    }
    
    class HealthResponse {
        +str status
        +str version
        +int models_available
        +int features_supported
        +bool audio_support
    }
    
    PredictionResponse <|-- AnnotatedPredictionResponse
```

---

## 7. Component Details

### Pragmatic and Conversational Component (Implemented)

```mermaid
graph TD
    subgraph PRAG_COMPONENT["Pragmatic and Conversational Component"]
        MAIN[Main Extractor]
        
        subgraph TEXT_MODS["Text-Based Modules"]
            MOD1["turn_taking.py: 45 features"]
            MOD2["topic_coherence.py: 28 features"]
            MOD3["pause_latency.py: 34 features"]
            MOD4["repair_detection.py: 35 features"]
            MOD5["pragmatic_linguistic.py: 35 features"]
        end
        
        subgraph AUDIO_MOD["Audio-Based Module"]
            MOD6["audio_features.py: 30 features"]
        end
        
        MAIN --> MOD1
        MAIN --> MOD2
        MAIN --> MOD3
        MAIN --> MOD4
        MAIN --> MOD5
        MAIN --> MOD6
        
        MOD1 --> FEATURES1["Turn Frequency, Turn Length Stats, Inter-Turn Gaps, Overlaps, Interruptions"]
        
        MOD2 --> FEATURES2["Topic Coherence, Topic Shifts, Semantic Similarity, LDA Metrics"]
        
        MOD3 --> FEATURES3["Response Latency, Filled Pauses, Unfilled Pauses, Speaking Ratio"]
        
        MOD4 --> FEATURES4["Self Repairs, Other Repairs, Clarifications, Reformulations"]
        
        MOD5 --> FEATURES5["MLU, Vocabulary Diversity, Echolalia, Pronouns"]
        
        MOD6 --> FEATURES6["Audio Pause Duration, Speaking Rate, Response Timing, Silence Patterns"]
    end
    
    style MAIN fill:#90EE90
    style MOD1 fill:#90EE90
    style MOD2 fill:#90EE90
    style MOD3 fill:#90EE90
    style MOD4 fill:#90EE90
    style MOD5 fill:#90EE90
    style MOD6 fill:#87CEEB
```

### Acoustic and Prosodic Component (Placeholder)

```mermaid
graph TD
    subgraph ACOU_COMPONENT["Acoustic and Prosodic Component - Team Member A"]
        ACOU_MAIN[Main Extractor]
        
        ACOU_MAIN --> ACOU_MOD["audio_features.py PLACEHOLDER"]
        
        ACOU_MOD --> FUTURE["To Be Implemented"]
        
        FUTURE -.->|Future| ACOU_FEATURES["Pitch Mean/Std, Jitter/Shimmer, Spectral Centroid, MFCC Features, Energy Distribution"]
    end
    
    style ACOU_MAIN fill:#FFB6C1
    style ACOU_MOD fill:#FFB6C1
    style FUTURE fill:#FFE4E1
```

### Syntactic and Semantic Component (Placeholder)

```mermaid
graph TD
    subgraph SYNT_COMPONENT["Syntactic and Semantic Component - Team Member B"]
        SYNT_MAIN[Main Extractor]
        
        SYNT_MAIN --> SYNT_MOD["audio_features.py PLACEHOLDER"]
        
        SYNT_MOD --> FUTURE["To Be Implemented"]
        
        FUTURE -.->|Future| SYNT_FEATURES["POS Distribution, Dependency Depths, Clause Complexity, Semantic Similarity, Entity Recognition"]
    end
    
    style SYNT_MAIN fill:#FFB6C1
    style SYNT_MOD fill:#FFB6C1
    style FUTURE fill:#FFE4E1
```

---

## 8. Data Flow

### Training Data Flow

```mermaid
flowchart LR
    subgraph INPUT_DATA["Input Data"]
        CHAT_FILES[("CHAT Files: .cha")]
        AUDIO_FILES[("Audio Files: .wav")]
    end
    
    subgraph PROCESSING["Processing"]
        PARSE[Parse or Transcribe]
        EXTRACT[Extract Features]
    end
    
    subgraph FEATURE_STORE["Feature Storage"]
        CSV[("Feature CSVs in output folder")]
    end
    
    subgraph MODEL_TRAINING["Model Training"]
        TRAIN[Train Models]
        EVAL[Evaluate]
    end
    
    subgraph MODEL_STORE["Model Storage"]
        MODELS[("Trained Models in models folder")]
        META[Model Metadata: metrics and version]
    end
    
    CHAT_FILES --> PARSE
    AUDIO_FILES --> PARSE
    PARSE --> EXTRACT
    EXTRACT --> CSV
    CSV --> TRAIN
    TRAIN --> EVAL
    EVAL --> MODELS
    EVAL --> META
    
    style CSV fill:#FFE4B5
    style MODELS fill:#98FB98
```

### Prediction Data Flow

```mermaid
flowchart LR
    subgraph USER_INPUT["User Input"]
        INPUT["Audio, Text, or CHAT File"]
    end
    
    subgraph PROC_PIPELINE["Processing Pipeline"]
        PROC[Input Handler]
        FE[Feature Extractor]
    end
    
    subgraph INFERENCE["Model Inference"]
        LOAD[Load Model from Registry]
        PRED[Predict]
    end
    
    subgraph GEN_OUTPUT["Output Generation"]
        ANNOT[Generate Annotations]
        RESULT[Prediction Result]
    end
    
    subgraph USER_OUTPUT["User Output"]
        DISPLAY["Display: Prediction, Confidence, Annotated Transcript"]
    end
    
    INPUT --> PROC
    PROC --> FE
    FE --> LOAD
    LOAD --> PRED
    FE --> ANNOT
    PRED --> RESULT
    ANNOT --> RESULT
    RESULT --> DISPLAY
    
    style DISPLAY fill:#98FB98
```

---

## 9. Annotation System Flow

```mermaid
flowchart TD
    START([Transcript and Features]) --> ANNOT[TranscriptAnnotator]
    
    ANNOT --> ANALYZE[Analyze Features for Markers]
    
    ANALYZE --> MARKERS{Identify Marker Types}
    
    MARKERS --> TURN["Turn Boundaries: TURN"]
    MARKERS --> PAUSE["Long Pauses: PAUSE"]
    MARKERS --> LATENCY["Response Delays: LATENCY"]
    MARKERS --> REPAIR["Self-Corrections: REPAIR"]
    MARKERS --> TOPIC["Topic Changes: TOPIC"]
    MARKERS --> MARKER["Discourse Markers: MARKER"]
    
    TURN --> POS1[Position in Transcript]
    PAUSE --> POS2[Position in Transcript]
    LATENCY --> POS3[Position in Transcript]
    REPAIR --> POS4[Position in Transcript]
    TOPIC --> POS5[Position in Transcript]
    MARKER --> POS6[Position in Transcript]
    
    POS1 --> HTML[Generate HTML]
    POS2 --> HTML
    POS3 --> HTML
    POS4 --> HTML
    POS5 --> HTML
    POS6 --> HTML
    
    HTML --> STYLE[Apply Styling with Colors and Symbols]
    
    STYLE --> OUTPUT["Annotated HTML, Plain Text, and JSON Format"]
    
    OUTPUT --> USER([Display to User])
    
    style TURN fill:#FFE4B5
    style PAUSE fill:#FFB6C1
    style LATENCY fill:#DDA0DD
    style REPAIR fill:#F0E68C
    style TOPIC fill:#87CEEB
    style MARKER fill:#90EE90
```

---

## 10. Model Fusion Architecture

```mermaid
flowchart TD
    START([New Input for Prediction]) --> PROCESS[Process Input]
    
    PROCESS --> COMP_PRED{Predict with Each Component}
    
    subgraph COMP_PREDICTIONS["Component Predictions"]
        COMP_PRED --> PRAG_P["Pragmatic Model: Prediction and Probability"]
        COMP_PRED --> ACOU_P["Acoustic Model: Prediction and Probability"]
        COMP_PRED --> SYNT_P["Syntactic Model: Prediction and Probability"]
    end
    
    PRAG_P --> FUSION[Model Fusion Layer]
    ACOU_P --> FUSION
    SYNT_P --> FUSION
    
    FUSION --> METHOD{Fusion Method?}
    
    METHOD -->|Weighted| WEIGHTED["Weighted Average: w1*P1 + w2*P2 + w3*P3"]
    METHOD -->|Voting| VOTING[Majority Voting: Most Common Prediction]
    METHOD -->|Stacking| STACKING[Meta-Learner: Train on Component Outputs]
    
    WEIGHTED --> WEIGHTS["Component Weights: Pragmatic 0.5, Acoustic 0.25, Syntactic 0.25"]
    
    WEIGHTS --> CALC[Calculate Final Prediction]
    VOTING --> CALC
    STACKING --> CALC
    
    CALC --> CONFIDENCE[Calculate Confidence Score]
    
    CONFIDENCE --> FINAL["Final Prediction: Class, Confidence, Component Breakdown"]
    
    FINAL --> OUTPUT([Return to User])
    
    style PRAG_P fill:#90EE90
    style ACOU_P fill:#FFB6C1
    style SYNT_P fill:#FFB6C1
    style WEIGHTED fill:#87CEEB
```

---

## 11. File System Structure

```mermaid
graph TD
    ROOT[Artistic Project Root]
    
    ROOT --> SRC[src folder]
    ROOT --> DATA[data folder]
    ROOT --> MODELS_DIR[models folder]
    ROOT --> OUTPUT_DIR[output folder]
    ROOT --> FRONTEND_FILES[frontend files]
    ROOT --> CONFIG[config.py]
    
    SRC --> API["api: app.py"]
    SRC --> AUDIO["audio: transcriber.py, audio_processor.py"]
    SRC --> FEATURES_DIR[features folder]
    SRC --> MODELS_CODE["models: model_trainer.py, model_registry.py"]
    SRC --> PARSERS["parsers: chat_parser.py"]
    SRC --> PIPELINE["pipeline: input_handler.py, annotated_transcript.py, model_fusion.py"]
    SRC --> UTILS["utils: logger.py, helpers.py"]
    
    FEATURES_DIR --> FEAT_BASE["base_features.py, feature_extractor.py"]
    FEATURES_DIR --> FEAT_PRAG["pragmatic_conversational: 6 modules IMPLEMENTED"]
    FEATURES_DIR --> FEAT_ACOU["acoustic_prosodic: audio_features.py PLACEHOLDER"]
    FEATURES_DIR --> FEAT_SYNT["syntactic_semantic: audio_features.py PLACEHOLDER"]
    
    DATA --> DATASET1["asdbank_eigsti: .cha files"]
    DATA --> DATASET2["asdbank_aac: .wav files"]
    DATA --> DATASET3["other datasets"]
    
    MODELS_DIR --> MODEL_FILES["trained models .pkl and metadata .json"]
    
    OUTPUT_DIR --> CSV_FILES["feature set .csv files"]
    
    FRONTEND_FILES --> HTML["frontend.html"]
    FRONTEND_FILES --> TAILWIND["tailwind.config.js"]
    
    style FEAT_PRAG fill:#90EE90
    style FEAT_ACOU fill:#FFB6C1
    style FEAT_SYNT fill:#FFB6C1
```

---

## 12. Key Data Structures

```mermaid
classDiagram
    class TranscriptData {
        +str participant_id
        +str diagnosis
        +List~Utterance~ utterances
        +List~Utterance~ child_utterances
        +int total_utterances
        +Dict metadata
        +float duration
    }
    
    class Utterance {
        +str speaker
        +str text
        +int index
        +float timestamp
        +float duration
        +Dict annotations
    }
    
    class ProcessedInput {
        +InputType input_type
        +TranscriptData transcript_data
        +Optional~Path~ audio_path
        +Optional~TranscriptionResult~ transcription_result
        +str raw_text
        +Optional~Path~ source_path
        +Dict metadata
        +bool has_audio
    }
    
    class TranscriptionResult {
        +str full_text
        +List~Segment~ segments
        +str language
        +Dict metadata
    }
    
    class Segment {
        +str text
        +float start_time
        +float end_time
        +float confidence
        +str speaker
    }
    
    class FeatureSet {
        +Dict~str,float~ features
        +str feature_type
        +Dict metadata
        +int feature_count
    }
    
    class ComponentPrediction {
        +str component
        +str prediction
        +float probability
        +Dict probabilities
        +Dict feature_importance
        +Dict metadata
    }
    
    class FusionResult {
        +str final_prediction
        +float confidence
        +Dict class_probabilities
        +List~ComponentPrediction~ component_predictions
        +Dict fusion_metadata
    }
    
    class AnnotatedTranscript {
        +TranscriptData original_transcript
        +List~Annotation~ annotations
        +to_html() str
        +to_plain_text() str
        +to_json() Dict
    }
    
    TranscriptData "1" *-- "many" Utterance
    ProcessedInput "1" *-- "1" TranscriptData
    ProcessedInput "1" *-- "0..1" TranscriptionResult
    TranscriptionResult "1" *-- "many" Segment
    FusionResult "1" *-- "many" ComponentPrediction
    AnnotatedTranscript "1" *-- "1" TranscriptData
```

---

## Summary

This architecture implements a **modular, multimodal ASD detection system** with:

1. **Flexible Input Handling**: Supports audio (WAV, MP3), text, and CHAT files
2. **Independent Components**: Three feature extraction components that can work independently
3. **Feature Extraction**: 207+ features from pragmatic/conversational analysis (implemented), with provisions for acoustic and syntactic components
4. **Model Fusion**: Combines predictions from multiple component models
5. **Visual Annotations**: Generates intuitive HTML representations showing where features were extracted
6. **Dual-Mode Interface**: User mode for predictions, Training mode for model development
7. **REST API**: Comprehensive FastAPI backend with endpoints for all operations
8. **Extensibility**: Clear structure for team members to implement their components

### Color Legend
- **Green**: Implemented components (Pragmatic and Conversational)
- **Pink**: Placeholder components (Acoustic and Syntactic - to be implemented by team members)
- **Blue**: Audio-related components
- **Yellow**: Storage and data components

### Key Features
- **207 Features** from Pragmatic component across 6 modules
- **Multimodal Processing** for both audio and text inputs
- **Model Registry** for managing trained models
- **Annotated Transcripts** with visual markers (TURN, PAUSE, LATENCY, REPAIR, TOPIC, MARKER)
- **Dual Frontend Modes** for end users and model trainers
- **Component Independence** allowing parallel development by multiple team members
