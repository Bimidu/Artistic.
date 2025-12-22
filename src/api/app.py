"""
FastAPI Application for ASD Detection

This module provides a REST API for ASD classification with support for:
- Audio file uploads and processing
- Text/transcript analysis
- Model training management
- Feature extraction with annotations
- Multi-component model fusion

Endpoints are organized into:
- Prediction endpoints (for users)
- Training endpoints (for model development)
- Feature inspection endpoints
- Health and status endpoints

Author: Bimidu Gunathilake
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import io
import json

from src.models.model_registry import ModelRegistry
from src.parsers.chat_parser import CHATParser
from src.features.feature_extractor import FeatureExtractor
from src.pipeline.input_handler import InputHandler, InputType
from src.pipeline.annotated_transcript import TranscriptAnnotator
from src.pipeline.model_fusion import ModelFusion, ComponentPrediction
from src.utils.logger import get_logger
from config import config

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ASD Detection API",
    description="Multimodal ASD Detection using audio and text analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_registry = ModelRegistry()
chat_parser = CHATParser()
feature_extractor = FeatureExtractor(categories='pragmatic_conversational')
input_handler = None  # Lazy-loaded due to heavy model loading
transcript_annotator = TranscriptAnnotator()
model_fusion = ModelFusion(method='weighted')


def get_input_handler():
    """Lazy-load input handler."""
    global input_handler
    if input_handler is None:
        input_handler = InputHandler()
    return input_handler


# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for predictions from features."""
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature names and values"
    )
    model_name: Optional[str] = Field(
        None,
        description="Name of model to use (None = best model)"
    )


class TextPredictionRequest(BaseModel):
    """Request for prediction from text."""
    text: str = Field(..., description="Text content to analyze")
    participant_id: Optional[str] = Field("CHI", description="Participant ID")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: str = Field(..., description="Predicted class (ASD or TD)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    model_used: str = Field(..., description="Name of model used")


class AnnotatedPredictionResponse(BaseModel):
    """Response with prediction and annotated transcript."""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    features_extracted: int
    annotated_transcript_html: str
    annotation_summary: Dict[str, int]


class FeatureExtractionRequest(BaseModel):
    """Request for feature extraction."""
    dataset_paths: List[str] = Field(..., description="Paths to dataset folders")
    output_filename: str = Field(
        default="training_features.csv",
        description="Output CSV filename"
    )


class TrainingRequest(BaseModel):
    """Request for model training."""
    dataset_paths: List[str] = Field(..., description="Paths to dataset folders")
    model_types: List[str] = Field(
        default=['random_forest', 'xgboost'],
        description="Model types to train"
    )
    component: str = Field(
        default='pragmatic_conversational',
        description="Component to train"
    )


class TrainingStatus(BaseModel):
    """Training status response."""
    status: str
    component: str
    models_trained: List[str]
    best_model: Optional[str]
    metrics: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_available: int
    features_supported: int
    audio_support: bool


# Helper functions
def get_model_and_preprocessor(model_name: Optional[str] = None):
    """Get model and preprocessor from registry."""
    try:
        if model_name is None:
            model_name, _ = model_registry.get_best_model()
            logger.info(f"Using best model: {model_name}")
        
        model, preprocessor = model_registry.load_model(
            model_name,
            load_preprocessor=True
        )
        
        return model, preprocessor, model_name
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )


def make_prediction(
    model: Any,
    features: pd.DataFrame,
    model_name: str
) -> Dict[str, Any]:
    """Make prediction with model."""
    try:
        prediction = model.predict(features)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba))
            
            classes = model.classes_ if hasattr(model, 'classes_') else ['ASD', 'TD']
            probabilities = {
                str(cls): float(prob)
                for cls, prob in zip(classes, proba)
            }
        else:
            confidence = 1.0
            probabilities = {str(prediction): 1.0}
        
        return {
            'prediction': str(prediction),
            'confidence': confidence,
            'probabilities': probabilities,
            'model_used': model_name
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


# ============================================================================
# General Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "ASD Detection API",
        "version": "2.0.0",
        "docs": "/docs",
        "modes": ["user", "training"],
        "supported_inputs": ["audio (.wav)", "transcript (.cha)", "text"]
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        models = model_registry.list_models()
        feature_names = feature_extractor.all_feature_names
        
        return {
            "status": "healthy",
            "version": "2.0.0",
            "models_available": len(models),
            "features_supported": len(feature_names),
            "audio_support": True
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )


# ============================================================================
# User Mode Endpoints - Prediction
# ============================================================================

@app.post("/predict/audio", tags=["User Mode"])
async def predict_from_audio(
    file: UploadFile = File(...),
    participant_id: Optional[str] = Form("CHI")
):
    """
    Predict ASD from uploaded audio file.
    
    Processes the audio through:
    1. Speech-to-text transcription
    2. Feature extraction
    3. ASD prediction
    4. Annotated transcript generation
    """
    logger.info(f"Audio prediction request: {file.filename}")
    
    # Validate file type
    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only audio files (.wav, .mp3, .flac) are supported"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(file.filename).suffix
        ) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = Path(tmp_file.name)
        
        # Process audio
        handler = get_input_handler()
        processed = handler.process(
            tmp_path,
            participant_id=participant_id
        )
        
        # Extract features with audio
        feature_set = feature_extractor.extract_with_audio(
            processed.transcript_data,
            audio_path=processed.audio_path,
            transcription_result=processed.transcription_result
        )
        
        features_df = pd.DataFrame([feature_set.features])
        
        # Get model and make prediction
        model, preprocessor, model_name = get_model_and_preprocessor()
        
        if preprocessor is not None:
            features_df = preprocessor.transform(features_df)
        
        result = make_prediction(model, features_df, model_name)
        
        # Generate annotated transcript
        annotated = transcript_annotator.annotate(
            processed.transcript_data,
            features=feature_set.features
        )
        
        # Clean up temp file
        tmp_path.unlink()
        
        return {
            **result,
            'features_extracted': len(feature_set.features),
            'transcript': processed.raw_text,
            'annotated_transcript_html': annotated.to_html(),
            'annotation_summary': annotated._get_annotation_summary(),
            'input_type': 'audio',
            'duration': processed.metadata.get('duration', 0),
        }
        
    except Exception as e:
        logger.error(f"Audio prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio prediction failed: {str(e)}"
        )


@app.post("/predict/text", tags=["User Mode"])
async def predict_from_text(request: TextPredictionRequest):
    """
    Predict ASD from text input.
    
    Analyzes the provided text and returns prediction with annotations.
    """
    logger.info("Text prediction request")
    
    try:
        # Process text
        handler = get_input_handler()
        processed = handler.process(
            request.text,
            participant_id=request.participant_id
        )
        
        # Extract features
        feature_set = feature_extractor.extract_from_transcript(processed.transcript_data)
        features_df = pd.DataFrame([feature_set.features])
        
        # Get model and make prediction
        model, preprocessor, model_name = get_model_and_preprocessor()
        
        if preprocessor is not None:
            features_df = preprocessor.transform(features_df)
        
        result = make_prediction(model, features_df, model_name)
        
        # Generate annotated transcript
        annotated = transcript_annotator.annotate(
            processed.transcript_data,
            features=feature_set.features
        )
        
        return {
            **result,
            'features_extracted': len(feature_set.features),
            'annotated_transcript_html': annotated.to_html(),
            'annotation_summary': annotated._get_annotation_summary(),
            'input_type': 'text',
        }
        
    except Exception as e:
        logger.error(f"Text prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text prediction failed: {str(e)}"
        )


@app.post("/predict/transcript", tags=["User Mode"])
async def predict_from_transcript(
    file: UploadFile = File(...)
):
    """
    Predict ASD from uploaded CHAT transcript file (.cha).
    """
    logger.info(f"Transcript prediction request: {file.filename}")
    
    if not file.filename.endswith('.cha'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .cha (CHAT) files are supported"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.cha') as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = Path(tmp_file.name)
        
        # Parse transcript
        transcript = chat_parser.parse_file(tmp_path)
        
        # Extract features
        feature_set = feature_extractor.extract_from_transcript(transcript)
        features_df = pd.DataFrame([feature_set.features])
        
        # Get model and make prediction
        model, preprocessor, model_name = get_model_and_preprocessor()
        
        if preprocessor is not None:
            features_df = preprocessor.transform(features_df)
        
        result = make_prediction(model, features_df, model_name)
        
        # Generate annotated transcript
        annotated = transcript_annotator.annotate(
            transcript,
            features=feature_set.features
        )
        
        # Clean up temp file
        tmp_path.unlink()
        
        return {
            **result,
            'participant_id': transcript.participant_id,
            'features_extracted': len(feature_set.features),
            'annotated_transcript_html': annotated.to_html(),
            'annotation_summary': annotated._get_annotation_summary(),
            'input_type': 'chat_file',
        }
        
    except Exception as e:
        logger.error(f"Transcript prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Transcript prediction failed: {str(e)}"
        )


# ============================================================================
# Training Mode Endpoints
# ============================================================================

@app.get("/training/datasets", tags=["Training Mode"])
async def list_datasets():
    """List available dataset folders for training."""
    data_dir = config.paths.data_dir
    
    datasets = []
    for item in data_dir.iterdir():
        if item.is_dir() and item.name.startswith('asdbank'):
            cha_files = list(item.rglob('*.cha'))
            wav_files = list(item.rglob('*.wav'))
            
            datasets.append({
                'name': item.name,
                'path': str(item),
                'chat_files': len(cha_files),
                'audio_files': len(wav_files),
            })
    
    return {
        'data_directory': str(data_dir),
        'datasets': datasets,
        'total_datasets': len(datasets)
    }


@app.post("/training/extract-features", tags=["Training Mode"])
async def extract_features_for_training(request: FeatureExtractionRequest):
    """
    Extract features from specified datasets for training.
    
    Returns the path to the generated feature CSV file.
    """
    logger.info(f"Feature extraction request for {len(request.dataset_paths)} datasets")
    
    all_dfs = []
    
    for dataset_path in request.dataset_paths:
        path = Path(dataset_path)
        if not path.exists():
            path = config.paths.data_dir / dataset_path
        
        if not path.exists():
            logger.warning(f"Dataset path not found: {dataset_path}")
            continue
        
        try:
            df = feature_extractor.extract_from_directory(path)
            if not df.empty:
                df['dataset'] = path.name
                all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error extracting from {dataset_path}: {e}")
    
    if not all_dfs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No features extracted from any dataset"
        )
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to output
    output_path = config.paths.output_dir / request.output_filename
    combined_df.to_csv(output_path, index=False)
    
    return {
        'status': 'success',
        'output_file': str(output_path),
        'total_samples': len(combined_df),
        'features_count': len(feature_extractor.all_feature_names),
        'datasets_processed': len(all_dfs)
    }


@app.post("/training/train", tags=["Training Mode"])
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Initiate model training for a component.
    
    Training runs in the background. Check status with /training/status.
    """
    logger.info(f"Training request for component: {request.component}")
    
    # This would typically start a background training task
    # For now, return a placeholder response
    
    return {
        'status': 'training_initiated',
        'component': request.component,
        'model_types': request.model_types,
        'datasets': request.dataset_paths,
        'message': 'Training started in background. Check /training/status for progress.'
    }


@app.get("/training/status", tags=["Training Mode"])
async def training_status():
    """Get current training status."""
    return {
        'status': 'idle',
        'component': None,
        'models_training': [],
        'progress': 0,
        'message': 'No training in progress'
    }


@app.post("/training/inspect-features", tags=["Training Mode"])
async def inspect_features(
    file: UploadFile = File(...),
    file_type: str = Form("auto")
):
    """
    Inspect feature extraction for a specific file.
    
    Shows where each feature was extracted from, useful for
    verifying feature extraction algorithms.
    """
    logger.info(f"Feature inspection request: {file.filename}")
    
    try:
        # Determine file type
        suffix = Path(file.filename).suffix.lower()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = Path(tmp_file.name)
        
        # Process based on type
        handler = get_input_handler()
        processed = handler.process(tmp_path)
        
        # Extract features
        if processed.has_audio:
            feature_set = feature_extractor.extract_with_audio(
                processed.transcript_data,
                audio_path=processed.audio_path,
                transcription_result=processed.transcription_result
            )
        else:
            feature_set = feature_extractor.extract_from_transcript(
                processed.transcript_data
            )
        
        # Generate detailed annotation
        annotated = transcript_annotator.annotate(
            processed.transcript_data,
            features=feature_set.features,
            include_patterns=True
        )
        
        # Clean up
        tmp_path.unlink()
        
        return {
            'participant_id': processed.transcript_data.participant_id,
            'input_type': processed.input_type.value,
            'total_features': len(feature_set.features),
            'features': feature_set.features,
            'utterance_count': processed.transcript_data.total_utterances,
            'annotated_transcript_html': annotated.to_html(),
            'annotated_transcript_text': annotated.to_plain_text(),
            'annotation_details': annotated.to_json(),
        }
        
    except Exception as e:
        logger.error(f"Feature inspection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature inspection failed: {str(e)}"
        )


# ============================================================================
# Feature and Model Information Endpoints
# ============================================================================

@app.get("/features", tags=["Information"])
async def list_features():
    """List all supported features by category."""
    try:
        feature_names = feature_extractor.all_feature_names
        
        return {
            "features": feature_names,
            "count": len(feature_names),
            "by_category": feature_extractor.feature_count_by_category,
            "categories": {
                "pragmatic_conversational": {
                    "status": "implemented",
                    "description": "Turn-taking, linguistic, pragmatic, and conversational features",
                    "includes_audio": True
                },
                "acoustic_prosodic": {
                    "status": "placeholder",
                    "description": "Acoustic and prosodic features from audio (Team Member A)"
                },
                "syntactic_semantic": {
                    "status": "placeholder", 
                    "description": "Syntactic and semantic features from text (Team Member B)"
                }
            }
        }
    except Exception as e:
        logger.error(f"Error listing features: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing features: {str(e)}"
        )


@app.get("/models", tags=["Information"])
async def list_models():
    """List all available trained models."""
    try:
        models = model_registry.list_models()
        
        model_info = []
        for model_name in models:
            try:
                metadata = model_registry.get_model_metadata(model_name)
                model_info.append({
                    'name': model_name,
                    'type': metadata.model_type,
                    'accuracy': metadata.accuracy,
                    'f1_score': metadata.f1_score,
                    'version': metadata.version,
                })
            except:
                model_info.append({'name': model_name, 'type': 'unknown'})
        
        return {
            "models": model_info,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )


@app.get("/components", tags=["Information"])
async def list_components():
    """List all feature extraction components and their status."""
    return {
        "components": {
            "pragmatic_conversational": {
                "name": "Pragmatic & Conversational",
                "status": "implemented",
                "features": {
                    "turn_taking": 45,
                    "topic_coherence": 28,
                    "pause_latency": 34,
                    "repair_detection": 35,
                    "pragmatic_linguistic": 35,
                    "pragmatic_audio": 30
                },
                "audio_support": True
            },
            "acoustic_prosodic": {
                "name": "Acoustic & Prosodic",
                "status": "placeholder",
                "team": "Team Member A",
                "audio_support": True
            },
            "syntactic_semantic": {
                "name": "Syntactic & Semantic",
                "status": "placeholder",
                "team": "Team Member B",
                "audio_support": False
            }
        },
        "fusion_method": model_fusion.method,
        "fusion_weights": model_fusion.component_weights
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("ASD Detection API v2.0 starting up...")
    logger.info(f"Models directory: {model_registry.registry_dir}")
    logger.info(f"Available models: {len(model_registry.list_models())}")
    logger.info(f"Supported features: {len(feature_extractor.all_feature_names)}")
    logger.info("API ready to serve requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("ASD Detection API shutting down...")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
