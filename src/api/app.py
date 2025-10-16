"""
FastAPI Application for ASD Detection

This module provides a REST API for ASD classification predictions
using pragmatic and conversational features.

Endpoints:
- POST /predict: Predict ASD from features
- POST /predict/file: Predict from uploaded CSV file
- POST /predict/transcript: Predict from CHAT transcript
- GET /models: List available models
- GET /models/{model_name}: Get model information
- GET /health: Health check

Author: Bimidu Gunathilake
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import io

from src.models.model_registry import ModelRegistry
from src.parsers.chat_parser import CHATParser
from src.features.feature_extractor import FeatureExtractor
from src.utils.logger import get_logger
from config import config

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ASD Detection API",
    description="API for detecting Autism Spectrum Disorder using pragmatic and conversational features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
model_registry = ModelRegistry()
chat_parser = CHATParser()
feature_extractor = FeatureExtractor(categories='pragmatic_conversational')


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


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: str = Field(..., description="Predicted class (ASD or TD)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Dict[str, float] = Field(..., description="Class probabilities")
    model_used: str = Field(..., description="Name of model used")


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    model_type: str
    version: str
    accuracy: float
    f1_score: float
    n_features: int
    created_at: str
    description: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    models_available: int
    features_supported: int


# Helper functions
def get_model_and_preprocessor(model_name: Optional[str] = None):
    """
    Get model and preprocessor from registry.
    
    Args:
        model_name: Name of model (None = best model)
    
    Returns:
        Tuple of (model, preprocessor, model_name)
    """
    try:
        if model_name is None:
            # Get best model
            model_name, _ = model_registry.get_best_model()
            logger.info(f"Using best model: {model_name}")
        
        # Load model and preprocessor
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
    """
    Make prediction with model.
    
    Args:
        model: Trained model
        features: Feature DataFrame
        model_name: Name of model
    
    Returns:
        Dict with prediction results
    """
    try:
        # Get prediction
        prediction = model.predict(features)[0]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba))
            
            # Get class labels
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


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "ASD Detection API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    try:
        models = model_registry.list_models()
        feature_names = feature_extractor.all_feature_names
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "models_available": len(models),
            "features_supported": len(feature_names)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_from_features(request: PredictionRequest):
    """
    Make prediction from feature dictionary.
    
    Provide feature values as a dictionary.
    """
    logger.info(f"Prediction request received with {len(request.features)} features")
    
    # Get model and preprocessor
    model, preprocessor, model_name = get_model_and_preprocessor(request.model_name)
    
    try:
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Preprocess if preprocessor available
        if preprocessor is not None:
            features_df = preprocessor.transform(features_df)
        
        # Make prediction
        result = make_prediction(model, features_df, model_name)
        
        logger.info(f"Prediction successful: {result['prediction']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/file", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_from_file(
    file: UploadFile = File(...),
    model_name: Optional[str] = None
):
    """
    Make predictions from uploaded CSV file.
    
    CSV should contain feature columns.
    Returns predictions for all rows.
    """
    logger.info(f"File upload request: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are supported"
        )
    
    # Get model and preprocessor
    model, preprocessor, model_name = get_model_and_preprocessor(model_name)
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        logger.info(f"CSV loaded with shape: {df.shape}")
        
        # Preprocess
        if preprocessor is not None:
            df_processed = preprocessor.transform(df)
        else:
            df_processed = df
        
        # Make predictions
        predictions = model.predict(df_processed)
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df_processed)
            confidences = np.max(probabilities, axis=1)
            classes = model.classes_ if hasattr(model, 'classes_') else ['ASD', 'TD']
        else:
            probabilities = np.eye(len(set(predictions)))[predictions]
            confidences = np.ones(len(predictions))
            classes = list(set(predictions))
        
        # Format results
        results = []
        for i, (pred, conf, proba) in enumerate(zip(predictions, confidences, probabilities)):
            results.append({
                'prediction': str(pred),
                'confidence': float(conf),
                'probabilities': {
                    str(cls): float(prob)
                    for cls, prob in zip(classes, proba)
                },
                'model_used': model_name
            })
        
        logger.info(f"Batch prediction successful: {len(results)} predictions")
        
        return results
    
    except Exception as e:
        logger.error(f"File prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File prediction failed: {str(e)}"
        )


@app.post("/predict/transcript", response_model=PredictionResponse, tags=["Prediction"])
async def predict_from_transcript(
    file: UploadFile = File(...),
    model_name: Optional[str] = None
):
    """
    Make prediction from uploaded CHAT transcript file (.cha).
    
    Extracts features and makes prediction.
    """
    logger.info(f"Transcript upload request: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.cha'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .cha (CHAT) files are supported"
        )
    
    # Get model and preprocessor
    model, preprocessor, model_name = get_model_and_preprocessor(model_name)
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.cha') as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # Parse transcript
        transcript = chat_parser.parse_file(tmp_path)
        logger.info(f"Transcript parsed: {transcript.participant_id}")
        
        # Extract features
        feature_set = feature_extractor.extract_from_transcript(transcript)
        features_df = pd.DataFrame([feature_set.features])
        
        logger.info(f"Extracted {len(feature_set.features)} features")
        
        # Preprocess
        if preprocessor is not None:
            features_df = preprocessor.transform(features_df)
        
        # Make prediction
        result = make_prediction(model, features_df, model_name)
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        logger.info(f"Transcript prediction successful: {result['prediction']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Transcript prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Transcript prediction failed: {str(e)}"
        )


@app.get("/models", tags=["Models"])
async def list_models():
    """List all available models."""
    try:
        models = model_registry.list_models()
        
        return {
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )


@app.get("/models/{model_name}", response_model=ModelInfo, tags=["Models"])
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    try:
        metadata = model_registry.get_model_metadata(model_name)
        
        return {
            "model_name": metadata.model_name,
            "model_type": metadata.model_type,
            "version": metadata.version,
            "accuracy": metadata.accuracy,
            "f1_score": metadata.f1_score,
            "n_features": metadata.n_features,
            "created_at": metadata.created_at,
            "description": metadata.description
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}"
        )


@app.get("/features", tags=["Features"])
async def list_features():
    """List all supported features by category."""
    try:
        feature_names = feature_extractor.all_feature_names
        
        return {
            "features": feature_names,
            "count": len(feature_names),
            "categories": {
                "pragmatic_conversational": {
                    "count": len(feature_names),
                    "status": "implemented",
                    "description": "Turn-taking, linguistic, pragmatic, and conversational features"
                },
                "acoustic_prosodic": {
                    "count": 12,
                    "status": "placeholder",
                    "description": "Acoustic and prosodic features from audio (Team Member A)"
                },
                "syntactic_semantic": {
                    "count": 12,
                    "status": "placeholder", 
                    "description": "Syntactic and semantic features from text (Team Member B)"
                }
            },
            "total_features": {
                "implemented": len(feature_names),
                "placeholders": 24,
                "total": len(feature_names) + 24
            }
        }
    except Exception as e:
        logger.error(f"Error listing features: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing features: {str(e)}"
        )


@app.get("/categories", tags=["Features"])
async def list_categories():
    """List all feature categories and their implementation status."""
    try:
        return {
            "categories": {
                "acoustic_prosodic": {
                    "name": "Acoustic & Prosodic Features",
                    "status": "placeholder",
                    "team": "Team Member A",
                    "description": "Features extracted from audio data",
                    "examples": [
                        "mean_pitch", "pitch_std", "speaking_rate",
                        "pause_rate", "intonation_variability"
                    ],
                    "implementation_notes": "Requires audio files and audio processing libraries (librosa, praat)"
                },
                "syntactic_semantic": {
                    "name": "Syntactic & Semantic Features", 
                    "status": "placeholder",
                    "team": "Team Member B",
                    "description": "Features extracted from text and linguistic analysis",
                    "examples": [
                        "avg_dependency_depth", "clause_complexity",
                        "grammatical_error_rate", "semantic_coherence"
                    ],
                    "implementation_notes": "Requires NLP libraries (spaCy, NLTK) and linguistic analysis"
                },
                "pragmatic_conversational": {
                    "name": "Pragmatic & Conversational Features",
                    "status": "implemented",
                    "team": "Current Implementation",
                    "description": "Features extracted from conversational patterns and social language use",
                    "examples": [
                        "mlu_words", "echolalia_ratio", "turn_taking_patterns",
                        "topic_shift_ratio", "pronoun_reversal_count"
                    ],
                    "implementation_notes": "Fully implemented and ready for production use"
                }
            },
            "summary": {
                "total_categories": 3,
                "implemented": 1,
                "placeholders": 2,
                "total_features": {
                    "implemented": 61,
                    "placeholders": 24,
                    "total": 85
                }
            }
        }
    except Exception as e:
        logger.error(f"Error listing categories: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing categories: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("ASD Detection API starting up...")
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

