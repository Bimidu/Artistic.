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
import os
import pandas as pd
import numpy as np
import ast
from src.interpretability.explainability.shap_manager import SHAPManager
from pathlib import Path
import tempfile
import io
import json
import shutil
import uuid
from fastapi.staticfiles import StaticFiles

from src.models.model_registry import ModelRegistry, ModelMetadata
from src.parsers.chat_parser import CHATParser
from src.features.feature_extractor import FeatureExtractor
from src.pipeline.input_handler import InputHandler, InputType
from src.pipeline.annotated_transcript import TranscriptAnnotator
from src.pipeline.model_fusion import ModelFusion, ComponentPrediction
from src.utils.logger import get_logger
from src.interpretability.counterfactuals.cf_service import generate_counterfactual
from src.interpretability.counterfactuals.train_autoencoder import train_autoencoder
from config import config

logger = get_logger(__name__)
ASSETS_DIR = Path("assets")

# Initialize FastAPI app
app = FastAPI(
    title="ASD Detection API",
    description="Multimodal ASD Detection using audio and text analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.mount(
    "/assets",
    StaticFiles(directory=Path("assets")),
    name="assets"
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
# Include both pragmatic and acoustic features for prediction
# This ensures models trained with acoustic features can be used for prediction
feature_extractor = FeatureExtractor(
    categories=['turn_taking', 'topic_coherence', 'pause_latency', 'repair_detection', 
                'pragmatic_linguistic', 'pragmatic_audio', 'acoustic_prosodic']
)
input_handler = None  # Lazy-loaded due to heavy model loading
transcript_annotator = TranscriptAnnotator()
model_fusion = ModelFusion(method='weighted')

FEATURE_CSV_PATH = Path("assets/feature_explanations/feature_explanations_literal.csv")

def get_input_handler():
    """Lazy-load input handler with smart backend selection."""
    global input_handler
    if input_handler is None:
        import platform
        is_macos = platform.system() == 'Darwin'
        
        # Use faster-whisper on macOS to avoid PyTorch crashes
        # On other platforms, try faster-whisper first, fallback to whisper
        if is_macos:
            logger.info("macOS detected: Using faster-whisper for audio transcription (avoids PyTorch crashes)")
            backend = 'faster-whisper'
        else:
            backend = 'faster-whisper'  # Prefer faster-whisper everywhere
        
        try:
            input_handler = InputHandler(
                transcriber_backend=backend,
                whisper_model_size='tiny',  # Use tiny for faster loading
                device='cpu',
                language='en'
            )
        except Exception as e:
            logger.warning(f"Failed to initialize {backend}, trying fallback backends: {e}")
            # Try fallback backends
            for fallback_backend in ['google', 'vosk']:
                try:
                    logger.info(f"Trying {fallback_backend} as fallback...")
                    input_handler = InputHandler(
                        transcriber_backend=fallback_backend,
                        language='en'
                    )
                    logger.info(f"✓ Initialized with {fallback_backend}")
                    break
                except Exception as fallback_error:
                    logger.debug(f"{fallback_backend} failed: {fallback_error}")
                    continue
            
            if input_handler is None:
                logger.error("All transcription backends failed!")
                raise RuntimeError(
                    "Could not initialize any transcription backend. "
                    "Install faster-whisper: pip install faster-whisper"
                )
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
        description="Output CSV filename (component-specific: pragmatic_conversational_features.csv, etc.)"
    )
    component: Optional[str] = Field(
        default=None,
        description="Component name (pragmatic_conversational, acoustic_prosodic, syntactic_semantic). Auto-detected from filename if not provided."
    )
    max_samples_per_dataset: Optional[int] = Field(
        default=None,
        description="Maximum samples per dataset (for large datasets like TD)"
    )


class TrainingRequest(BaseModel):
    """Request for model training."""
    dataset_names: List[str] = Field(..., description="Dataset names to use from CSV (not paths)")
    model_types: List[str] = Field(
        default=['random_forest', 'xgboost'],
        description="Model types to train"
    )
    component: str = Field(
        default='pragmatic_conversational',
        description="Component to train"
    )
    n_features: Optional[int] = Field(
        default=30,
        description="Number of features to select (None = use all)"
    )
    feature_selection: bool = Field(
        default=True,
        description="Whether to perform feature selection"
    )
    test_size: float = Field(
        default=0.2,
        description="Fraction of data for test set (0.1 to 0.4)"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    custom_hyperparameters: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Custom hyperparameters for each model type"
    )
    enable_autoencoder: Optional[bool] = Field(
        default=None,
        description="Enable counterfactual autoencoder training (None = auto-detect based on OS)"
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
def preprocess_with_dict(df: pd.DataFrame, preprocessor_dict: Dict) -> pd.DataFrame:
    """
    Apply preprocessing using dict format preprocessor.
    
    Args:
        df: Input DataFrame with features
        preprocessor_dict: Dict with 'selected_features', 'cleaner', 'scaler'
    
    Returns:
        Preprocessed DataFrame with selected features
    """
    try:
        # Get selected features
        selected_features = preprocessor_dict.get('selected_features', [])
        feature_columns = preprocessor_dict.get('feature_columns', [])
        
        if not selected_features:
            logger.warning("No selected features in preprocessor, returning all features")
            return df
        
        # Clean data
        cleaner = preprocessor_dict.get('cleaner')
        if cleaner:
            # Fix logger if it's None (can happen after unpickling)
            if not hasattr(cleaner, 'logger') or cleaner.logger is None:
                cleaner.logger = logger
            df = cleaner.clean(df, target_column=None, feature_columns=feature_columns)
        
        # Select only the features the model was trained on
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = [f for f in selected_features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
            # Add missing features with zeros
            for feature in missing_features:
                df[feature] = 0.0
        
        df_selected = df[selected_features]
        
        # Scale features
        scaler = preprocessor_dict.get('scaler')
        if scaler:
            # Fix logger if it's None (can happen after unpickling)
            if not hasattr(scaler, 'logger') or scaler.logger is None:
                scaler.logger = logger
            df_selected = scaler.transform(df_selected, feature_columns=selected_features)
        
        logger.info(f"Preprocessed to {len(selected_features)} selected features")
        return df_selected
        
    except Exception as e:
        logger.error(f"Error in dict preprocessing: {e}", exc_info=True)
        raise


def get_model_and_preprocessor(model_name: Optional[str] = None, component: Optional[str] = None):
    """Get model and preprocessor from registry."""
    try:
        if model_name is None:
            # Get best model, optionally filtered by component
            if component:
                # Find best model for specific component
                models = model_registry.list_models()
                component_models = [m for m in models if m.startswith(component)]
                if component_models:
                    # Get metadata for each and find best F1
                    best_f1 = -1
                    best_model = None
                    for m in component_models:
                        try:
                            metadata = model_registry.get_model_metadata(m)
                            if metadata.f1_score > best_f1:
                                best_f1 = metadata.f1_score
                                best_model = m
                        except:
                            pass
                    model_name = best_model if best_model else component_models[0]
                else:
                    model_name, _ = model_registry.get_best_model()
            else:
                model_name, _ = model_registry.get_best_model()
            logger.info(f"Using best model: {model_name}")
        
        # Validate model exists
        if model_name not in model_registry.list_models():
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model, preprocessor = model_registry.load_model(
            model_name,
            load_preprocessor=True
        )
        
        return model, preprocessor, model_name
    
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )


def get_model_component(model_name: str) -> Optional[str]:
    """Extract component name from model name."""
    if not model_name:
        return None
    parts = model_name.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return None


def is_model_compatible_with_input(model_name: str, input_type: str) -> bool:
    """
    Check if a model is compatible with the input type.

    Args:
        model_name: Name of the model
        input_type: 'audio', 'text', or 'chat_file'

    Returns:
        True if compatible, False otherwise
    """
    component = get_model_component(model_name)

    if input_type == 'audio':
        # Audio can use pragmatic or acoustic models
        return component in ['pragmatic_conversational', 'acoustic_prosodic']
    elif input_type in ['text', 'chat_file']:
        # Text/chat can use pragmatic or semantic models (not acoustic)
        return component in ['pragmatic_conversational', 'syntactic_semantic']

    return True  # Unknown input type, allow it


def get_feature_csv_path(component: str) -> Path:
    """
    Get the path to the feature CSV file for a component.
    
    Args:
        component: Component name (e.g., 'pragmatic_conversational')
        
    Returns:
        Path to the feature CSV file
    """
    filename = f"{component}_features.csv"
    return config.paths.output_dir / filename


def load_features_from_csv(csv_path: Path, dataset_names: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Load features from CSV file, optionally filtered by dataset names.
    
    Args:
        csv_path: Path to the CSV file
        dataset_names: Optional list of dataset names to filter by
        
    Returns:
        DataFrame with features, or None if file doesn't exist
    """
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if dataset_names:
            # Filter to only specified datasets
            if 'dataset' in df.columns:
                df = df[df['dataset'].isin(dataset_names)]
        return df
    except Exception as e:
        logger.warning(f"Error loading features from {csv_path}: {e}")
        return None


def update_features_csv(csv_path: Path, new_features_df: pd.DataFrame, dataset_names: List[str]):
    """
    Update feature CSV by replacing rows for specified datasets with new features.
    
    Args:
        csv_path: Path to the CSV file
        new_features_df: DataFrame with new features (must have 'dataset' column)
        dataset_names: List of dataset names to update
    """
    # Load existing CSV if it exists
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            # Remove rows for datasets being updated
            if 'dataset' in existing_df.columns:
                existing_df = existing_df[~existing_df['dataset'].isin(dataset_names)]
            # Combine with new features
            combined_df = pd.concat([existing_df, new_features_df], ignore_index=True)
        except Exception as e:
            logger.warning(f"Error reading existing CSV, creating new one: {e}")
            combined_df = new_features_df
    else:
        combined_df = new_features_df
    
    # Save updated CSV
    combined_df.to_csv(csv_path, index=False)
    logger.info(f"Updated feature CSV: {csv_path} ({len(combined_df)} total samples)")


def get_component_weights_for_input_type(input_type: str) -> Dict[str, float]:
    """
    Get appropriate component weights based on input type.

    Rules:
    - For audio: acoustic_prosodic has weight, syntactic_semantic has 0 (no semantic features from audio alone)
    - For text/chat: acoustic_prosodic has 0 (no audio), syntactic_semantic has weight
    - pragmatic_conversational works for both

    Args:
        input_type: 'audio', 'text', or 'chat_file'

    Returns:
        Dictionary of component weights
    """
    if input_type == 'audio':
        # Audio input: acoustic works, semantic doesn't (no text analysis)
        return {
            'pragmatic_conversational': 0.4,
            'acoustic_prosodic': 0.4,
            'syntactic_semantic': 0.0,  # No semantic features from audio alone
        }
    elif input_type in ['text', 'chat_file']:
        # Text/chat input: semantic works, acoustic doesn't (no audio)
        return {
            'pragmatic_conversational': 0.4,
            'acoustic_prosodic': 0.0,  # No audio features from text
            'syntactic_semantic': 0.6,
        }
    else:
        # Default: all components have weight
        return {
            'pragmatic_conversational': 0.5,
            'acoustic_prosodic': 0.25,
            'syntactic_semantic': 0.25,
        }


def make_prediction(
    model: Any,
    features: pd.DataFrame,
    model_name: str
) -> Dict[str, Any]:
    """Make prediction with model."""
    try:
        prediction = model.predict(features)[0]
        
        # Convert numeric prediction back to string labels if needed
        if isinstance(prediction, (int, np.integer)):
            label_map = {0: 'TD', 1: 'ASD'}
            prediction_label = label_map.get(prediction, str(prediction))
        else:
            prediction_label = str(prediction)
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba))
            
            # Get class labels
            if hasattr(model, 'classes_'):
                classes = model.classes_
                # Convert numeric classes to string labels
                if isinstance(classes[0], (int, np.integer)):
                    label_map = {0: 'TD', 1: 'ASD'}
                    class_labels = [label_map.get(c, str(c)) for c in classes]
                else:
                    class_labels = [str(c) for c in classes]
            else:
                class_labels = ['ASD', 'TD']
            
            probabilities = {
                str(cls): float(prob)
                for cls, prob in zip(class_labels, proba)
            }
        else:
            confidence = 1.0
            probabilities = {prediction_label: 1.0}
        
        return {
            'prediction': prediction_label,
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
    participant_id: Optional[str] = Form("CHI"),
    model_name: Optional[str] = Form(None),
    use_fusion: bool = Form(False)
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
        
        if use_fusion:
            # Multi-component prediction with fusion for audio
            component_predictions = []

            # Get weights for audio input (acoustic works, semantic doesn't)
            component_weights = get_component_weights_for_input_type('audio')

            # Try each component
            for component in ['pragmatic_conversational', 'acoustic_prosodic', 'syntactic_semantic']:
                # Skip components with zero weight
                if component_weights.get(component, 0) == 0:
                    logger.info(f"Skipping {component} for audio input (weight=0)")
                    continue

                try:
                    # Select feature extractor
                    if component == 'acoustic_prosodic':
                        from src.features.acoustic_prosodic.acoustic_extractor import AcousticFeatureExtractor
                        extractor = AcousticFeatureExtractor()
                        features = extractor.extract_with_audio(
                            processed.transcript_data,
                            audio_path=processed.audio_path,
                            transcription_result=processed.transcription_result
                        ).features
                    elif component == 'syntactic_semantic':
                        # Skip semantic for audio (no text analysis)
                        continue
                    else:  # pragmatic_conversational
                        features = feature_extractor.extract_with_audio(
                            processed.transcript_data,
                            audio_path=processed.audio_path,
                            transcription_result=processed.transcription_result
                        ).features

                    features_df = pd.DataFrame([features])

                    # Get best model for this component (fusion always uses best model per component)
                    # Ignore model_name when fusion is enabled - fusion uses best model from each component
                    model, preprocessor, used_model_name = get_model_and_preprocessor(component=component)

                    if preprocessor is not None:
                        if isinstance(preprocessor, dict):
                            features_df = preprocess_with_dict(features_df, preprocessor)
                        else:
                            features_df = preprocessor.transform(features_df)

                    # Make prediction
                    prediction = model.predict(features_df)[0]
                    proba = model.predict_proba(features_df)[0] if hasattr(model, 'predict_proba') else None

                    if proba is not None:
                        classes = model.classes_ if hasattr(model, 'classes_') else ['ASD', 'TD']
                        # Convert numeric classes to string labels
                        if isinstance(classes[0], (int, np.integer)):
                            label_map = {0: 'TD', 1: 'ASD'}
                            class_labels = [label_map.get(c, str(c)) for c in classes]
                        else:
                            class_labels = [str(c) for c in classes]
                        probabilities = {str(cls): float(prob) for cls, prob in zip(class_labels, proba)}
                        confidence = float(np.max(proba))
                        asd_prob = probabilities.get('ASD', proba[1] if len(proba) > 1 else proba[0])
                    else:
                        probabilities = {str(prediction): 1.0}
                        confidence = 1.0
                        asd_prob = 1.0 if str(prediction).upper() == 'ASD' else 0.0

                    component_predictions.append(ComponentPrediction(
                        component=component,
                        prediction=str(prediction),
                        probability=asd_prob,
                        probabilities=probabilities,
                        confidence=confidence,
                        model_name=used_model_name
                    ))

                    logger.info(f"{component}: {prediction} ({confidence:.2f})")

                except Exception as e:
                    logger.warning(f"Component {component} failed: {e}")
                    continue

            if not component_predictions:
                raise ValueError("No components available for prediction")

            # Fuse predictions with audio-specific weights
            fused = model_fusion.fuse(component_predictions, component_weights_override=component_weights)

            # Generate annotated transcript (from pragmatic component)
            feature_set = feature_extractor.extract_with_audio(
                processed.transcript_data,
                audio_path=processed.audio_path,
                transcription_result=processed.transcription_result
            )
            annotated = transcript_annotator.annotate(
                processed.transcript_data,
                features=feature_set.features
            )

            # Clean up temp file
            tmp_path.unlink()

            return {
                'prediction': fused.final_prediction,
                'confidence': fused.confidence,
                'probabilities': fused.final_probabilities,
                'model_used': 'fusion',
                'models_used': [cp.model_name for cp in component_predictions],  # List all models used
                'component_breakdown': [
                    {
                        'component': cp.component,
                        'prediction': cp.prediction,
                        'confidence': cp.confidence,
                        'probabilities': cp.probabilities,
                        'model_name': cp.model_name
                    }
                    for cp in component_predictions
                ],
                'features_extracted': len(feature_set.features),
                'transcript': processed.raw_text,
                'annotated_transcript_html': annotated.to_html(),
                'annotation_summary': annotated._get_annotation_summary(),
                'input_type': 'audio',
                'duration': processed.metadata.get('duration', 0),
            }
        else:
            # Single component prediction
            # Determine which component/model to use and extract appropriate features
            selected_component = None
            if model_name:
                selected_component = get_model_component(model_name)
                # Validate compatibility
                if not is_model_compatible_with_input(model_name, 'audio'):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model '{model_name}' ({selected_component}) is not compatible with audio input. "
                               f"Audio input requires pragmatic_conversational or acoustic_prosodic models. "
                               f"Please select a compatible model or use 'Best Model (Auto)'."
                    )

            # Extract features based on selected model component
            if selected_component == 'acoustic_prosodic':
                # Extract acoustic features
                from src.features.acoustic_prosodic.acoustic_extractor import AcousticFeatureExtractor
                acoustic_extractor = AcousticFeatureExtractor()
                feature_set = acoustic_extractor.extract_with_audio(
                    processed.transcript_data,
                    audio_path=processed.audio_path,
                    transcription_result=processed.transcription_result
                )
            else:
                # Extract pragmatic features (default for audio)
                feature_set = feature_extractor.extract_with_audio(
                    processed.transcript_data,
                    audio_path=processed.audio_path,
                    transcription_result=processed.transcription_result
                )
            
            features_df = pd.DataFrame([feature_set.features])
            
            # Get model and make prediction (use specified model or best compatible model)
            if model_name:
                model, preprocessor, used_model_name = get_model_and_preprocessor(model_name=model_name)
            else:
                # Get best compatible model for audio
                models = model_registry.list_models()
                compatible_models = [m for m in models if is_model_compatible_with_input(m, 'audio')]
                if not compatible_models:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No compatible models found for audio input. "
                               "Please train a pragmatic_conversational or acoustic_prosodic model first."
                    )
                # Find best compatible model
                best_f1 = -1
                best_model = None
                for m in compatible_models:
                    try:
                        metadata = model_registry.get_model_metadata(m)
                        if metadata.f1_score > best_f1:
                            best_f1 = metadata.f1_score
                            best_model = m
                    except:
                        pass
                model_name = best_model or compatible_models[0]
                model, preprocessor, used_model_name = get_model_and_preprocessor(model_name=model_name)
        
        if preprocessor is not None:
            if isinstance(preprocessor, dict):
                features_df = preprocess_with_dict(features_df, preprocessor)
            else:
                features_df = preprocessor.transform(features_df)
        
            result = make_prediction(model, features_df, used_model_name)

            # LOCAL SHAP (AUDIO)
            request_id = str(uuid.uuid4())
            local_shap_dir = Path("assets/shap/local") / request_id
            local_shap_dir.mkdir(parents=True, exist_ok=True)

            background = np.load(
                Path("assets/shap") / used_model_name / "background.npy"
            )

            predicted_class = 1 if result["prediction"] == "ASD" else 0

            shap_manager = SHAPManager(
                model=model,
                background_data=background,
                feature_names=list(features_df.columns),
                model_type=used_model_name.split("_")[-1]
            )

            shap_manager.generate_local_waterfall(
                X_instance=features_df.values[0],
                save_dir=local_shap_dir,
                predicted_class=predicted_class
            )

            # COUNTERFACTUAL
            component = "_".join(used_model_name.split("_")[:-1])
            logger.info(f"Counterfactual component: {component}")

            cf_result = None
            try:
                cf_result = generate_counterfactual(
                    model=model,
                    x_instance=features_df.values[0],
                    feature_names=list(features_df.columns),
                    component=component,
                    predicted_class=predicted_class
                )
            except FileNotFoundError as e:
                logger.warning(f"Counterfactual skipped: {e}")

            # Generate annotated transcript (use pragmatic features for annotation)
            pragmatic_feature_set = feature_extractor.extract_with_audio(
                processed.transcript_data,
                audio_path=processed.audio_path,
                transcription_result=processed.transcription_result
            )
        annotated = transcript_annotator.annotate(
            processed.transcript_data,
                features=pragmatic_feature_set.features
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
                'model_used': used_model_name,  # Explicitly state which model was used
                'component': get_model_component(used_model_name),
            "local_shap": {
                "request_id": request_id,
                "waterfall": f"/assets/shap/local/{request_id}/waterfall.png"
            },
            "counterfactual": cf_result,
        }
        
    except Exception as e:
        logger.error(f"Audio prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio prediction failed: {str(e)}"
        )


class TextPredictionRequestWithOptions(BaseModel):
    """Request for prediction from text with model selection."""
    text: str = Field(..., description="Text content to analyze")
    participant_id: Optional[str] = Field("CHI", description="Participant ID")
    model_name: Optional[str] = Field(None, description="Specific model to use (None = best model)")
    use_fusion: bool = Field(False, description="Use multi-component fusion")


@app.post("/predict/text", tags=["User Mode"])
async def predict_from_text(request: TextPredictionRequestWithOptions):
    """
    Predict ASD from text input.
    
    Analyzes the provided text and returns prediction with annotations.
    Supports model selection and multi-component fusion.
    """
    logger.info(f"Text prediction request (fusion={request.use_fusion}, model={request.model_name})")
    
    try:
        # Process text
        handler = get_input_handler()
        processed = handler.process(
            request.text,
            participant_id=request.participant_id
        )
        
        if request.use_fusion:
            # Multi-component prediction with fusion for text
            component_predictions = []

            # Get weights for text input (semantic works, acoustic doesn't)
            component_weights = get_component_weights_for_input_type('text')

            # Try each component
            for component in ['pragmatic_conversational', 'acoustic_prosodic', 'syntactic_semantic']:
                # Skip components with zero weight
                if component_weights.get(component, 0) == 0:
                    logger.info(f"Skipping {component} for text input (weight=0)")
                    continue

                try:
                    # Select feature extractor
                    if component == 'acoustic_prosodic':
                        # Skip acoustic for text (no audio)
                        continue
                    elif component == 'syntactic_semantic':
                        from src.features.syntactic_semantic.syntactic_extractor import SyntacticFeatureExtractor
                        extractor = SyntacticFeatureExtractor()
                        features = extractor.extract_from_transcript(processed.transcript_data).features
                    else:  # pragmatic_conversational
                        features = feature_extractor.extract_from_transcript(processed.transcript_data).features

                    features_df = pd.DataFrame([features])

                    # Get best model for this component (or use specified model if it matches)
                    if request.model_name and request.model_name.startswith(component):
                        model, preprocessor, used_model_name = get_model_and_preprocessor(model_name=request.model_name)
                    else:
                        model, preprocessor, used_model_name = get_model_and_preprocessor(component=component)

                    if preprocessor is not None:
                        if isinstance(preprocessor, dict):
                            features_df = preprocess_with_dict(features_df, preprocessor)
                        else:
                            features_df = preprocessor.transform(features_df)

                    # Make prediction
                    prediction = model.predict(features_df)[0]
                    proba = model.predict_proba(features_df)[0] if hasattr(model, 'predict_proba') else None

                    if proba is not None:
                        classes = model.classes_ if hasattr(model, 'classes_') else ['ASD', 'TD']
                        # Convert numeric classes to string labels
                        if isinstance(classes[0], (int, np.integer)):
                            label_map = {0: 'TD', 1: 'ASD'}
                            class_labels = [label_map.get(c, str(c)) for c in classes]
                        else:
                            class_labels = [str(c) for c in classes]
                        probabilities = {str(cls): float(prob) for cls, prob in zip(class_labels, proba)}
                        confidence = float(np.max(proba))
                        asd_prob = probabilities.get('ASD', proba[1] if len(proba) > 1 else proba[0])
                    else:
                        probabilities = {str(prediction): 1.0}
                        confidence = 1.0
                        asd_prob = 1.0 if str(prediction).upper() == 'ASD' else 0.0

                    component_predictions.append(ComponentPrediction(
                        component=component,
                        prediction=str(prediction),
                        probability=asd_prob,
                        probabilities=probabilities,
                        confidence=confidence,
                        model_name=used_model_name
                    ))

                    logger.info(f"{component}: {prediction} ({confidence:.2f})")

                except Exception as e:
                    logger.warning(f"Component {component} failed: {e}")
                    continue

            if not component_predictions:
                raise ValueError("No components available for prediction")

            # Fuse predictions with text-specific weights
            fused = model_fusion.fuse(component_predictions, component_weights_override=component_weights)

            # Generate annotated transcript (from pragmatic component)
            feature_set = feature_extractor.extract_from_transcript(processed.transcript_data)
            annotated = transcript_annotator.annotate(
                processed.transcript_data,
                features=feature_set.features
            )

            return {
                'prediction': fused.final_prediction,
                'confidence': fused.confidence,
                'probabilities': fused.final_probabilities,
                'model_used': 'fusion',
                'models_used': [cp.model_name for cp in component_predictions],  # List all models used
                'component_breakdown': [
                    {
                        'component': cp.component,
                        'prediction': cp.prediction,
                        'confidence': cp.confidence,
                        'probabilities': cp.probabilities,
                        'model_name': cp.model_name
                    }
                    for cp in component_predictions
                ],
                'features_extracted': len(feature_set.features),
                'annotated_transcript_html': annotated.to_html(),
                'annotation_summary': annotated._get_annotation_summary(),
                'input_type': 'text',
            }
        else:
            # Single component prediction
            # Extract features
            feature_set = feature_extractor.extract_from_transcript(processed.transcript_data)
            features_df = pd.DataFrame([feature_set.features])
            
            # Get model and make prediction (use specified model or best model)
            model, preprocessor, used_model_name = get_model_and_preprocessor(model_name=request.model_name)
            
            if preprocessor is not None:
                if isinstance(preprocessor, dict):
                    features_df = preprocess_with_dict(features_df, preprocessor)
                    selected_features = preprocessor["selected_features"]
                else:
                    features_df = preprocessor.transform(features_df)
                    selected_features = preprocessor.selected_features_
            
            result = make_prediction(model, features_df, used_model_name)

            # ============================
            # LOCAL SHAP
            # ============================
            request_id = str(uuid.uuid4())
            local_shap_dir = Path("assets/shap/local") / request_id
            local_shap_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Load background data saved during training
                background_path = Path("assets/shap") / used_model_name / "background.npy"
                if background_path.exists():
                    background = np.load(background_path)

                    predicted_class = 1 if result["prediction"] == "ASD" else 0

                    shap_manager = SHAPManager(
                        model=model,
                        background_data=background,
                        feature_names=selected_features,
                        model_type=used_model_name.split("_")[-1]
                    )

                    shap_manager.generate_local_waterfall(
                        X_instance=features_df.values,
                        save_dir=local_shap_dir,
                        predicted_class=predicted_class
                    )

                    local_shap_data = {
                        'request_id': request_id,
                        'waterfall': f"/assets/shap/local/{request_id}/waterfall.png"
                    }
            except Exception as shap_error:
                logger.warning(f"SHAP explanation not available: {shap_error}")
                # Continue without SHAP

            # ============================
            # COUNTERFACTUAL
            # ============================
            component = "_".join(used_model_name .split("_")[:-1])
            logger.info(component)

            cf_result = generate_counterfactual(
                model=model,
                x_instance=features_df.values[0],
                feature_names=selected_features,
                component=component,
                predicted_class=predicted_class
            )
            
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
                'model_used': used_model_name,  # Explicitly state which model was used
                'component': get_model_component(used_model_name),
                "local_shap": {
                    "request_id": request_id,
                    "waterfall": f"/assets/shap/local/{request_id}/waterfall.png"
                },
                "counterfactual": cf_result,
            }
        
    except Exception as e:
        logger.error(f"Text prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text prediction failed: {str(e)}"
        )


@app.post("/predict/transcript", tags=["User Mode"])
async def predict_from_transcript(
    file: UploadFile = File(...),
    use_fusion: bool = Form(False),
    model_name: Optional[str] = Form(None)
):
    """
    Predict ASD from uploaded CHAT transcript file (.cha).
    
    Args:
        file: CHAT file
        use_fusion: If True, use all available components and fuse predictions
    """
    logger.info(f"Transcript prediction request: {file.filename} (fusion={use_fusion})")
    
    if not file.filename.lower().endswith('.cha'):
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
        
        if use_fusion:
            # Multi-component prediction with fusion for chat file
            component_predictions = []
            
            # Get weights for chat file input (semantic works, acoustic doesn't)
            component_weights = get_component_weights_for_input_type('chat_file')
            
            # Try each component
            for component in ['pragmatic_conversational', 'acoustic_prosodic', 'syntactic_semantic']:
                # Skip components with zero weight
                if component_weights.get(component, 0) == 0:
                    logger.info(f"Skipping {component} for chat file input (weight=0)")
                    continue

                try:
                    # Select feature extractor
                    if component == 'acoustic_prosodic':
                        # Skip acoustic for chat file (no audio)
                        continue
                    elif component == 'syntactic_semantic':
                        from src.features.syntactic_semantic.syntactic_extractor import SyntacticFeatureExtractor
                        extractor = SyntacticFeatureExtractor()
                        features = extractor.extract_from_transcript(transcript).features
                    else:
                        features = feature_extractor.extract_from_transcript(transcript).features
                    
                    features_df = pd.DataFrame([features])
                    
                    # Get best model for this component (fusion always uses best model per component)
                    # Ignore model_name when fusion is enabled - fusion uses best model from each component
                    model, preprocessor, used_model_name = get_model_and_preprocessor(component=component)
                    
                    if preprocessor is not None:
                        if isinstance(preprocessor, dict):
                            features_df = preprocess_with_dict(features_df, preprocessor)
                        else:
                            features_df = preprocessor.transform(features_df)
                    
                    # Make prediction
                    prediction = model.predict(features_df)[0]
                    proba = model.predict_proba(features_df)[0] if hasattr(model, 'predict_proba') else None
                    
                    if proba is not None:
                        classes = model.classes_ if hasattr(model, 'classes_') else ['ASD', 'TD']
                        # Convert numeric classes to string labels
                        if isinstance(classes[0], (int, np.integer)):
                            label_map = {0: 'TD', 1: 'ASD'}
                            class_labels = [label_map.get(c, str(c)) for c in classes]
                        else:
                            class_labels = [str(c) for c in classes]
                        probabilities = {str(cls): float(prob) for cls, prob in zip(class_labels, proba)}
                        confidence = float(np.max(proba))
                        asd_prob = probabilities.get('ASD', proba[1] if len(proba) > 1 else proba[0])
                    else:
                        probabilities = {str(prediction): 1.0}
                        confidence = 1.0
                        asd_prob = 1.0 if str(prediction).upper() == 'ASD' else 0.0
                    
                    component_predictions.append(ComponentPrediction(
                        component=component,
                        prediction=str(prediction),
                        probability=asd_prob,
                        probabilities=probabilities,
                        confidence=confidence,
                        model_name=used_model_name
                    ))
                    
                    logger.info(f"{component}: {prediction} ({confidence:.2f})")
                
                except Exception as e:
                    logger.warning(f"Component {component} failed: {e}")
                    continue
            
            if not component_predictions:
                # Check which components have models available
                available_components = []
                for comp in ['pragmatic_conversational', 'syntactic_semantic']:
                    if component_weights.get(comp, 0) > 0:
                        models = model_registry.list_models()
                        comp_models = [m for m in models if m.startswith(comp)]
                        if comp_models:
                            available_components.append(comp)

                if not available_components:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No compatible models found for CHAT file input with fusion enabled. "
                               "CHAT files require pragmatic_conversational or syntactic_semantic models. "
                               "Please train models for these components first."
                    )
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Fusion failed: Could not get predictions from any component. "
                               f"Available components with models: {', '.join(available_components)}. "
                               f"Please check that models are properly trained and loaded."
                    )
            
            # Fuse predictions with chat file-specific weights
            fused = model_fusion.fuse(component_predictions, component_weights_override=component_weights)
            
            # Generate annotated transcript (from pragmatic component)
            annotated = transcript_annotator.annotate(
                transcript,
                features=feature_extractor.extract_from_transcript(transcript).features
            )
            
            # Clean up temp file
            tmp_path.unlink()
            
            return {
                'prediction': fused.final_prediction,
                'confidence': fused.confidence,
                'probabilities': fused.final_probabilities,
                'model_used': 'fusion',
                'models_used': [cp.model_name for cp in component_predictions],  # List all models used
                'component_breakdown': [
                    {
                        'component': cp.component,
                        'prediction': cp.prediction,
                        'confidence': cp.confidence,
                        'probabilities': cp.probabilities,
                        'model_name': cp.model_name
                    }
                    for cp in component_predictions
                ],
                'participant_id': transcript.participant_id,
                'features_extracted': len(feature_extractor.extract_from_transcript(transcript).features),
                'annotated_transcript_html': annotated.to_html(),
                'annotation_summary': annotated._get_annotation_summary(),
                'input_type': 'chat_file',
            }
        
        else:
            # Single component prediction
            # Validate model compatibility with input type
            if model_name:
                if not is_model_compatible_with_input(model_name, 'chat_file'):
                    component = get_model_component(model_name)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model '{model_name}' ({component}) is not compatible with CHAT file input. "
                               f"CHAT files don't contain audio, so acoustic models cannot be used. "
                               f"Please select a pragmatic_conversational or syntactic_semantic model, "
                               f"or use 'Best Model (Auto)' to automatically select a compatible model."
                    )

            feature_set = feature_extractor.extract_from_transcript(transcript)
            features_df = pd.DataFrame([feature_set.features])
            
            # Get model and make prediction (use specified model or best compatible model)
            if model_name:
                # Validate that the model exists in the registry
                available_models = model_registry.list_models()
                if model_name not in available_models:
                    # Try to find a matching model
                    matching_models = [m for m in available_models if model_name in m]
                    if matching_models:
                        # If there's exactly one match, use it
                        if len(matching_models) == 1:
                            logger.info(f"Model '{model_name}' not found, using matching model: {matching_models[0]}")
                            model_name = matching_models[0]
                        else:
                            # Multiple matches - find compatible ones
                            compatible_matches = [m for m in matching_models if is_model_compatible_with_input(m, 'chat_file')]
                            if compatible_matches:
                                # Use the best compatible match
                                best_f1 = -1
                                best_match = None
                                for m in compatible_matches:
                                    try:
                                        metadata = model_registry.get_model_metadata(m)
                                        if metadata.f1_score > best_f1:
                                            best_f1 = metadata.f1_score
                                            best_match = m
                                    except:
                                        pass
                                if best_match:
                                    logger.info(f"Model '{model_name}' not found, using best compatible match: {best_match}")
                                    model_name = best_match
                                else:
                                    raise HTTPException(
                                        status_code=status.HTTP_400_BAD_REQUEST,
                                        detail=f"Model '{model_name}' not found. Found {len(matching_models)} similar models: {', '.join(matching_models[:5])}. "
                                               f"Please select a full model name from the dropdown."
                                    )
                            else:
                                raise HTTPException(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"Model '{model_name}' not found and no compatible matches. "
                                           f"Available models: {', '.join(available_models[:10])}. "
                                           f"Please select a full model name from the dropdown."
                                )
                    else:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Model '{model_name}' not found in registry. "
                                   f"Available models: {', '.join(available_models[:10])}. "
                                   f"Please select a model from the dropdown."
                        )

                # Now validate compatibility
                if not is_model_compatible_with_input(model_name, 'chat_file'):
                    component = get_model_component(model_name)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Model '{model_name}' ({component}) is not compatible with CHAT file input. "
                               f"CHAT files don't contain audio, so acoustic models cannot be used. "
                               f"Please select a pragmatic_conversational or syntactic_semantic model, "
                               f"or use 'Best Model (Auto)' to automatically select a compatible model."
                    )

                model, preprocessor, used_model_name = get_model_and_preprocessor(model_name=model_name)
            else:
                # Get best model, but only from compatible components
                models = model_registry.list_models()
                compatible_models = [m for m in models if is_model_compatible_with_input(m, 'chat_file')]
                if not compatible_models:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="No compatible models found for CHAT file input. "
                               "Please train a pragmatic_conversational or syntactic_semantic model first."
                    )
                # Find best compatible model
                best_f1 = -1
                best_model = None
                for m in compatible_models:
                    try:
                        metadata = model_registry.get_model_metadata(m)
                        if metadata.f1_score > best_f1:
                            best_f1 = metadata.f1_score
                            best_model = m
                    except:
                        pass
                model_name = best_model or compatible_models[0]
                model, preprocessor, used_model_name = get_model_and_preprocessor(model_name=model_name)
            
            if preprocessor is not None:
                if isinstance(preprocessor, dict):
                    features_df = preprocess_with_dict(features_df, preprocessor)
                    selected_features = preprocessor["selected_features"]
                else:
                    features_df = preprocessor.transform(features_df)
                    selected_features = preprocessor.selected_features_
            
            result = make_prediction(model, features_df, used_model_name)

            # SHAP explanation (optional, may not be available for all models)
            request_id = str(uuid.uuid4())
            local_shap_dir = Path("assets/shap/local") / request_id
            local_shap_data = None

            try:
                # Load background data saved during training
                background_path = Path("assets/shap") / used_model_name / "background.npy"
                if background_path.exists():
                    background = np.load(background_path)

                    predicted_class = 1 if result["prediction"] == "ASD" else 0

                    shap_manager = SHAPManager(
                        model=model,
                        background_data=background,
                        feature_names=selected_features,
                        model_type=used_model_name.split("_")[-1]
                    )

                    shap_manager.generate_local_waterfall(
                        X_instance=features_df.values,
                        save_dir=local_shap_dir,
                        predicted_class=predicted_class
                    )

                    local_shap_data = {
                        'request_id': request_id,
                        'waterfall': f"/assets/shap/local/{request_id}/waterfall.png"
                    }
            except Exception as shap_error:
                logger.warning(f"SHAP explanation not available: {shap_error}")
                # Continue without SHAP

            #Generate Counterfactuals
            component = "_".join(used_model_name.split("_")[:-1])
            logger.info(component)
            cf_result = generate_counterfactual(
                model=model,
                x_instance=features_df.values[0],
                feature_names=selected_features,
                component=component,
                predicted_class=predicted_class
            )
            
            # Generate annotated transcript
            annotated = transcript_annotator.annotate(
                transcript,
                features=feature_set.features
            )
            
            # Clean up temp file
            tmp_path.unlink()
            
            response_data = {
                **result,
                'participant_id': transcript.participant_id,
                'features_extracted': len(feature_set.features),
                'annotated_transcript_html': annotated.to_html(),
                'annotation_summary': annotated._get_annotation_summary(),
                'input_type': 'chat_file',
                'model_used': used_model_name,  # Explicitly state which model was used
                'component': get_model_component(used_model_name),
            }

            # Add SHAP data if available
            if local_shap_data:
                response_data['local_shap'] = local_shap_data

            if cf_result:
                response_data['counterfactual'] = cf_result

            return response_data

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions as-is (they already have proper error messages)
        # But ensure detail is not empty
        if not http_exc.detail or http_exc.detail.strip() == '':
            http_exc.detail = f"Transcript prediction failed: {type(http_exc).__name__}"
        raise
    except Exception as e:
        logger.error(f"Transcript prediction failed: {e}", exc_info=True)
        # Get error message - try multiple ways
        error_msg = str(e) if str(e) else repr(e)
        if not error_msg or error_msg.strip() == '':
            error_msg = f"{type(e).__name__}: An error occurred during prediction"

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Transcript prediction failed: {error_msg}"
        )


# ============================================================================
# Training Mode Endpoints
# ============================================================================

@app.get("/training/datasets", tags=["Training Mode"])
async def list_datasets():
    """List available dataset folders for feature extraction."""
    data_dir = config.paths.data_dir
    datasets = []

    allowed_names = set(config.datasets.datasets)
    allowed_prefixes = ("asdbank", "td", "child")

    for item in data_dir.iterdir():
        if not item.is_dir():
            continue

        name_matches = (
            item.name in allowed_names
            or any(item.name.startswith(prefix) for prefix in allowed_prefixes)
        )

        if not name_matches:
            continue

        cha_files = list(item.rglob("*.cha"))
        wav_files = list(item.rglob("*.wav"))

        datasets.append({
            "name": item.name,
            "path": str(item),
            "chat_files": len(cha_files),
            "audio_files": len(wav_files),
        })

    return {
        "data_directory": str(data_dir),
        "datasets": datasets,
        "total_datasets": len(datasets),
    }


@app.get("/training/available-datasets/{component}", tags=["Training Mode"])
async def list_available_datasets_in_csv(component: str):
    """
    List datasets available in the feature CSV for a component.
    
    This is used for training - shows which datasets have features available.
    """
    feature_csv_path = get_feature_csv_path(component)
    
    if not feature_csv_path.exists():
        return {
            "component": component,
            "csv_exists": False,
            "datasets": [],
            "total_datasets": 0,
            "total_samples": 0,
            "message": f"No features CSV found for {component}. Please extract features first."
        }
    
    try:
        df = pd.read_csv(feature_csv_path)
        
        if 'dataset' not in df.columns:
            return {
                "component": component,
                "csv_exists": True,
                "datasets": [],
                "total_datasets": 0,
                "total_samples": len(df),
                "message": "CSV exists but has no 'dataset' column. Cannot identify datasets."
            }
        
        # Get unique datasets with sample counts
        dataset_counts = df['dataset'].value_counts().to_dict()
        datasets = [
            {
                "name": name,
                "samples": int(count)
            }
            for name, count in dataset_counts.items()
        ]
        
        return {
            "component": component,
            "csv_exists": True,
            "csv_path": str(feature_csv_path),
            "datasets": datasets,
            "total_datasets": len(datasets),
            "total_samples": len(df),
            "message": f"Found {len(datasets)} dataset(s) with {len(df)} total samples"
        }
    except Exception as e:
        logger.error(f"Error reading feature CSV: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading feature CSV: {str(e)}"
        )



@app.post("/training/extract-features", tags=["Training Mode"])
async def extract_features_for_training(request: FeatureExtractionRequest):
    """
    Extract features from specified datasets for training.
    
    Supports max_samples_per_dataset for large datasets (e.g., TD with 4000+ files).
    Updates existing CSV by overwriting only the relevant dataset's features.
    
    Returns the path to the generated/updated feature CSV file.
    """
    logger.info(f"Feature extraction request for {len(request.dataset_paths)} datasets")
    if request.max_samples_per_dataset:
        logger.info(f"Max samples per dataset: {request.max_samples_per_dataset}")
    
    # Determine component from request, filename, or default to pragmatic
    if request.component:
        component = request.component
    elif 'acoustic' in request.output_filename.lower():
        component = 'acoustic_prosodic'
    elif 'syntactic' in request.output_filename.lower() or 'semantic' in request.output_filename.lower():
        component = 'syntactic_semantic'
    else:
        component = 'pragmatic_conversational'  # Default
    
    logger.info(f"Using component: {component}")
    
    # Select appropriate feature extractor
    if component == 'acoustic_prosodic':
        from src.features.acoustic_prosodic.acoustic_extractor import AcousticFeatureExtractor
        extractor = AcousticFeatureExtractor()
    elif component == 'syntactic_semantic':
        from src.features.syntactic_semantic.syntactic_extractor import SyntacticFeatureExtractor
        extractor = SyntacticFeatureExtractor()
    else:
        extractor = feature_extractor
    
    all_dfs = []
    dataset_names = []
    
    for dataset_path in request.dataset_paths:
        path = Path(dataset_path)
        if not path.exists():
            path = config.paths.data_dir / dataset_path
        
        if not path.exists():
            logger.warning(f"Dataset path not found: {dataset_path}")
            continue
        
        dataset_name = path.name
        dataset_names.append(dataset_name)
        
        try:
            # Check if this is a large dataset (TD)
            is_td_dataset = 'td' in path.name.lower()
            
            # Use request parameter, or fall back to config default
            max_samples = request.max_samples_per_dataset or config.datasets.max_samples_td
            max_samples_for_extraction = max_samples if is_td_dataset else None
            
            logger.info(f"Extracting {component} features from {dataset_name}...")
            
            # Extract features
            if component == 'acoustic_prosodic' and hasattr(extractor, 'extract_from_directory'):
                df = extractor.extract_from_directory(path, max_samples=max_samples_for_extraction)
            else:
                df = extractor.extract_from_directory(path)
                # Sample after extraction for other extractors
                if is_td_dataset and max_samples and len(df) > max_samples:
                    logger.info(f"Sampling {max_samples} from {len(df)} TD samples")
                    df = df.sample(n=max_samples, random_state=42)
            
            if not df.empty:
                df['dataset'] = dataset_name
                all_dfs.append(df)
                logger.info(f"Extracted {len(df)} samples from {dataset_name}")
        except Exception as e:
            logger.error(f"Error extracting from {dataset_path}: {e}")
    
    if not all_dfs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No features extracted from any dataset"
        )
    
    # Combine all DataFrames
    new_features_df = pd.concat(all_dfs, ignore_index=True)
    
    # Update CSV (overwrite only relevant datasets)
    output_path = config.paths.output_dir / request.output_filename
    update_features_csv(output_path, new_features_df, dataset_names)
    
    # Count actual features in the dataframe (exclude metadata columns)
    metadata_cols = ['participant_id', 'file_path', 'diagnosis', 'age_months', 'dataset']
    actual_feature_cols = [col for col in new_features_df.columns if col not in metadata_cols]
    
    # Get total samples in CSV after update
    total_samples = len(pd.read_csv(output_path)) if output_path.exists() else len(new_features_df)
    
    return {
        'status': 'success',
        'output_file': str(output_path),
        'total_samples': total_samples,
        'new_samples': len(new_features_df),
        'features_count': len(actual_feature_cols),
        'datasets_processed': len(all_dfs),
        'datasets_updated': dataset_names
    }

@app.get("/explain/shap/global/{model_name}", tags=["Interpretability"])
async def get_global_shap(model_name: str):
    shap_dir = Path("assets/shap") / model_name

    if not shap_dir.exists():
        raise HTTPException(404, "Global SHAP not found")

    return {
        "beeswarm": f"/static/shap/{model_name}/global_beeswarm.png",
        "bar": f"/static/shap/{model_name}/global_bar.png"
    }

# Global training state
training_state = {
    'status': 'idle',  # idle, training, completed, error
    'component': None,
    'model_types': [],
    'progress': 0,
    'current_model': None,
    'total_models': 0,
    'message': 'No training in progress',
    'results': {},
    'error': None
}

def force_numeric_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """
    HARD sanitize dataframe → guarantees float values.
    Handles:
    - "[0.77]"
    - "[7.7227724E-1]"
    - "['0.77']"
    - numpy arrays
    - lists / tuples
    """

    X = X.copy()

    def sanitize(v):
        # Already numeric
        if isinstance(v, (int, float, np.number)):
            return float(v)

        # numpy array / list / tuple
        if isinstance(v, (list, tuple, np.ndarray)):
            return float(v[0])

        # string case
        if isinstance(v, str):
            try:
                parsed = ast.literal_eval(v)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return float(parsed[0])
                return float(parsed)
            except Exception:
                return float(v)

        # fallback
        return float(v)

    for col in X.columns:
        try:
            X[col] = X[col].apply(sanitize)
        except Exception as e:
            bad_vals = X[col][~X[col].apply(lambda x: isinstance(x, (int, float, np.number)))].head(5)
            raise ValueError(
                f"❌ Non-numeric values remain in column '{col}': {bad_vals.tolist()}"
            ) from e

    return X

def run_training_task(dataset_names: List[str], model_types: List[str], component: str, n_features: int = 30, feature_selection: bool = True, test_size: float = 0.2, random_state: int = 42, custom_hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None, enable_autoencoder: Optional[bool] = None):
    """Background task for model training."""
    global training_state
    
    try:
        training_state['status'] = 'training'
        training_state['component'] = component
        training_state['model_types'] = model_types
        training_state['total_models'] = len(model_types)
        training_state['progress'] = 0
        training_state['message'] = 'Loading feature data...'
        training_state['results'] = {}
        training_state['error'] = None
        
        logger.info(f"Starting training: component={component}, models={model_types}, n_features={n_features}, feature_selection={feature_selection}")
        
        # Step 1: Load features from CSV (training NEVER extracts features)
        feature_csv_path = get_feature_csv_path(component)
        
        if not feature_csv_path.exists():
            raise ValueError(
                f"No features CSV found for {component}. "
                f"Please extract features first using the 'Extract Features' button. "
                f"Expected CSV: {feature_csv_path}"
            )
        
        logger.info(f"Loading features from CSV: {feature_csv_path}")
        training_state['message'] = f'Loading features from CSV for {component}...'
        
        # Load features for selected datasets
        existing_df = load_features_from_csv(feature_csv_path, dataset_names)
        
        if existing_df is None or existing_df.empty:
            available_datasets = []
            if feature_csv_path.exists():
                try:
                    full_df = pd.read_csv(feature_csv_path)
                    if 'dataset' in full_df.columns:
                        available_datasets = full_df['dataset'].unique().tolist()
                except:
                    pass
            
            raise ValueError(
                f"No features found for selected datasets: {', '.join(dataset_names)}. "
                f"Available datasets in CSV: {', '.join(available_datasets) if available_datasets else 'none'}. "
                f"Please extract features for these datasets first."
            )
        
        # Filter to only selected datasets
        if 'dataset' in existing_df.columns:
            existing_datasets = set(existing_df['dataset'].unique())
            missing_datasets = [d for d in dataset_names if d not in existing_datasets]
            
            if missing_datasets:
                raise ValueError(
                    f"Some selected datasets are not in the CSV: {', '.join(missing_datasets)}. "
                    f"Available datasets: {', '.join(sorted(existing_datasets))}. "
                    f"Please extract features for missing datasets first."
                )
            
            # Filter to only selected datasets
            all_dfs = []
            for dataset_name in dataset_names:
                dataset_df = existing_df[existing_df['dataset'] == dataset_name].copy()
                if not dataset_df.empty:
                    all_dfs.append(dataset_df)
                    logger.info(f"Loaded {len(dataset_df)} samples for {dataset_name} from CSV")
                else:
                    logger.warning(f"No samples found for {dataset_name} in CSV")
        else:
            # No dataset column, use all features
            all_dfs = [existing_df]
            logger.warning("CSV has no 'dataset' column, using all features")
        
        if not all_dfs:
            raise ValueError("No features available for selected datasets. Please extract features first.")
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined features: {len(combined_df)} samples")
        
        # Clean up the data before preprocessing
        # 1. Drop non-numeric columns that aren't needed for training
        cols_to_drop = ['participant_id', 'file_path', 'age_months', 'dataset']
        combined_df = combined_df.drop(columns=[col for col in cols_to_drop if col in combined_df.columns])
        
        # 2. Filter out samples with missing diagnosis
        if 'diagnosis' in combined_df.columns:
            before_count = len(combined_df)
            combined_df = combined_df[combined_df['diagnosis'].notna()]
            after_count = len(combined_df)
            dropped = before_count - after_count
            if dropped > 0:
                logger.warning(f"Dropped {dropped} samples with missing diagnosis")
                training_state['message'] = f'Filtered data: removed {dropped} samples without labels'
            
            # 3. Filter out samples where diagnosis looks like age (contains : or .)
            combined_df = combined_df[~combined_df['diagnosis'].astype(str).str.contains(r'[:\.]', na=False)]
            
            # 4. Normalize diagnosis values
            # Common variations: ASD, TD, TYP (typically developing)
            diagnosis_map = {
                'TYP': 'TD',  # Map TYP to TD for consistency
                'TYPICAL': 'TD',
                'CONTROL': 'TD',
            }
            combined_df['diagnosis'] = combined_df['diagnosis'].replace(diagnosis_map)
            
            # 5. For binary classification, keep only ASD and TD/TYP
            unique_diagnoses = combined_df['diagnosis'].unique()
            logger.info(f"Diagnosis labels found: {unique_diagnoses}")
            
            # If we have multiple classes, filter to binary (ASD vs TD)
            if len(unique_diagnoses) > 2:
                logger.info("Multiple diagnosis labels found, filtering to binary classification (ASD vs TD)")
                combined_df = combined_df[combined_df['diagnosis'].isin(['ASD', 'TD'])]
                training_state['message'] = 'Filtered to binary classification: ASD vs TD'
            
            # Convert string labels to numeric for XGBoost/LightGBM compatibility
            # ASD = 1, TD = 0
            label_map = {'TD': 0, 'ASD': 1}
            combined_df['diagnosis'] = combined_df['diagnosis'].map(label_map)
            
            logger.info(f"After cleaning: {len(combined_df)} samples with labels {combined_df['diagnosis'].unique()}")
            
            # Check if we have at least 2 classes for binary classification
            unique_labels = combined_df['diagnosis'].unique()
            if len(unique_labels) < 2:
                class_name = 'TD' if 0 in unique_labels else 'ASD'
                raise ValueError(
                    f"Cannot train binary classifier with only one class: {class_name}. "
                    f"Need both ASD and TD samples. "
                    f"Found {len(combined_df)} samples, all labeled as {class_name}. "
                    f"Please include datasets with both ASD and TD samples for training."
                )
        
        if len(combined_df) < 10:
            raise ValueError(f"Insufficient samples after filtering: {len(combined_df)} (need at least 10)")
        
        # Step 2: Preprocess data
        training_state['progress'] = 10
        training_state['message'] = 'Preprocessing data...'
        
        from src.preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor(
            target_column='diagnosis',
            test_size=test_size,
            random_state=random_state,
            feature_selection=feature_selection,
            n_features=n_features if n_features else 218  # Use all if None
        )
        
        # Fit and transform - skip validation since we already cleaned the data
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(combined_df, validate=False)
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Feature selection: {feature_selection}, Features used: {X_train.shape[1]}")
        
        # Save preprocessor as dict to avoid pickling issues
        # Remove logger references to make it picklable
        preprocessor_dict = {
            'feature_columns': preprocessor.feature_columns_,
            'selected_features': preprocessor.selected_features_,
            'scaler': preprocessor.scaler,
            'cleaner': preprocessor.cleaner,
            'target_column': preprocessor.target_column
        }
        
        # Remove logger attributes to make objects picklable
        if hasattr(preprocessor_dict['cleaner'], 'logger'):
            preprocessor_dict['cleaner'].logger = None
        if hasattr(preprocessor_dict['scaler'], 'logger'):
            preprocessor_dict['scaler'].logger = None

        
        # Step 3: Train models
        from src.models import ModelTrainer, ModelConfig, ModelEvaluator
        trainer = ModelTrainer()
        evaluator = ModelEvaluator()
        
        trained_models = {}
        model_reports = {}
        
        for i, model_type in enumerate(model_types):
            training_state['current_model'] = model_type
            training_state['progress'] = 10 + int((i / len(model_types)) * 80)
            training_state['message'] = f'Training {model_type}...'
            
            logger.info(f"Training model: {model_type}")
            
            # Get custom hyperparameters if provided
            hyperparams = {}
            if custom_hyperparameters and model_type in custom_hyperparameters:
                hyperparams = custom_hyperparameters[model_type]
                logger.info(f"Using custom hyperparameters for {model_type}: {hyperparams}")
            
            config_obj = ModelConfig(
                model_type=model_type,
                hyperparameters=hyperparams,
                tune_hyperparameters=False
            )
            
            model = trainer.train_model(X_train, y_train, config_obj)
            trained_models[model_type] = model
            
            # Evaluate
            training_state['message'] = f'Evaluating {model_type}...'
            report = evaluator.evaluate(
                model,
                X_test,
                y_test,
                model_name=model_type,
                X_train=X_train,
                y_train=y_train
            )
            model_reports[model_type] = report
            
            logger.info(f"{model_type} - Accuracy: {report.accuracy:.4f}, F1: {report.f1_score:.4f}")
        
        # Step 4: Save models to registry
        training_state['progress'] = 90
        training_state['message'] = 'Saving models...'
        
        for model_type, model in trained_models.items():
            report = model_reports[model_type]
            
            # Create model name with component prefix
            model_name = f"{component}_{model_type}"

            # =====================================================
            # TRAIN COUNTERFACTUAL AUTOENCODER (ONCE PER COMPONENT)
            # =====================================================
            # Note: This is optional and may crash on some systems (e.g., macOS ARM64 with PyTorch)
            # Disabled by default on macOS due to PyTorch segfault issues
            # Can be controlled via UI checkbox or ENABLE_COUNTERFACTUAL_AE environment variable

            import platform
            is_macos = platform.system() == "Darwin"

            # Priority: UI setting > environment variable > OS-based default
            if enable_autoencoder is None:
                env_setting = os.getenv("ENABLE_COUNTERFACTUAL_AE", "").lower()
                if env_setting == "":
                    # Default: disabled on macOS, enabled elsewhere
                    enable_autoencoder_flag = not is_macos
                else:
                    enable_autoencoder_flag = env_setting == "true"
            else:
                enable_autoencoder_flag = enable_autoencoder

            if enable_autoencoder_flag:
                ae_dir = Path("models/counterfactuals")
                ae_dir.mkdir(parents=True, exist_ok=True)
                ae_path = ae_dir / f"{model_name}_ae.pt"

                # Train only if not already trained
                if not ae_path.exists():
                    try:
                        logger.info(f"Training counterfactual autoencoder for {component}")
                        train_autoencoder(
                            X_train.values,  # IMPORTANT: already preprocessed + feature-selected
                            model_name,
                            ae_dir
                        )
                        logger.info(f"Counterfactual autoencoder trained successfully for {component}")
                    except Exception as ae_error:
                        # Autoencoder training is optional - log warning but continue training
                        logger.warning(
                            f"Failed to train counterfactual autoencoder for {model_name}: {ae_error}. "
                            f"Training will continue without counterfactual support. "
                            f"If you see segmentation faults, set ENABLE_COUNTERFACTUAL_AE=false to disable. "
                            f"Counterfactual explanations will not be available for predictions."
                        )
                else:
                    logger.info(f"Autoencoder already exists for {model_name}, skipping training")
            else:
                logger.info(f"Counterfactual autoencoder training is disabled (ENABLE_COUNTERFACTUAL_AE=false)")

            SHAP_SUPPORTED_MODELS = {
                "random_forest",
                "gradient_boosting",
                "adaboost",
                "svm",
                "lightgbm",
                "xgboost",
                "logistic",
            }

            logger.warning(f"Calling SHAP for model {model_name}, X_train shape: {X_train.shape}")

            # ================================
            # GLOBAL SHAP (TRAINING TIME)
            # ================================

            if model_type in SHAP_SUPPORTED_MODELS:
                try:
                    shap_dir = config.paths.shap_dir / model_name
                    shap_dir.mkdir(parents=True, exist_ok=True)

                    # Small, safe background (important for Kernel SHAP)
                    background = X_train.sample(
                        n=min(50, len(X_train)),
                        random_state=42
                    )

                    np.save(shap_dir / "background.npy", background.values)

                    logger.info(f"Saving global SHAP to: {shap_dir}")

                    shap_manager = SHAPManager(
                        model=model,
                        background_data=X_train,
                        feature_names=preprocessor.selected_features_,
                        model_type=model_type
                    )

                    shap_manager.generate_global_explanations(
                        X_train=X_train,
                        save_dir=shap_dir
                    )

                    logger.info(f" SHAP generated for {model_name}")

                except Exception as shap_error:
                    logger.warning(
                        f" SHAP failed for {model_name}: {shap_error}"
                    )

            else:
                logger.info(
                    f" Skipping SHAP for model {model_name} "
                    f"(unsupported model type: {model_type})"
                )
            
            metadata = ModelMetadata(
                model_name=model_name,
                model_type=model_type,
                accuracy=float(report.accuracy),
                f1_score=float(report.f1_score),
                precision=float(report.precision),
                recall=float(report.recall),
                roc_auc=float(report.roc_auc) if report.roc_auc is not None else None,
                matthews_corr=float(report.matthews_corr),
                confusion_matrix=report.confusion_matrix.tolist() if len(report.confusion_matrix) > 0 else [],
                n_features=len(preprocessor.selected_features_),
                training_samples=len(X_train),
                feature_names=preprocessor.selected_features_,
                description=f"{component} component - {model_type}",
                component=component
            )
            
            model_registry.register_model(model, metadata, preprocessor=preprocessor_dict)
            
            training_state['results'][model_name] = {
                'accuracy': float(report.accuracy),
                'f1_score': float(report.f1_score),
                'precision': float(report.precision),
                'recall': float(report.recall),
                'component': component
            }
        
        # Complete
        training_state['status'] = 'completed'
        training_state['progress'] = 100
        training_state['message'] = f'Training completed! Trained {len(trained_models)} models.'
        training_state['current_model'] = None
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        training_state['status'] = 'error'
        training_state['error'] = str(e)
        training_state['message'] = f'Training failed: {str(e)}'


@app.post("/training/train", tags=["Training Mode"])
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Initiate model training for a component.
    
    Training runs in the background. Check status with /training/status.
    """
    global training_state
    
    if training_state['status'] == 'training':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Training already in progress"
        )
    
    logger.info(f"Training request for component: {request.component}")
    
    # Start training in background
    background_tasks.add_task(
        run_training_task,
        request.dataset_names,
        request.model_types,
        request.component,
        request.n_features,
        request.feature_selection,
        request.test_size,
        request.random_state,
        request.custom_hyperparameters,
        request.enable_autoencoder
    )
    
    return {
        'status': 'training_initiated',
        'component': request.component,
        'model_types': request.model_types,
        'datasets': request.dataset_names,
        'message': 'Training started in background. Check /training/status for progress.'
    }


@app.get("/training/status", tags=["Training Mode"])
async def training_status():
    """Get current training status."""
    return training_state


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
                    "status": "implemented",
                    "description": "Acoustic and prosodic features from audio (child-only extraction)",
                    "includes_audio": True
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

def get_shap_assets(model_name: str):
    shap_dir = Path("assets/shap") / model_name

    if not shap_dir.exists():
        return None

    beeswarm = shap_dir / "global_beeswarm.png"
    bar = shap_dir / "global_bar.png"

    if not beeswarm.exists() or not bar.exists():
        return None

    return {
        "beeswarm": f"/assets/shap/{model_name}/global_beeswarm.png",
        "bar": f"/assets/shap/{model_name}/global_bar.png"
    }

@app.get("/models", tags=["Information"])
async def list_models():
    """List all available trained models."""
    try:
        models = model_registry.list_models()
        
        model_info = []
        for model_name in models:
            try:
                metadata = model_registry.get_model_metadata(model_name)
                shap_assets = get_shap_assets(model_name)
                model_info.append({
                    'name': model_name,
                    'type': metadata.model_type,
                    'accuracy': metadata.accuracy,
                    'f1_score': metadata.f1_score,
                    'precision': metadata.precision,
                    'recall': metadata.recall,
                    'roc_auc': metadata.roc_auc,
                    'matthews_corr': metadata.matthews_corr,
                    'confusion_matrix': metadata.confusion_matrix,
                    'version': metadata.version,
                    'n_features': metadata.n_features,
                    'training_samples': metadata.training_samples,
                    'component': metadata.component,
                    'created_at': metadata.created_at,
                    'shap': shap_assets
                })
            except:
                model_info.append({'name': model_name, 'type': 'unknown'})
        
        # Get best model
        best_model_name = None
        if model_info:
            best_model_name = max(model_info, key=lambda x: x.get('f1_score', 0))['name']
        
        return {
            "models": model_info,
            "count": len(models),
            "best_model": best_model_name
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )


@app.delete("/models/{model_name}", tags=["Information"])
async def delete_model(model_name: str):
    """Delete a trained model."""
    try:
        models = model_registry.list_models()
        if model_name not in models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        # Delete model directory
        model_dir = model_registry.registry_dir / model_name
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        
        # Remove from registry
        model_registry.models_.pop(model_name, None)
        model_registry._save_registry()
        
        logger.info(f"Deleted model: {model_name}")
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting model: {str(e)}"
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
                "status": "implemented",
                "features": {
                    "acoustic_audio": 60
                },
                "audio_support": True,
                "child_only_extraction": True
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

@app.get("/features/guidelines")
def get_feature_guidelines():
    if not FEATURE_CSV_PATH.exists():
        return {"error": "Feature guideline CSV not found"}

    df = pd.read_csv(FEATURE_CSV_PATH)

    return {
        "columns": list(df.columns),
        "rows": df.fillna("").to_dict(orient="records")
    }


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
