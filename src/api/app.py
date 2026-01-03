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
feature_extractor = FeatureExtractor(categories='pragmatic_conversational')
input_handler = None  # Lazy-loaded due to heavy model loading
transcript_annotator = TranscriptAnnotator()
model_fusion = ModelFusion(method='weighted')


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
        description="Output CSV filename"
    )
    max_samples_per_dataset: Optional[int] = Field(
        default=None,
        description="Maximum samples per dataset (for large datasets like TD)"
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
    max_samples_per_dataset: Optional[int] = Field(
        default=None,
        description="Maximum samples per dataset (useful for large datasets like TD with 4000+ files)"
    )
    custom_hyperparameters: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Custom hyperparameters for each model type"
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
            if isinstance(preprocessor, dict):
                features_df = preprocess_with_dict(features_df, preprocessor)
            else:
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
            if isinstance(preprocessor, dict):
                features_df = preprocess_with_dict(features_df, preprocessor)
            else:
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
    file: UploadFile = File(...),
    use_fusion: bool = Form(False)
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
            # Multi-component prediction with fusion
            component_predictions = []
            
            # Try each component
            for component in ['pragmatic_conversational', 'acoustic_prosodic', 'syntactic_semantic']:
                try:
                    # Select feature extractor
                    if component == 'acoustic_prosodic':
                        from src.features.acoustic_prosodic.acoustic_extractor import AcousticFeatureExtractor
                        extractor = AcousticFeatureExtractor()
                        features = extractor.extract_from_transcript(transcript)
                    elif component == 'syntactic_semantic':
                        from src.features.syntactic_semantic.syntactic_extractor import SyntacticFeatureExtractor
                        extractor = SyntacticFeatureExtractor()
                        features = extractor.extract_from_transcript(transcript)
                    else:
                        features = feature_extractor.extract_from_transcript(transcript).features
                    
                    features_df = pd.DataFrame([features])
                    
                    # Get best model for this component
                    model, preprocessor, model_name = get_model_and_preprocessor(component=component)
                    
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
                        probabilities = {str(cls): float(prob) for cls, prob in zip(classes, proba)}
                        confidence = float(np.max(proba))
                    else:
                        probabilities = {str(prediction): 1.0}
                        confidence = 1.0
                    
                    component_predictions.append(ComponentPrediction(
                        component=component,
                        prediction=str(prediction),
                        probabilities=probabilities,
                        confidence=confidence,
                        model_name=model_name
                    ))
                    
                    logger.info(f"{component}: {prediction} ({confidence:.2f})")
                
                except Exception as e:
                    logger.warning(f"Component {component} failed: {e}")
                    continue
            
            if not component_predictions:
                raise ValueError("No components available for prediction")
            
            # Fuse predictions
            fused = model_fusion.fuse(component_predictions)
            
            # Generate annotated transcript (from pragmatic component)
            annotated = transcript_annotator.annotate(
                transcript,
                features=feature_extractor.extract_from_transcript(transcript).features
            )
            
            # Clean up temp file
            tmp_path.unlink()
            
            return {
                'prediction': fused.prediction,
                'confidence': fused.confidence,
                'probabilities': fused.probabilities,
                'model_used': 'fusion',
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
                'features_extracted': sum(len(cp.probabilities) for cp in component_predictions),
                'annotated_transcript_html': annotated.to_html(),
                'annotation_summary': annotated._get_annotation_summary(),
                'input_type': 'chat_file',
            }
        
        else:
            # Single component prediction (original logic)
            feature_set = feature_extractor.extract_from_transcript(transcript)
            features_df = pd.DataFrame([feature_set.features])
            
            # Get model and make prediction
            model, preprocessor, model_name = get_model_and_preprocessor()
            
            if preprocessor is not None:
                if isinstance(preprocessor, dict):
                    features_df = preprocess_with_dict(features_df, preprocessor)
                    selected_features = preprocessor["selected_features"]
                else:
                    features_df = preprocessor.transform(features_df)
                    selected_features = preprocessor.selected_features_
            
            result = make_prediction(model, features_df, model_name)

            #SHAP explanation
            request_id = str(uuid.uuid4())
            local_shap_dir = Path("assets/shap/local") / request_id

            # Load background data saved during training
            background = np.load(
                Path("assets/shap") / model_name / "background.npy"
            )

            predicted_class = 1 if result["prediction"] == "ASD" else 0

            shap_manager = SHAPManager(
                model=model,
                background_data=background,
                feature_names=selected_features,
                model_type=model_name.split("_")[-1]
            )

            shap_manager.generate_local_waterfall(
                X_instance=features_df.values,
                save_dir=local_shap_dir,
                predicted_class=predicted_class
            )
            
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
                'local_shap': {
                    'request_id': request_id,
                    'waterfall': f"/assets/shap/local/{request_id}/waterfall.png"
                },
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



@app.post("/training/extract-features", tags=["Training Mode"])
async def extract_features_for_training(request: FeatureExtractionRequest):
    """
    Extract features from specified datasets for training.
    
    Supports max_samples_per_dataset for large datasets (e.g., TD with 4000+ files).
    
    Returns the path to the generated feature CSV file.
    """
    logger.info(f"Feature extraction request for {len(request.dataset_paths)} datasets")
    if request.max_samples_per_dataset:
        logger.info(f"Max samples per dataset: {request.max_samples_per_dataset}")
    
    all_dfs = []
    
    for dataset_path in request.dataset_paths:
        path = Path(dataset_path)
        if not path.exists():
            path = config.paths.data_dir / dataset_path
        
        if not path.exists():
            logger.warning(f"Dataset path not found: {dataset_path}")
            continue
        
        try:
            # Check if this is a large dataset (TD)
            is_td_dataset = 'td' in path.name.lower()
            
            # Use request parameter, or fall back to config default
            max_samples = request.max_samples_per_dataset or config.datasets.max_samples_td
            max_samples_for_extraction = max_samples if is_td_dataset else None
            
            # Extract features - pragmatic extractor doesn't support max_samples param
            df = feature_extractor.extract_from_directory(path)
            
            # Sample after extraction for pragmatic features
            if is_td_dataset and max_samples and len(df) > max_samples:
                logger.info(f"Sampling {max_samples} from {len(df)} TD samples (config: max_samples_td={config.datasets.max_samples_td})")
                df = df.sample(n=max_samples, random_state=42)
            
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
    
    # Count actual features in the dataframe (exclude metadata columns)
    metadata_cols = ['participant_id', 'file_path', 'diagnosis', 'age_months', 'dataset']
    actual_feature_cols = [col for col in combined_df.columns if col not in metadata_cols]
    
    return {
        'status': 'success',
        'output_file': str(output_path),
        'total_samples': len(combined_df),
        'features_count': len(actual_feature_cols),  # ← FIXED: Count actual features
        'datasets_processed': len(all_dfs)
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

def run_training_task(dataset_paths: List[str], model_types: List[str], component: str, n_features: int = 30, feature_selection: bool = True, test_size: float = 0.2, random_state: int = 42, custom_hyperparameters: Optional[Dict[str, Dict[str, Any]]] = None, max_samples_per_dataset: Optional[int] = None):
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
        
        # Select appropriate feature extractor based on component
        if component == 'acoustic_prosodic':
            from src.features.acoustic_prosodic.acoustic_extractor import AcousticFeatureExtractor
            extractor = AcousticFeatureExtractor()
        elif component == 'syntactic_semantic':
            from src.features.syntactic_semantic.syntactic_extractor import SyntacticFeatureExtractor
            extractor = SyntacticFeatureExtractor()
        else:  # pragmatic_conversational (default)
            extractor = feature_extractor
        
        # Step 1: Load or extract features
        all_dfs = []
        for i, dataset_path in enumerate(dataset_paths):
            path = Path(dataset_path)
            if not path.exists():
                path = config.paths.data_dir / dataset_path
            
            if not path.exists():
                logger.warning(f"Dataset path not found: {dataset_path}")
                continue
            
            # Check if this is TD dataset and apply sampling BEFORE extraction
            is_td_dataset = 'td' in path.name.lower()
            max_samples = max_samples_per_dataset or config.datasets.max_samples_td
            max_samples_for_extraction = max_samples if is_td_dataset else None
            
            training_state['message'] = f'Extracting {component} features from dataset {i+1}/{len(dataset_paths)}...'
            
            # Pass max_samples to acoustic extractor to limit files BEFORE processing
            if component == 'acoustic_prosodic' and hasattr(extractor, 'extract_from_directory'):
                df = extractor.extract_from_directory(path, max_samples=max_samples_for_extraction)
            else:
                df = extractor.extract_from_directory(path)
                # For other extractors, sample after extraction
                if is_td_dataset and max_samples and len(df) > max_samples:
                    logger.info(f"Sampling {max_samples} from {len(df)} TD samples")
                    df = df.sample(n=max_samples, random_state=42)
            
            if not df.empty:
                df['dataset'] = path.name
                all_dfs.append(df)
        
        if not all_dfs:
            raise ValueError("No features extracted from any dataset")
        
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

            SHAP_SUPPORTED_MODELS = {
                "random_forest",
                "gradient_boosting",
                "adaboost",
                "svm",
                "lightgbm"
            }

            logger.warning(f"Calling SHAP for model {model_name}, X_train shape: {X_train.shape}")

            # ================================
            # 🔍 GLOBAL SHAP (TRAINING TIME)
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
        request.dataset_paths,
        request.model_types,
        request.component,
        request.n_features,
        request.feature_selection,
        request.test_size,
        request.random_state,
        request.custom_hyperparameters,
        request.max_samples_per_dataset
    )
    
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
