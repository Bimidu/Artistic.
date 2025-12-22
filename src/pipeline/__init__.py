"""
ASD Detection Pipeline Module

This module provides the unified pipeline for processing inputs
(audio or text) through feature extraction and prediction.

Components:
- InputHandler: Unified handling of audio and text inputs
- AnnotatedTranscript: Feature-annotated transcript generation
- ModelFusion: Fusion of component models for final prediction

Flow:
1. InputHandler receives audio or text input
2. Audio is transcribed (if audio input)
3. Features are extracted by each component independently
4. Each component trains/uses its own model
5. ModelFusion combines component predictions
6. AnnotatedTranscript shows where features were extracted

Author: Bimidu Gunathilake
"""

from .input_handler import InputHandler, ProcessedInput, InputType
from .annotated_transcript import (
    AnnotatedTranscript,
    FeatureAnnotation,
    AnnotationType,
    TranscriptAnnotator,
)
from .model_fusion import ModelFusion, FusionResult, ComponentPrediction

__all__ = [
    # Input handling
    "InputHandler",
    "ProcessedInput",
    "InputType",
    # Annotation
    "AnnotatedTranscript",
    "FeatureAnnotation",
    "AnnotationType",
    "TranscriptAnnotator",
    # Model fusion
    "ModelFusion",
    "FusionResult",
    "ComponentPrediction",
]

__version__ = "1.0.0"

