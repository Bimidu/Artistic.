"""
Machine Learning Models Package

This package contains ML models for ASD classification using pragmatic
and conversational features.

Modules:
    - model_trainer: Model training pipeline
    - model_evaluator: Model evaluation and metrics
    - model_registry: Model persistence and management
    - ensemble: Ensemble methods

Author: Bimidu Gunathilake
"""

from .model_trainer import ModelTrainer, ModelConfig
from .model_evaluator import ModelEvaluator, EvaluationReport
from .model_registry import ModelRegistry

# Import component trainers
from .acoustic_prosodic import AcousticProsodicTrainer
from .syntactic_semantic import SyntacticSemanticTrainer
from .pragmatic_conversational import PragmaticConversationalTrainer

__all__ = [
    "ModelTrainer",
    "ModelConfig",
    "ModelEvaluator",
    "EvaluationReport",
    "ModelRegistry",
    "AcousticProsodicTrainer",
    "SyntacticSemanticTrainer", 
    "PragmaticConversationalTrainer",
]

__version__ = "1.0.0"

