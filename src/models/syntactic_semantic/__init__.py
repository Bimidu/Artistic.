"""
Syntactic & Semantic Model Training

This module provides model training specifically for syntactic and semantic features.

Status: ✅ FULLY IMPLEMENTED

Supported Models:
- LightGBM (Primary): Fast gradient boosting optimized for syntactic patterns
- Gradient Boosting (Secondary): Handles non-linear patterns when real features added

Features:
- 27 syntactic/semantic features
- Component-specific hyperparameter optimization
- Feature importance analysis with clinical interpretation
- Model evaluation and comparison

Author: Randil Haturusinghe
"""

from .model_trainer import SyntacticSemanticTrainer, SyntacticSemanticModelConfig
from .preprocessor import SyntacticSemanticPreprocessor

__all__ = [
    "SyntacticSemanticTrainer",
    "SyntacticSemanticModelConfig",
    "SyntacticSemanticPreprocessor",
]

__version__ = "2.0.0"
__status__ = "implemented"
__author__ = "Randil Haturusinghe"
__feature_count__ = 27
