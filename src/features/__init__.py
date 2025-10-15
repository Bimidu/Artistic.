"""
Feature Extraction Package

This package contains modules for extracting three main categories of features
from parsed CHAT transcripts for ASD detection:

1. ACOUSTIC & PROSODIC FEATURES (Implemented by Team Member A)
   - Pitch variations, speech rate, pause patterns, intonation
   - To be integrated later

2. SYNTACTIC & SEMANTIC FEATURES (Implemented by Team Member B)
   - Grammar structures, semantic relationships, sentence complexity
   - To be integrated later

3. PRAGMATIC & CONVERSATIONAL FEATURES (Fully Implemented)
   - Turn-taking patterns
   - Linguistic complexity and diversity
   - Pragmatic language use (echolalia, questions, pronouns)
   - Conversational management (topic, discourse, repairs)
   - Behavioral/non-verbal markers

Modules:
    - base_features: Base class for all feature extractors
    - feature_extractor: Main orchestrator for all feature types
    
    Category 1 - acoustic_prosodic/: Acoustic & prosodic features (Team Member A)
        - acoustic_prosodic.py: Placeholder implementation
    
    Category 2 - syntactic_semantic/: Syntactic & semantic features (Team Member B)  
        - syntactic_semantic.py: Placeholder implementation
        
    Category 3 - pragmatic_conversational/: Pragmatic & conversational features (Implemented)
        - turn_taking.py: Turn-taking pattern features
        - linguistic.py: Linguistic complexity features
        - pragmatic.py: Pragmatic language features
        - conversational.py: Conversational management features
"""

from .feature_extractor import FeatureExtractor, FeatureSet

# Import from organized submodules
from .acoustic_prosodic import AcousticProsodicFeatures
from .syntactic_semantic import SyntacticSemanticFeatures
from .pragmatic_conversational import (
    TurnTakingFeatures,
    LinguisticFeatures,
    PragmaticFeatures,
    ConversationalFeatures,
)

__all__ = [
    "FeatureExtractor",
    "FeatureSet",
    "AcousticProsodicFeatures",
    "SyntacticSemanticFeatures",
    "TurnTakingFeatures",
    "LinguisticFeatures",
    "PragmaticFeatures",
    "ConversationalFeatures",
]
