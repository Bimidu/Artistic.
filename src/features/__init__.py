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
        - turn_taking.py: Turn-taking metrics (3.3.1)
        - topic_coherence.py: Topic maintenance & semantic coherence (3.3.2)
        - pause_latency.py: Pause and latency analysis (3.3.3)
        - repair_detection.py: Conversational repair detection (3.3.4)
        - pragmatic_linguistic.py: General pragmatic & linguistic features
"""

from .feature_extractor import FeatureExtractor, FeatureSet

# Import from organized submodules
# Temporarily disabled: acoustic_prosodic module not fully implemented
# from .acoustic_prosodic import AcousticProsodicFeatures

try:
    from .syntactic_semantic import SyntacticSemanticFeatures
except ImportError:
    SyntacticSemanticFeatures = None

from .pragmatic_conversational import (
    TurnTakingFeatures,
    TopicCoherenceFeatures,
    PauseLatencyFeatures,
    RepairDetectionFeatures,
    PragmaticLinguisticFeatures,
)

__all__ = [
    "FeatureExtractor",
    "FeatureSet",
    # "AcousticProsodicFeatures",  # Temporarily disabled
    "SyntacticSemanticFeatures",
    "TurnTakingFeatures",
    "TopicCoherenceFeatures",
    "PauseLatencyFeatures",
    "RepairDetectionFeatures",
    "PragmaticLinguisticFeatures",
]
