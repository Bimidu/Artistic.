"""
Pragmatic & Conversational Feature Extractors

Feature extraction module organized according to research methodology:

=== PRIMARY EXTRACTORS  ===

Turn-Taking Metrics (turn_taking.py)
  - Turn frequency, length, variability
  - Inter-turn gaps and response latency
  - Overlap and interruption detection

Topic Maintenance and Semantic Coherence (topic_coherence.py)
  - LDA topic modeling
  - Semantic similarity using word embeddings
  - Topic shift detection

Pause and Latency Analysis (pause_latency.py)
  - Response latency distribution
  - Filled and unfilled pauses
  - Speaking vs silence ratio

Conversational Repair Detection (repair_detection.py)
  - Self-repair and other-repair
  - Clarification requests
  - Repair success rate

=== SUPPORTING EXTRACTORS ===

Pragmatic & Linguistic Features (pragmatic_linguistic.py)
  - MLU and vocabulary diversity
  - Echolalia patterns
  - Pronoun usage and reversal
  - Question usage, social language
  - Discourse markers, behavioral markers

Audio Features (audio_features.py)
  - Pause patterns from audio timing
  - Response latency from audio
  - Speaking rate from audio
  - Turn-taking timing from audio segments

Note: Syntactic/semantic features (POS analysis, dependency parsing) are 
handled by the dedicated syntactic_semantic module.

Author: Bimidu Gunathilake
"""

# Shared constants (patterns, thresholds, vocabulary lists)
from . import constants

# Primary methodology-aligned extractors 
from .turn_taking import TurnTakingFeatures
from .topic_coherence import TopicCoherenceFeatures
from .pause_latency import PauseLatencyFeatures
from .repair_detection import RepairDetectionFeatures

# Consolidated supporting extractors
from .pragmatic_linguistic import PragmaticLinguisticFeatures
from .audio_features import PragmaticAudioFeatures, PauseInfo

__all__ = [
    # Shared constants module
    "constants",

    # Methodology-aligned 
    "TurnTakingFeatures",        
    "TopicCoherenceFeatures",   
    "PauseLatencyFeatures",       
    "RepairDetectionFeatures",   
    
    # Supporting extractors
    "PragmaticLinguisticFeatures",
    "PragmaticAudioFeatures",     # Audio-derived pragmatic features
    "PauseInfo",                  # Pause data structure
]

__version__ = "2.1.0"
__status__ = "implemented"
__team__ = "ASD Detection Team"

# Feature counts by category
FEATURE_COUNTS = {
    'turn_taking': 45,           
    'topic_coherence': 28,       
    'pause_latency': 34,         
    'repair_detection': 35,      
    'pragmatic_linguistic': 35,  # Supporting (text)
    'pragmatic_audio': 30,       # Supporting (audio)
}

__feature_count__ = sum(FEATURE_COUNTS.values())
