"""
Pragmatic & Conversational Feature Extractors

This module contains feature extractors for pragmatic and conversational analysis,
organized according to the research methodology sections:

Section 3.3.1: Turn-Taking Metrics
- Turn frequency, length, variability
- Inter-turn gaps and response latency
- Overlap and interruption detection

Section 3.3.2: Topic Maintenance and Semantic Coherence
- LDA topic modeling
- Semantic similarity using word embeddings
- Topic shift detection

Section 3.3.3: Pause and Latency Analysis
- Response latency distribution
- Filled and unfilled pauses
- Speaking vs silence ratio

Section 3.3.4: Conversational Repair Detection
- Self-repair and other-repair
- Clarification requests
- Repair success rate

Additional Features:
- Linguistic features (MLU, vocabulary diversity)
- Pragmatic features (echolalia, pronouns, questions)

Author: Bimidu Gunathilake
"""

# Core methodology-aligned extractors (Sections 3.3.1 - 3.3.4)
from .turn_taking import TurnTakingFeatures
from .topic_coherence import TopicCoherenceFeatures
from .pause_latency import PauseLatencyFeatures
from .repair_detection import RepairDetectionFeatures

# Additional supporting extractors
from .linguistic import LinguisticFeatures
from .pragmatic import PragmaticFeatures
from .conversational import ConversationalFeatures

__all__ = [
    # Methodology-aligned (primary)
    "TurnTakingFeatures",       # Section 3.3.1
    "TopicCoherenceFeatures",   # Section 3.3.2
    "PauseLatencyFeatures",     # Section 3.3.3
    "RepairDetectionFeatures",  # Section 3.3.4
    
    # Supporting extractors
    "LinguisticFeatures",
    "PragmaticFeatures",
    "ConversationalFeatures",
]

__version__ = "2.0.0"
__status__ = "implemented"
__team__ = "ASD Detection Team"

# Feature counts by category
FEATURE_COUNTS = {
    'turn_taking': 45,       # Section 3.3.1
    'topic_coherence': 28,   # Section 3.3.2
    'pause_latency': 34,     # Section 3.3.3
    'repair_detection': 35,  # Section 3.3.4
    'linguistic': 15,        # Supporting
    'pragmatic': 17,         # Supporting
    'conversational': 17,    # Supporting (legacy)
}

__feature_count__ = sum(FEATURE_COUNTS.values())
