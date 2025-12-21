"""
Pragmatic & Conversational Feature Extractors

Feature extraction module organized according to research methodology:

=== PRIMARY EXTRACTORS (Methodology Sections 3.3.1 - 3.3.4) ===

Section 3.3.1: Turn-Taking Metrics (turn_taking.py)
  - Turn frequency, length, variability
  - Inter-turn gaps and response latency
  - Overlap and interruption detection

Section 3.3.2: Topic Maintenance and Semantic Coherence (topic_coherence.py)
  - LDA topic modeling
  - Semantic similarity using word embeddings
  - Topic shift detection

Section 3.3.3: Pause and Latency Analysis (pause_latency.py)
  - Response latency distribution
  - Filled and unfilled pauses
  - Speaking vs silence ratio

Section 3.3.4: Conversational Repair Detection (repair_detection.py)
  - Self-repair and other-repair
  - Clarification requests
  - Repair success rate

=== SUPPORTING EXTRACTOR ===

Pragmatic & Linguistic Features (pragmatic_linguistic.py)
  - MLU and vocabulary diversity
  - Echolalia patterns
  - Pronoun usage and reversal
  - Question usage, social language
  - Discourse markers, behavioral markers

Note: Syntactic/semantic features (POS analysis, dependency parsing) are 
handled by the dedicated syntactic_semantic module.

Author: Bimidu Gunathilake
"""

# Primary methodology-aligned extractors (Sections 3.3.1 - 3.3.4)
from .turn_taking import TurnTakingFeatures
from .topic_coherence import TopicCoherenceFeatures
from .pause_latency import PauseLatencyFeatures
from .repair_detection import RepairDetectionFeatures

# Consolidated supporting extractor
from .pragmatic_linguistic import PragmaticLinguisticFeatures

__all__ = [
    # Methodology-aligned (Sections 3.3.1 - 3.3.4)
    "TurnTakingFeatures",         # Section 3.3.1
    "TopicCoherenceFeatures",     # Section 3.3.2
    "PauseLatencyFeatures",       # Section 3.3.3
    "RepairDetectionFeatures",    # Section 3.3.4
    
    # Consolidated supporting extractor
    "PragmaticLinguisticFeatures",
]

__version__ = "2.0.0"
__status__ = "implemented"
__team__ = "ASD Detection Team"

# Feature counts by category
FEATURE_COUNTS = {
    'turn_taking': 45,           # Section 3.3.1
    'topic_coherence': 28,       # Section 3.3.2
    'pause_latency': 34,         # Section 3.3.3
    'repair_detection': 35,      # Section 3.3.4
    'pragmatic_linguistic': 35,  # Supporting (consolidated)
}

__feature_count__ = sum(FEATURE_COUNTS.values())
