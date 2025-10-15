"""
Pragmatic & Conversational Feature Extractors

This module contains feature extractors for pragmatic and conversational analysis.
These features analyze social language use, turn-taking, and conversation management.

Status: Fully Implemented - Ready for production use

Features (61 total):
- Turn-taking features (15): Turn frequency, response latency, initiation patterns
- Linguistic features (14): MLU, vocabulary diversity, grammatical complexity
- Pragmatic features (16): Echolalia, pronouns, questions, social language
- Conversational features (16): Topic management, discourse markers, repairs

Author: Bimidu Gunathilake
"""

from .turn_taking import TurnTakingFeatures
from .linguistic import LinguisticFeatures
from .pragmatic import PragmaticFeatures
from .conversational import ConversationalFeatures

__all__ = [
    "TurnTakingFeatures",
    "LinguisticFeatures", 
    "PragmaticFeatures",
    "ConversationalFeatures",
]

__version__ = "1.0.0"
__status__ = "implemented"
__team__ = "ASD Detection Team"
__feature_count__ = 61
