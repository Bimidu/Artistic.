"""
Syntactic & Semantic Feature Extractors

This module contains feature extractors for syntactic and semantic analysis.
These features analyze grammar structures, semantic relationships, and language complexity.

Status: ✅ FULLY IMPLEMENTED

Features (27 total):
- Syntactic Complexity (6): dependency depth, clause complexity, subordination
- Grammatical Accuracy (5): error rates, tense consistency, POS diversity
- Sentence Structure (4): parse tree height, phrase complexity
- Semantic Features (4): coherence, density, thematic consistency
- Vocabulary Semantics (4): abstractness, semantic fields, word senses
- Advanced Semantics (4): semantic roles, entity density, verb arguments

Tools:
- spaCy (en_core_web_sm): Dependency parsing, POS tagging, NER
- NLTK WordNet: Semantic analysis, word sense disambiguation
- ClinicalInterpreter: Clinical interpretation of feature values

Author: Randil Haturusinghe
Enhanced: Clinical interpretation module added
"""

from .syntactic_semantic import SyntacticSemanticFeatures
from .clinical_interpretation import ClinicalInterpreter, FeatureInterpretation

__all__ = [
    "SyntacticSemanticFeatures",
    "ClinicalInterpreter",
    "FeatureInterpretation"
]

__version__ = "2.0.0"
__status__ = "implemented"
__author__ = "Randil Haturusinghe"
__feature_count__ = 27
