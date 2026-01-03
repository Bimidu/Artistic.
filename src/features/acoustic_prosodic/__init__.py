"""
Acoustic & Prosodic Feature Extractors

This module contains feature extractors for acoustic and prosodic analysis.
These features analyze audio characteristics like pitch, speech rate, prosody, and pauses.

Status: Implemented

Features (60+ total):
- Pitch features (F0 mean, std, range, slope, contour)
- Prosody features (intonation, rhythm, stress variability)
- Voice quality (jitter, shimmer, HNR)
- Spectral features (MFCCs, spectral centroid, rolloff, bandwidth)
- Energy/intensity patterns
- Formant-like features

Author: Implementation based on pragmatic features pattern
"""

from .audio_features import AcousticAudioFeatures
from .acoustic_extractor import AcousticFeatureExtractor
from .child_audio_extractor import ChildAudioExtractor

__all__ = ["AcousticAudioFeatures", "AcousticFeatureExtractor", "ChildAudioExtractor"]

__version__ = "2.0.0"
__status__ = "implemented"
__feature_count__ = 60
