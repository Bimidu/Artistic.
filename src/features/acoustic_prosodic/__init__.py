"""
Acoustic & Prosodic Feature Extractors

This module contains feature extractors for acoustic and prosodic analysis.
These features analyze audio characteristics like pitch, speech rate, prosody, and pauses.

Status: Implemented

Features (110+ total):
- Pitch features (F0 mean, std, range, slope, contour)
- Prosody features (intonation, rhythm, stress variability)
- Voice quality (jitter, shimmer, HNR)
- Spectral features (MFCCs 1-13, spectral centroid, rolloff, bandwidth)
- Energy/intensity patterns
- Formant-like features
- Extended MFCC features (coefficients 6-13)
- Chroma features (12 pitch classes)
- Temporal dynamics (pitch/energy trajectories)
- Spectral contrast features
- Tonnetz features (harmonic network)
- Rhythm and timing features

Author: Implementation based on pragmatic features pattern
"""

from .audio_features import AcousticAudioFeatures
from .acoustic_extractor import AcousticFeatureExtractor
from .child_audio_extractor import ChildAudioExtractor

__all__ = ["AcousticAudioFeatures", "AcousticFeatureExtractor", "ChildAudioExtractor"]

__version__ = "2.0.0"
__status__ = "implemented"
__feature_count__ = 110
