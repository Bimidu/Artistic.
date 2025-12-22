"""
Audio Preprocessing Module for ASD Detection

This module provides COMMON audio preprocessing capabilities:
- Speech-to-text transcription using advanced models
- Audio loading and normalization
- Basic audio utilities

NOTE: Audio FEATURE EXTRACTION happens inside each feature module:
- src/features/pragmatic_conversational/ - Pause patterns, timing features
- src/features/acoustic_prosodic/ - Pitch, prosody features (placeholder)
- src/features/syntactic_semantic/ - Text-derived features from audio (placeholder)

Author: Bimidu Gunathilake
"""

from .transcriber import AudioTranscriber, TranscriptionResult, Segment, WordTimestamp
from .audio_processor import AudioProcessor, AudioProcessingResult

__all__ = [
    # Common transcription
    "AudioTranscriber",
    "TranscriptionResult",
    "Segment",
    "WordTimestamp",
    # Common processing
    "AudioProcessor",
    "AudioProcessingResult",
]

__version__ = "1.0.0"
__status__ = "implemented"

