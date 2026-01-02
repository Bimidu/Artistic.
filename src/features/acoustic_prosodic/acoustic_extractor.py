"""
Acoustic & Prosodic Feature Extractor

This module provides a wrapper for acoustic and prosodic feature extraction.
It uses the AcousticAudioFeatures class to extract real features from audio.

Features include:
- Pitch features (F0 mean, std, range, slope)
- Prosody features (intonation, rhythm, stress)
- Voice quality (jitter, shimmer, HNR)
- Spectral features (MFCCs, spectral centroid, rolloff)
- Energy/intensity patterns

Author: Implementation based on pragmatic features pattern
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from src.utils.logger import get_logger
from src.parsers.chat_parser import TranscriptData
from .audio_features import AcousticAudioFeatures

logger = get_logger(__name__)


class AcousticFeatureExtractor:
    """
    Wrapper for acoustic and prosodic feature extraction.
    
    Uses AcousticAudioFeatures to extract real features from audio files.
    Provides compatibility interface for API usage.
    """
    
    def __init__(self):
        """Initialize acoustic feature extractor."""
        self.audio_feature_extractor = AcousticAudioFeatures()
        self.feature_names = self.audio_feature_extractor.feature_names
        logger.info(f"AcousticFeatureExtractor initialized with {len(self.feature_names)} features")
    
    def extract_from_audio(self, audio_path: Path) -> Dict[str, float]:
        """
        Extract acoustic features from audio file.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary of feature values
        """
        logger.info(f"Extracting acoustic features from: {audio_path}")
        
        # Create a dummy transcript for the extractor
        from src.parsers.chat_parser import TranscriptData, Utterance
        dummy_transcript = TranscriptData(
            file_path=audio_path,
            participant_id="CHI",
            utterances=[],
            metadata={}
        )
        
        # Extract features using AcousticAudioFeatures
        result = self.audio_feature_extractor.extract(
            transcript=dummy_transcript,
            audio_path=audio_path
        )
        
        logger.info(f"Extracted {len(result.features)} acoustic features")
        return result.features
    
    def extract_from_transcript(self, transcript_data: Any) -> Dict[str, float]:
        """
        Extract acoustic features from transcript (requires audio file).
        
        Args:
            transcript_data: Transcript data (should have file_path with audio)
        
        Returns:
            Dictionary of feature values
        """
        logger.info("Extracting acoustic features from transcript")
        
        # Try to find associated audio file
        audio_path = None
        if isinstance(transcript_data, TranscriptData):
            # Check if transcript has audio path in metadata
            if hasattr(transcript_data, 'file_path') and transcript_data.file_path:
                # Try to find .wav file with same name
                base_path = Path(transcript_data.file_path)
                audio_path = base_path.with_suffix('.wav')
                
                # If not found, try common audio extensions
                if not audio_path.exists():
                    for ext in ['.mp3', '.flac', '.m4a']:
                        audio_path = base_path.with_suffix(ext)
                        if audio_path.exists():
                            break
                    else:
                        audio_path = None
        
        # Extract features
        result = self.audio_feature_extractor.extract(
            transcript=transcript_data if isinstance(transcript_data, TranscriptData) else None,
            audio_path=audio_path
        )
        
        return result.features
    
    def extract_from_directory(self, directory: Path) -> pd.DataFrame:
        """
        Extract features from all files in directory.
        
        Args:
            directory: Directory path
        
        Returns:
            DataFrame with features
        """
        logger.info(f"Extracting acoustic features from directory: {directory}")
        
        # Find audio files
        audio_files = list(directory.rglob('*.wav'))
        audio_files.extend(directory.rglob('*.mp3'))
        audio_files.extend(directory.rglob('*.flac'))
        
        if not audio_files:
            logger.warning(f"No audio files found in {directory}")
            # Return empty DataFrame with correct columns
            data = []
            features_dict = {name: 0.0 for name in self.feature_names}
            features_dict['diagnosis'] = None
            features_dict['file_path'] = None
            features_dict['participant_id'] = None
            return pd.DataFrame([features_dict])
        
        # Extract from actual files
        data = []
        for audio_file in audio_files:
            try:
                features = self.extract_from_audio(audio_file)
                
                # Try to infer diagnosis from directory structure or filename
                path_str = str(audio_file).upper()
                if '/ASD/' in path_str or '_ASD_' in path_str or '\\ASD\\' in path_str:
                    features['diagnosis'] = 'ASD'
                elif '/TD/' in path_str or '/TYP/' in path_str or '_TD_' in path_str or '\\TD\\' in path_str or '\\TYP\\' in path_str:
                    features['diagnosis'] = 'TD'
                else:
                    features['diagnosis'] = None
                
                features['file_path'] = str(audio_file)
                features['participant_id'] = audio_file.stem
                data.append(features)
                
            except Exception as e:
                logger.error(f"Error extracting features from {audio_file}: {e}")
                continue
        
        if not data:
            logger.warning("No features extracted from any files")
            return pd.DataFrame()
        
        logger.info(f"Extracted features from {len(data)} audio files")
        return pd.DataFrame(data)

