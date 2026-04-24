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
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from src.utils.logger import get_logger
from src.parsers.chat_parser import TranscriptData, CHATParser
from .audio_features import AcousticAudioFeatures

logger = get_logger(__name__)


class AcousticFeatureExtractor:
    """
    Wrapper for acoustic and prosodic feature extraction.
    
    Uses AcousticAudioFeatures to extract real features from audio files.
    Provides compatibility interface for API usage.
    """
    
    def __init__(self, extract_child_only: bool = True):
        """
        Initialize acoustic feature extractor.
        
        Args:
            extract_child_only: If True, extract only child speech from audio
        """
        self.audio_feature_extractor = AcousticAudioFeatures(extract_child_only=extract_child_only)
        self.parser = CHATParser()
        self.feature_names = self.audio_feature_extractor.feature_names
        logger.info(
            f"AcousticFeatureExtractor initialized with {len(self.feature_names)} features "
            f"(child_only={extract_child_only})"
        )
    
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

        # PLASTER FIX: Ensure all expected features are present
        expected_features = self.feature_names
        final_features = {}
        for feature_name in expected_features:
            final_features[feature_name] = result.features.get(feature_name, 0.0)

        logger.debug(f"Final feature set has {len(final_features)} features")
        return final_features

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
    
    def extract_from_directory(self, directory: Path, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Extract features from all files in directory.
        
        Args:
            directory: Directory path
            max_samples: Maximum number of samples to process (random sample if exceeded)
        
        Returns:
            DataFrame with features
        """
        logger.info(f"Extracting acoustic features from directory: {directory}")
        
        # Find audio files (exclude child_only directories from old extraction script)
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            for audio_file in directory.rglob(ext):
                # Skip files in child_only folders (from old standalone script)
                if 'child_only' in str(audio_file):
                    logger.debug(f"Skipping pre-extracted file: {audio_file.name}")
                    continue
                audio_files.append(audio_file)
        
        total_files = len(audio_files)
        logger.info(f"Found {total_files} audio files in {directory}")
        
        # Random sampling if too many files
        if max_samples and total_files > max_samples:
            import random
            random.seed(42)  # For reproducibility
            audio_files = random.sample(audio_files, max_samples)
            logger.info(f"Randomly sampled {max_samples} files from {total_files} total files")
        
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
                # Try to find corresponding transcript file
                transcript = None
                cha_file = audio_file.with_suffix('.cha')
                
                if cha_file.exists():
                    try:
                        transcript = self.parser.parse_file(cha_file)
                        logger.debug(f"Found transcript for {audio_file.name}")
                    except Exception as e:
                        logger.warning(f"Could not parse transcript {cha_file.name}: {e}")
                
                # Extract features (with transcript if available)
                if transcript:
                    features = self.extract_from_transcript(transcript)
                else:
                    features = self.extract_from_audio(audio_file)
                
                # Try to infer diagnosis from transcript first, then path
                if transcript and transcript.diagnosis:
                    features['diagnosis'] = transcript.diagnosis
                else:
                    # Try to infer from directory structure or filename
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

    def extract_from_prepared_groups(
        self,
        prepared_items: List[Tuple[List[Path], str, str]],
        sample_rate: int = 16000,
    ) -> pd.DataFrame:
        """
        Extract features from prepared groups with different strategies for ASD and TD.

        - ASD groups: Single file per group (individual processing with child-only extraction)
        - TD groups: Multiple files may be provided, but no merging is performed;
          only one valid file is used for extraction.

        Args:
            prepared_items: List of (list of audio paths, diagnosis, dataset_name) where:
                           - ASD items have 1 file per list
                           - TD items may have multiple files per list (no merging performed)
            sample_rate: Kept for backwards compatibility (not used for TD merging).

        Returns:
            DataFrame with one row per prepared group (individual ASD files and non-merged TD).
        """
        data = []
        total_groups = len(prepared_items)
        logger.info(f"Starting feature extraction for {total_groups} groups")

        for idx, (group_paths, diagnosis, dataset_name) in enumerate(prepared_items):
            try:
                # Progress logging
                if (idx + 1) % 10 == 0 or idx == 0:
                    logger.info(f"Processing group {idx + 1}/{total_groups} ({diagnosis})")
                elif (idx + 1) % 5 == 0:
                    logger.info(f"Progress: {idx + 1}/{total_groups} groups completed")
                if diagnosis == 'ASD':
                    # ASD Strategy: Individual file processing (no merging)
                    if len(group_paths) != 1:
                        logger.warning(f"ASD group {idx} has {len(group_paths)} files, expected 1. Using first file.")

                    audio_file = group_paths[0]
                    if not audio_file.exists():
                        logger.warning(f"Missing ASD file: {audio_file}")
                        continue

                    # Extract features from individual ASD file (with child-only extraction)
                    features = self.extract_from_audio(audio_file)
                    features["file_path"] = str(audio_file)
                    features["participant_id"] = audio_file.stem

                    logger.debug(f"Processed individual ASD file: {audio_file.name}")

                elif diagnosis == 'TD':
                    # TD Strategy (no merge): pick one valid file and process directly
                    valid_files = [Path(ap) for ap in group_paths if Path(ap).exists()]
                    if not valid_files:
                        logger.warning(f"No valid files in TD group {idx}")
                        continue

                    if len(valid_files) > 1:
                        logger.info(
                            f"TD group {idx + 1}: Received {len(valid_files)} files; "
                            f"merging is disabled, using first file only: {valid_files[0].name}"
                        )

                    audio_file = valid_files[0]
                    features = self.extract_from_audio(audio_file)
                    features["file_path"] = str(audio_file)
                    features["participant_id"] = audio_file.stem

                else:
                    logger.warning(f"Unknown diagnosis: {diagnosis}")
                    continue

                # Add common metadata
                features["diagnosis"] = diagnosis
                features["dataset"] = dataset_name
                data.append(features)

            except Exception as e:
                logger.error(f"Error processing group {idx} ({diagnosis}): {e}")
                continue

        if not data:
            logger.warning("No features extracted from any prepared groups")
            return pd.DataFrame()

        asd_count = sum(1 for d in data if d.get('diagnosis') == 'ASD')
        td_count = sum(1 for d in data if d.get('diagnosis') == 'TD')
        logger.info(f"Extracted features from {len(data)} groups (ASD: {asd_count} individual, TD: {td_count} non-merged)")
        return pd.DataFrame(data)

    def extract_with_audio(
        self,
        transcript: TranscriptData,
        audio_path: Optional[Path] = None,
        transcription_result: Optional[Any] = None
    ) -> Any:
        """
        Extract acoustic features with audio support (API compatibility method).
        
        This method matches the interface expected by the API for feature extraction
        with audio files. It extracts child-only audio segments if enabled.
        
        Args:
            transcript: Parsed transcript data
            audio_path: Optional path to audio file
            transcription_result: Optional TranscriptionResult with timing information
        
        Returns:
            FeatureSet-like object with features attribute
        """
        logger.debug(f"Extracting acoustic features with audio for {transcript.participant_id}")
        
        # Extract features using the internal audio feature extractor
        result = self.audio_feature_extractor.extract(
            transcript=transcript,
            audio_path=audio_path,
            transcription_result=transcription_result
        )
        
        # Return a FeatureSet-like object for API compatibility
        from src.features.feature_extractor import FeatureSet
        feature_set = FeatureSet(
            participant_id=transcript.participant_id,
            file_path=transcript.file_path,
            diagnosis=transcript.diagnosis,
            age_months=transcript.age_months,
            features=result.features,
            metadata={
                'total_utterances': transcript.total_utterances,
                'extraction_metadata': result.metadata,
                'audio_path': str(audio_path) if audio_path else None,
                'has_audio': audio_path is not None,
            },
            feature_categories=['acoustic_prosodic']
        )
        
        logger.debug(f"Extracted {len(result.features)} acoustic features")
        return feature_set