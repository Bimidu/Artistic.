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
        
        if not audio_files:
            logger.warning(f"No audio files found in {directory}")
            # Return empty DataFrame with correct columns
            data = []
            features_dict = {name: 0.0 for name in self.feature_names}
            features_dict['diagnosis'] = None
            features_dict['file_path'] = None
            features_dict['participant_id'] = None
            return pd.DataFrame([features_dict])
        
        # Separate TD and ASD files FIRST (before max_samples)
        td_files = []
        asd_files = []
        other_files = []
        
        for audio_file in audio_files:
            # Try to infer diagnosis from path
            path_str = str(audio_file).upper()
            filename = audio_file.name.upper()
            
            # Check for ASD patterns
            asd_patterns = ['/ASD/', '_ASD_', '\\ASD\\', 'ASDBANK', 'ASD_']
            is_asd = any(pattern in path_str for pattern in asd_patterns)
            
            # Check for TD patterns (more flexible)
            td_patterns = [
                '/TD/', '/TYP/', '\\TD\\', '\\TYP\\', 
                '_TD_', '_TYP_', 
                'TD_', 'TYP_', '_TD', '_TYP',
                'TYPICAL', 'TYPICALLY', 'CONTROL',
                'TD\\', 'TD/', 'TYP\\', 'TYP/'
            ]
            is_td = any(pattern in path_str for pattern in td_patterns)
            
            # Also check filename if path doesn't match
            if not is_td and not is_asd:
                is_td = any(pattern in filename for pattern in ['TD', 'TYP', 'TYPICAL', 'CONTROL'])
                is_asd = any(pattern in filename for pattern in ['ASD'])
            
            if is_asd:
                asd_files.append(audio_file)
            elif is_td:
                td_files.append(audio_file)
            else:
                other_files.append(audio_file)
        
        logger.info(f"Found {len(td_files)} TD files, {len(asd_files)} ASD files, {len(other_files)} other files")
        
        # If we have many "other" files, they might be TD files - check if directory name suggests TD
        if len(other_files) > 0:
            # Check if the directory itself suggests TD (e.g., "td", "typical", "control" in parent dirs)
            dir_path_str = str(directory).upper()
            td_dir_patterns = ['TD', 'TYP', 'TYPICAL', 'CONTROL', 'NORMAL', 'HEALTHY']
            is_td_directory = any(pattern in dir_path_str for pattern in td_dir_patterns)
            
            if is_td_directory:
                logger.info(f"📁 Directory name suggests TD dataset (found '{[p for p in td_dir_patterns if p in dir_path_str][0]}' in path).")
                logger.info(f"   Treating {len(other_files)} unclassified files as TD.")
                td_files.extend(other_files)
                other_files = []
            elif len(other_files) > 100 and len(td_files) < 100:
                # Log sample paths for debugging
                logger.warning(f"⚠️ Many files ({len(other_files)}) not classified as TD/ASD. Sample paths:")
                for i, f in enumerate(other_files[:5]):
                    logger.warning(f"   {i+1}. {f}")
                logger.warning(f"   ... and {len(other_files) - 5} more")
        
        logger.info(f"Final counts: {len(td_files)} TD files, {len(asd_files)} ASD files, {len(other_files)} other files")
        
        # Apply max_samples AFTER separation (only if not doing TD combination)
        # If we're combining TD files, we need all available TD files
        files_per_combined = 10
        target_combined_count = 80
        required_td_files = target_combined_count * files_per_combined
        
        # Only apply max_samples if we're NOT combining TD files (i.e., not enough TD files)
        if max_samples and len(td_files) < required_td_files:
            # Limit total files if we don't have enough TD files for combination
            total_available = len(td_files) + len(asd_files) + len(other_files)
            if total_available > max_samples:
                import random
                random.seed(42)
                # Sample proportionally: keep all TD, sample from others
                if len(td_files) < max_samples:
                    remaining = max_samples - len(td_files)
                    other_files_sample = random.sample(other_files + asd_files, min(remaining, len(other_files) + len(asd_files)))
                    asd_files = [f for f in other_files_sample if f in asd_files]
                    other_files = [f for f in other_files_sample if f in other_files]
                    logger.info(f"Applied max_samples: kept {len(td_files)} TD files, sampled {len(asd_files)} ASD + {len(other_files)} other files")
        
        # Extract from actual files
        data = []
        
        # COMBINE TD FILES: Create 80 combined samples, each from 10 TD audio files
        
        if len(td_files) >= required_td_files:
            logger.info(f"🔗 Combining TD audio files: creating {target_combined_count} combined samples (each from {files_per_combined} files)")
            logger.info(f"   Using {required_td_files} TD files out of {len(td_files)} available")
            
            # Randomly shuffle and take required number
            import random
            random.seed(42)
            selected_td_files = random.sample(td_files, required_td_files)
            
            # Group into batches of 10 and combine audio
            for batch_idx in range(0, len(selected_td_files), files_per_combined):
                batch_files = selected_td_files[batch_idx:batch_idx + files_per_combined]
                
                if len(batch_files) == files_per_combined:
                    try:
                        # Combine audio files
                        combined_audio_path = self._combine_audio_files(batch_files, batch_idx // files_per_combined)
                        
                        # Extract features from combined audio
                        features = self.extract_from_audio(combined_audio_path)
                        features['diagnosis'] = 'TD'
                        features['file_path'] = f"combined_{files_per_combined}_files"
                        features['participant_id'] = f"TD_combined_{batch_idx // files_per_combined}"
                        data.append(features)
                        
                        # Clean up temporary combined audio file
                        if combined_audio_path.exists() and 'temp' in str(combined_audio_path):
                            combined_audio_path.unlink()
                        
                    except Exception as e:
                        logger.error(f"Error combining TD files batch {batch_idx // files_per_combined}: {e}")
                        continue
            
            logger.info(f"✅ Created {len([d for d in data if d.get('diagnosis') == 'TD'])} combined TD samples")
            # TD files are already processed, don't process them again
            td_files_to_process = []
        else:
            logger.info(f"Not enough TD files for combination: have {len(td_files)}, need {required_td_files}")
            logger.info("Extracting features from individual TD files instead")
            td_files_to_process = td_files
        
        # Process remaining TD files (if not enough for combination) and ASD files
        files_to_process = td_files_to_process + asd_files + other_files
        
        for audio_file in files_to_process:
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
    
    def _combine_audio_files(self, audio_files: list, batch_id: int) -> Path:
        """
        Combine multiple audio files into one by concatenating them.
        
        Args:
            audio_files: List of audio file paths to combine
            batch_id: Batch identifier for naming
            
        Returns:
            Path to temporary combined audio file
        """
        import librosa
        import soundfile as sf
        import tempfile
        
        logger.debug(f"Combining {len(audio_files)} audio files for batch {batch_id}")
        
        # Load all audio files and concatenate
        combined_audio = []
        target_sr = 22050  # librosa default
        
        for audio_file in audio_files:
            try:
                audio, sr = librosa.load(str(audio_file), sr=target_sr, mono=True)
                combined_audio.append(audio)
            except Exception as e:
                logger.warning(f"Could not load {audio_file}: {e}")
                continue
        
        if not combined_audio:
            raise ValueError(f"No audio could be loaded from {len(audio_files)} files")
        
        # Concatenate all audio
        final_audio = np.concatenate(combined_audio)
        
        # Create temporary file for combined audio
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.wav',
            delete=False,
            prefix=f'td_combined_{batch_id}_'
        )
        temp_path = Path(temp_file.name)
        temp_file.close()
        
        # Save combined audio
        sf.write(str(temp_path), final_audio, target_sr)
        
        logger.debug(f"Combined {len(combined_audio)} files into {temp_path.name} ({len(final_audio)/target_sr:.1f}s)")
        
        return temp_path

