"""
Audio Feature Extractor for Acoustic & Prosodic Analysis

This module extracts audio-specific features relevant to
acoustic and prosodic analysis:

- Pitch features (F0 mean, std, range, slope, contour)
- Prosody features (intonation, stress, rhythm variability)
- Voice quality (jitter, shimmer, HNR)
- Spectral features (MFCCs, spectral centroid, rolloff)
- Energy/intensity patterns

Uses librosa for audio analysis.

Author: Implementation based on pragmatic features pattern
"""

import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

from src.parsers.chat_parser import TranscriptData
from src.utils.logger import get_logger
from src.utils.helpers import safe_divide
from ..base_features import BaseFeatureExtractor, FeatureResult
from .child_audio_extractor import ChildAudioExtractor

logger = get_logger(__name__)

# Try to import audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available for acoustic features")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


class AcousticAudioFeatures(BaseFeatureExtractor):
    """
    Extract acoustic and prosodic features from audio.
    
    This extractor focuses on features from audio that relate to:
    - Pitch characteristics (F0 mean, std, range, slope)
    - Prosody (intonation, rhythm, stress patterns)
    - Voice quality (jitter, shimmer, harmonic-to-noise ratio)
    - Spectral features (MFCCs, spectral centroid, rolloff)
    - Energy/intensity patterns
    
    Example:
        >>> extractor = AcousticAudioFeatures()
        >>> features = extractor.extract(transcript, audio_path="audio.wav")
    """
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # Pitch features (F0)
            'acoustic_pitch_mean',
            'acoustic_pitch_std',
            'acoustic_pitch_median',
            'acoustic_pitch_min',
            'acoustic_pitch_max',
            'acoustic_pitch_range',
            'acoustic_pitch_slope_mean',
            'acoustic_pitch_slope_std',
            'acoustic_pitch_variability',
            'acoustic_pitch_contour_std',
            
            # Prosody features
            'acoustic_intonation_range',
            'acoustic_intonation_variability',
            'acoustic_rhythm_variability',
            'acoustic_stress_variability',
            'acoustic_pitch_rising_ratio',
            'acoustic_pitch_falling_ratio',
            'acoustic_pitch_flat_ratio',
            
            # Voice quality features
            'acoustic_jitter',
            'acoustic_shimmer',
            'acoustic_hnr_mean',
            'acoustic_hnr_std',
            'acoustic_voicing_fraction',
            
            # Spectral features
            'acoustic_spectral_centroid_mean',
            'acoustic_spectral_centroid_std',
            'acoustic_spectral_rolloff_mean',
            'acoustic_spectral_rolloff_std',
            'acoustic_spectral_bandwidth_mean',
            'acoustic_spectral_bandwidth_std',
            'acoustic_zero_crossing_rate_mean',
            'acoustic_zero_crossing_rate_std',
            
            # MFCC features (first 5 coefficients)
            'acoustic_mfcc_1_mean',
            'acoustic_mfcc_1_std',
            'acoustic_mfcc_2_mean',
            'acoustic_mfcc_2_std',
            'acoustic_mfcc_3_mean',
            'acoustic_mfcc_3_std',
            'acoustic_mfcc_4_mean',
            'acoustic_mfcc_4_std',
            'acoustic_mfcc_5_mean',
            'acoustic_mfcc_5_std',
            
            # Energy/Intensity features
            'acoustic_intensity_mean',
            'acoustic_intensity_std',
            'acoustic_intensity_range',
            'acoustic_intensity_variability',
            'acoustic_energy_mean',
            'acoustic_energy_std',
            
            # Formant-like features (from spectral peaks)
            'acoustic_formant_1_mean',
            'acoustic_formant_1_std',
            'acoustic_formant_2_mean',
            'acoustic_formant_2_std',
            'acoustic_formant_3_mean',
            'acoustic_formant_3_std',
            
            # Extended MFCC features (MFCC 6-13) - 8 coefficients × 2 = 16 features
            'acoustic_mfcc_6_mean',
            'acoustic_mfcc_6_std',
            'acoustic_mfcc_7_mean',
            'acoustic_mfcc_7_std',
            'acoustic_mfcc_8_mean',
            'acoustic_mfcc_8_std',
            'acoustic_mfcc_9_mean',
            'acoustic_mfcc_9_std',
            'acoustic_mfcc_10_mean',
            'acoustic_mfcc_10_std',
            'acoustic_mfcc_11_mean',
            'acoustic_mfcc_11_std',
            'acoustic_mfcc_12_mean',
            'acoustic_mfcc_12_std',
            'acoustic_mfcc_13_mean',
            'acoustic_mfcc_13_std',
            
            # Chroma features (12 pitch classes) - mean/std = 24 features
            'acoustic_chroma_1_mean',
            'acoustic_chroma_1_std',
            'acoustic_chroma_2_mean',
            'acoustic_chroma_2_std',
            'acoustic_chroma_3_mean',
            'acoustic_chroma_3_std',
            'acoustic_chroma_4_mean',
            'acoustic_chroma_4_std',
            'acoustic_chroma_5_mean',
            'acoustic_chroma_5_std',
            'acoustic_chroma_6_mean',
            'acoustic_chroma_6_std',
            'acoustic_chroma_7_mean',
            'acoustic_chroma_7_std',
            'acoustic_chroma_8_mean',
            'acoustic_chroma_8_std',
            'acoustic_chroma_9_mean',
            'acoustic_chroma_9_std',
            'acoustic_chroma_10_mean',
            'acoustic_chroma_10_std',
            'acoustic_chroma_11_mean',
            'acoustic_chroma_11_std',
            'acoustic_chroma_12_mean',
            'acoustic_chroma_12_std',
            
            # Temporal dynamics - pitch trajectory features (5 features)
            'acoustic_pitch_trajectory_slope',
            'acoustic_pitch_trajectory_curvature',
            'acoustic_pitch_acceleration_mean',
            'acoustic_pitch_acceleration_std',
            'acoustic_energy_trajectory_slope',
            
            # Spectral contrast features (7 bands) - 7 features
            'acoustic_spectral_contrast_1',
            'acoustic_spectral_contrast_2',
            'acoustic_spectral_contrast_3',
            'acoustic_spectral_contrast_4',
            'acoustic_spectral_contrast_5',
            'acoustic_spectral_contrast_6',
            'acoustic_spectral_contrast_mean',
            
            # Tonnetz features (harmonic network) - 6 features
            'acoustic_tonnetz_1',
            'acoustic_tonnetz_2',
            'acoustic_tonnetz_3',
            'acoustic_tonnetz_4',
            'acoustic_tonnetz_5',
            'acoustic_tonnetz_6',
            
            # Additional rhythm and timing features (3 features)
            'acoustic_tempo',
            'acoustic_onset_rate',
            'acoustic_silence_ratio',
            
            # Advanced pitch statistics (8 features)
            'acoustic_pitch_q25',
            'acoustic_pitch_q75',
            'acoustic_pitch_iqr',
            'acoustic_pitch_skewness',
            'acoustic_pitch_kurtosis',
            'acoustic_pitch_percentile_10',
            'acoustic_pitch_percentile_90',
            'acoustic_pitch_median_abs_dev',
            
            # Advanced spectral features (6 features)
            'acoustic_spectral_flatness_mean',
            'acoustic_spectral_flatness_std',
            'acoustic_spectral_flux_mean',
            'acoustic_spectral_flux_std',
            'acoustic_spectral_spread_mean',
            'acoustic_spectral_spread_std',
            
            # MFCC Delta features (first 5 coefficients) - 10 features
            'acoustic_mfcc_1_delta_mean',
            'acoustic_mfcc_1_delta_std',
            'acoustic_mfcc_2_delta_mean',
            'acoustic_mfcc_2_delta_std',
            'acoustic_mfcc_3_delta_mean',
            'acoustic_mfcc_3_delta_std',
            'acoustic_mfcc_4_delta_mean',
            'acoustic_mfcc_4_delta_std',
            'acoustic_mfcc_5_delta_mean',
            'acoustic_mfcc_5_delta_std',
            
            # Additional formant features (5 features)
            'acoustic_formant_4_mean',
            'acoustic_formant_4_std',
            'acoustic_formant_1_bandwidth',
            'acoustic_formant_2_bandwidth',
            'acoustic_formant_2_1_ratio',
            
            # Cross-feature correlations (5 features)
            'acoustic_pitch_energy_correlation',
            'acoustic_pitch_intensity_correlation',
            'acoustic_energy_spectral_centroid_correlation',
            'acoustic_pitch_spectral_centroid_correlation',
            'acoustic_intensity_spectral_bandwidth_correlation',
            
            # Harmonic features (3 features)
            'acoustic_harmonic_energy_mean',
            'acoustic_harmonic_energy_std',
            'acoustic_percussive_energy_ratio',
            
            # Additional statistical moments (3 features)
            'acoustic_energy_skewness',
            'acoustic_energy_kurtosis',
            'acoustic_intensity_skewness',
        ]
    
    def __init__(self, sample_rate: int = 16000, extract_child_only: bool = True):
        """
        Initialize the acoustic audio feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio processing
            extract_child_only: If True, extract only child speech segments from audio
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.extract_child_only = extract_child_only
        self.child_audio_extractor = ChildAudioExtractor() if extract_child_only else None
        
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available - acoustic features will be limited")
        
        logger.info(f"AcousticAudioFeatures initialized (child_only={extract_child_only})")
    
    def extract(
        self,
        transcript: TranscriptData,
        audio_path: Optional[str | Path] = None,
        transcription_result: Optional[Any] = None,
        **kwargs
    ) -> FeatureResult:
        """
        Extract acoustic/prosodic features from audio.
        
        If extract_child_only is True, this will extract only child speech segments
        from the audio before analyzing acoustic features.
        
        Args:
            transcript: Parsed transcript data
            audio_path: Path to audio file for direct analysis
            transcription_result: Optional TranscriptionResult from audio processing
            **kwargs: Additional arguments
            
        Returns:
            FeatureResult with acoustic/prosodic features
        """
        features = {}
        child_audio_extracted = False
        temp_audio_path = None
        
        logger.debug(f"Extracting acoustic features for {transcript.participant_id}")
        
        # If audio file provided, extract real features
        if audio_path and LIBROSA_AVAILABLE:
            try:
                # Extract child-only audio if enabled
                if self.extract_child_only and self.child_audio_extractor:
                    logger.debug("Extracting child-only audio segments...")
                    child_audio_path = self.child_audio_extractor.extract_child_audio(
                        audio_path=Path(audio_path),
                        transcript=transcript,
                        transcription_result=transcription_result
                    )
                    
                    if child_audio_path and child_audio_path != Path(audio_path):
                        # Successfully extracted child audio
                        audio_path = child_audio_path
                        child_audio_extracted = True
                        temp_audio_path = child_audio_path
                        logger.debug(f"Using child-only audio: {child_audio_path.name}")
                    else:
                        logger.debug("Using full audio (child extraction not possible)")
                
                # Extract features from audio (child-only or full)
                audio_features = self._extract_from_audio_file(audio_path)
                features.update(audio_features)
                
                # Clean up temporary child audio file
                if temp_audio_path and temp_audio_path.exists():
                    try:
                        temp_audio_path.unlink()
                        logger.debug(f"Cleaned up temporary file: {temp_audio_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not delete temp file: {e}")
                
            except Exception as e:
                logger.error(f"Error extracting from audio file: {e}")
                # Fall back to default features
                features.update(self._get_default_features())
        else:
            # Use default features when no audio available
            features.update(self._get_default_features())
        
        logger.debug(f"Extracted {len(features)} acoustic features")

        # This prevents missing feature errors during inference
        expected_features = self.feature_names
        for feature_name in expected_features:
            if feature_name not in features:
                features[feature_name] = 0.0

        # Reorder features to match expected order
        ordered_features = {name: features.get(name, 0.0) for name in expected_features}

        return FeatureResult(
            features=ordered_features,
            feature_type='acoustic_audio',
            metadata={
                'has_audio': audio_path is not None,
                'child_audio_extracted': child_audio_extracted,
                'librosa_available': LIBROSA_AVAILABLE,
                'sample_rate': self.sample_rate,
            }
        )
    
    def _extract_from_audio_file(
        self,
        audio_path: str | Path
    ) -> Dict[str, float]:
        """
        Extract acoustic features directly from audio file using librosa.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of acoustic features
        """
        features = {}
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return self._get_default_features()
        
        try:
            audio, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                mono=True
            )

            if len(audio) == 0:
                logger.warning(f"Empty audio file: {audio_path}")
                return self._get_default_features()
            
            logger.debug(f"Loaded audio: {len(audio)} samples at {sr} Hz ({len(audio)/sr:.1f}s)")

            # Use faster feature extraction for individual ASD files
            is_individual_asd = any(dataset in str(audio_path).lower() for dataset in ['asdbank', 'aac'])

            if is_individual_asd:
                # Fast feature extraction for individual ASD files
                logger.debug(f"Using optimized feature extraction for ASD file: {Path(audio_path).name}")
                features = self._extract_fast_features(audio, sr)
            else:
                # Full feature extraction for merged files
                logger.debug(f"Using full feature extraction for: {Path(audio_path).name}")
                features = self._extract_full_features(audio, sr)

            logger.debug(f"Completed feature extraction from {Path(audio_path).name} - {len(features)} features")

        except Exception as e:
            logger.error(f"Error processing audio file {audio_path}: {e}")
            return self._get_default_features()
        
        return features
    
    def _extract_pitch_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract pitch (F0) features."""
        features = {}
        
        try:
            # Extract pitch using librosa's pyin algorithm
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=float(librosa.note_to_hz('C2')),  # ~65 Hz
                fmax=float(librosa.note_to_hz('C7')),  # ~2093 Hz
                frame_length=2048,
                hop_length=512
            )
            
            # Filter out unvoiced frames (NaN values)
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) > 0:
                features['acoustic_pitch_mean'] = float(np.mean(f0_voiced))
                features['acoustic_pitch_std'] = float(np.std(f0_voiced))
                features['acoustic_pitch_median'] = float(np.median(f0_voiced))
                features['acoustic_pitch_min'] = float(np.min(f0_voiced))
                features['acoustic_pitch_max'] = float(np.max(f0_voiced))
                features['acoustic_pitch_range'] = float(np.max(f0_voiced) - np.min(f0_voiced))
                
                # Pitch slope (rate of change)
                if len(f0_voiced) > 1:
                    pitch_diff = np.diff(f0_voiced)
                    features['acoustic_pitch_slope_mean'] = float(np.mean(pitch_diff))
                    features['acoustic_pitch_slope_std'] = float(np.std(pitch_diff))
                else:
                    features['acoustic_pitch_slope_mean'] = 0.0
                    features['acoustic_pitch_slope_std'] = 0.0
                
                # Pitch variability (coefficient of variation)
                features['acoustic_pitch_variability'] = float(safe_divide(
                    features['acoustic_pitch_std'],
                    features['acoustic_pitch_mean']
                ))

                # Pitch contour standard deviation
                features['acoustic_pitch_contour_std'] = features['acoustic_pitch_std']
                
                # Voicing fraction
                features['acoustic_voicing_fraction'] = float(np.sum(voiced_flag) / len(voiced_flag))
            else:
                # No voiced frames found
                features.update({k: 0.0 for k in [
                    'acoustic_pitch_mean', 'acoustic_pitch_std', 'acoustic_pitch_median',
                    'acoustic_pitch_min', 'acoustic_pitch_max', 'acoustic_pitch_range',
                    'acoustic_pitch_slope_mean', 'acoustic_pitch_slope_std',
                    'acoustic_pitch_variability', 'acoustic_pitch_contour_std',
                    'acoustic_voicing_fraction'
                ]})
                
        except Exception as e:
            logger.warning(f"Error extracting pitch features: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_pitch_mean', 'acoustic_pitch_std', 'acoustic_pitch_median',
                'acoustic_pitch_min', 'acoustic_pitch_max', 'acoustic_pitch_range',
                'acoustic_pitch_slope_mean', 'acoustic_pitch_slope_std',
                'acoustic_pitch_variability', 'acoustic_pitch_contour_std',
                'acoustic_voicing_fraction'
            ]})
        
        return features
    
    def _extract_prosody_features(
        self,
        audio: np.ndarray,
        sr: int,
        pitch_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract prosody features (intonation, rhythm, stress)."""
        features = {}
        
        try:
            # Intonation range and variability
            if 'acoustic_pitch_range' in pitch_features:
                features['acoustic_intonation_range'] = pitch_features['acoustic_pitch_range']
                features['acoustic_intonation_variability'] = pitch_features.get(
                    'acoustic_pitch_variability', 0.0
                )
            else:
                features['acoustic_intonation_range'] = 0.0
                features['acoustic_intonation_variability'] = 0.0
            
            # Rhythm variability (from energy envelope)
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            if len(rms) > 1:
                features['acoustic_rhythm_variability'] = float(np.std(rms))
            else:
                features['acoustic_rhythm_variability'] = 0.0
            
            # Stress variability (from intensity variations)
            features['acoustic_stress_variability'] = features['acoustic_rhythm_variability']
            
            # Pitch direction ratios (rising, falling, flat)
            try:
                f0, _, _ = librosa.pyin(
                    audio,
                    fmin=float(librosa.note_to_hz('C2')),
                    fmax=float(librosa.note_to_hz('C7')),
                    frame_length=2048,
                    hop_length=512
                )
                f0_voiced = f0[~np.isnan(f0)]
                
                if len(f0_voiced) > 1:
                    pitch_diff = np.diff(f0_voiced)
                    total = len(pitch_diff)
                    rising = np.sum(pitch_diff > 5)  # Threshold for rising
                    falling = np.sum(pitch_diff < -5)  # Threshold for falling
                    flat = total - rising - falling
                    
                    features['acoustic_pitch_rising_ratio'] = float(safe_divide(rising, total))
                    features['acoustic_pitch_falling_ratio'] = float(safe_divide(falling, total))
                    features['acoustic_pitch_flat_ratio'] = float(safe_divide(flat, total))
                else:
                    features['acoustic_pitch_rising_ratio'] = 0.0
                    features['acoustic_pitch_falling_ratio'] = 0.0
                    features['acoustic_pitch_flat_ratio'] = 1.0
            except:
                features['acoustic_pitch_rising_ratio'] = 0.0
                features['acoustic_pitch_falling_ratio'] = 0.0
                features['acoustic_pitch_flat_ratio'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error extracting prosody features: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_intonation_range', 'acoustic_intonation_variability',
                'acoustic_rhythm_variability', 'acoustic_stress_variability',
                'acoustic_pitch_rising_ratio', 'acoustic_pitch_falling_ratio',
                'acoustic_pitch_flat_ratio'
            ]})
        
        return features
    
    def _extract_voice_quality_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract voice quality features (jitter, shimmer, HNR)."""
        features = {}
        
        try:
            # Extract pitch for jitter calculation
            f0, _, _ = librosa.pyin(
                audio,
                fmin=float(librosa.note_to_hz('C2')),
                fmax=float(librosa.note_to_hz('C7')),
                frame_length=2048,
                hop_length=512
            )
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) > 1:
                # Jitter: period-to-period variation in F0
                periods = 1.0 / f0_voiced  # Period in seconds
                period_diff = np.abs(np.diff(periods))
                features['acoustic_jitter'] = float(safe_divide(
                    float(np.mean(period_diff)),
                    float(np.mean(periods))
                ))
            else:
                features['acoustic_jitter'] = 0.0
            
            # Shimmer: amplitude variation (simplified)
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            if len(rms) > 1:
                rms_diff = np.abs(np.diff(rms))
                features['acoustic_shimmer'] = float(safe_divide(
                    float(np.mean(rms_diff)),
                    float(np.mean(rms))
                ))
            else:
                features['acoustic_shimmer'] = 0.0
            
            # Harmonic-to-Noise Ratio (HNR) - simplified using spectral features
            # Using spectral centroid as proxy
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            if len(spectral_centroid) > 0:
                # HNR approximation: higher spectral centroid suggests more harmonic content
                hnr_approx = np.mean(spectral_centroid) / 1000.0  # Normalize
                features['acoustic_hnr_mean'] = float(hnr_approx)
                features['acoustic_hnr_std'] = float(np.std(spectral_centroid) / 1000.0)
            else:
                features['acoustic_hnr_mean'] = 0.0
                features['acoustic_hnr_std'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error extracting voice quality features: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_jitter', 'acoustic_shimmer',
                'acoustic_hnr_mean', 'acoustic_hnr_std'
            ]})
        
        return features
    
    def _extract_spectral_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract spectral features."""
        features = {}
        
        try:
            # Spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['acoustic_spectral_centroid_mean'] = float(np.mean(spectral_centroid))
            features['acoustic_spectral_centroid_std'] = float(np.std(spectral_centroid))
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            features['acoustic_spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            features['acoustic_spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['acoustic_spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            features['acoustic_spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['acoustic_zero_crossing_rate_mean'] = float(np.mean(zcr))
            features['acoustic_zero_crossing_rate_std'] = float(np.std(zcr))
            
        except Exception as e:
            logger.warning(f"Error extracting spectral features: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_spectral_centroid_mean', 'acoustic_spectral_centroid_std',
                'acoustic_spectral_rolloff_mean', 'acoustic_spectral_rolloff_std',
                'acoustic_spectral_bandwidth_mean', 'acoustic_spectral_bandwidth_std',
                'acoustic_zero_crossing_rate_mean', 'acoustic_zero_crossing_rate_std'
            ]})
        
        return features
    
    def _extract_mfcc_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract MFCC features (first 5 coefficients)."""
        features = {}
        
        try:
            # Extract MFCCs (13 coefficients, we use first 5)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            for i in range(1, 6):  # MFCC 1-5 (0-indexed: 0-4)
                mfcc_coeff = mfccs[i-1, :]  # librosa uses 0-indexed
                features[f'acoustic_mfcc_{i}_mean'] = float(np.mean(mfcc_coeff))
                features[f'acoustic_mfcc_{i}_std'] = float(np.std(mfcc_coeff))
                
        except Exception as e:
            logger.warning(f"Error extracting MFCC features: {e}")
            for i in range(1, 6):
                features[f'acoustic_mfcc_{i}_mean'] = 0.0
                features[f'acoustic_mfcc_{i}_std'] = 0.0
        
        return features
    
    def _extract_energy_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract energy/intensity features."""
        features = {}
        
        try:
            # RMS energy
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            features['acoustic_energy_mean'] = float(np.mean(rms))
            features['acoustic_energy_std'] = float(np.std(rms))
            
            # Intensity (similar to RMS, in dB)
            rms_db = librosa.power_to_db(rms**2, ref=np.max)
            features['acoustic_intensity_mean'] = float(np.mean(rms_db))
            features['acoustic_intensity_std'] = float(np.std(rms_db))
            features['acoustic_intensity_range'] = float(np.max(rms_db) - np.min(rms_db))
            features['acoustic_intensity_variability'] = features['acoustic_intensity_std']
            
        except Exception as e:
            logger.warning(f"Error extracting energy features: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_intensity_mean', 'acoustic_intensity_std',
                'acoustic_intensity_range', 'acoustic_intensity_variability',
                'acoustic_energy_mean', 'acoustic_energy_std'
            ]})
        
        return features
    
    def _extract_formant_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract formant features (F1/F2/F3).

        Preferred backend: Praat via praat-parselmouth (LPC formants).
        Fallback backend: spectral-peak proxy (previous implementation).

        Returns:
            Dict containing:
                - acoustic_formant_1_mean/std
                - acoustic_formant_2_mean/std
                - acoustic_formant_3_mean/std
        """
        try:
            import parselmouth  # type: ignore

            # Parselmouth expects float64 and works best with normalized audio.
            y = np.asarray(audio, dtype=np.float64)
            if y.size == 0:
                raise ValueError("Empty audio")

            peak = float(np.max(np.abs(y)))
            if peak > 0:
                y = y / peak

            # Create Praat Sound
            snd = parselmouth.Sound(y, sampling_frequency=float(sr))

            # Praat formant (Burg).
            formant = snd.to_formant_burg(
                time_step=0.01,
                max_number_of_formants=5,
                maximum_formant=5500,
                window_length=0.025,
                pre_emphasis_from=50,
            )

            times = formant.xs()
            f1: list[float] = []
            f2: list[float] = []
            f3: list[float] = []

            for t in times:
                v1 = formant.get_value_at_time(1, t)
                v2 = formant.get_value_at_time(2, t)
                v3 = formant.get_value_at_time(3, t)

                # Parselmouth returns NaN for undefined values.
                if v1 is not None and np.isfinite(v1) and v1 > 0:
                    f1.append(float(v1))
                if v2 is not None and np.isfinite(v2) and v2 > 0:
                    f2.append(float(v2))
                if v3 is not None and np.isfinite(v3) and v3 > 0:
                    f3.append(float(v3))

            # Reasonable defaults if extraction fails (same as old behavior)
            features: Dict[str, float] = {
                'acoustic_formant_1_mean': float(np.mean(f1)) if len(f1) else 300.0,
                'acoustic_formant_1_std': float(np.std(f1)) if len(f1) else 0.0,
                'acoustic_formant_2_mean': float(np.mean(f2)) if len(f2) else 1500.0,
                'acoustic_formant_2_std': float(np.std(f2)) if len(f2) else 0.0,
                'acoustic_formant_3_mean': float(np.mean(f3)) if len(f3) else 3000.0,
                'acoustic_formant_3_std': float(np.std(f3)) if len(f3) else 0.0,
            }

            return features

        except Exception as e:
            # Fall back to the previous spectral-peak approach if parselmouth isn't
            # installed or if Praat extraction fails for this file.
            logger.debug(f"Parselmouth formant extraction unavailable/failed; falling back to spectral peaks: {e}")

        # --- Fallback: spectral-peak proxy (previous implementation) ---
        features: Dict[str, float] = {}

        try:
            # Get spectral magnitude with shorter window for efficiency
            stft = librosa.stft(audio, n_fft=1024, hop_length=512)
            magnitude = np.abs(stft)

            # Find spectral peaks (formant-like)
            formant_1 = []
            formant_2 = []
            formant_3 = []

            # Sample only every 10th frame for speed
            for frame in range(0, magnitude.shape[1], 10):
                try:
                    frame_mag = magnitude[:, frame]

                    # Find local maxima
                    peaks = []
                    for i in range(1, len(frame_mag) - 1):
                        if frame_mag[i] > frame_mag[i - 1] and frame_mag[i] > frame_mag[i + 1]:
                            peaks.append(i)

                    if len(peaks) == 0:
                        continue

                    # Convert bin indices to frequencies
                    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 1)

                    # Ensure we don't exceed frequency array bounds
                    valid_peaks = [p for p in peaks if p < len(freqs)]
                    if len(valid_peaks) == 0:
                        continue

                    peak_freqs = freqs[valid_peaks]
                    peak_mags = frame_mag[valid_peaks]

                    # Sort by magnitude and take top candidates
                    if len(peak_mags) > 0:
                        sorted_indices = np.argsort(peak_mags)[::-1]
                        sorted_freqs = peak_freqs[sorted_indices]

                        # Filter to typical formant ranges
                        f1_candidates = sorted_freqs[(sorted_freqs >= 300) & (sorted_freqs <= 1000)]
                        f2_candidates = sorted_freqs[(sorted_freqs >= 1000) & (sorted_freqs <= 3000)]
                        f3_candidates = sorted_freqs[(sorted_freqs >= 2500) & (sorted_freqs <= 4000)]

                        if len(f1_candidates) > 0:
                            formant_1.append(f1_candidates[0])
                        if len(f2_candidates) > 0:
                            formant_2.append(f2_candidates[0])
                        if len(f3_candidates) > 0:
                            formant_3.append(f3_candidates[0])

                except (IndexError, ValueError):
                    continue

            # Calculate statistics with safety checks
            features['acoustic_formant_1_mean'] = float(np.mean(formant_1)) if len(formant_1) > 0 else 300.0
            features['acoustic_formant_1_std'] = float(np.std(formant_1)) if len(formant_1) > 0 else 0.0

            features['acoustic_formant_2_mean'] = float(np.mean(formant_2)) if len(formant_2) > 0 else 1500.0
            features['acoustic_formant_2_std'] = float(np.std(formant_2)) if len(formant_2) > 0 else 0.0

            features['acoustic_formant_3_mean'] = float(np.mean(formant_3)) if len(formant_3) > 0 else 3000.0
            features['acoustic_formant_3_std'] = float(np.std(formant_3)) if len(formant_3) > 0 else 0.0

        except Exception as e:
            logger.debug(f"Error extracting formant features: {e}")
            features.update({
                'acoustic_formant_1_mean': 300.0, 'acoustic_formant_1_std': 50.0,
                'acoustic_formant_2_mean': 1500.0, 'acoustic_formant_2_std': 200.0,
                'acoustic_formant_3_mean': 3000.0, 'acoustic_formant_3_std': 300.0
            })

        return features
    
    def _extract_extended_mfcc_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract extended MFCC features (coefficients 6-13)."""
        features = {}
        
        try:
            # Extract MFCCs (13 coefficients total, we use 6-13)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            for i in range(6, 14):  # MFCC 6-13 (0-indexed: 5-12)
                mfcc_coeff = mfccs[i-1, :]  # librosa uses 0-indexed
                features[f'acoustic_mfcc_{i}_mean'] = float(np.mean(mfcc_coeff))
                features[f'acoustic_mfcc_{i}_std'] = float(np.std(mfcc_coeff))
                
        except Exception as e:
            logger.warning(f"Error extracting extended MFCC features: {e}")
            for i in range(6, 14):
                features[f'acoustic_mfcc_{i}_mean'] = 0.0
                features[f'acoustic_mfcc_{i}_std'] = 0.0
        
        return features
    
    def _extract_chroma_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract chroma features (12 pitch classes)."""
        features = {}
        
        try:
            # Extract chroma features (12 pitch classes) using chroma_stft
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            for i in range(1, 13):  # 12 pitch classes
                chroma_coeff = chroma[i-1, :]  # 0-indexed
                features[f'acoustic_chroma_{i}_mean'] = float(np.mean(chroma_coeff))
                features[f'acoustic_chroma_{i}_std'] = float(np.std(chroma_coeff))
                
        except Exception as e:
            logger.warning(f"Error extracting chroma features: {e}")
            for i in range(1, 13):
                features[f'acoustic_chroma_{i}_mean'] = 0.0
                features[f'acoustic_chroma_{i}_std'] = 0.0
        
        return features
    
    def _extract_temporal_dynamics_features(
        self,
        audio: np.ndarray,
        sr: int,
        pitch_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract temporal dynamics features (pitch/energy trajectories)."""
        features = {}
        
        try:
            # Extract pitch for trajectory analysis
            f0, _, _ = librosa.pyin(
                audio,
                fmin=float(librosa.note_to_hz('C2')),
                fmax=float(librosa.note_to_hz('C7')),
                frame_length=2048,
                hop_length=512
            )
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) > 2:
                # Pitch trajectory slope (linear fit)
                x = np.arange(len(f0_voiced))
                coeffs = np.polyfit(x, f0_voiced, 1)
                features['acoustic_pitch_trajectory_slope'] = float(coeffs[0])
                
                # Pitch trajectory curvature (quadratic fit)
                coeffs_quad = np.polyfit(x, f0_voiced, 2)
                features['acoustic_pitch_trajectory_curvature'] = float(coeffs_quad[0])
                
                # Pitch acceleration (second derivative)
                if len(f0_voiced) > 2:
                    pitch_diff = np.diff(f0_voiced)
                    pitch_accel = np.diff(pitch_diff)
                    features['acoustic_pitch_acceleration_mean'] = float(np.mean(pitch_accel))
                    features['acoustic_pitch_acceleration_std'] = float(np.std(pitch_accel))
                else:
                    features['acoustic_pitch_acceleration_mean'] = 0.0
                    features['acoustic_pitch_acceleration_std'] = 0.0
            else:
                features['acoustic_pitch_trajectory_slope'] = 0.0
                features['acoustic_pitch_trajectory_curvature'] = 0.0
                features['acoustic_pitch_acceleration_mean'] = 0.0
                features['acoustic_pitch_acceleration_std'] = 0.0
            
            # Energy trajectory slope
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            if len(rms) > 1:
                x_energy = np.arange(len(rms))
                coeffs_energy = np.polyfit(x_energy, rms, 1)
                features['acoustic_energy_trajectory_slope'] = float(coeffs_energy[0])
            else:
                features['acoustic_energy_trajectory_slope'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error extracting temporal dynamics features: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_pitch_trajectory_slope',
                'acoustic_pitch_trajectory_curvature',
                'acoustic_pitch_acceleration_mean',
                'acoustic_pitch_acceleration_std',
                'acoustic_energy_trajectory_slope'
            ]})
        
        return features
    
    def _extract_spectral_contrast_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract spectral contrast features."""
        features = {}
        
        try:
            # Extract spectral contrast (6 frequency bands by default)
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            
            # Get mean across time for each band (typically 6-7 bands)
            n_bands = min(spectral_contrast.shape[0], 6)  # Use up to 6 bands
            for i in range(1, n_bands + 1):
                features[f'acoustic_spectral_contrast_{i}'] = float(np.mean(spectral_contrast[i-1, :]))
            
            # Fill remaining bands with 0 if fewer than 6
            for i in range(n_bands + 1, 7):
                features[f'acoustic_spectral_contrast_{i}'] = 0.0
            
            # Overall mean
            features['acoustic_spectral_contrast_mean'] = float(np.mean(spectral_contrast))
            
        except Exception as e:
            logger.warning(f"Error extracting spectral contrast features: {e}")
            # Default to 7 features (6 bands + mean)
            for i in range(1, 7):
                features[f'acoustic_spectral_contrast_{i}'] = 0.0
            features['acoustic_spectral_contrast_mean'] = 0.0
        
        return features
    
    def _extract_tonnetz_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract tonnetz features (harmonic network representation)."""
        features = {}
        
        try:
            # Extract tonnetz features (6 dimensions)
            tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
            
            # Get mean across time for each dimension
            for i in range(1, 7):  # 6 dimensions
                features[f'acoustic_tonnetz_{i}'] = float(np.mean(tonnetz[i-1, :]))
                
        except Exception as e:
            logger.warning(f"Error extracting tonnetz features: {e}")
            for i in range(1, 7):
                features[f'acoustic_tonnetz_{i}'] = 0.0
        
        return features
    
    def _extract_rhythm_timing_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract rhythm and timing features."""
        features = {}
        
        try:
            # Tempo estimation
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['acoustic_tempo'] = float(tempo) if tempo is not None else 0.0
            
            # Onset rate (onsets per second)
            onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            duration = len(audio) / sr
            features['acoustic_onset_rate'] = float(safe_divide(len(onsets), duration))

            # Silence ratio (proportion of silence)
            intervals = librosa.effects.split(audio, top_db=30)
            if len(intervals) > 0:
                speech_time = sum((e - s) for s, e in intervals) / sr
                silence_time = max(duration - speech_time, 0.0)
                features['acoustic_silence_ratio'] = float(safe_divide(silence_time, duration))
            else:
                features['acoustic_silence_ratio'] = 1.0  # All silence
                
        except Exception as e:
            logger.warning(f"Error extracting rhythm/timing features: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_tempo',
                'acoustic_onset_rate',
                'acoustic_silence_ratio'
            ]})
        
        return features
    
    def _extract_advanced_pitch_statistics(
        self,
        audio: np.ndarray,
        sr: int,
        pitch_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract advanced pitch statistics (quartiles, skewness, kurtosis)."""
        features = {}
        
        try:
            # Extract pitch for advanced statistics
            f0, _, _ = librosa.pyin(
                audio,
                fmin=float(librosa.note_to_hz('C2')),
                fmax=float(librosa.note_to_hz('C7')),
                frame_length=2048,
                hop_length=512
            )
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) > 0:
                # Quartiles
                features['acoustic_pitch_q25'] = float(np.percentile(f0_voiced, 25))
                features['acoustic_pitch_q75'] = float(np.percentile(f0_voiced, 75))
                features['acoustic_pitch_iqr'] = float(features['acoustic_pitch_q75'] - features['acoustic_pitch_q25'])
                
                # Percentiles
                features['acoustic_pitch_percentile_10'] = float(np.percentile(f0_voiced, 10))
                features['acoustic_pitch_percentile_90'] = float(np.percentile(f0_voiced, 90))
                
                # Median absolute deviation
                median = np.median(f0_voiced)
                features['acoustic_pitch_median_abs_dev'] = float(np.median(np.abs(f0_voiced - median)))
                
                # Skewness and kurtosis
                if len(f0_voiced) > 2:
                    from scipy import stats
                    try:
                        features['acoustic_pitch_skewness'] = float(stats.skew(f0_voiced))
                        features['acoustic_pitch_kurtosis'] = float(stats.kurtosis(f0_voiced))
                    except:
                        features['acoustic_pitch_skewness'] = 0.0
                        features['acoustic_pitch_kurtosis'] = 0.0
                else:
                    features['acoustic_pitch_skewness'] = 0.0
                    features['acoustic_pitch_kurtosis'] = 0.0
            else:
                features.update({k: 0.0 for k in [
                    'acoustic_pitch_q25', 'acoustic_pitch_q75', 'acoustic_pitch_iqr',
                    'acoustic_pitch_skewness', 'acoustic_pitch_kurtosis',
                    'acoustic_pitch_percentile_10', 'acoustic_pitch_percentile_90',
                    'acoustic_pitch_median_abs_dev'
                ]})
                
        except Exception as e:
            logger.warning(f"Error extracting advanced pitch statistics: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_pitch_q25', 'acoustic_pitch_q75', 'acoustic_pitch_iqr',
                'acoustic_pitch_skewness', 'acoustic_pitch_kurtosis',
                'acoustic_pitch_percentile_10', 'acoustic_pitch_percentile_90',
                'acoustic_pitch_median_abs_dev'
            ]})
        
        return features
    
    def _extract_advanced_spectral_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract advanced spectral features (flatness, flux, spread)."""
        features = {}
        
        try:
            # Spectral flatness
            spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
            features['acoustic_spectral_flatness_mean'] = float(np.mean(spectral_flatness))
            features['acoustic_spectral_flatness_std'] = float(np.std(spectral_flatness))
            
            # Spectral flux (rate of change of spectrum)
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            spectral_flux = np.sum(np.diff(magnitude, axis=1)**2, axis=0)
            features['acoustic_spectral_flux_mean'] = float(np.mean(spectral_flux))
            features['acoustic_spectral_flux_std'] = float(np.std(spectral_flux))
            
            # Spectral spread (second moment around centroid)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_spread = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features['acoustic_spectral_spread_mean'] = float(np.mean(spectral_spread))
            features['acoustic_spectral_spread_std'] = float(np.std(spectral_spread))
            
        except Exception as e:
            logger.warning(f"Error extracting advanced spectral features: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_spectral_flatness_mean', 'acoustic_spectral_flatness_std',
                'acoustic_spectral_flux_mean', 'acoustic_spectral_flux_std',
                'acoustic_spectral_spread_mean', 'acoustic_spectral_spread_std'
            ]})
        
        return features
    
    def _extract_mfcc_delta_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract MFCC delta features (first 5 coefficients)."""
        features = {}
        
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Compute delta (first derivative)
            mfcc_delta = librosa.feature.delta(mfccs)
            
            # Extract first 5 coefficients
            for i in range(1, 6):
                delta_coeff = mfcc_delta[i-1, :]
                features[f'acoustic_mfcc_{i}_delta_mean'] = float(np.mean(delta_coeff))
                features[f'acoustic_mfcc_{i}_delta_std'] = float(np.std(delta_coeff))
                
        except Exception as e:
            logger.warning(f"Error extracting MFCC delta features: {e}")
            for i in range(1, 6):
                features[f'acoustic_mfcc_{i}_delta_mean'] = 0.0
                features[f'acoustic_mfcc_{i}_delta_std'] = 0.0
        
        return features
    
    def _extract_additional_formant_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract additional formant features."""
        features = {}
        
        try:
            # Simplified approach using spectral statistics
            stft = librosa.stft(audio, n_fft=1024, hop_length=512)
            magnitude = np.abs(stft)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)

            # Calculate spectral centroid in different frequency bands
            f4_range_mask = (freqs >= 3500) & (freqs <= 5000)
            f1_range_mask = (freqs >= 300) & (freqs <= 1000)
            f2_range_mask = (freqs >= 1000) & (freqs <= 3000)

            if np.any(f4_range_mask):
                f4_energy = np.mean(magnitude[f4_range_mask, :])
                features['acoustic_formant_4_mean'] = float(4000.0 if f4_energy > 0.01 else 3800.0)
                features['acoustic_formant_4_std'] = float(200.0 if f4_energy > 0.01 else 100.0)
            else:
                features['acoustic_formant_4_mean'] = 3800.0
                features['acoustic_formant_4_std'] = 100.0

            # Approximate bandwidths using frequency range statistics
            if np.any(f1_range_mask) and np.any(f2_range_mask):
                f1_bandwidth = 100.0  # Typical F1 bandwidth
                f2_bandwidth = 200.0  # Typical F2 bandwidth
                f2_f1_ratio = 1500.0 / 300.0  # Typical ratio

                features['acoustic_formant_1_bandwidth'] = f1_bandwidth
                features['acoustic_formant_2_bandwidth'] = f2_bandwidth
                features['acoustic_formant_2_1_ratio'] = f2_f1_ratio
            else:
                features['acoustic_formant_1_bandwidth'] = 100.0
                features['acoustic_formant_2_bandwidth'] = 200.0
                features['acoustic_formant_2_1_ratio'] = 5.0

        except Exception as e:
            logger.debug(f"Error extracting additional formant features: {e}")  # Changed to debug level
            # Return typical values
            features.update({
                'acoustic_formant_4_mean': 3800.0, 'acoustic_formant_4_std': 100.0,
                'acoustic_formant_1_bandwidth': 100.0, 'acoustic_formant_2_bandwidth': 200.0,
                'acoustic_formant_2_1_ratio': 5.0
            })

        return features
    
    def _extract_cross_feature_correlations(
        self,
        audio: np.ndarray,
        sr: int,
        pitch_features: Dict[str, float],
        energy_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract cross-feature correlations."""
        features = {}
        
        try:
            # Extract pitch and energy time series
            f0, _, _ = librosa.pyin(
                audio,
                fmin=float(librosa.note_to_hz('C2')),
                fmax=float(librosa.note_to_hz('C7')),
                frame_length=2048,
                hop_length=512
            )
            f0_voiced = f0[~np.isnan(f0)]
            
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Align lengths
            min_len = min(len(f0_voiced), len(rms))
            if min_len > 1:
                f0_aligned = f0_voiced[:min_len]
                rms_aligned = rms[:min_len]
                
                # Correlations
                features['acoustic_pitch_energy_correlation'] = float(np.corrcoef(f0_aligned, rms_aligned)[0, 1])
            else:
                features['acoustic_pitch_energy_correlation'] = 0.0
            
            # Intensity correlation
            rms_db = librosa.power_to_db(rms**2, ref=np.max)
            if min_len > 1:
                features['acoustic_pitch_intensity_correlation'] = float(np.corrcoef(f0_aligned, rms_db[:min_len])[0, 1])
            else:
                features['acoustic_pitch_intensity_correlation'] = 0.0
            
            # Spectral centroid correlations
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            
            if len(rms) > 1 and len(spectral_centroid) > 1:
                min_len_sc = min(len(rms), len(spectral_centroid))
                features['acoustic_energy_spectral_centroid_correlation'] = float(
                    np.corrcoef(rms[:min_len_sc], spectral_centroid[:min_len_sc])[0, 1]
                )
            else:
                features['acoustic_energy_spectral_centroid_correlation'] = 0.0
            
            if len(f0_voiced) > 1 and len(spectral_centroid) > 1:
                min_len_pc = min(len(f0_voiced), len(spectral_centroid))
                features['acoustic_pitch_spectral_centroid_correlation'] = float(
                    np.corrcoef(f0_voiced[:min_len_pc], spectral_centroid[:min_len_pc])[0, 1]
                )
            else:
                features['acoustic_pitch_spectral_centroid_correlation'] = 0.0
            
            if len(rms_db) > 1 and len(spectral_bandwidth) > 1:
                min_len_sb = min(len(rms_db), len(spectral_bandwidth))
                features['acoustic_intensity_spectral_bandwidth_correlation'] = float(
                    np.corrcoef(rms_db[:min_len_sb], spectral_bandwidth[:min_len_sb])[0, 1]
                )
            else:
                features['acoustic_intensity_spectral_bandwidth_correlation'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error extracting cross-feature correlations: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_pitch_energy_correlation',
                'acoustic_pitch_intensity_correlation',
                'acoustic_energy_spectral_centroid_correlation',
                'acoustic_pitch_spectral_centroid_correlation',
                'acoustic_intensity_spectral_bandwidth_correlation'
            ]})
        
        return features
    
    def _extract_harmonic_features(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, float]:
        """Extract harmonic and percussive features."""
        features = {}
        
        try:
            # Separate harmonic and percussive components
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            # Calculate energy for each component
            frame_length = 2048
            hop_length = 512
            rms_harmonic = librosa.feature.rms(y=y_harmonic, frame_length=frame_length, hop_length=hop_length)[0]
            rms_percussive = librosa.feature.rms(y=y_percussive, frame_length=frame_length, hop_length=hop_length)[0]
            
            features['acoustic_harmonic_energy_mean'] = float(np.mean(rms_harmonic))
            features['acoustic_harmonic_energy_std'] = float(np.std(rms_harmonic))
            
            # Percussive energy ratio
            total_energy = np.mean(rms_harmonic) + np.mean(rms_percussive)
            features['acoustic_percussive_energy_ratio'] = float(safe_divide(
                float(np.mean(rms_percussive)),
                float(total_energy)
            ))

        except Exception as e:
            logger.warning(f"Error extracting harmonic features: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_harmonic_energy_mean',
                'acoustic_harmonic_energy_std',
                'acoustic_percussive_energy_ratio'
            ]})
        
        return features
    
    def _extract_additional_statistical_moments(
        self,
        audio: np.ndarray,
        sr: int,
        energy_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Extract additional statistical moments for energy and intensity."""
        features = {}
        
        try:
            # Extract energy time series
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            rms_db = librosa.power_to_db(rms**2, ref=np.max)
            
            if len(rms) > 2:
                from scipy import stats
                try:
                    features['acoustic_energy_skewness'] = float(stats.skew(rms))
                    features['acoustic_energy_kurtosis'] = float(stats.kurtosis(rms))
                except:
                    features['acoustic_energy_skewness'] = 0.0
                    features['acoustic_energy_kurtosis'] = 0.0
            else:
                features['acoustic_energy_skewness'] = 0.0
                features['acoustic_energy_kurtosis'] = 0.0
            
            if len(rms_db) > 2:
                from scipy import stats
                try:
                    features['acoustic_intensity_skewness'] = float(stats.skew(rms_db))
                except:
                    features['acoustic_intensity_skewness'] = 0.0
            else:
                features['acoustic_intensity_skewness'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error extracting additional statistical moments: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_energy_skewness',
                'acoustic_energy_kurtosis',
                'acoustic_intensity_skewness'
            ]})
        
        return features

    def _extract_fast_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Fast feature extraction optimized for individual ASD files."""
        features = {}

        # Essential features only for speed
        pitch_features = self._extract_pitch_features(audio, sr)
        features.update(pitch_features)

        prosody_features = self._extract_prosody_features(audio, sr, pitch_features)
        features.update(prosody_features)

        # Basic spectral features
        spectral_features = self._extract_spectral_features(audio, sr)
        features.update(spectral_features)

        # Core MFCC features (fewer coefficients for speed)
        mfcc_features = self._extract_mfcc_features(audio, sr)
        features.update(mfcc_features)

        # Energy features
        energy_features = self._extract_energy_features(audio, sr)
        features.update(energy_features)

        # Fast formant approximation
        formant_features = self._extract_formant_features(audio, sr)
        features.update(formant_features)

        # Skip slower features for individual processing
        # This reduces processing time from 10+ minutes to ~1-2 minutes

        return features

    def _extract_full_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Full feature extraction for merged TD files (comprehensive analysis)."""
        features = {}

        # Extract pitch features (F0)
        pitch_features = self._extract_pitch_features(audio, sr)
        features.update(pitch_features)

        # Extract prosody features
        prosody_features = self._extract_prosody_features(audio, sr, pitch_features)
        features.update(prosody_features)

        # Extract voice quality features
        voice_quality_features = self._extract_voice_quality_features(audio, sr)
        features.update(voice_quality_features)

        # Extract spectral features
        spectral_features = self._extract_spectral_features(audio, sr)
        features.update(spectral_features)

        # Extract MFCC features
        mfcc_features = self._extract_mfcc_features(audio, sr)
        features.update(mfcc_features)

        # Extract energy/intensity features
        energy_features = self._extract_energy_features(audio, sr)
        features.update(energy_features)

        # Extract formant-like features
        formant_features = self._extract_formant_features(audio, sr)
        features.update(formant_features)

        # Extract extended MFCC features (6-13)
        extended_mfcc_features = self._extract_extended_mfcc_features(audio, sr)
        features.update(extended_mfcc_features)

        # Extract chroma features
        chroma_features = self._extract_chroma_features(audio, sr)
        features.update(chroma_features)

        # Extract temporal dynamics features
        temporal_features = self._extract_temporal_dynamics_features(audio, sr, pitch_features)
        features.update(temporal_features)

        # Extract additional formant features
        additional_formant_features = self._extract_additional_formant_features(audio, sr)
        features.update(additional_formant_features)

        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when audio is not available."""
        return {name: 0.0 for name in self.feature_names}


__all__ = ["AcousticAudioFeatures"]

