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
        
        return FeatureResult(
            features=features,
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
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            
            if len(audio) == 0:
                logger.warning(f"Empty audio file: {audio_path}")
                return self._get_default_features()
            
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
            
            # Extract spectral contrast features
            spectral_contrast_features = self._extract_spectral_contrast_features(audio, sr)
            features.update(spectral_contrast_features)
            
            # Extract tonnetz features
            tonnetz_features = self._extract_tonnetz_features(audio, sr)
            features.update(tonnetz_features)
            
            # Extract rhythm and timing features
            rhythm_features = self._extract_rhythm_timing_features(audio, sr)
            features.update(rhythm_features)
            
            # Extract advanced pitch statistics
            advanced_pitch_features = self._extract_advanced_pitch_statistics(audio, sr, pitch_features)
            features.update(advanced_pitch_features)
            
            # Extract advanced spectral features
            advanced_spectral_features = self._extract_advanced_spectral_features(audio, sr)
            features.update(advanced_spectral_features)
            
            # Extract MFCC delta features
            mfcc_delta_features = self._extract_mfcc_delta_features(audio, sr)
            features.update(mfcc_delta_features)
            
            # Extract additional formant features
            additional_formant_features = self._extract_additional_formant_features(audio, sr)
            features.update(additional_formant_features)
            
            # Extract cross-feature correlations
            correlation_features = self._extract_cross_feature_correlations(audio, sr, pitch_features, energy_features)
            features.update(correlation_features)
            
            # Extract harmonic features
            harmonic_features = self._extract_harmonic_features(audio, sr)
            features.update(harmonic_features)
            
            # Extract additional statistical moments
            statistical_moments = self._extract_additional_statistical_moments(audio, sr, energy_features)
            features.update(statistical_moments)
            
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
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
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
                features['acoustic_pitch_variability'] = safe_divide(
                    features['acoustic_pitch_std'],
                    features['acoustic_pitch_mean']
                )
                
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
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
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
                    
                    features['acoustic_pitch_rising_ratio'] = safe_divide(rising, total)
                    features['acoustic_pitch_falling_ratio'] = safe_divide(falling, total)
                    features['acoustic_pitch_flat_ratio'] = safe_divide(flat, total)
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
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                frame_length=2048,
                hop_length=512
            )
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) > 1:
                # Jitter: period-to-period variation in F0
                periods = 1.0 / f0_voiced  # Period in seconds
                period_diff = np.abs(np.diff(periods))
                features['acoustic_jitter'] = safe_divide(
                    np.mean(period_diff),
                    np.mean(periods)
                )
            else:
                features['acoustic_jitter'] = 0.0
            
            # Shimmer: amplitude variation (simplified)
            frame_length = 2048
            hop_length = 512
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            if len(rms) > 1:
                rms_diff = np.abs(np.diff(rms))
                features['acoustic_shimmer'] = safe_divide(
                    np.mean(rms_diff),
                    np.mean(rms)
                )
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
        """Extract formant-like features from spectral peaks."""
        features = {}
        
        try:
            # Get spectral magnitude
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Find spectral peaks (formant-like)
            # For each frame, find peaks in the magnitude spectrum
            formant_1 = []
            formant_2 = []
            formant_3 = []
            
            for frame in range(magnitude.shape[1]):
                frame_mag = magnitude[:, frame]
                # Find peaks
                peaks, _ = librosa.util.peak_pick(
                    frame_mag,
                    pre_max=3,
                    post_max=3,
                    pre_avg=3,
                    post_avg=5,
                    delta=0.1,
                    wait=10
                )
                
                if len(peaks) > 0:
                    # Convert bin indices to frequencies
                    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0] * 2 - 1)
                    peak_freqs = freqs[peaks]
                    
                    # Sort by magnitude and take top 3
                    peak_mags = frame_mag[peaks]
                    sorted_indices = np.argsort(peak_mags)[::-1]
                    sorted_freqs = peak_freqs[sorted_indices]
                    
                    # Filter to typical formant ranges
                    # F1: 300-1000 Hz, F2: 1000-3000 Hz, F3: 2500-4000 Hz
                    f1_candidates = sorted_freqs[(sorted_freqs >= 300) & (sorted_freqs <= 1000)]
                    f2_candidates = sorted_freqs[(sorted_freqs >= 1000) & (sorted_freqs <= 3000)]
                    f3_candidates = sorted_freqs[(sorted_freqs >= 2500) & (sorted_freqs <= 4000)]
                    
                    if len(f1_candidates) > 0:
                        formant_1.append(f1_candidates[0])
                    if len(f2_candidates) > 0:
                        formant_2.append(f2_candidates[0])
                    if len(f3_candidates) > 0:
                        formant_3.append(f3_candidates[0])
            
            # Calculate statistics
            if len(formant_1) > 0:
                features['acoustic_formant_1_mean'] = float(np.mean(formant_1))
                features['acoustic_formant_1_std'] = float(np.std(formant_1))
            else:
                features['acoustic_formant_1_mean'] = 0.0
                features['acoustic_formant_1_std'] = 0.0
            
            if len(formant_2) > 0:
                features['acoustic_formant_2_mean'] = float(np.mean(formant_2))
                features['acoustic_formant_2_std'] = float(np.std(formant_2))
            else:
                features['acoustic_formant_2_mean'] = 0.0
                features['acoustic_formant_2_std'] = 0.0
            
            if len(formant_3) > 0:
                features['acoustic_formant_3_mean'] = float(np.mean(formant_3))
                features['acoustic_formant_3_std'] = float(np.std(formant_3))
            else:
                features['acoustic_formant_3_mean'] = 0.0
                features['acoustic_formant_3_std'] = 0.0
                
        except Exception as e:
            logger.warning(f"Error extracting formant features: {e}")
            features.update({k: 0.0 for k in [
                'acoustic_formant_1_mean', 'acoustic_formant_1_std',
                'acoustic_formant_2_mean', 'acoustic_formant_2_std',
                'acoustic_formant_3_mean', 'acoustic_formant_3_std'
            ]})
        
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
            # Extract chroma features (12 pitch classes)
            chroma = librosa.feature.chroma(y=audio, sr=sr)
            
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
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
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
            features['acoustic_onset_rate'] = safe_divide(len(onsets), duration)
            
            # Silence ratio (proportion of silence)
            intervals = librosa.effects.split(audio, top_db=30)
            if len(intervals) > 0:
                speech_time = sum((e - s) for s, e in intervals) / sr
                silence_time = max(duration - speech_time, 0.0)
                features['acoustic_silence_ratio'] = safe_divide(silence_time, duration)
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
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when audio is not available."""
        return {name: 0.0 for name in self.feature_names}


__all__ = ["AcousticAudioFeatures"]

