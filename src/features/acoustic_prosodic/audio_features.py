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
        ]
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the acoustic audio feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        super().__init__()
        self.sample_rate = sample_rate
        
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available - acoustic features will be limited")
        
        logger.info("AcousticAudioFeatures initialized")
    
    def extract(
        self,
        transcript: TranscriptData,
        audio_path: Optional[str | Path] = None,
        transcription_result: Optional[Any] = None,
        **kwargs
    ) -> FeatureResult:
        """
        Extract acoustic/prosodic features from audio.
        
        Args:
            transcript: Parsed transcript data
            audio_path: Path to audio file for direct analysis
            transcription_result: Optional TranscriptionResult from audio processing
            **kwargs: Additional arguments
            
        Returns:
            FeatureResult with acoustic/prosodic features
        """
        features = {}
        
        logger.debug(f"Extracting acoustic features for {transcript.participant_id}")
        
        # If audio file provided, extract real features
        if audio_path and LIBROSA_AVAILABLE:
            try:
                audio_features = self._extract_from_audio_file(audio_path)
                features.update(audio_features)
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
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default features when audio is not available."""
        return {name: 0.0 for name in self.feature_names}


__all__ = ["AcousticAudioFeatures"]

