"""
Audio Feature Extractor for Pragmatic & Conversational Analysis

This module extracts audio-specific features that are relevant to
pragmatic and conversational analysis:

- Pause patterns (duration, frequency, distribution)
- Response latency from audio timing
- Speaking rate and rhythm
- Turn-taking timing from audio segments

These features complement the text-based pragmatic features.

Author: Bimidu Gunathilake
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.logger import get_logger
from src.utils.helpers import safe_divide, calculate_ratio
from ..base_features import BaseFeatureExtractor, FeatureResult

logger = get_logger(__name__)

# Try to import audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available for advanced audio features")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


@dataclass
class PauseInfo:
    """Information about a detected pause in speech."""
    start_time: float
    end_time: float
    duration: float
    pause_type: str = "unfilled"  # "filled" (um, uh) or "unfilled" (silence)
    before_utterance_idx: int = -1
    after_utterance_idx: int = -1
    context_before: str = ""
    context_after: str = ""
    
    @property
    def is_long(self) -> bool:
        """Check if this is a long pause (>1 second)."""
        return self.duration > 1.0
    
    @property
    def is_very_long(self) -> bool:
        """Check if this is a very long pause (>2 seconds)."""
        return self.duration > 2.0


class PragmaticAudioFeatures(BaseFeatureExtractor):
    """
    Extract audio features relevant to pragmatic/conversational analysis.
    
    This extractor focuses on features from audio that relate to:
    - Pause patterns (important for ASD detection)
    - Response latency timing
    - Speaking rate variability
    - Turn-taking timing
    
    NOTE: This module extracts pragmatic-relevant features from audio.
    Pitch/prosody features should be handled by acoustic_prosodic module.
    
    Example:
        >>> extractor = PragmaticAudioFeatures()
        >>> features = extractor.extract(transcript, audio_path="audio.wav")
    """
    
    # Pause detection thresholds
    MIN_PAUSE_DURATION = 0.2  # Minimum pause to detect (seconds)
    LONG_PAUSE_THRESHOLD = 1.0  # Threshold for "long" pause (seconds)
    VERY_LONG_PAUSE_THRESHOLD = 2.0  # Threshold for "very long" pause
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # Pause features from audio
            'audio_pause_count',
            'audio_pause_total_duration',
            'audio_pause_mean_duration',
            'audio_pause_median_duration',
            'audio_pause_std_duration',
            'audio_pause_max_duration',
            'audio_pause_min_duration',
            'audio_long_pause_count',
            'audio_very_long_pause_count',
            'audio_pause_ratio',
            'audio_speaking_ratio',
            'audio_pause_rate_per_minute',
            
            # Filled vs unfilled pauses
            'audio_filled_pause_count',
            'audio_unfilled_pause_count',
            'audio_filled_pause_ratio',
            
            # Speaking rate features
            'audio_speaking_rate_wpm',
            'audio_articulation_rate',
            'audio_speech_rate_variability',
            
            # Segment/turn timing features
            'audio_segment_duration_mean',
            'audio_segment_duration_std',
            'audio_segment_duration_max',
            'audio_segment_duration_min',
            
            # Response latency features
            'audio_response_latency_mean',
            'audio_response_latency_std',
            'audio_response_latency_max',
            
            # Temporal features
            'audio_total_duration',
            'audio_speech_duration',
            'audio_silence_duration',
            'audio_speech_to_silence_ratio',
        ]
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize the audio feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.pauses: List[PauseInfo] = []  # Store detected pauses for annotation
        
        logger.info("PragmaticAudioFeatures initialized")
    
    def extract(
        self,
        transcript: TranscriptData,
        audio_path: Optional[str | Path] = None,
        transcription_result: Optional[Any] = None
    ) -> FeatureResult:
        """
        Extract audio features from transcript with optional audio file.
        
        Args:
            transcript: Parsed transcript data (with timing if available)
            audio_path: Optional path to audio file for direct analysis
            transcription_result: Optional TranscriptionResult from audio processing
            
        Returns:
            FeatureResult with audio-derived pragmatic features
        """
        features = {}
        self.pauses = []
        
        logger.debug(f"Extracting audio features for {transcript.participant_id}")
        
        # Check if we have timing information
        has_timing = self._has_timing_info(transcript.utterances)
        total_duration = self._estimate_duration(transcript, transcription_result)
        
        # Extract pause features
        if has_timing or transcription_result:
            pause_info = self._extract_pause_features(
                transcript, transcription_result, total_duration
            )
            features.update(pause_info['features'])
            self.pauses = pause_info['pauses']
        else:
            # Use default values when no timing available
            features.update(self._get_default_pause_features())
        
        # Extract speaking rate features
        rate_features = self._extract_rate_features(
            transcript, transcription_result, total_duration
        )
        features.update(rate_features)
        
        # Extract segment timing features
        segment_features = self._extract_segment_features(transcript)
        features.update(segment_features)
        
        # Extract response latency features
        latency_features = self._extract_response_latency(transcript)
        features.update(latency_features)
        
        # Extract temporal features
        temporal_features = self._extract_temporal_features(
            transcript, total_duration
        )
        features.update(temporal_features)
        
        # If audio file provided, extract additional features
        if audio_path and LIBROSA_AVAILABLE:
            audio_direct_features = self._extract_from_audio_file(
                audio_path, transcript
            )
            # Merge, preferring direct audio analysis
            features.update(audio_direct_features)
        
        logger.debug(f"Extracted {len(features)} audio features")
        
        return FeatureResult(
            features=features,
            feature_type='pragmatic_audio',
            metadata={
                'has_timing': has_timing,
                'total_duration': total_duration,
                'pause_count': len(self.pauses),
                'audio_file_used': audio_path is not None,
            }
        )
    
    def get_detected_pauses(self) -> List[PauseInfo]:
        """Get the list of detected pauses (for annotation)."""
        return self.pauses
    
    def _has_timing_info(self, utterances: List[Utterance]) -> bool:
        """Check if utterances have timing information."""
        timed_count = sum(1 for u in utterances if u.timing is not None)
        return timed_count > len(utterances) * 0.5
    
    def _estimate_duration(
        self,
        transcript: TranscriptData,
        transcription_result: Optional[Any]
    ) -> float:
        """Estimate total duration from available sources."""
        # Try transcription result first
        if transcription_result and hasattr(transcription_result, 'duration'):
            return transcription_result.duration
        
        # Try timing from utterances
        timed = [u for u in transcript.utterances if u.timing is not None]
        if timed:
            max_time = max(u.end_timing or u.timing for u in timed)
            return max_time
        
        # Estimate from word count (rough: 150 wpm)
        total_words = sum(u.word_count for u in transcript.utterances)
        return total_words / 2.5  # words per second
    
    def _extract_pause_features(
        self,
        transcript: TranscriptData,
        transcription_result: Optional[Any],
        total_duration: float
    ) -> Dict[str, Any]:
        """Extract pause-related features."""
        pauses = []
        
        # Get pauses from transcription result if available
        if transcription_result and hasattr(transcription_result, 'get_pauses'):
            raw_pauses = transcription_result.get_pauses(
                min_pause=self.MIN_PAUSE_DURATION
            )
            for p in raw_pauses:
                pause = PauseInfo(
                    start_time=p['start_time'],
                    end_time=p['end_time'],
                    duration=p['duration'],
                    before_utterance_idx=p.get('before_segment', -1),
                    after_utterance_idx=p.get('after_segment', -1),
                )
                pauses.append(pause)
        else:
            # Extract from transcript timing
            pauses = self._extract_pauses_from_transcript(transcript)
        
        # Detect filled pauses from text
        filled_count = self._count_filled_pauses(transcript)
        
        # Calculate features
        features = self._calculate_pause_features(
            pauses, total_duration, filled_count
        )
        
        return {'features': features, 'pauses': pauses}
    
    def _extract_pauses_from_transcript(
        self,
        transcript: TranscriptData
    ) -> List[PauseInfo]:
        """Extract pauses from transcript timing information."""
        pauses = []
        utterances = transcript.utterances
        
        for i in range(1, len(utterances)):
            prev = utterances[i - 1]
            curr = utterances[i]
            
            # Get end time of previous utterance
            prev_end = prev.end_timing or prev.timing
            curr_start = curr.timing
            
            if prev_end is not None and curr_start is not None:
                gap = curr_start - prev_end
                
                if gap >= self.MIN_PAUSE_DURATION:
                    pause = PauseInfo(
                        start_time=prev_end,
                        end_time=curr_start,
                        duration=gap,
                        before_utterance_idx=i - 1,
                        after_utterance_idx=i,
                        context_before=prev.text[-50:] if prev.text else "",
                        context_after=curr.text[:50] if curr.text else "",
                    )
                    pauses.append(pause)
        
        return pauses
    
    def _count_filled_pauses(self, transcript: TranscriptData) -> int:
        """Count filled pauses (um, uh, etc.) in transcript text."""
        filled_patterns = ['um', 'uh', 'hmm', 'er', 'ah', 'erm']
        count = 0
        
        for utterance in transcript.utterances:
            text_lower = utterance.text.lower()
            for pattern in filled_patterns:
                count += text_lower.count(pattern)
        
        return count
    
    def _calculate_pause_features(
        self,
        pauses: List[PauseInfo],
        total_duration: float,
        filled_count: int
    ) -> Dict[str, float]:
        """Calculate pause-related features from detected pauses."""
        features = {}
        
        if not pauses:
            features['audio_pause_count'] = 0
            features['audio_pause_total_duration'] = 0.0
            features['audio_pause_mean_duration'] = 0.0
            features['audio_pause_median_duration'] = 0.0
            features['audio_pause_std_duration'] = 0.0
            features['audio_pause_max_duration'] = 0.0
            features['audio_pause_min_duration'] = 0.0
            features['audio_long_pause_count'] = 0
            features['audio_very_long_pause_count'] = 0
            features['audio_pause_ratio'] = 0.0
            features['audio_speaking_ratio'] = 1.0
            features['audio_pause_rate_per_minute'] = 0.0
            features['audio_filled_pause_count'] = filled_count
            features['audio_unfilled_pause_count'] = 0
            features['audio_filled_pause_ratio'] = 0.0
            return features
        
        durations = [p.duration for p in pauses]
        total_pause_time = sum(durations)
        
        features['audio_pause_count'] = len(pauses)
        features['audio_pause_total_duration'] = total_pause_time
        features['audio_pause_mean_duration'] = float(np.mean(durations))
        features['audio_pause_median_duration'] = float(np.median(durations))
        features['audio_pause_std_duration'] = float(np.std(durations)) if len(durations) > 1 else 0.0
        features['audio_pause_max_duration'] = float(np.max(durations))
        features['audio_pause_min_duration'] = float(np.min(durations))
        
        # Long pause counts
        features['audio_long_pause_count'] = sum(1 for p in pauses if p.is_long)
        features['audio_very_long_pause_count'] = sum(1 for p in pauses if p.is_very_long)
        
        # Ratios
        features['audio_pause_ratio'] = total_pause_time / total_duration if total_duration > 0 else 0.0
        features['audio_speaking_ratio'] = 1.0 - features['audio_pause_ratio']
        
        # Pause rate
        duration_minutes = total_duration / 60.0
        features['audio_pause_rate_per_minute'] = len(pauses) / duration_minutes if duration_minutes > 0 else 0.0
        
        # Filled vs unfilled
        unfilled_count = len(pauses)
        features['audio_filled_pause_count'] = filled_count
        features['audio_unfilled_pause_count'] = unfilled_count
        features['audio_filled_pause_ratio'] = calculate_ratio(filled_count, filled_count + unfilled_count)
        
        return features
    
    def _extract_rate_features(
        self,
        transcript: TranscriptData,
        transcription_result: Optional[Any],
        total_duration: float
    ) -> Dict[str, float]:
        """Extract speaking rate features."""
        features = {}
        
        # Count total words
        total_words = sum(u.word_count for u in transcript.utterances)
        
        # Speaking rate (words per minute)
        duration_minutes = total_duration / 60.0
        features['audio_speaking_rate_wpm'] = total_words / duration_minutes if duration_minutes > 0 else 0.0
        
        # Articulation rate (words per minute of actual speech)
        # Estimate speech time by subtracting pauses
        pause_time = sum(p.duration for p in self.pauses)
        speech_time = max(total_duration - pause_time, 1.0)
        speech_minutes = speech_time / 60.0
        features['audio_articulation_rate'] = total_words / speech_minutes if speech_minutes > 0 else 0.0
        
        # Speech rate variability across utterances
        utterance_rates = []
        for u in transcript.utterances:
            if u.timing is not None and u.end_timing is not None:
                duration = u.end_timing - u.timing
                if duration > 0:
                    rate = u.word_count / (duration / 60.0)
                    utterance_rates.append(rate)
        
        features['audio_speech_rate_variability'] = float(np.std(utterance_rates)) if utterance_rates else 0.0
        
        return features
    
    def _extract_segment_features(
        self,
        transcript: TranscriptData
    ) -> Dict[str, float]:
        """Extract segment/utterance timing features."""
        features = {}
        
        # Calculate segment durations
        durations = []
        for u in transcript.utterances:
            if u.timing is not None and u.end_timing is not None:
                duration = u.end_timing - u.timing
                if duration > 0:
                    durations.append(duration)
        
        if durations:
            features['audio_segment_duration_mean'] = float(np.mean(durations))
            features['audio_segment_duration_std'] = float(np.std(durations)) if len(durations) > 1 else 0.0
            features['audio_segment_duration_max'] = float(np.max(durations))
            features['audio_segment_duration_min'] = float(np.min(durations))
        else:
            features['audio_segment_duration_mean'] = 0.0
            features['audio_segment_duration_std'] = 0.0
            features['audio_segment_duration_max'] = 0.0
            features['audio_segment_duration_min'] = 0.0
        
        return features
    
    def _extract_response_latency(
        self,
        transcript: TranscriptData
    ) -> Dict[str, float]:
        """Extract response latency features (child responding to adult)."""
        features = {}
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        child_latencies = []
        
        utterances = transcript.utterances
        for i in range(1, len(utterances)):
            prev = utterances[i - 1]
            curr = utterances[i]
            
            # Check if child responding to adult
            if prev.speaker in adult_codes and curr.speaker == 'CHI':
                prev_end = prev.end_timing or prev.timing
                curr_start = curr.timing
                
                if prev_end is not None and curr_start is not None:
                    latency = curr_start - prev_end
                    if latency >= 0:
                        child_latencies.append(latency)
        
        if child_latencies:
            features['audio_response_latency_mean'] = float(np.mean(child_latencies))
            features['audio_response_latency_std'] = float(np.std(child_latencies)) if len(child_latencies) > 1 else 0.0
            features['audio_response_latency_max'] = float(np.max(child_latencies))
        else:
            features['audio_response_latency_mean'] = 0.0
            features['audio_response_latency_std'] = 0.0
            features['audio_response_latency_max'] = 0.0
        
        return features
    
    def _extract_temporal_features(
        self,
        transcript: TranscriptData,
        total_duration: float
    ) -> Dict[str, float]:
        """Extract temporal features."""
        features = {}
        
        # Calculate speech vs silence time
        pause_time = sum(p.duration for p in self.pauses)
        speech_time = max(total_duration - pause_time, 0.0)
        
        features['audio_total_duration'] = total_duration
        features['audio_speech_duration'] = speech_time
        features['audio_silence_duration'] = pause_time
        features['audio_speech_to_silence_ratio'] = safe_divide(speech_time, pause_time)
        
        return features
    
    def _extract_from_audio_file(
        self,
        audio_path: str | Path,
        transcript: TranscriptData
    ) -> Dict[str, float]:
        """
        Extract additional features directly from audio file.
        
        Uses librosa for energy-based pause detection.
        """
        features = {}
        
        if not LIBROSA_AVAILABLE:
            return features
        
        try:
            audio_path = Path(audio_path)
            
            # Load audio
            audio, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
            
            # Compute energy for pause detection
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)  # 10ms hop
            
            energy = librosa.feature.rms(
                y=audio,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            # Normalize and threshold
            energy_norm = energy / (energy.max() + 1e-10)
            threshold = 0.1
            
            # Calculate speaking ratio from energy
            speaking_frames = np.sum(energy_norm > threshold)
            total_frames = len(energy_norm)
            
            if total_frames > 0:
                features['audio_speaking_ratio'] = speaking_frames / total_frames
                features['audio_pause_ratio'] = 1.0 - features['audio_speaking_ratio']
            
        except Exception as e:
            logger.warning(f"Error extracting from audio file: {e}")
        
        return features
    
    def _get_default_pause_features(self) -> Dict[str, float]:
        """Return default pause features when no timing available."""
        return {
            'audio_pause_count': 0,
            'audio_pause_total_duration': 0.0,
            'audio_pause_mean_duration': 0.0,
            'audio_pause_median_duration': 0.0,
            'audio_pause_std_duration': 0.0,
            'audio_pause_max_duration': 0.0,
            'audio_pause_min_duration': 0.0,
            'audio_long_pause_count': 0,
            'audio_very_long_pause_count': 0,
            'audio_pause_ratio': 0.0,
            'audio_speaking_ratio': 1.0,
            'audio_pause_rate_per_minute': 0.0,
            'audio_filled_pause_count': 0,
            'audio_unfilled_pause_count': 0,
            'audio_filled_pause_ratio': 0.0,
        }


__all__ = ["PragmaticAudioFeatures", "PauseInfo"]

