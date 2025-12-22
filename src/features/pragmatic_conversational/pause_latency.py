"""
Pause and Latency Analysis Feature Extractor (Section 3.3.3)

This module extracts features related to pauses and response latency,
which are often different in children with ASD. Based on methodology section 3.3.3.

Features implemented:
- Response latency (time from interviewer turn-end to child turn-start)
- Filled pauses ("um", "uh", "er", "ah")
- Unfilled pause duration
- Pause distribution statistics
- Speaking vs silence ratio

References:
- Analysis at two levels: between turns and within utterances
- Statistical modeling of pause length distributions

Author: Bimidu Gunathilake
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.helpers import safe_divide, calculate_ratio
from src.utils.logger import get_logger
from ..base_features import BaseFeatureExtractor, FeatureResult

logger = get_logger(__name__)

# Try to import scipy for distribution fitting
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Distribution fitting features will be limited.")


class PauseLatencyFeatures(BaseFeatureExtractor):
    """
    Extract pause and latency features from transcripts (Section 3.3.3).
    
    Features capture:
    - Response latency between turns
    - Filled pauses (hesitation markers)
    - Pause duration and frequency
    - Speaking vs silence time ratio
    - Distribution parameters of pause lengths
    
    Example:
        >>> extractor = PauseLatencyFeatures()
        >>> features = extractor.extract(transcript)
        >>> print(features.features['filled_pause_ratio'])
    """
    
    # Filled pause patterns (hesitation markers)
    FILLED_PAUSE_PATTERNS = [
        r'\bum\b', r'\buh\b', r'\ber\b', r'\bah\b', r'\behm\b',
        r'\bhmm\b', r'\bmm\b', r'\buhm\b', r'\bumm\b',
        r'&-um', r'&-uh', r'&-er', r'&-ah',  # CHAT format
    ]
    
    # CHAT pause markers
    PAUSE_MARKERS = {
        '(.)': 0.5,      # Short pause (~0.5 sec)
        '(..)': 1.0,     # Medium pause (~1 sec)
        '(...)': 1.5,    # Long pause (~1.5 sec)
        '(pause)': 2.0,  # Extended pause
    }
    
    # Thresholds
    # Thresholds (Derived from Unsupervised ML Clustering (GMM) on ASDBank dataset)
    # The GMM identified 3 latent clusters: 1. Rapid (mean ~0.2s), 2. Processing (mean ~1.25s), 3. Disengaged (mean ~4.3s).
    # Boundaries were calculated at the intersection of these clusters.
    NORMAL_RESPONSE_TIME = 0.45  # Upper bound of "Rapid" cluster
    LONG_PAUSE_THRESHOLD = 2.00  # Boundary between "Processing" and "Disengaged"
    VERY_LONG_PAUSE_THRESHOLD = 4.32 # Mean of "Disengaged" cluster (Center of the long pause distribution)
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # Response latency (between turns)
            'response_latency_mean',
            'response_latency_median',
            'response_latency_std',
            'response_latency_max',
            'response_latency_min',
            'child_response_latency_mean',
            'child_response_latency_median',
            'child_response_latency_std',
            'adult_response_latency_mean',
            
            # Delayed response indicators
            'delayed_response_count',
            'delayed_response_ratio',
            'very_delayed_response_count',
            'immediate_response_ratio',
            
            # Filled pauses (hesitation markers)
            'filled_pause_count',
            'filled_pause_ratio',
            'filled_pause_per_utterance',
            'child_filled_pause_count',
            'child_filled_pause_ratio',
            'um_count',
            'uh_count',
            'other_filler_count',
            
            # Unfilled pauses (within utterances from CHAT markers)
            'unfilled_pause_count',
            'unfilled_pause_total_duration',
            'unfilled_pause_mean_duration',
            'long_pause_count',
            'very_long_pause_count',
            'pause_per_utterance',
            
            # Child-specific pause features
            'child_pause_count',
            'child_pause_ratio',
            'child_long_pause_ratio',
            
            # Speaking vs silence (estimated)
            'estimated_speaking_time',
            'estimated_silence_time',
            'speaking_silence_ratio',
            'fluency_score',
            
            # Pause distribution statistics
            'pause_distribution_skewness',
            'pause_distribution_kurtosis',
            'pause_cv',  # Coefficient of variation
            
            # Latency distribution parameters
            'latency_exponential_lambda',  # Rate parameter if exponential fit
            'latency_percentile_75',
            'latency_percentile_90',
            'latency_iqr',  # Interquartile range
            
            # Turn-internal hesitation
            'hesitation_density',
            'false_start_count',
            'word_repetition_count',
        ]
    
    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract pause and latency features from transcript.
        
        Args:
            transcript: Parsed transcript data
            
        Returns:
            FeatureResult with pause and latency features
        """
        features = {}
        
        all_utterances = transcript.valid_utterances
        child_utterances = self.get_child_utterances(transcript)
        adult_utterances = self.get_adult_utterances(transcript)
        
        logger.debug(f"Extracting pause/latency features from {len(all_utterances)} utterances")
        
        # Response latency features (between turns)
        latency_features = self._calculate_response_latency(
            all_utterances, child_utterances, adult_utterances
        )
        features.update(latency_features)
        
        # Delayed response indicators
        delay_features = self._calculate_delay_indicators(all_utterances)
        features.update(delay_features)
        
        # Filled pause features (hesitation markers)
        filled_features = self._calculate_filled_pauses(
            all_utterances, child_utterances
        )
        features.update(filled_features)
        
        # Unfilled pause features (from CHAT markers)
        unfilled_features = self._calculate_unfilled_pauses(
            all_utterances, child_utterances
        )
        features.update(unfilled_features)
        
        # Speaking vs silence ratio
        speaking_features = self._calculate_speaking_silence_ratio(
            all_utterances
        )
        features.update(speaking_features)
        
        # Pause distribution statistics
        distribution_features = self._calculate_pause_distribution(all_utterances)
        features.update(distribution_features)
        
        # Turn-internal hesitation features
        hesitation_features = self._calculate_hesitation_features(
            all_utterances, child_utterances
        )
        features.update(hesitation_features)
        
        logger.debug(f"Extracted {len(features)} pause/latency features")
        
        return FeatureResult(
            features=features,
            feature_type='pause_latency',
            metadata={
                'total_utterances': len(all_utterances),
                'child_utterances': len(child_utterances),
                'has_timing': any(u.timing is not None for u in all_utterances),
                'scipy_available': SCIPY_AVAILABLE
            }
        )
    
    def _calculate_response_latency(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance],
        adult_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate response latency features.
        
        Response latency = time from one speaker's turn end to next speaker's start.
        """
        features = {
            'response_latency_mean': 0.0,
            'response_latency_median': 0.0,
            'response_latency_std': 0.0,
            'response_latency_max': 0.0,
            'response_latency_min': 0.0,
            'child_response_latency_mean': 0.0,
            'child_response_latency_median': 0.0,
            'child_response_latency_std': 0.0,
            'adult_response_latency_mean': 0.0,
        }
        
        # Extract timing gaps
        all_gaps = self.extract_timing_gaps(all_utterances)
        
        if all_gaps:
            features['response_latency_mean'] = float(np.mean(all_gaps))
            features['response_latency_median'] = float(np.median(all_gaps))
            features['response_latency_std'] = float(np.std(all_gaps))
            features['response_latency_max'] = float(np.max(all_gaps))
            features['response_latency_min'] = float(np.min(all_gaps))
        
        # Child-specific latency
        child_latencies = self._extract_speaker_latencies(all_utterances, 'CHI')
        if child_latencies:
            features['child_response_latency_mean'] = float(np.mean(child_latencies))
            features['child_response_latency_median'] = float(np.median(child_latencies))
            if len(child_latencies) > 1:
                features['child_response_latency_std'] = float(np.std(child_latencies))
        
        # Adult latency
        adult_latencies = self._extract_speaker_latencies(all_utterances, 'ADULT')
        if adult_latencies:
            features['adult_response_latency_mean'] = float(np.mean(adult_latencies))
        
        return features
    
    def _extract_speaker_latencies(
        self,
        utterances: List[Utterance],
        speaker_type: str
    ) -> List[float]:
        """Extract response latencies for a specific speaker type."""
        latencies = []
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        for i in range(1, len(utterances)):
            prev = utterances[i - 1]
            curr = utterances[i]
            
            # Determine if this is a response from target speaker
            if speaker_type == 'CHI':
                is_response = curr.speaker == 'CHI' and prev.speaker in adult_codes
            else:
                is_response = curr.speaker in adult_codes and prev.speaker == 'CHI'
            
            if is_response and prev.timing is not None and curr.timing is not None:
                gap = curr.timing - prev.timing
                if gap >= 0:
                    latencies.append(gap)
        
        return latencies
    
    def _calculate_delay_indicators(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate delayed response indicators."""
        features = {
            'delayed_response_count': 0,
            'delayed_response_ratio': 0.0,
            'very_delayed_response_count': 0,
            'immediate_response_ratio': 0.0,
        }
        
        gaps = self.extract_timing_gaps(utterances)
        
        if not gaps:
            return features
        
        delayed = sum(1 for g in gaps if g > self.LONG_PAUSE_THRESHOLD)
        very_delayed = sum(1 for g in gaps if g > self.VERY_LONG_PAUSE_THRESHOLD)
        immediate = sum(1 for g in gaps if g < self.NORMAL_RESPONSE_TIME)
        
        features['delayed_response_count'] = delayed
        features['delayed_response_ratio'] = delayed / len(gaps)
        features['very_delayed_response_count'] = very_delayed
        features['immediate_response_ratio'] = immediate / len(gaps)
        
        return features
    
    def _calculate_filled_pauses(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate filled pause (hesitation marker) features.
        
        Filled pauses: "um", "uh", "er", "ah", etc.
        """
        features = {
            'filled_pause_count': 0,
            'filled_pause_ratio': 0.0,
            'filled_pause_per_utterance': 0.0,
            'child_filled_pause_count': 0,
            'child_filled_pause_ratio': 0.0,
            'um_count': 0,
            'uh_count': 0,
            'other_filler_count': 0,
        }
        
        if not all_utterances:
            return features
        
        total_filled = 0
        child_filled = 0
        um_count = 0
        uh_count = 0
        other_count = 0
        total_words = 0
        child_words = 0
        
        for u in all_utterances:
            text = u.text.lower()
            word_count = u.word_count
            total_words += word_count
            
            # Count specific fillers
            um_matches = len(re.findall(r'\bum+\b', text))
            uh_matches = len(re.findall(r'\buh+\b', text))
            
            um_count += um_matches
            uh_count += uh_matches
            
            # Count all filled pauses
            utterance_filled = 0
            for pattern in self.FILLED_PAUSE_PATTERNS:
                matches = len(re.findall(pattern, text))
                utterance_filled += matches
            
            # Subtract double counting from specific patterns
            other = max(0, utterance_filled - um_matches - uh_matches)
            other_count += other
            total_filled += utterance_filled
            
            if u.speaker == 'CHI':
                child_filled += utterance_filled
                child_words += word_count
        
        features['filled_pause_count'] = total_filled
        features['filled_pause_ratio'] = total_filled / total_words if total_words > 0 else 0.0
        features['filled_pause_per_utterance'] = total_filled / len(all_utterances)
        features['child_filled_pause_count'] = child_filled
        features['child_filled_pause_ratio'] = child_filled / child_words if child_words > 0 else 0.0
        features['um_count'] = um_count
        features['uh_count'] = uh_count
        features['other_filler_count'] = other_count
        
        return features
    
    def _calculate_unfilled_pauses(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate unfilled pause features from CHAT pause markers.
        
        CHAT uses (.), (..), (...) for pauses of increasing duration.
        """
        features = {
            'unfilled_pause_count': 0,
            'unfilled_pause_total_duration': 0.0,
            'unfilled_pause_mean_duration': 0.0,
            'long_pause_count': 0,
            'very_long_pause_count': 0,
            'pause_per_utterance': 0.0,
            'child_pause_count': 0,
            'child_pause_ratio': 0.0,
            'child_long_pause_ratio': 0.0,
        }
        
        if not all_utterances:
            return features
        
        total_pauses = 0
        total_duration = 0.0
        long_pauses = 0
        very_long_pauses = 0
        child_pauses = 0
        child_long_pauses = 0
        pause_durations = []
        
        for u in all_utterances:
            text = u.text
            utterance_pauses = 0
            
            for marker, duration in self.PAUSE_MARKERS.items():
                count = text.count(marker)
                if count > 0:
                    utterance_pauses += count
                    total_duration += duration * count
                    pause_durations.extend([duration] * count)
                    
                    if duration >= self.LONG_PAUSE_THRESHOLD:
                        long_pauses += count
                    if duration >= self.VERY_LONG_PAUSE_THRESHOLD:
                        very_long_pauses += count
            
            total_pauses += utterance_pauses
            
            if u.speaker == 'CHI':
                child_pauses += utterance_pauses
                for marker, duration in self.PAUSE_MARKERS.items():
                    if duration >= self.LONG_PAUSE_THRESHOLD:
                        child_long_pauses += text.count(marker)
        
        features['unfilled_pause_count'] = total_pauses
        features['unfilled_pause_total_duration'] = total_duration
        features['unfilled_pause_mean_duration'] = (
            total_duration / total_pauses if total_pauses > 0 else 0.0
        )
        features['long_pause_count'] = long_pauses
        features['very_long_pause_count'] = very_long_pauses
        features['pause_per_utterance'] = total_pauses / len(all_utterances)
        features['child_pause_count'] = child_pauses
        features['child_pause_ratio'] = child_pauses / total_pauses if total_pauses > 0 else 0.0
        features['child_long_pause_ratio'] = (
            child_long_pauses / child_pauses if child_pauses > 0 else 0.0
        )
        
        return features
    
    def _calculate_speaking_silence_ratio(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Estimate speaking vs silence ratio.
        
        Uses word count and average speaking rate to estimate speaking time.
        """
        features = {
            'estimated_speaking_time': 0.0,
            'estimated_silence_time': 0.0,
            'speaking_silence_ratio': 0.0,
            'fluency_score': 0.0,
        }
        
        if not utterances:
            return features
        
        # Constants for estimation
        WORDS_PER_SECOND = 2.5  # Approximate child speaking rate
        
        # Estimate total speaking time from word counts
        total_words = sum(u.word_count for u in utterances)
        estimated_speaking = total_words / WORDS_PER_SECOND
        
        # Estimate silence from gaps between utterances
        gaps = self.extract_timing_gaps(utterances)
        estimated_silence = sum(gaps) if gaps else 0.0
        
        # Add pause markers duration
        for u in utterances:
            for marker, duration in self.PAUSE_MARKERS.items():
                estimated_silence += u.text.count(marker) * duration
        
        features['estimated_speaking_time'] = estimated_speaking
        features['estimated_silence_time'] = estimated_silence
        features['speaking_silence_ratio'] = (
            estimated_speaking / estimated_silence if estimated_silence > 0 else 0.0
        )
        
        # Fluency score (higher = more fluent, less pausing)
        total_time = estimated_speaking + estimated_silence
        features['fluency_score'] = estimated_speaking / total_time if total_time > 0 else 0.0
        
        return features
    
    def _calculate_pause_distribution(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate pause distribution statistics.
        
        Fits statistical models to pause length distribution.
        """
        features = {
            'pause_distribution_skewness': 0.0,
            'pause_distribution_kurtosis': 0.0,
            'pause_cv': 0.0,
            'latency_exponential_lambda': 0.0,
            'latency_percentile_75': 0.0,
            'latency_percentile_90': 0.0,
            'latency_iqr': 0.0,
        }
        
        # Collect all pause durations
        gaps = self.extract_timing_gaps(utterances)
        
        # Add within-utterance pauses
        pause_durations = list(gaps) if gaps else []
        for u in utterances:
            for marker, duration in self.PAUSE_MARKERS.items():
                count = u.text.count(marker)
                pause_durations.extend([duration] * count)
        
        if len(pause_durations) < 3:
            return features
        
        pause_array = np.array(pause_durations)
        
        # Basic statistics
        mean = np.mean(pause_array)
        std = np.std(pause_array)
        
        if std > 0:
            features['pause_cv'] = std / mean
        
        # Distribution shape
        if SCIPY_AVAILABLE and len(pause_array) >= 4:
            features['pause_distribution_skewness'] = float(stats.skew(pause_array))
            features['pause_distribution_kurtosis'] = float(stats.kurtosis(pause_array))
            
            # Fit exponential distribution (common for pause durations)
            try:
                # Exponential rate parameter (lambda)
                lambda_param = 1.0 / mean if mean > 0 else 0.0
                features['latency_exponential_lambda'] = lambda_param
            except Exception:
                pass
        
        # Percentiles
        features['latency_percentile_75'] = float(np.percentile(pause_array, 75))
        features['latency_percentile_90'] = float(np.percentile(pause_array, 90))
        features['latency_iqr'] = float(
            np.percentile(pause_array, 75) - np.percentile(pause_array, 25)
        )
        
        return features
    
    def _calculate_hesitation_features(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate turn-internal hesitation features.
        
        Includes false starts, word repetitions, and hesitation density.
        """
        features = {
            'hesitation_density': 0.0,
            'false_start_count': 0,
            'word_repetition_count': 0,
        }
        
        if not all_utterances:
            return features
        
        total_hesitations = 0
        false_starts = 0
        word_repetitions = 0
        total_words = 0
        
        # CHAT markers for hesitation
        false_start_patterns = [r'\[/\]', r'\[//\]', r'\+/\.']  # Retrace markers
        repetition_pattern = r'\b(\w+)\s+\1\b'  # Immediate word repetition
        
        for u in all_utterances:
            text = u.text
            word_count = u.word_count
            total_words += word_count
            
            # Count false starts
            for pattern in false_start_patterns:
                false_starts += len(re.findall(pattern, text))
            
            # Count word repetitions
            word_reps = len(re.findall(repetition_pattern, text.lower()))
            word_repetitions += word_reps
            
            # Total hesitations (false starts + repetitions + filled pauses)
            utterance_hesitations = false_starts + word_reps
            for pattern in self.FILLED_PAUSE_PATTERNS:
                utterance_hesitations += len(re.findall(pattern, text.lower()))
            
            total_hesitations += utterance_hesitations
        
        features['false_start_count'] = false_starts
        features['word_repetition_count'] = word_repetitions
        features['hesitation_density'] = total_hesitations / total_words if total_words > 0 else 0.0
        
        return features


__all__ = ["PauseLatencyFeatures"]






