"""
Turn-Taking Feature Extractor (Section 3.3.1)

This module extracts features related to turn-taking patterns in conversations,
which are often impaired in children with ASD. Based on methodology section 3.3.1.

Features implemented:
- Turn Lengths (duration and word count)
- Inter-Turn Gaps (silence between turns)
- Overlap Duration (simultaneous speech)
- Interruption Count
- Turn Variability (standard deviation)
- Turn-level temporal features

References:
- Wehrle (2023): ASD speakers often have longer and more variable inter-turn gaps

Author: Bimidu Gunathilake
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.helpers import safe_divide, calculate_ratio
from src.utils.logger import get_logger
from ..base_features import BaseFeatureExtractor, FeatureResult

logger = get_logger(__name__)


class TurnTakingFeatures(BaseFeatureExtractor):
    """
    Extract turn-taking pattern features from transcripts (Section 3.3.1).
    
    Features capture:
    - Turn frequency and distribution
    - Turn lengths (words and duration)
    - Response latency / inter-turn gaps
    - Turn overlap and interruptions
    - Turn length variability
    - Speaker switching patterns
    
    Example:
        >>> extractor = TurnTakingFeatures()
        >>> features = extractor.extract(transcript)
        >>> print(features.features['inter_turn_gap_mean'])
    """
    
    # Thresholds for detecting overlaps and interruptions
    OVERLAP_THRESHOLD_MS = 100  # Minimum overlap to count as overlap
    INTERRUPTION_THRESHOLD_MS = 500  # Max gap before next turn to count as interruption
    LONG_PAUSE_THRESHOLD_SEC = 1.0  # Threshold for "long" pauses
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # Basic turn counts
            'total_turns',
            'child_turns',
            'adult_turns',
            'turns_per_minute',
            'child_turn_ratio',
            
            # Turn length features (word-based)
            'avg_turn_length_words',
            'avg_child_turn_length',
            'avg_adult_turn_length',
            'max_child_turn_length',
            'min_child_turn_length',
            
            # Turn length variability (key ASD marker)
            'child_turn_length_std',
            'child_turn_length_cv',  # Coefficient of variation
            'adult_turn_length_std',
            
            # Turn length (duration-based when timing available)
            'avg_turn_duration_sec',
            'child_turn_duration_mean',
            'child_turn_duration_std',
            
            # Inter-turn gaps / Response latency (Section 3.3.1)
            'inter_turn_gap_mean',
            'inter_turn_gap_median',
            'inter_turn_gap_std',
            'inter_turn_gap_max',
            'child_response_latency_mean',
            'child_response_latency_std',
            'adult_response_latency_mean',
            'long_pause_count',
            'long_pause_ratio',
            
            # Overlap features (Section 3.3.1)
            'overlap_count',
            'overlap_duration_total',
            'overlap_ratio',
            'child_overlaps_adult_count',
            'adult_overlaps_child_count',
            
            # Interruption features (Section 3.3.1)
            'interruption_count',
            'child_interruption_count',
            'adult_interruption_count',
            'interruption_ratio',
            
            # Turn initiation features
            'child_initiated_turns',
            'adult_initiated_turns',
            'child_initiation_ratio',
            
            # Turn switching patterns
            'turn_switches',
            'avg_turns_before_switch',
            'turn_switch_rate',
            
            # Consecutive turn features
            'max_consecutive_child_turns',
            'max_consecutive_adult_turns',
            'child_monologue_ratio',
        ]
    
    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract turn-taking features from transcript.
        
        Args:
            transcript: Parsed transcript data
            
        Returns:
            FeatureResult with turn-taking features
        """
        features = {}
        
        # Get valid utterances
        all_utterances = transcript.valid_utterances
        child_utterances = self.get_child_utterances(transcript)
        adult_utterances = self.get_adult_utterances(transcript)
        
        logger.debug(f"Extracting turn-taking features from {len(all_utterances)} utterances")
        
        # Check if we have timing information
        has_timing = self._has_timing_info(all_utterances)
        
        # Basic turn counts
        basic_features = self._calculate_basic_counts(
            all_utterances, child_utterances, adult_utterances
        )
        features.update(basic_features)
        
        # Turn length features (word-based)
        length_features = self._calculate_turn_lengths(
            all_utterances, child_utterances, adult_utterances
        )
        features.update(length_features)
        
        # Turn length variability (key ASD marker per Wehrle 2023)
        variability_features = self._calculate_turn_variability(
            child_utterances, adult_utterances
        )
        features.update(variability_features)
        
        # Duration-based features (when timing available)
        duration_features = self._calculate_duration_features(
            all_utterances, child_utterances, has_timing
        )
        features.update(duration_features)
        
        # Inter-turn gaps / Response latency
        gap_features = self._calculate_inter_turn_gaps(
            all_utterances, child_utterances, adult_utterances, has_timing
        )
        features.update(gap_features)
        
        # Overlap detection
        overlap_features = self._calculate_overlaps(all_utterances, has_timing)
        features.update(overlap_features)
        
        # Interruption detection
        interruption_features = self._calculate_interruptions(
            all_utterances, has_timing
        )
        features.update(interruption_features)
        
        # Turn initiation features
        initiation_features = self._calculate_turn_initiation(all_utterances)
        features.update(initiation_features)
        
        # Turn switching patterns
        switch_features = self._calculate_turn_switches(all_utterances)
        features.update(switch_features)
        
        # Consecutive turn features
        consecutive_features = self._calculate_consecutive_turns(all_utterances)
        features.update(consecutive_features)
        
        logger.debug(f"Extracted {len(features)} turn-taking features")
        
        return FeatureResult(
            features=features,
            feature_type='turn_taking',
            metadata={
                'total_utterances': len(all_utterances),
                'has_timing': has_timing,
                'child_utterances': len(child_utterances),
                'adult_utterances': len(adult_utterances)
            }
        )
    
    def _has_timing_info(self, utterances: List[Utterance]) -> bool:
        """Check if utterances have timing information."""
        timed_count = sum(1 for u in utterances if u.timing is not None)
        return timed_count > len(utterances) * 0.5  # At least 50% have timing
    
    def _calculate_basic_counts(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance],
        adult_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate basic turn counts."""
        features = {
            'total_turns': len(all_utterances),
            'child_turns': len(child_utterances),
            'adult_turns': len(adult_utterances),
            'turns_per_minute': 0.0,
            'child_turn_ratio': 0.0,
        }
        
        # Turn ratio
        features['child_turn_ratio'] = calculate_ratio(
            len(child_utterances), len(all_utterances)
        )
        
        # Turns per minute
        features['turns_per_minute'] = self._calculate_turns_per_minute(all_utterances)
        
        return features
    
    def _calculate_turns_per_minute(self, utterances: List[Utterance]) -> float:
        """Calculate number of turns per minute."""
        if not utterances:
            return 0.0
        
        timed_utterances = [u for u in utterances if u.timing is not None]
        
        if len(timed_utterances) < 2:
            # Estimate based on typical 15-minute session
            return len(utterances) / 15.0
        
        start_time = min(u.timing for u in timed_utterances)
        end_time = max(u.timing for u in timed_utterances)
        duration_minutes = (end_time - start_time) / 60.0
        
        if duration_minutes <= 0:
            return len(utterances) / 15.0
        
        return len(utterances) / duration_minutes
    
    def _calculate_turn_lengths(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance],
        adult_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate turn length features in words."""
        features = {
            'avg_turn_length_words': 0.0,
            'avg_child_turn_length': 0.0,
            'avg_adult_turn_length': 0.0,
            'max_child_turn_length': 0,
            'min_child_turn_length': 0,
        }
        
        # All utterances
        all_lengths = self.get_utterance_lengths(all_utterances, in_words=True)
        if all_lengths:
            features['avg_turn_length_words'] = float(np.mean(all_lengths))
        
        # Child utterances
        child_lengths = self.get_utterance_lengths(child_utterances, in_words=True)
        if child_lengths:
            features['avg_child_turn_length'] = float(np.mean(child_lengths))
            features['max_child_turn_length'] = max(child_lengths)
            features['min_child_turn_length'] = min(child_lengths)
        
        # Adult utterances
        adult_lengths = self.get_utterance_lengths(adult_utterances, in_words=True)
        if adult_lengths:
            features['avg_adult_turn_length'] = float(np.mean(adult_lengths))
        
        return features
    
    def _calculate_turn_variability(
        self,
        child_utterances: List[Utterance],
        adult_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate turn length variability features.
        
        Key ASD marker: Higher variability in turn lengths suggests less
        consistent conversational patterns (Wehrle 2023).
        """
        features = {
            'child_turn_length_std': 0.0,
            'child_turn_length_cv': 0.0,
            'adult_turn_length_std': 0.0,
        }
        
        # Child variability
        child_lengths = self.get_utterance_lengths(child_utterances, in_words=True)
        if len(child_lengths) > 1:
            std = float(np.std(child_lengths))
            mean = float(np.mean(child_lengths))
            features['child_turn_length_std'] = std
            # Coefficient of variation (normalized variability)
            features['child_turn_length_cv'] = std / mean if mean > 0 else 0.0
        
        # Adult variability
        adult_lengths = self.get_utterance_lengths(adult_utterances, in_words=True)
        if len(adult_lengths) > 1:
            features['adult_turn_length_std'] = float(np.std(adult_lengths))
        
        return features
    
    def _calculate_duration_features(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance],
        has_timing: bool
    ) -> Dict[str, float]:
        """Calculate duration-based turn length features."""
        features = {
            'avg_turn_duration_sec': 0.0,
            'child_turn_duration_mean': 0.0,
            'child_turn_duration_std': 0.0,
        }
        
        if not has_timing:
            return features
        
        # Extract turn durations from timing information
        # Note: CHAT timing often provides start time; duration may need estimation
        # For now, estimate from word count (avg ~150 words/minute for children)
        WORDS_PER_SECOND = 2.5  # Approximate speaking rate
        
        all_durations = [u.word_count / WORDS_PER_SECOND for u in all_utterances if u.word_count > 0]
        if all_durations:
            features['avg_turn_duration_sec'] = float(np.mean(all_durations))
        
        child_durations = [u.word_count / WORDS_PER_SECOND for u in child_utterances if u.word_count > 0]
        if child_durations:
            features['child_turn_duration_mean'] = float(np.mean(child_durations))
            if len(child_durations) > 1:
                features['child_turn_duration_std'] = float(np.std(child_durations))
        
        return features
    
    def _calculate_inter_turn_gaps(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance],
        adult_utterances: List[Utterance],
        has_timing: bool
    ) -> Dict[str, float]:
        """
        Calculate inter-turn gap features (Section 3.3.1).
        
        Key finding from Wehrle (2023): ASD speakers often have longer and
        more variable inter-turn gaps.
        """
        features = {
            'inter_turn_gap_mean': 0.0,
            'inter_turn_gap_median': 0.0,
            'inter_turn_gap_std': 0.0,
            'inter_turn_gap_max': 0.0,
            'child_response_latency_mean': 0.0,
            'child_response_latency_std': 0.0,
            'adult_response_latency_mean': 0.0,
            'long_pause_count': 0,
            'long_pause_ratio': 0.0,
        }
        
        # Extract all gaps
        all_gaps = self.extract_timing_gaps(all_utterances)
        
        if all_gaps:
            features['inter_turn_gap_mean'] = float(np.mean(all_gaps))
            features['inter_turn_gap_median'] = float(np.median(all_gaps))
            features['inter_turn_gap_std'] = float(np.std(all_gaps))
            features['inter_turn_gap_max'] = float(np.max(all_gaps))
            
            # Long pauses (>1 second)
            long_pauses = [g for g in all_gaps if g > self.LONG_PAUSE_THRESHOLD_SEC]
            features['long_pause_count'] = len(long_pauses)
            features['long_pause_ratio'] = len(long_pauses) / len(all_gaps)
        
        # Child response latency (time from adult turn end to child turn start)
        child_latencies = self._extract_speaker_response_latency(all_utterances, 'CHI')
        if child_latencies:
            features['child_response_latency_mean'] = float(np.mean(child_latencies))
            if len(child_latencies) > 1:
                features['child_response_latency_std'] = float(np.std(child_latencies))
        
        # Adult response latency
        adult_latencies = self._extract_speaker_response_latency(all_utterances, 'ADULT')
        if adult_latencies:
            features['adult_response_latency_mean'] = float(np.mean(adult_latencies))
        
        return features
    
    def _extract_speaker_response_latency(
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
            
            # Check if this is the target speaker responding
            if speaker_type == 'CHI':
                is_response = (prev.speaker in adult_codes and curr.speaker == 'CHI')
            else:  # ADULT
                is_response = (prev.speaker == 'CHI' and curr.speaker in adult_codes)
            
            if is_response and prev.timing is not None and curr.timing is not None:
                gap = curr.timing - prev.timing
                if gap >= 0:  # Valid positive gap
                    latencies.append(gap)
        
        return latencies
    
    def _calculate_overlaps(
        self,
        utterances: List[Utterance],
        has_timing: bool
    ) -> Dict[str, float]:
        """
        Calculate overlap features (Section 3.3.1).
        
        Overlap = moments where speakers interrupt each other.
        """
        features = {
            'overlap_count': 0,
            'overlap_duration_total': 0.0,
            'overlap_ratio': 0.0,
            'child_overlaps_adult_count': 0,
            'adult_overlaps_child_count': 0,
        }
        
        if not has_timing or len(utterances) < 2:
            # Estimate overlaps from text markers if no timing
            return self._estimate_overlaps_from_text(utterances)
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        overlap_count = 0
        child_overlaps = 0
        adult_overlaps = 0
        
        for i in range(1, len(utterances)):
            prev = utterances[i - 1]
            curr = utterances[i]
            
            if prev.timing is not None and curr.timing is not None:
                gap = curr.timing - prev.timing
                
                # Negative gap or very small gap indicates overlap
                if gap < (self.OVERLAP_THRESHOLD_MS / 1000.0):
                    overlap_count += 1
                    
                    # Track who overlapped whom
                    if curr.speaker == 'CHI' and prev.speaker in adult_codes:
                        child_overlaps += 1
                    elif curr.speaker in adult_codes and prev.speaker == 'CHI':
                        adult_overlaps += 1
        
        features['overlap_count'] = overlap_count
        features['child_overlaps_adult_count'] = child_overlaps
        features['adult_overlaps_child_count'] = adult_overlaps
        features['overlap_ratio'] = calculate_ratio(overlap_count, len(utterances) - 1)
        
        return features
    
    def _estimate_overlaps_from_text(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Estimate overlaps from CHAT text markers when timing unavailable."""
        features = {
            'overlap_count': 0,
            'overlap_duration_total': 0.0,
            'overlap_ratio': 0.0,
            'child_overlaps_adult_count': 0,
            'adult_overlaps_child_count': 0,
        }
        
        # CHAT uses <> for overlapping speech and [>] [<] for overlap markers
        overlap_markers = ['<', '>', '[>]', '[<]', '[/]', '[//]']
        
        overlap_count = 0
        for u in utterances:
            if any(marker in u.text for marker in overlap_markers):
                overlap_count += 1
        
        features['overlap_count'] = overlap_count
        features['overlap_ratio'] = calculate_ratio(overlap_count, len(utterances))
        
        return features
    
    def _calculate_interruptions(
        self,
        utterances: List[Utterance],
        has_timing: bool
    ) -> Dict[str, float]:
        """
        Calculate interruption features (Section 3.3.1).
        
        Interruption = speaker change with minimal or negative gap,
        often cutting off the previous speaker.
        """
        features = {
            'interruption_count': 0,
            'child_interruption_count': 0,
            'adult_interruption_count': 0,
            'interruption_ratio': 0.0,
        }
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        # Look for interruption markers in CHAT format
        interruption_markers = ['[//]', '+/', '+//', '<', '>', '[<]', '[>]']
        
        interruption_count = 0
        child_interruptions = 0
        adult_interruptions = 0
        
        for i in range(1, len(utterances)):
            prev = utterances[i - 1]
            curr = utterances[i]
            
            # Check text markers for interruptions
            has_interruption_marker = any(
                marker in curr.text or marker in prev.text
                for marker in interruption_markers
            )
            
            # Check timing if available
            timing_interruption = False
            if has_timing and prev.timing is not None and curr.timing is not None:
                gap = curr.timing - prev.timing
                # Very short or negative gap with speaker change = interruption
                if gap < (self.INTERRUPTION_THRESHOLD_MS / 1000.0) and prev.speaker != curr.speaker:
                    timing_interruption = True
            
            if has_interruption_marker or timing_interruption:
                if prev.speaker != curr.speaker:  # Only count actual speaker changes
                    interruption_count += 1
                    
                    if curr.speaker == 'CHI':
                        child_interruptions += 1
                    elif curr.speaker in adult_codes:
                        adult_interruptions += 1
        
        features['interruption_count'] = interruption_count
        features['child_interruption_count'] = child_interruptions
        features['adult_interruption_count'] = adult_interruptions
        features['interruption_ratio'] = calculate_ratio(
            interruption_count, len(utterances) - 1
        )
        
        return features
    
    def _calculate_turn_initiation(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate turn initiation features.
        
        Children with ASD often have difficulty initiating conversation.
        """
        features = {
            'child_initiated_turns': 0,
            'adult_initiated_turns': 0,
            'child_initiation_ratio': 0.0,
        }
        
        if not utterances:
            return features
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        child_initiated = 0
        adult_initiated = 0
        
        # First utterance counts as initiation
        if utterances[0].speaker == 'CHI':
            child_initiated += 1
        elif utterances[0].speaker in adult_codes:
            adult_initiated += 1
        
        # Speaker changes indicate turn initiation
        for i in range(1, len(utterances)):
            prev_speaker = utterances[i - 1].speaker
            curr_speaker = utterances[i].speaker
            
            if prev_speaker != curr_speaker:
                if curr_speaker == 'CHI':
                    child_initiated += 1
                elif curr_speaker in adult_codes:
                    adult_initiated += 1
        
        features['child_initiated_turns'] = child_initiated
        features['adult_initiated_turns'] = adult_initiated
        features['child_initiation_ratio'] = calculate_ratio(
            child_initiated, child_initiated + adult_initiated
        )
        
        return features
    
    def _calculate_turn_switches(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate turn switching patterns.
        
        Perseveration (staying on own turns) can indicate ASD.
        """
        features = {
            'turn_switches': 0,
            'avg_turns_before_switch': 0.0,
            'turn_switch_rate': 0.0,
        }
        
        if len(utterances) < 2:
            return features
        
        switches = 0
        turns_in_sequence = []
        current_sequence = 1
        
        for i in range(1, len(utterances)):
            if utterances[i].speaker != utterances[i - 1].speaker:
                switches += 1
                turns_in_sequence.append(current_sequence)
                current_sequence = 1
            else:
                current_sequence += 1
        
        # Add last sequence
        turns_in_sequence.append(current_sequence)
        
        features['turn_switches'] = switches
        features['avg_turns_before_switch'] = float(np.mean(turns_in_sequence)) if turns_in_sequence else 0.0
        features['turn_switch_rate'] = calculate_ratio(switches, len(utterances) - 1)
        
        return features
    
    def _calculate_consecutive_turns(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate consecutive turn patterns (monologue detection)."""
        features = {
            'max_consecutive_child_turns': 0,
            'max_consecutive_adult_turns': 0,
            'child_monologue_ratio': 0.0,
        }
        
        if not utterances:
            return features
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        max_child_consecutive = 0
        max_adult_consecutive = 0
        current_child_run = 0
        current_adult_run = 0
        child_monologue_turns = 0  # Turns in sequences of 3+
        
        for u in utterances:
            if u.speaker == 'CHI':
                current_child_run += 1
                current_adult_run = 0
                max_child_consecutive = max(max_child_consecutive, current_child_run)
            elif u.speaker in adult_codes:
                current_adult_run += 1
                if current_child_run >= 3:
                    child_monologue_turns += current_child_run
                current_child_run = 0
                max_adult_consecutive = max(max_adult_consecutive, current_adult_run)
            else:
                # Other speaker, reset both
                if current_child_run >= 3:
                    child_monologue_turns += current_child_run
                current_child_run = 0
                current_adult_run = 0
        
        # Final check for last sequence
        if current_child_run >= 3:
            child_monologue_turns += current_child_run
        
        child_total = sum(1 for u in utterances if u.speaker == 'CHI')
        
        features['max_consecutive_child_turns'] = max_child_consecutive
        features['max_consecutive_adult_turns'] = max_adult_consecutive
        features['child_monologue_ratio'] = calculate_ratio(child_monologue_turns, child_total)
        
        return features


__all__ = ["TurnTakingFeatures"]
