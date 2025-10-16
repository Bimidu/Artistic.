"""
Turn-Taking Feature Extractor

This module extracts features related to turn-taking patterns in conversations,
which are often impaired in children with ASD. These features capture:
- Turn frequency and distribution
- Response latency
- Turn overlap/interruptions
- Turn appropriateness

Author: Bimidu Gunathilake
"""

import numpy as np
from typing import List, Dict, Any

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.helpers import safe_divide, calculate_ratio
from ..base_features import BaseFeatureExtractor, FeatureResult


class TurnTakingFeatures(BaseFeatureExtractor):
    """
    Extract turn-taking pattern features from transcripts.
    
    Features include:
    - Turn frequency (turns per minute)
    - Average turn duration
    - Response latency (time between turns)
    - Turn overlap frequency
    - Child-initiated vs adult-initiated turns
    - Turn distribution (child vs adult ratio)
    
    Example:
        >>> extractor = TurnTakingFeatures()
        >>> features = extractor.extract(transcript)
        >>> print(features.features['turns_per_minute'])
    """
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            'total_turns',
            'child_turns',
            'adult_turns',
            'turns_per_minute',
            'child_turn_ratio',
            'avg_turn_length_words',
            'avg_child_turn_length',
            'avg_adult_turn_length',
            'avg_response_latency',
            'median_response_latency',
            'child_initiated_turns',
            'adult_initiated_turns',
            'child_initiation_ratio',
            'turn_switches',
            'avg_turns_before_switch',
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
        
        # Basic turn counts
        features['total_turns'] = len(all_utterances)
        features['child_turns'] = len(child_utterances)
        features['adult_turns'] = len(adult_utterances)
        
        # Calculate turns per minute (if timing available)
        features['turns_per_minute'] = self._calculate_turns_per_minute(
            all_utterances
        )
        
        # Turn distribution
        features['child_turn_ratio'] = calculate_ratio(
            features['child_turns'],
            features['total_turns']
        )
        
        # Average turn lengths (in words)
        features['avg_turn_length_words'] = self._calculate_avg_length(
            all_utterances
        )
        features['avg_child_turn_length'] = self._calculate_avg_length(
            child_utterances
        )
        features['avg_adult_turn_length'] = self._calculate_avg_length(
            adult_utterances
        )
        
        # Response latency features
        latency_features = self._calculate_response_latency(all_utterances)
        features.update(latency_features)
        
        # Turn initiation features
        initiation_features = self._calculate_turn_initiation(all_utterances)
        features.update(initiation_features)
        
        # Turn switching patterns
        switch_features = self._calculate_turn_switches(all_utterances)
        features.update(switch_features)
        
        return FeatureResult(
            features=features,
            feature_type='turn_taking',
            metadata={
                'total_utterances': len(all_utterances),
                'has_timing': any(u.timing is not None for u in all_utterances)
            }
        )
    
    def _calculate_turns_per_minute(
        self,
        utterances: List[Utterance]
    ) -> float:
        """
        Calculate number of turns per minute.
        
        Args:
            utterances: List of utterances
            
        Returns:
            Turns per minute, or 0 if timing unavailable
        """
        if not utterances:
            return 0.0
        
        # Get utterances with timing information
        timed_utterances = [u for u in utterances if u.timing is not None]
        
        if not timed_utterances or len(timed_utterances) < 2:
            # If no timing, estimate based on typical conversation rate
            # Assume 15-minute session if no timing available
            estimated_duration_minutes = 15.0
            return len(utterances) / estimated_duration_minutes
        
        # Calculate duration in minutes
        start_time = min(u.timing for u in timed_utterances)
        end_time = max(u.timing for u in timed_utterances)
        duration_minutes = (end_time - start_time) / 60.0
        
        if duration_minutes <= 0:
            # Fallback to estimated duration
            estimated_duration_minutes = 15.0
            return len(utterances) / estimated_duration_minutes
        
        # Calculate turns per minute
        return len(utterances) / duration_minutes
    
    def _calculate_avg_length(self, utterances: List[Utterance]) -> float:
        """
        Calculate average utterance length in words.
        
        Args:
            utterances: List of utterances
            
        Returns:
            Average length in words
        """
        if not utterances:
            return 0.0
        
        lengths = self.get_utterance_lengths(utterances, in_words=True)
        
        if not lengths:
            return 0.0
        
        return np.mean(lengths)
    
    def _calculate_response_latency(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate response latency features.
        
        Response latency is the time gap between turns, which can
        indicate processing speed and conversational engagement.
        
        Args:
            utterances: List of utterances with timing
            
        Returns:
            Dictionary of latency features
        """
        features = {
            'avg_response_latency': 0.0,
            'median_response_latency': 0.0,
        }
        
        # Extract time gaps
        gaps = self.extract_timing_gaps(utterances)
        
        if not gaps:
            return features
        
        # Calculate statistics
        features['avg_response_latency'] = np.mean(gaps)
        features['median_response_latency'] = np.median(gaps)
        
        return features
    
    def _calculate_turn_initiation(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, Any]:
        """
        Calculate turn initiation features.
        
        Determines who initiates conversational turns (child vs adult).
        Children with ASD often have difficulty initiating conversation.
        
        Args:
            utterances: List of utterances
            
        Returns:
            Dictionary of initiation features
        """
        features = {
            'child_initiated_turns': 0,
            'adult_initiated_turns': 0,
            'child_initiation_ratio': 0.0,
        }
        
        if not utterances:
            return features
        
        child_initiated = 0
        adult_initiated = 0
        
        # First utterance
        if utterances[0].speaker == 'CHI':
            child_initiated += 1
        else:
            adult_initiated += 1
        
        # Check speaker changes (new topic/turn initiation)
        for i in range(1, len(utterances)):
            prev_speaker = utterances[i-1].speaker
            curr_speaker = utterances[i].speaker
            
            # Speaker change indicates potential turn initiation
            if prev_speaker != curr_speaker:
                if curr_speaker == 'CHI':
                    child_initiated += 1
                else:
                    adult_initiated += 1
        
        features['child_initiated_turns'] = child_initiated
        features['adult_initiated_turns'] = adult_initiated
        features['child_initiation_ratio'] = calculate_ratio(
            child_initiated,
            child_initiated + adult_initiated
        )
        
        return features
    
    def _calculate_turn_switches(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, Any]:
        """
        Calculate turn switching patterns.
        
        Analyzes how often speakers switch and average turns before switch.
        Perseveration (staying on own turns) can indicate ASD.
        
        Args:
            utterances: List of utterances
            
        Returns:
            Dictionary of turn switch features
        """
        features = {
            'turn_switches': 0,
            'avg_turns_before_switch': 0.0,
        }
        
        if len(utterances) < 2:
            return features
        
        switches = 0
        turns_in_sequence = []
        current_sequence = 1
        
        for i in range(1, len(utterances)):
            if utterances[i].speaker != utterances[i-1].speaker:
                switches += 1
                turns_in_sequence.append(current_sequence)
                current_sequence = 1
            else:
                current_sequence += 1
        
        # Add last sequence
        turns_in_sequence.append(current_sequence)
        
        features['turn_switches'] = switches
        features['avg_turns_before_switch'] = (
            np.mean(turns_in_sequence) if turns_in_sequence else 0.0
        )
        
        return features


__all__ = ["TurnTakingFeatures"]

