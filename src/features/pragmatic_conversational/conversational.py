"""
Conversational Feature Extractor

This module extracts features related to conversational management and
discourse patterns, which are often impaired in children with ASD.

Features include:
- Topic management (initiation, maintenance, shifts)
- Conversational repair strategies
- Discourse markers usage
- Behavioral/non-verbal communication patterns

Author: Bimidu Gunathilake
"""

import re
import numpy as np
from typing import List, Dict, Any, Set
from collections import Counter

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.helpers import calculate_ratio, safe_divide
from ..base_features import BaseFeatureExtractor, FeatureResult


class ConversationalFeatures(BaseFeatureExtractor):
    """
    Extract conversational and discourse management features.
    
    Features capture:
    - Topic continuity and shifts
    - Discourse markers
    - Conversational repairs
    - Non-verbal/behavioral markers
    - Turn relevance
    
    Example:
        >>> extractor = ConversationalFeatures()
        >>> features = extractor.extract(transcript)
        >>> print(f"Topic shifts: {features.features['topic_shift_ratio']}")
    """
    
    # Discourse markers that indicate topic management
    DISCOURSE_MARKERS = {
        'topic_intro': ['so', 'well', 'anyway', 'by the way'],
        'topic_continuation': ['and', 'also', 'too', 'then'],
        'topic_shift': ['but', 'however', 'although', 'though'],
        'repair': ['i mean', 'sorry', 'wait', 'no', 'actually'],
        'acknowledgment': ['okay', 'yeah', 'yes', 'mhm', 'uh huh'],
    }
    
    # Non-verbal/behavioral markers from CHAT
    BEHAVIORAL_MARKERS = [
        '&=laughs', '&=cries', '&=screams', '&=sighs',
        '&=gasps', '&=whispers', '&=hums', '&=sings',
        '&=squeals', '&=yells', '&=breathes'
    ]
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # Topic management
            'topic_shift_ratio',
            'topic_maintenance_score',
            'topic_intro_marker_ratio',
            'avg_topic_duration',
            
            # Discourse markers
            'discourse_marker_ratio',
            'continuation_marker_ratio',
            'repair_marker_ratio',
            'acknowledgment_ratio',
            
            # Conversational repair
            'self_repair_count',
            'other_repair_count',
            'clarification_request_ratio',
            
            # Behavioral features
            'nonverbal_behavior_ratio',
            'laughter_ratio',
            'vocal_behavior_diversity',
            
            # Turn relevance
            'topic_relevance_score',
            'off_topic_ratio',
        ]
    
    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract conversational features from transcript.
        
        Args:
            transcript: Parsed transcript data
            
        Returns:
            FeatureResult with conversational features
        """
        features = {}
        
        # Get utterances
        child_utterances = self.get_child_utterances(transcript)
        all_utterances = transcript.valid_utterances
        
        if not child_utterances:
            return FeatureResult(
                features={name: 0.0 for name in self.feature_names},
                feature_type='conversational',
                metadata={'error': 'No valid child utterances'}
            )
        
        # Extract topic management features
        topic_features = self._calculate_topic_features(child_utterances, all_utterances)
        features.update(topic_features)
        
        # Extract discourse marker features
        discourse_features = self._calculate_discourse_markers(child_utterances)
        features.update(discourse_features)
        
        # Extract repair features
        repair_features = self._calculate_repair_features(child_utterances, all_utterances)
        features.update(repair_features)
        
        # Extract behavioral features
        behavioral_features = self._calculate_behavioral_features(child_utterances)
        features.update(behavioral_features)
        
        # Extract relevance features
        relevance_features = self._calculate_relevance_features(child_utterances, all_utterances)
        features.update(relevance_features)
        
        return FeatureResult(
            features=features,
            feature_type='conversational',
            metadata={
                'num_child_utterances': len(child_utterances),
                'total_utterances': len(all_utterances)
            }
        )
    
    def _calculate_topic_features(
        self,
        child_utterances: List[Utterance],
        all_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate topic management features.
        
        Estimates topic shifts by detecting major vocabulary changes
        between consecutive utterances.
        
        Args:
            child_utterances: Child's utterances
            all_utterances: All utterances for context
            
        Returns:
            Dictionary of topic features
        """
        features = {
            'topic_shift_ratio': 0.0,
            'topic_maintenance_score': 0.0,
            'topic_intro_marker_ratio': 0.0,
            'avg_topic_duration': 0.0,
        }
        
        if len(all_utterances) < 2:
            return features
        
        # Detect topic shifts using word overlap
        topic_shifts = 0
        topic_durations = []
        current_topic_length = 1
        
        intro_markers = 0
        
        for i in range(1, len(all_utterances)):
            prev_utterance = all_utterances[i-1]
            curr_utterance = all_utterances[i]
            
            # Check for topic introduction markers (child only)
            if curr_utterance.speaker == 'CHI':
                text = curr_utterance.text.lower()
                if any(marker in text for marker in self.DISCOURSE_MARKERS['topic_intro']):
                    intro_markers += 1
            
            # Detect topic shift using word overlap
            if self._is_topic_shift(prev_utterance, curr_utterance):
                topic_shifts += 1
                topic_durations.append(current_topic_length)
                current_topic_length = 1
            else:
                current_topic_length += 1
        
        # Add last topic duration
        topic_durations.append(current_topic_length)
        
        # Calculate features
        features['topic_shift_ratio'] = calculate_ratio(topic_shifts, len(all_utterances))
        features['topic_maintenance_score'] = 1.0 - features['topic_shift_ratio']
        features['topic_intro_marker_ratio'] = calculate_ratio(intro_markers, len(child_utterances))
        features['avg_topic_duration'] = np.mean(topic_durations) if topic_durations else 0.0
        
        return features
    
    def _is_topic_shift(
        self,
        prev_utterance: Utterance,
        curr_utterance: Utterance,
        threshold: float = 0.3
    ) -> bool:
        """
        Detect if there's a topic shift between utterances.
        
        Uses word overlap to detect topic changes. Low overlap suggests
        a new topic.
        
        Args:
            prev_utterance: Previous utterance
            curr_utterance: Current utterance
            threshold: Minimum overlap to consider same topic
            
        Returns:
            True if topic shift detected
        """
        if not prev_utterance.tokens or not curr_utterance.tokens:
            return False
        
        # Get content words (exclude function words)
        function_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'to', 'in', 'on', 'at'}
        
        # Extract word strings from tokens
        prev_token_words = []
        for token in prev_utterance.tokens:
            if hasattr(token, 'word') and token.word:
                prev_token_words.append(token.word.lower())
        
        curr_token_words = []
        for token in curr_utterance.tokens:
            if hasattr(token, 'word') and token.word:
                curr_token_words.append(token.word.lower())
        
        prev_words = set(
            w for w in prev_token_words
            if w not in function_words
        )
        curr_words = set(
            w for w in curr_token_words
            if w not in function_words
        )
        
        if not prev_words or not curr_words:
            return False
        
        # Calculate Jaccard similarity
        intersection = prev_words.intersection(curr_words)
        union = prev_words.union(curr_words)
        
        overlap = len(intersection) / len(union) if union else 0
        
        # Topic shift if overlap is below threshold
        return overlap < threshold
    
    def _calculate_discourse_markers(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate discourse marker usage.
        
        Discourse markers help structure conversation and show
        pragmatic competence.
        
        Args:
            utterances: List of utterances
            
        Returns:
            Dictionary of discourse marker features
        """
        features = {
            'discourse_marker_ratio': 0.0,
            'continuation_marker_ratio': 0.0,
            'repair_marker_ratio': 0.0,
            'acknowledgment_ratio': 0.0,
        }
        
        if not utterances:
            return features
        
        total_markers = 0
        continuation_count = 0
        repair_count = 0
        acknowledgment_count = 0
        
        for utterance in utterances:
            text = utterance.text.lower()
            
            # Check each marker category
            for marker in self.DISCOURSE_MARKERS['topic_continuation']:
                if f" {marker} " in f" {text} " or text.startswith(f"{marker} "):
                    continuation_count += 1
                    total_markers += 1
                    break
            
            for marker in self.DISCOURSE_MARKERS['repair']:
                if marker in text:
                    repair_count += 1
                    total_markers += 1
                    break
            
            for marker in self.DISCOURSE_MARKERS['acknowledgment']:
                if text.strip() == marker or text.startswith(f"{marker} "):
                    acknowledgment_count += 1
                    total_markers += 1
                    break
        
        features['discourse_marker_ratio'] = calculate_ratio(total_markers, len(utterances))
        features['continuation_marker_ratio'] = calculate_ratio(continuation_count, len(utterances))
        features['repair_marker_ratio'] = calculate_ratio(repair_count, len(utterances))
        features['acknowledgment_ratio'] = calculate_ratio(acknowledgment_count, len(utterances))
        
        return features
    
    def _calculate_repair_features(
        self,
        child_utterances: List[Utterance],
        all_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate conversational repair features.
        
        Repair strategies indicate awareness of communication breakdown:
        - Self-repair: Child corrects own utterance
        - Other-repair: Response to clarification request
        - Clarification requests: Child asks for clarification
        
        Args:
            child_utterances: Child's utterances
            all_utterances: All utterances for context
            
        Returns:
            Dictionary of repair features
        """
        features = {
            'self_repair_count': 0,
            'other_repair_count': 0,
            'clarification_request_ratio': 0.0,
        }
        
        self_repair_count = 0
        other_repair_count = 0
        clarification_count = 0
        
        # Self-repair markers
        repair_markers = ['i mean', 'wait', 'no', 'sorry']
        
        # Clarification markers
        clarification_markers = ['what', 'huh', 'pardon', 'again']
        
        for i, utterance in enumerate(all_utterances):
            if utterance.speaker != 'CHI':
                continue
            
            text = utterance.text.lower()
            
            # Check for self-repair
            for marker in repair_markers:
                if marker in text:
                    self_repair_count += 1
                    break
            
            # Check for clarification requests
            if any(marker in text for marker in clarification_markers):
                if '?' in text or len(text.split()) <= 3:
                    clarification_count += 1
            
            # Check for other-repair (response to adult clarification)
            if i > 0:
                prev_text = all_utterances[i-1].text.lower()
                if all_utterances[i-1].speaker != 'CHI':
                    if any(word in prev_text for word in ['what', 'huh', 'pardon', 'mean']):
                        other_repair_count += 1
        
        features['self_repair_count'] = self_repair_count
        features['other_repair_count'] = other_repair_count
        features['clarification_request_ratio'] = calculate_ratio(
            clarification_count,
            len(child_utterances)
        )
        
        return features
    
    def _calculate_behavioral_features(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate non-verbal and behavioral features.
        
        CHAT uses markers like &=laughs for non-verbal behaviors.
        
        Args:
            utterances: List of utterances
            
        Returns:
            Dictionary of behavioral features
        """
        features = {
            'nonverbal_behavior_ratio': 0.0,
            'laughter_ratio': 0.0,
            'vocal_behavior_diversity': 0.0,
        }
        
        if not utterances:
            return features
        
        behavior_count = 0
        laughter_count = 0
        behavior_types = set()
        
        for utterance in utterances:
            text = utterance.text
            
            # Check for behavioral markers
            for marker in self.BEHAVIORAL_MARKERS:
                if marker in text:
                    behavior_count += 1
                    behavior_types.add(marker)
                    
                    if 'laugh' in marker:
                        laughter_count += 1
        
        features['nonverbal_behavior_ratio'] = calculate_ratio(behavior_count, len(utterances))
        features['laughter_ratio'] = calculate_ratio(laughter_count, len(utterances))
        features['vocal_behavior_diversity'] = len(behavior_types) / len(self.BEHAVIORAL_MARKERS)
        
        return features
    
    def _calculate_relevance_features(
        self,
        child_utterances: List[Utterance],
        all_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate topic relevance features.
        
        Measures how well child maintains topic relevance.
        
        Args:
            child_utterances: Child's utterances
            all_utterances: All utterances
            
        Returns:
            Dictionary of relevance features
        """
        features = {
            'topic_relevance_score': 0.0,
            'off_topic_ratio': 0.0,
        }
        
        if len(all_utterances) < 2:
            return features
        
        relevant_count = 0
        off_topic_count = 0
        
        for i, utterance in enumerate(all_utterances):
            if utterance.speaker != 'CHI' or i == 0:
                continue
            
            # Compare with previous adult utterance
            prev_adult_idx = i - 1
            while prev_adult_idx >= 0 and all_utterances[prev_adult_idx].speaker == 'CHI':
                prev_adult_idx -= 1
            
            if prev_adult_idx >= 0:
                prev_adult = all_utterances[prev_adult_idx]
                
                # Check topic relevance using word overlap
                if not self._is_topic_shift(prev_adult, utterance, threshold=0.2):
                    relevant_count += 1
                else:
                    off_topic_count += 1
        
        total_responses = relevant_count + off_topic_count
        
        features['topic_relevance_score'] = calculate_ratio(relevant_count, total_responses)
        features['off_topic_ratio'] = calculate_ratio(off_topic_count, total_responses)
        
        return features


__all__ = ["ConversationalFeatures"]

