"""
Pragmatic Feature Extractor

This module extracts pragmatic language features that are often impaired in ASD.
Pragmatic language refers to the social use of language in context.

Key features:
- Echolalia (immediate and delayed repetition)
- Question usage and appropriateness
- Pronoun usage and errors
- Social/communicative intent markers
- Inappropriate responses

Author: Bimidu Gunathilake
"""

import re
import numpy as np
from typing import List, Dict, Any
from collections import Counter

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.helpers import calculate_ratio
from ..base_features import BaseFeatureExtractor, FeatureResult


class PragmaticFeatures(BaseFeatureExtractor):
    """
    Extract pragmatic language use features.
    
    Pragmatic features capture social language use:
    - Echolalia (repetition patterns)
    - Question formation and use
    - Pronoun usage (including reversal)
    - Social/communicative functions
    - Response appropriateness
    
    Example:
        >>> extractor = PragmaticFeatures()
        >>> features = extractor.extract(transcript)
        >>> print(f"Echolalia rate: {features.features['echolalia_ratio']}")
    """
    
    # Common social phrases children should use
    SOCIAL_PHRASES = [
        'please', 'thank you', 'sorry', 'excuse me',
        'hello', 'hi', 'bye', 'goodbye',
        'yes please', 'no thank you'
    ]
    
    # Question markers
    QUESTION_WORDS = [
        'what', 'where', 'when', 'who', 'why', 'how',
        'which', 'whose', 'whom'
    ]
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # Echolalia features
            'echolalia_ratio',
            'immediate_echolalia_count',
            'delayed_echolalia_count',
            'partial_repetition_ratio',
            
            # Question features
            'question_ratio',
            'question_diversity',
            'yes_no_question_ratio',
            'wh_question_ratio',
            
            # Pronoun features
            'pronoun_usage_ratio',
            'first_person_pronoun_ratio',
            'pronoun_error_ratio',
            'pronoun_reversal_count',
            
            # Social language
            'social_phrase_ratio',
            'greeting_count',
            'politeness_marker_count',
            
            # Response features
            'appropriate_response_ratio',
            'unintelligible_ratio',
        ]
    
    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract pragmatic features from transcript.
        
        Args:
            transcript: Parsed transcript data
            
        Returns:
            FeatureResult with pragmatic features
        """
        features = {}
        
        # Get child utterances and adult utterances for context
        child_utterances = self.get_child_utterances(transcript)
        adult_utterances = self.get_adult_utterances(transcript)
        all_utterances = transcript.valid_utterances
        
        if not child_utterances:
            return FeatureResult(
                features={name: 0.0 for name in self.feature_names},
                feature_type='pragmatic',
                metadata={'error': 'No valid child utterances'}
            )
        
        # Extract echolalia features
        echolalia_features = self._calculate_echolalia(
            child_utterances,
            adult_utterances,
            all_utterances
        )
        features.update(echolalia_features)
        
        # Extract question features
        question_features = self._calculate_question_features(child_utterances)
        features.update(question_features)
        
        # Extract pronoun features
        pronoun_features = self._calculate_pronoun_features(child_utterances)
        features.update(pronoun_features)
        
        # Extract social language features
        social_features = self._calculate_social_language(child_utterances)
        features.update(social_features)
        
        # Extract response appropriateness
        response_features = self._calculate_response_features(
            child_utterances,
            all_utterances
        )
        features.update(response_features)
        
        return FeatureResult(
            features=features,
            feature_type='pragmatic',
            metadata={'num_child_utterances': len(child_utterances)}
        )
    
    def _calculate_echolalia(
        self,
        child_utterances: List[Utterance],
        adult_utterances: List[Utterance],
        all_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate echolalia (repetition) features.
        
        Echolalia types:
        - Immediate: Repeating immediately after adult
        - Delayed: Repeating something said earlier
        - Partial: Repeating part of an utterance
        
        Args:
            child_utterances: Child's utterances
            adult_utterances: Adult's utterances
            all_utterances: All utterances in order
            
        Returns:
            Dictionary of echolalia features
        """
        features = {
            'echolalia_ratio': 0.0,
            'immediate_echolalia_count': 0,
            'delayed_echolalia_count': 0,
            'partial_repetition_ratio': 0.0,
        }
        
        if not child_utterances:
            return features
        
        immediate_count = 0
        delayed_count = 0
        partial_count = 0
        
        # Create map of utterances for context
        utterance_texts = [u.text.lower().strip() for u in all_utterances]
        
        for i, utterance in enumerate(all_utterances):
            if utterance.speaker != 'CHI':
                continue
            
            child_text = utterance.text.lower().strip()
            
            # Skip very short utterances
            if len(child_text.split()) < 2:
                continue
            
            # Check for immediate echolalia (previous utterance)
            if i > 0:
                prev_text = utterance_texts[i-1]
                
                # Exact match = immediate echolalia
                if child_text == prev_text:
                    immediate_count += 1
                    continue
                
                # Partial match
                if self._is_partial_repetition(child_text, prev_text):
                    partial_count += 1
            
            # Check for delayed echolalia (earlier in conversation)
            for j in range(max(0, i-10), i-1):  # Look back up to 10 turns
                if utterance_texts[j] == child_text:
                    delayed_count += 1
                    break
        
        # Calculate ratios
        total_child = len(child_utterances)
        total_echolalia = immediate_count + delayed_count
        
        features['immediate_echolalia_count'] = immediate_count
        features['delayed_echolalia_count'] = delayed_count
        features['echolalia_ratio'] = calculate_ratio(total_echolalia, total_child)
        features['partial_repetition_ratio'] = calculate_ratio(partial_count, total_child)
        
        return features
    
    def _is_partial_repetition(self, text1: str, text2: str) -> bool:
        """
        Check if text1 is a partial repetition of text2.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            True if partial repetition detected
        """
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        # Calculate overlap
        overlap = words1.intersection(words2)
        overlap_ratio = len(overlap) / len(words2)
        
        # Consider partial if >60% overlap
        return overlap_ratio > 0.6
    
    def _calculate_question_features(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate question-related features.
        
        Children with ASD may:
        - Ask fewer questions
        - Use limited question types
        - Have difficulty with wh-questions
        
        Args:
            utterances: List of utterances
            
        Returns:
            Dictionary of question features
        """
        features = {
            'question_ratio': 0.0,
            'question_diversity': 0.0,
            'yes_no_question_ratio': 0.0,
            'wh_question_ratio': 0.0,
        }
        
        if not utterances:
            return features
        
        question_count = 0
        yes_no_count = 0
        wh_count = 0
        question_types = set()
        
        for utterance in utterances:
            text = utterance.text.lower().strip()
            
            # Check if it's a question (ends with ?)
            if text.endswith('?'):
                question_count += 1
                
                # Classify question type
                words = text.split()
                if words:
                    first_word = words[0]
                    
                    # Wh-question
                    if first_word in self.QUESTION_WORDS:
                        wh_count += 1
                        question_types.add(first_word)
                    # Yes/no question (auxiliary verb first)
                    elif first_word in ['is', 'are', 'do', 'does', 'did', 'can', 'will', 'would']:
                        yes_no_count += 1
                        question_types.add('yes_no')
        
        total_utterances = len(utterances)
        
        features['question_ratio'] = calculate_ratio(question_count, total_utterances)
        features['yes_no_question_ratio'] = calculate_ratio(yes_no_count, total_utterances)
        features['wh_question_ratio'] = calculate_ratio(wh_count, total_utterances)
        features['question_diversity'] = len(question_types) / len(self.QUESTION_WORDS)
        
        return features
    
    def _calculate_pronoun_features(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate pronoun usage features.
        
        Pronoun reversal (saying "you" instead of "I") is common in ASD.
        Also track overall pronoun usage patterns.
        
        Args:
            utterances: List of utterances
            
        Returns:
            Dictionary of pronoun features
        """
        features = {
            'pronoun_usage_ratio': 0.0,
            'first_person_pronoun_ratio': 0.0,
            'pronoun_error_ratio': 0.0,
            'pronoun_reversal_count': 0,
        }
        
        if not utterances:
            return features
        
        total_words = 0
        pronoun_count = 0
        first_person_count = 0
        reversal_count = 0
        
        first_person = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours'}
        second_person = {'you', 'your', 'yours', 'yourself'}
        all_pronouns = first_person.union(second_person).union({'he', 'she', 'it', 'they', 'them', 'their'})
        
        for utterance in utterances:
            if not utterance.tokens:
                continue
            
            words = [w.lower() for w in utterance.tokens]
            total_words += len(words)
            
            for word in words:
                if word in all_pronouns:
                    pronoun_count += 1
                    
                    if word in first_person:
                        first_person_count += 1
                    
                    # Detect potential pronoun reversal
                    # Using "you" in contexts where "I" should be used
                    if word == 'you':
                        # Simple heuristic: "you want" instead of "I want"
                        text = utterance.text.lower()
                        if any(phrase in text for phrase in ['you want', 'you like', 'you need']):
                            reversal_count += 1
        
        features['pronoun_usage_ratio'] = calculate_ratio(pronoun_count, total_words)
        features['first_person_pronoun_ratio'] = calculate_ratio(first_person_count, pronoun_count) if pronoun_count > 0 else 0.0
        features['pronoun_reversal_count'] = reversal_count
        features['pronoun_error_ratio'] = calculate_ratio(reversal_count, pronoun_count) if pronoun_count > 0 else 0.0
        
        return features
    
    def _calculate_social_language(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate social language usage.
        
        Track use of social phrases, greetings, and politeness markers.
        
        Args:
            utterances: List of utterances
            
        Returns:
            Dictionary of social language features
        """
        features = {
            'social_phrase_ratio': 0.0,
            'greeting_count': 0,
            'politeness_marker_count': 0,
        }
        
        if not utterances:
            return features
        
        social_count = 0
        greeting_count = 0
        politeness_count = 0
        
        greetings = ['hello', 'hi', 'bye', 'goodbye', 'good morning', 'good night']
        politeness = ['please', 'thank you', 'thanks', 'sorry', 'excuse me']
        
        for utterance in utterances:
            text = utterance.text.lower().strip()
            
            # Check for social phrases
            for phrase in self.SOCIAL_PHRASES:
                if phrase in text:
                    social_count += 1
                    break
            
            # Check for greetings
            for greeting in greetings:
                if greeting in text:
                    greeting_count += 1
                    break
            
            # Check for politeness markers
            for marker in politeness:
                if marker in text:
                    politeness_count += 1
                    break
        
        features['social_phrase_ratio'] = calculate_ratio(social_count, len(utterances))
        features['greeting_count'] = greeting_count
        features['politeness_marker_count'] = politeness_count
        
        return features
    
    def _calculate_response_features(
        self,
        child_utterances: List[Utterance],
        all_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate response appropriateness features.
        
        Track unintelligible utterances and response patterns.
        
        Args:
            child_utterances: Child's utterances
            all_utterances: All utterances for context
            
        Returns:
            Dictionary of response features
        """
        features = {
            'appropriate_response_ratio': 0.0,
            'unintelligible_ratio': 0.0,
        }
        
        if not child_utterances:
            return features
        
        # Count unintelligible utterances (marked with xxx in CHAT)
        unintelligible_count = sum(
            1 for u in child_utterances
            if 'xxx' in u.text.lower()
        )
        
        features['unintelligible_ratio'] = calculate_ratio(
            unintelligible_count,
            len(child_utterances)
        )
        
        # Appropriate responses (responses to adult questions)
        # This is a simplified heuristic
        appropriate_count = 0
        response_opportunities = 0
        
        for i, utterance in enumerate(all_utterances):
            # Check if previous was adult question
            if i > 0 and all_utterances[i-1].text.endswith('?'):
                if all_utterances[i-1].speaker != 'CHI':
                    response_opportunities += 1
                    
                    # Check if child responded
                    if utterance.speaker == 'CHI':
                        # Simple check: response is appropriate if not too short or unintelligible
                        if len(utterance.tokens) >= 1 and 'xxx' not in utterance.text:
                            appropriate_count += 1
        
        features['appropriate_response_ratio'] = calculate_ratio(
            appropriate_count,
            response_opportunities
        )
        
        return features


__all__ = ["PragmaticFeatures"]

