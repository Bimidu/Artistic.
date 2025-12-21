"""
Pragmatic and Linguistic Feature Extractor (Supporting Module)

This consolidated module extracts pragmatic language features and conversational
linguistic features that support the main methodology-aligned extractors.

Features NOT covered by Sections 3.3.1-3.3.4:
- MLU and language development metrics
- Vocabulary diversity (TTR)
- Echolalia patterns
- Pronoun usage and reversal
- Question usage patterns
- Social language markers
- Discourse markers
- Non-verbal behavioral markers

Note: Syntactic features (POS ratios) are handled by syntactic_semantic module.
Note: Topic coherence/shifts are handled by topic_coherence.py (Section 3.3.2).
Note: Repair features are handled by repair_detection.py (Section 3.3.4).

Author: Bimidu Gunathilake
"""

import re
import numpy as np
from typing import List, Dict, Any
from collections import Counter

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.helpers import safe_divide, calculate_ratio
from src.utils.logger import get_logger
from ..base_features import BaseFeatureExtractor, FeatureResult

logger = get_logger(__name__)


class PragmaticLinguisticFeatures(BaseFeatureExtractor):
    """
    Extract pragmatic and conversational linguistic features.
    
    This is a consolidated extractor for supporting features that complement
    the main methodology-aligned extractors (Sections 3.3.1-3.3.4).
    
    Features include:
    - MLU (Mean Length of Utterance) - language development
    - Vocabulary diversity (TTR)
    - Echolalia patterns - ASD marker
    - Pronoun usage and reversal - ASD marker
    - Question formation - pragmatic competence
    - Social language markers - social communication
    - Discourse markers - conversational structure
    - Non-verbal behavioral markers - from CHAT annotations
    
    Example:
        >>> extractor = PragmaticLinguisticFeatures()
        >>> features = extractor.extract(transcript)
        >>> print(f"MLU: {features.features['mlu_words']}")
    """
    
    # Social phrases for pragmatic analysis
    SOCIAL_PHRASES = [
        'please', 'thank you', 'sorry', 'excuse me',
        'hello', 'hi', 'bye', 'goodbye',
        'yes please', 'no thank you'
    ]
    
    # Question words for question analysis
    QUESTION_WORDS = [
        'what', 'where', 'when', 'who', 'why', 'how',
        'which', 'whose', 'whom'
    ]
    
    # Discourse markers (excluding repair markers - handled by repair_detection.py)
    DISCOURSE_MARKERS = {
        'topic_intro': ['so', 'well', 'anyway', 'by the way'],
        'topic_continuation': ['and', 'also', 'too', 'then'],
        'acknowledgment': ['okay', 'yeah', 'yes', 'mhm', 'uh huh', 'right'],
        'hesitation': ['um', 'uh', 'er'],  # Simple markers, detailed in pause_latency.py
    }
    
    # Non-verbal behavioral markers from CHAT format
    BEHAVIORAL_MARKERS = [
        '&=laughs', '&=cries', '&=screams', '&=sighs',
        '&=gasps', '&=whispers', '&=hums', '&=sings',
        '&=squeals', '&=yells', '&=breathes', '&=groans',
        '&=claps', '&=points', '&=nods'
    ]
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # === MLU and Language Development ===
            'mlu_words',
            'mlu_morphemes',
            'avg_word_length_chars',
            'max_utterance_length',
            
            # === Vocabulary Diversity ===
            'total_words',
            'unique_words',
            'type_token_ratio',
            'corrected_ttr',
            'lexical_density',
            'utterance_complexity_score',
            
            # === Echolalia (ASD Marker) ===
            'echolalia_ratio',
            'immediate_echolalia_count',
            'delayed_echolalia_count',
            'partial_repetition_ratio',
            
            # === Question Usage (Pragmatic) ===
            'question_ratio',
            'question_diversity',
            'yes_no_question_ratio',
            'wh_question_ratio',
            
            # === Pronoun Usage (ASD Marker) ===
            'pronoun_usage_ratio',
            'first_person_pronoun_ratio',
            'pronoun_error_ratio',
            'pronoun_reversal_count',
            
            # === Social Language (Pragmatic) ===
            'social_phrase_ratio',
            'greeting_count',
            'politeness_marker_count',
            
            # === Response Quality ===
            'appropriate_response_ratio',
            'unintelligible_ratio',
            
            # === Discourse Markers ===
            'discourse_marker_ratio',
            'continuation_marker_ratio',
            'acknowledgment_ratio',
            
            # === Non-verbal Behavioral Markers ===
            'nonverbal_behavior_ratio',
            'laughter_ratio',
            'vocal_behavior_diversity',
        ]
    
    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract pragmatic and linguistic features from transcript.
        
        Args:
            transcript: Parsed transcript data
            
        Returns:
            FeatureResult with pragmatic and linguistic features
        """
        features = {}
        
        child_utterances = self.get_child_utterances(transcript)
        adult_utterances = self.get_adult_utterances(transcript)
        all_utterances = transcript.valid_utterances
        
        logger.debug(f"Extracting pragmatic/linguistic features from {len(all_utterances)} utterances")
        
        if not child_utterances:
            return FeatureResult(
                features={name: 0.0 for name in self.feature_names},
                feature_type='pragmatic_linguistic',
                metadata={'error': 'No valid child utterances'}
            )
        
        # MLU and language development
        mlu_features = self._calculate_mlu(child_utterances)
        features.update(mlu_features)
        
        # Vocabulary diversity
        vocab_features = self._calculate_vocabulary(child_utterances)
        features.update(vocab_features)
        
        # Echolalia patterns
        echolalia_features = self._calculate_echolalia(
            child_utterances, adult_utterances, all_utterances
        )
        features.update(echolalia_features)
        
        # Question usage
        question_features = self._calculate_questions(child_utterances)
        features.update(question_features)
        
        # Pronoun usage
        pronoun_features = self._calculate_pronouns(child_utterances)
        features.update(pronoun_features)
        
        # Social language
        social_features = self._calculate_social_language(child_utterances)
        features.update(social_features)
        
        # Response quality
        response_features = self._calculate_response_quality(
            child_utterances, all_utterances
        )
        features.update(response_features)
        
        # Discourse markers
        discourse_features = self._calculate_discourse_markers(child_utterances)
        features.update(discourse_features)
        
        # Non-verbal behavioral markers
        behavioral_features = self._calculate_behavioral_markers(child_utterances)
        features.update(behavioral_features)
        
        logger.debug(f"Extracted {len(features)} pragmatic/linguistic features")
        
        return FeatureResult(
            features=features,
            feature_type='pragmatic_linguistic',
            metadata={
                'num_child_utterances': len(child_utterances),
                'num_adult_utterances': len(adult_utterances),
                'has_morphology': any(u.morphology for u in child_utterances)
            }
        )
    
    # =========================================================================
    # MLU and Language Development
    # =========================================================================
    
    def _calculate_mlu(self, utterances: List[Utterance]) -> Dict[str, float]:
        """
        Calculate Mean Length of Utterance (MLU) metrics.
        
        MLU is a key measure of language development.
        """
        features = {
            'mlu_words': 0.0,
            'mlu_morphemes': 0.0,
            'avg_word_length_chars': 0.0,
            'max_utterance_length': 0,
        }
        
        if not utterances:
            return features
        
        # MLU in words
        word_lengths = self.get_utterance_lengths(utterances, in_words=True)
        if word_lengths:
            features['mlu_words'] = float(np.mean(word_lengths))
            features['max_utterance_length'] = max(word_lengths)
        
        # MLU in morphemes and average word length
        morpheme_counts = []
        word_char_lengths = []
        
        for utterance in utterances:
            # Morpheme count from %mor tier
            if utterance.morphology:
                morphemes = self._count_morphemes(utterance.morphology)
                if morphemes > 0:
                    morpheme_counts.append(morphemes)
            
            # Character length of words
            if utterance.tokens:
                for token in utterance.tokens:
                    if hasattr(token, 'word') and token.word:
                        word_char_lengths.append(len(token.word))
        
        if morpheme_counts:
            features['mlu_morphemes'] = float(np.mean(morpheme_counts))
        
        if word_char_lengths:
            features['avg_word_length_chars'] = float(np.mean(word_char_lengths))
        
        return features
    
    def _count_morphemes(self, morphology_str: str) -> int:
        """Count morphemes from CHAT %mor tier."""
        if not morphology_str:
            return 0
        
        analyses = morphology_str.split()
        morpheme_count = 0
        
        for analysis in analyses:
            morpheme_count += 1  # Base morpheme
            if '|' in analysis:
                word_part = analysis.split('|', 1)[1]
                morpheme_count += word_part.count('-')
                morpheme_count += word_part.count('~')
        
        return morpheme_count
    
    # =========================================================================
    # Vocabulary Diversity
    # =========================================================================
    
    def _calculate_vocabulary(self, utterances: List[Utterance]) -> Dict[str, float]:
        """
        Calculate vocabulary diversity metrics.
        
        Lower TTR (more repetitive) is often associated with ASD.
        """
        features = {
            'total_words': 0,
            'unique_words': 0,
            'type_token_ratio': 0.0,
            'corrected_ttr': 0.0,
            'lexical_density': 0.0,
            'utterance_complexity_score': 0.0,
        }
        
        # Collect all words
        all_words = []
        for utterance in utterances:
            if utterance.tokens:
                for token in utterance.tokens:
                    if hasattr(token, 'word') and token.word:
                        all_words.append(token.word.lower())
        
        if not all_words:
            return features
        
        total_words = len(all_words)
        unique_words = len(set(all_words))
        
        features['total_words'] = total_words
        features['unique_words'] = unique_words
        features['type_token_ratio'] = safe_divide(unique_words, total_words)
        features['corrected_ttr'] = safe_divide(unique_words, np.sqrt(2 * total_words))
        
        # Lexical density (content vs function words)
        function_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
            'he', 'she', 'it', 'they', 'we', 'you', 'i',
            'this', 'that', 'these', 'those'
        }
        content_words = sum(1 for w in all_words if w not in function_words)
        features['lexical_density'] = calculate_ratio(content_words, total_words)
        
        # Complexity score (composite)
        word_lengths = self.get_utterance_lengths(utterances, in_words=True)
        mlu = np.mean(word_lengths) if word_lengths else 0
        unique_ratio = unique_words / total_words
        
        complexity = (
            (min(mlu / 10, 1.0) * 0.4) +
            (unique_ratio * 0.3) +
            (features['lexical_density'] * 0.3)
        )
        features['utterance_complexity_score'] = complexity
        
        return features
    
    # =========================================================================
    # Echolalia (ASD Marker)
    # =========================================================================
    
    def _calculate_echolalia(
        self,
        child_utterances: List[Utterance],
        adult_utterances: List[Utterance],
        all_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate echolalia (repetition) features.
        
        Echolalia is a key ASD marker:
        - Immediate: Repeating right after someone
        - Delayed: Repeating something from earlier
        - Partial: Repeating part of an utterance
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
        
        utterance_texts = [u.text.lower().strip() for u in all_utterances]
        
        for i, utterance in enumerate(all_utterances):
            if utterance.speaker != 'CHI':
                continue
            
            child_text = utterance.text.lower().strip()
            if len(child_text.split()) < 2:
                continue
            
            # Immediate echolalia
            if i > 0:
                prev_text = utterance_texts[i - 1]
                
                if child_text == prev_text:
                    immediate_count += 1
                    continue
                
                # Partial match
                if self._is_partial_repetition(child_text, prev_text):
                    partial_count += 1
            
            # Delayed echolalia (look back up to 10 turns)
            for j in range(max(0, i - 10), i - 1):
                if utterance_texts[j] == child_text:
                    delayed_count += 1
                    break
        
        total_child = len(child_utterances)
        total_echolalia = immediate_count + delayed_count
        
        features['immediate_echolalia_count'] = immediate_count
        features['delayed_echolalia_count'] = delayed_count
        features['echolalia_ratio'] = calculate_ratio(total_echolalia, total_child)
        features['partial_repetition_ratio'] = calculate_ratio(partial_count, total_child)
        
        return features
    
    def _is_partial_repetition(self, text1: str, text2: str) -> bool:
        """Check if text1 is a partial repetition of text2."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = words1.intersection(words2)
        overlap_ratio = len(overlap) / len(words2)
        
        return overlap_ratio > 0.6
    
    # =========================================================================
    # Question Usage (Pragmatic)
    # =========================================================================
    
    def _calculate_questions(self, utterances: List[Utterance]) -> Dict[str, float]:
        """
        Calculate question usage features.
        
        Children with ASD may ask fewer questions or use limited question types.
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
            
            if text.endswith('?'):
                question_count += 1
                words = text.split()
                
                if words:
                    first_word = words[0]
                    
                    if first_word in self.QUESTION_WORDS:
                        wh_count += 1
                        question_types.add(first_word)
                    elif first_word in ['is', 'are', 'do', 'does', 'did', 'can', 'will', 'would', 'have', 'has']:
                        yes_no_count += 1
                        question_types.add('yes_no')
        
        total = len(utterances)
        features['question_ratio'] = calculate_ratio(question_count, total)
        features['yes_no_question_ratio'] = calculate_ratio(yes_no_count, total)
        features['wh_question_ratio'] = calculate_ratio(wh_count, total)
        features['question_diversity'] = len(question_types) / len(self.QUESTION_WORDS)
        
        return features
    
    # =========================================================================
    # Pronoun Usage (ASD Marker)
    # =========================================================================
    
    def _calculate_pronouns(self, utterances: List[Utterance]) -> Dict[str, float]:
        """
        Calculate pronoun usage features.
        
        Pronoun reversal (saying "you" instead of "I") is common in ASD.
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
        all_pronouns = first_person | second_person | {'he', 'she', 'it', 'they', 'them', 'their'}
        
        for utterance in utterances:
            if not utterance.tokens:
                continue
            
            words = []
            for token in utterance.tokens:
                if hasattr(token, 'word') and token.word:
                    words.append(token.word.lower())
            
            total_words += len(words)
            
            for word in words:
                if word in all_pronouns:
                    pronoun_count += 1
                    
                    if word in first_person:
                        first_person_count += 1
                    
                    # Detect potential reversal
                    if word == 'you':
                        text = utterance.text.lower()
                        if any(phrase in text for phrase in ['you want', 'you like', 'you need', 'you have']):
                            reversal_count += 1
        
        features['pronoun_usage_ratio'] = calculate_ratio(pronoun_count, total_words)
        features['first_person_pronoun_ratio'] = calculate_ratio(first_person_count, pronoun_count)
        features['pronoun_reversal_count'] = reversal_count
        features['pronoun_error_ratio'] = calculate_ratio(reversal_count, pronoun_count)
        
        return features
    
    # =========================================================================
    # Social Language (Pragmatic)
    # =========================================================================
    
    def _calculate_social_language(self, utterances: List[Utterance]) -> Dict[str, float]:
        """
        Calculate social language usage features.
        
        Tracks greetings, politeness markers, and social phrases.
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
            
            for phrase in self.SOCIAL_PHRASES:
                if phrase in text:
                    social_count += 1
                    break
            
            for greeting in greetings:
                if greeting in text:
                    greeting_count += 1
                    break
            
            for marker in politeness:
                if marker in text:
                    politeness_count += 1
                    break
        
        features['social_phrase_ratio'] = calculate_ratio(social_count, len(utterances))
        features['greeting_count'] = greeting_count
        features['politeness_marker_count'] = politeness_count
        
        return features
    
    # =========================================================================
    # Response Quality
    # =========================================================================
    
    def _calculate_response_quality(
        self,
        child_utterances: List[Utterance],
        all_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate response quality features."""
        features = {
            'appropriate_response_ratio': 0.0,
            'unintelligible_ratio': 0.0,
        }
        
        if not child_utterances:
            return features
        
        # Unintelligible utterances (marked with xxx in CHAT)
        unintelligible = sum(1 for u in child_utterances if 'xxx' in u.text.lower())
        features['unintelligible_ratio'] = calculate_ratio(unintelligible, len(child_utterances))
        
        # Appropriate responses to adult questions
        appropriate_count = 0
        response_opportunities = 0
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        for i, utterance in enumerate(all_utterances):
            if i > 0 and all_utterances[i - 1].text.endswith('?'):
                if all_utterances[i - 1].speaker in adult_codes:
                    response_opportunities += 1
                    
                    if utterance.speaker == 'CHI':
                        if utterance.word_count >= 1 and 'xxx' not in utterance.text:
                            appropriate_count += 1
        
        features['appropriate_response_ratio'] = calculate_ratio(
            appropriate_count, response_opportunities
        )
        
        return features
    
    # =========================================================================
    # Discourse Markers
    # =========================================================================
    
    def _calculate_discourse_markers(self, utterances: List[Utterance]) -> Dict[str, float]:
        """
        Calculate discourse marker usage.
        
        Discourse markers indicate pragmatic competence in structuring conversation.
        Note: Repair markers are handled by repair_detection.py
        """
        features = {
            'discourse_marker_ratio': 0.0,
            'continuation_marker_ratio': 0.0,
            'acknowledgment_ratio': 0.0,
        }
        
        if not utterances:
            return features
        
        total_markers = 0
        continuation_count = 0
        acknowledgment_count = 0
        
        for utterance in utterances:
            text = utterance.text.lower()
            
            # Continuation markers
            for marker in self.DISCOURSE_MARKERS['topic_continuation']:
                if f" {marker} " in f" {text} " or text.startswith(f"{marker} "):
                    continuation_count += 1
                    total_markers += 1
                    break
            
            # Acknowledgment markers
            for marker in self.DISCOURSE_MARKERS['acknowledgment']:
                if text.strip() == marker or text.startswith(f"{marker} "):
                    acknowledgment_count += 1
                    total_markers += 1
                    break
        
        features['discourse_marker_ratio'] = calculate_ratio(total_markers, len(utterances))
        features['continuation_marker_ratio'] = calculate_ratio(continuation_count, len(utterances))
        features['acknowledgment_ratio'] = calculate_ratio(acknowledgment_count, len(utterances))
        
        return features
    
    # =========================================================================
    # Non-verbal Behavioral Markers
    # =========================================================================
    
    def _calculate_behavioral_markers(self, utterances: List[Utterance]) -> Dict[str, float]:
        """
        Calculate non-verbal behavioral features from CHAT annotations.
        
        CHAT uses markers like &=laughs for paralinguistic behaviors.
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


__all__ = ["PragmaticLinguisticFeatures"]





