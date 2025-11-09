"""
Linguistic Feature Extractor

This module extracts linguistic complexity and diversity features from transcripts.
These features capture language development and complexity, which can differ
significantly between ASD and typically developing children.

Features include:
- MLU (Mean Length of Utterance)
- Vocabulary diversity (TTR - Type-Token Ratio)
- Grammatical complexity
- Word category distribution
- Sentence types

Author: Bimidu Gunathilake
"""

import numpy as np
from typing import List, Dict, Any, Set
from collections import Counter

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.helpers import safe_divide, calculate_ratio
from ..base_features import BaseFeatureExtractor, FeatureResult


class LinguisticFeatures(BaseFeatureExtractor):
    """
    Extract linguistic complexity and diversity features.
    
    Features capture:
    - Mean Length of Utterance (MLU) in words and morphemes
    - Vocabulary diversity (Type-Token Ratio)
    - Lexical density
    - Grammatical complexity metrics
    - Word category usage
    
    Example:
        >>> extractor = LinguisticFeatures()
        >>> features = extractor.extract(transcript)
        >>> print(f"MLU: {features.features['mlu_words']}")
    """
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # Length features
            'mlu_words',
            'mlu_morphemes',
            'avg_word_length',
            'max_utterance_length',
            
            # Vocabulary features
            'total_words',
            'unique_words',
            'type_token_ratio',
            'corrected_ttr',  # For longer texts
            
            # Grammatical features
            'noun_ratio',
            'verb_ratio',
            'adjective_ratio',
            'pronoun_ratio',
            'function_word_ratio',
            
            # Complexity metrics
            'lexical_density',
            'utterance_complexity_score',
        ]
    
    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract linguistic features from transcript.
        
        Args:
            transcript: Parsed transcript data
            
        Returns:
            FeatureResult with linguistic features
        """
        features = {}
        
        # Get child utterances only
        child_utterances = self.get_child_utterances(transcript)
        
        if not child_utterances:
            # Return zero features if no valid child utterances
            return FeatureResult(
                features={name: 0.0 for name in self.feature_names},
                feature_type='linguistic',
                metadata={'error': 'No valid child utterances'}
            )
        
        # Extract MLU features
        mlu_features = self._calculate_mlu(child_utterances)
        features.update(mlu_features)
        
        # Extract vocabulary features
        vocab_features = self._calculate_vocabulary_metrics(child_utterances)
        features.update(vocab_features)
        
        # Extract grammatical features (from morphology tier)
        gram_features = self._calculate_grammatical_features(child_utterances)
        features.update(gram_features)
        
        # Calculate complexity metrics
        complexity_features = self._calculate_complexity_metrics(child_utterances)
        features.update(complexity_features)
        
        return FeatureResult(
            features=features,
            feature_type='linguistic',
            metadata={
                'num_utterances': len(child_utterances),
                'has_morphology': any(u.morphology for u in child_utterances)
            }
        )
    
    def _calculate_mlu(self, utterances: List[Utterance]) -> Dict[str, float]:
        """
        Calculate Mean Length of Utterance (MLU) metrics.
        
        MLU is a key measure of language development:
        - MLU in words: average words per utterance
        - MLU in morphemes: average morphemes per utterance
        
        Args:
            utterances: List of child utterances
            
        Returns:
            Dictionary of MLU features
        """
        features = {
            'mlu_words': 0.0,
            'mlu_morphemes': 0.0,
            'avg_word_length': 0.0,
            'max_utterance_length': 0,
        }
        
        if not utterances:
            return features
        
        # Calculate MLU in words
        word_lengths = self.get_utterance_lengths(utterances, in_words=True)
        
        if word_lengths:
            features['mlu_words'] = np.mean(word_lengths)
            features['max_utterance_length'] = max(word_lengths)
        
        # Calculate MLU in morphemes (from %mor tier if available)
        morpheme_counts = []
        word_char_lengths = []
        
        for utterance in utterances:
            if utterance.morphology:
                # Count morphemes in %mor tier
                # Morphemes are typically separated by specific markers
                morphemes = self._count_morphemes(utterance.morphology)
                if morphemes > 0:
                    morpheme_counts.append(morphemes)
            
            # Calculate average word length in characters
            if utterance.tokens:
                # Extract word strings from tokens and calculate their character lengths
                word_lengths = []
                for token in utterance.tokens:
                    if hasattr(token, 'word') and token.word:
                        word_lengths.append(len(token.word))
                if word_lengths:
                    avg_len = np.mean(word_lengths)
                    word_char_lengths.append(avg_len)
        
        if morpheme_counts:
            features['mlu_morphemes'] = np.mean(morpheme_counts)
        
        if word_char_lengths:
            features['avg_word_length'] = np.mean(word_char_lengths)
        
        return features
    
    def _count_morphemes(self, morphology_str: str) -> int:
        """
        Count morphemes from CHAT %mor tier.
        
        %mor format: pos|word-morpheme analysis
        Example: "det|the-Def-Art noun|dog-Plur"
        
        Args:
            morphology_str: Morphology annotation string
            
        Returns:
            Number of morphemes
        """
        if not morphology_str:
            return 0
        
        # Split by whitespace to get individual word analyses
        analyses = morphology_str.split()
        
        morpheme_count = 0
        
        for analysis in analyses:
            # Each word contributes at least 1 morpheme (the stem)
            morpheme_count += 1
            
            # Count additional morphemes (indicated by - separators)
            if '|' in analysis:
                word_part = analysis.split('|', 1)[1]
                # Count morpheme markers (-, ~)
                morpheme_count += word_part.count('-')
                morpheme_count += word_part.count('~')
        
        return morpheme_count
    
    def _calculate_vocabulary_metrics(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate vocabulary diversity metrics.
        
        Type-Token Ratio (TTR) is a measure of lexical diversity:
        - Higher TTR = more diverse vocabulary
        - Lower TTR = more repetitive language (common in ASD)
        
        Args:
            utterances: List of utterances
            
        Returns:
            Dictionary of vocabulary features
        """
        features = {
            'total_words': 0,
            'unique_words': 0,
            'type_token_ratio': 0.0,
            'corrected_ttr': 0.0,
        }
        
        # Collect all words
        all_words = []
        for utterance in utterances:
            if utterance.tokens:
                # Extract word strings from tokens and normalize to lowercase for counting
                words = []
                for token in utterance.tokens:
                    if hasattr(token, 'word') and token.word:
                        words.append(token.word.lower())
                all_words.extend(words)
        
        if not all_words:
            return features
        
        # Calculate metrics
        total_words = len(all_words)
        unique_words = len(set(all_words))
        
        features['total_words'] = total_words
        features['unique_words'] = unique_words
        
        # Basic TTR
        features['type_token_ratio'] = safe_divide(unique_words, total_words)
        
        # Corrected TTR (CTTR) for longer texts
        # CTTR = types / sqrt(2 * tokens)
        features['corrected_ttr'] = safe_divide(
            unique_words,
            np.sqrt(2 * total_words)
        )
        
        return features
    
    def _calculate_grammatical_features(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate grammatical category usage.
        
        Analyzes distribution of word categories from %mor tier:
        - Nouns, verbs, adjectives, pronouns, etc.
        
        Args:
            utterances: List of utterances
            
        Returns:
            Dictionary of grammatical features
        """
        features = {
            'noun_ratio': 0.0,
            'verb_ratio': 0.0,
            'adjective_ratio': 0.0,
            'pronoun_ratio': 0.0,
            'function_word_ratio': 0.0,
        }
        
        # Count word categories from morphology
        category_counts = Counter()
        total_words = 0
        
        for utterance in utterances:
            if not utterance.morphology:
                continue
            
            # Parse morphology to extract POS tags
            analyses = utterance.morphology.split()
            
            for analysis in analyses:
                if '|' in analysis:
                    pos = analysis.split('|')[0]
                    category_counts[pos] += 1
                    total_words += 1
        
        if total_words == 0:
            return features
        
        # Map POS tags to categories
        # Note: CHAT uses specific POS tag conventions
        noun_tags = ['n', 'noun', 'pro', 'propn']
        verb_tags = ['v', 'verb', 'aux', 'cop']
        adj_tags = ['adj', 'adv']
        pronoun_tags = ['pro', 'pron']
        function_tags = ['det', 'prep', 'adp', 'conj', 'part']
        
        # Calculate ratios
        noun_count = sum(category_counts.get(tag, 0) for tag in noun_tags)
        verb_count = sum(category_counts.get(tag, 0) for tag in verb_tags)
        adj_count = sum(category_counts.get(tag, 0) for tag in adj_tags)
        pronoun_count = sum(category_counts.get(tag, 0) for tag in pronoun_tags)
        function_count = sum(category_counts.get(tag, 0) for tag in function_tags)
        
        features['noun_ratio'] = calculate_ratio(noun_count, total_words)
        features['verb_ratio'] = calculate_ratio(verb_count, total_words)
        features['adjective_ratio'] = calculate_ratio(adj_count, total_words)
        features['pronoun_ratio'] = calculate_ratio(pronoun_count, total_words)
        features['function_word_ratio'] = calculate_ratio(function_count, total_words)
        
        return features
    
    def _calculate_complexity_metrics(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate overall complexity metrics.
        
        Lexical density = content words / total words
        High lexical density indicates more complex language
        
        Args:
            utterances: List of utterances
            
        Returns:
            Dictionary of complexity features
        """
        features = {
            'lexical_density': 0.0,
            'utterance_complexity_score': 0.0,
        }
        
        # Collect all tokens
        all_tokens = []
        for utterance in utterances:
            if utterance.tokens:
                # Extract word strings from tokens
                words = []
                for token in utterance.tokens:
                    if hasattr(token, 'word') and token.word:
                        words.append(token.word.lower())
                all_tokens.extend(words)
        
        if not all_tokens:
            return features
        
        # Calculate lexical density (content vs function words)
        # Common function words in English
        function_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with',
            'he', 'she', 'it', 'they', 'we', 'you', 'i',
            'this', 'that', 'these', 'those'
        }
        
        content_words = sum(1 for word in all_tokens if word not in function_words)
        features['lexical_density'] = calculate_ratio(content_words, len(all_tokens))
        
        # Utterance complexity score (composite metric)
        # Combines MLU, TTR, and lexical density
        word_lengths = self.get_utterance_lengths(utterances, in_words=True)
        mlu = np.mean(word_lengths) if word_lengths else 0
        unique_ratio = len(set(all_tokens)) / len(all_tokens)
        
        # Normalized complexity score (0-1 scale)
        complexity = (
            (min(mlu / 10, 1.0) * 0.4) +  # MLU contribution (capped at 10)
            (unique_ratio * 0.3) +          # Vocabulary diversity
            (features['lexical_density'] * 0.3)  # Lexical density
        )
        
        features['utterance_complexity_score'] = complexity
        
        return features


__all__ = ["LinguisticFeatures"]

