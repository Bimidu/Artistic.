"""
Conversational Repair Detection Feature Extractor (Section 3.3.4)

This module extracts features related to conversational repair strategies,
which can indicate communication difficulties in children with ASD.
Based on methodology section 3.3.4.

Features implemented:
- Self-repair (speaker rephrases mid-utterance)
- Other-repair (response to clarification request)
- Clarification requests
- Repair frequency and success rate
- Repair effectiveness scoring

References:
- Repair attempts include self-repairs, clarification requests, and repetitions
- Effectiveness measured by semantic overlap with original prompt

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

# Try to import spaCy for semantic analysis
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Repair effectiveness analysis will be limited.")


class RepairDetectionFeatures(BaseFeatureExtractor):
    """
    Extract conversational repair features from transcripts (Section 3.3.4).
    
    Features capture:
    - Self-repair patterns (mid-utterance corrections)
    - Other-initiated repair (responding to clarification)
    - Clarification request patterns
    - Repair frequency and success rates
    - Repair strategy types
    
    Repair types:
    - Self-initiated self-repair: Speaker corrects own speech
    - Other-initiated self-repair: Speaker corrects after listener signals trouble
    - Self-initiated other-repair: Speaker asks listener to clarify
    - Other-initiated other-repair: Listener corrects speaker
    
    Example:
        >>> extractor = RepairDetectionFeatures()
        >>> features = extractor.extract(transcript)
        >>> print(features.features['repair_success_rate'])
    """
    
    # Self-repair markers (speaker corrects themselves)
    SELF_REPAIR_PATTERNS = [
        r'\bi mean\b',
        r'\bno wait\b',
        r'\bsorry\b',
        r'\bactually\b',
        r'\bno\s+i\s+mean\b',
        r'\bwell\s+not\b',
        r'\bor\s+rather\b',
        r'\blet me\s+rephrase\b',
    ]
    
    # CHAT retrace markers indicate self-correction
    CHAT_RETRACE_MARKERS = [
        r'\[/\]',      # Retrace without correction
        r'\[//\]',     # Retrace with correction
        r'\[///\]',    # Reformulation
        r'\[\?\]',     # Best guess
    ]
    
    # Clarification request markers
    CLARIFICATION_PATTERNS = [
        r'\bwhat\?',
        r'\bhuh\?',
        r'\bpardon\?',
        r'\bexcuse me\?',
        r'\bsay again\b',
        r'\bwhat did you\b',
        r'\bcan you repeat\b',
        r'\bi don\'?t understand\b',
        r'\bwhat do you mean\b',
        r'\bsorry\?',
    ]
    
    # Confirmation check patterns
    CONFIRMATION_PATTERNS = [
        r'\bdo you mean\b',
        r'\bso you\b',
        r'\blike\s+a\b',
        r'\byou mean\b',
        r'\bis that\b',
        r'\bright\?',
        r'\bokay\?',
    ]
    
    # Acknowledgment patterns (repair uptake)
    ACKNOWLEDGMENT_PATTERNS = [
        r'\boh\b',
        r'\bi see\b',
        r'\bokay\b',
        r'\byes\b',
        r'\boh okay\b',
        r'\bah\b',
        r'\bgo on\b',
        r'\bi got it\b',
    ]
    
    def __init__(self):
        """Initialize repair detection extractor."""
        super().__init__()
        self._nlp = None
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize spaCy for semantic analysis of repair effectiveness."""
        if not SPACY_AVAILABLE:
            return
        
        try:
            self._nlp = spacy.load("en_core_web_md")
            logger.info("Loaded spaCy model for repair effectiveness analysis")
        except OSError:
            try:
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy en_core_web_sm (limited semantic analysis)")
            except OSError:
                logger.warning("No spaCy model available for repair analysis")
                self._nlp = None
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [
            # Self-repair features
            'self_repair_count',
            'self_repair_ratio',
            'child_self_repair_count',
            'child_self_repair_ratio',
            'adult_self_repair_count',
            'retrace_count',
            'reformulation_count',
            
            # Other-initiated repair
            'other_initiated_repair_count',
            'child_repair_after_clarification',
            'adult_repair_after_clarification',
            
            # Clarification request features
            'clarification_request_count',
            'clarification_request_ratio',
            'child_clarification_count',
            'adult_clarification_count',
            'clarification_to_child_count',  # Adults asking child to clarify
            'clarification_to_adult_count',  # Child asking adults to clarify
            
            # Confirmation checks
            'confirmation_check_count',
            'child_confirmation_check_count',
            
            # Repetition-based repair
            'repetition_repair_count',
            'partial_repetition_count',
            'exact_repetition_count',
            'expansion_repair_count',  # Expanding on previous unclear utterance
            
            # Repair success metrics
            'repair_success_count',
            'repair_success_rate',
            'repair_failure_count',
            'repair_attempt_rate',
            
            # Repair sequences
            'avg_repair_sequence_length',
            'max_repair_sequence_length',
            'extended_repair_count',  # Repairs requiring multiple attempts
            
            # Acknowledgment after repair
            'repair_acknowledgment_count',
            'repair_uptake_ratio',
            
            # Child-specific repair effectiveness
            'child_repair_effectiveness',
            'child_needs_repair_ratio',
            'child_provides_repair_ratio',
            
            # Repair strategy diversity
            'repair_strategy_diversity',
            'dominant_repair_strategy',
            
            # Communication breakdown indicators
            'breakdown_count',
            'breakdown_resolution_rate',
            'unresolved_breakdown_count',
        ]
    
    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract repair detection features from transcript.
        
        Args:
            transcript: Parsed transcript data
            
        Returns:
            FeatureResult with repair detection features
        """
        features = {}
        
        all_utterances = transcript.valid_utterances
        child_utterances = self.get_child_utterances(transcript)
        adult_utterances = self.get_adult_utterances(transcript)
        
        logger.debug(f"Extracting repair features from {len(all_utterances)} utterances")
        
        # Self-repair features
        self_repair_features = self._calculate_self_repair(
            all_utterances, child_utterances, adult_utterances
        )
        features.update(self_repair_features)
        
        # Other-initiated repair
        other_repair_features = self._calculate_other_initiated_repair(all_utterances)
        features.update(other_repair_features)
        
        # Clarification request features
        clarification_features = self._calculate_clarification_requests(
            all_utterances, child_utterances, adult_utterances
        )
        features.update(clarification_features)
        
        # Confirmation check features
        confirmation_features = self._calculate_confirmation_checks(
            all_utterances, child_utterances
        )
        features.update(confirmation_features)
        
        # Repetition-based repair
        repetition_features = self._calculate_repetition_repairs(all_utterances)
        features.update(repetition_features)
        
        # Repair success metrics
        success_features = self._calculate_repair_success(all_utterances)
        features.update(success_features)
        
        # Repair sequences
        sequence_features = self._calculate_repair_sequences(all_utterances)
        features.update(sequence_features)
        
        # Child-specific repair effectiveness
        child_features = self._calculate_child_repair_effectiveness(
            all_utterances, child_utterances
        )
        features.update(child_features)
        
        # Repair strategy diversity
        strategy_features = self._calculate_repair_strategy_diversity(all_utterances)
        features.update(strategy_features)
        
        # Communication breakdown
        breakdown_features = self._calculate_communication_breakdowns(all_utterances)
        features.update(breakdown_features)
        
        logger.debug(f"Extracted {len(features)} repair detection features")
        
        return FeatureResult(
            features=features,
            feature_type='repair_detection',
            metadata={
                'total_utterances': len(all_utterances),
                'child_utterances': len(child_utterances),
                'has_spacy': self._nlp is not None
            }
        )
    
    def _calculate_self_repair(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance],
        adult_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate self-repair features.
        
        Self-repair = speaker corrects their own speech mid-utterance.
        """
        features = {
            'self_repair_count': 0,
            'self_repair_ratio': 0.0,
            'child_self_repair_count': 0,
            'child_self_repair_ratio': 0.0,
            'adult_self_repair_count': 0,
            'retrace_count': 0,
            'reformulation_count': 0,
        }
        
        if not all_utterances:
            return features
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        total_repairs = 0
        child_repairs = 0
        adult_repairs = 0
        retrace_count = 0
        reformulation_count = 0
        
        for u in all_utterances:
            text = u.text.lower()
            repair_found = False
            
            # Check for linguistic self-repair markers
            for pattern in self.SELF_REPAIR_PATTERNS:
                if re.search(pattern, text):
                    repair_found = True
                    break
            
            # Check for CHAT retrace markers
            for pattern in self.CHAT_RETRACE_MARKERS:
                matches = len(re.findall(pattern, u.text))
                if matches > 0:
                    repair_found = True
                    if pattern in [r'\[/\]', r'\[//\]']:
                        retrace_count += matches
                    elif pattern == r'\[///\]':
                        reformulation_count += matches
            
            if repair_found:
                total_repairs += 1
                if u.speaker == 'CHI':
                    child_repairs += 1
                elif u.speaker in adult_codes:
                    adult_repairs += 1
        
        features['self_repair_count'] = total_repairs
        features['self_repair_ratio'] = total_repairs / len(all_utterances)
        features['child_self_repair_count'] = child_repairs
        features['child_self_repair_ratio'] = (
            child_repairs / len(child_utterances) if child_utterances else 0.0
        )
        features['adult_self_repair_count'] = adult_repairs
        features['retrace_count'] = retrace_count
        features['reformulation_count'] = reformulation_count
        
        return features
    
    def _calculate_other_initiated_repair(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate other-initiated repair features.
        
        Other-initiated repair = speaker repairs after listener signals trouble.
        """
        features = {
            'other_initiated_repair_count': 0,
            'child_repair_after_clarification': 0,
            'adult_repair_after_clarification': 0,
        }
        
        if len(utterances) < 2:
            return features
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        other_initiated = 0
        child_after_clar = 0
        adult_after_clar = 0
        
        for i in range(1, len(utterances)):
            prev = utterances[i - 1]
            curr = utterances[i]
            prev_text = prev.text.lower()
            
            # Check if previous turn was a clarification request
            is_clarification = any(
                re.search(pattern, prev_text)
                for pattern in self.CLARIFICATION_PATTERNS
            )
            
            if is_clarification:
                # Current turn is likely a repair attempt
                other_initiated += 1
                
                if curr.speaker == 'CHI':
                    child_after_clar += 1
                elif curr.speaker in adult_codes:
                    adult_after_clar += 1
        
        features['other_initiated_repair_count'] = other_initiated
        features['child_repair_after_clarification'] = child_after_clar
        features['adult_repair_after_clarification'] = adult_after_clar
        
        return features
    
    def _calculate_clarification_requests(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance],
        adult_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate clarification request features."""
        features = {
            'clarification_request_count': 0,
            'clarification_request_ratio': 0.0,
            'child_clarification_count': 0,
            'adult_clarification_count': 0,
            'clarification_to_child_count': 0,
            'clarification_to_adult_count': 0,
        }
        
        if not all_utterances:
            return features
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        total_clarifications = 0
        child_clarifications = 0
        adult_clarifications = 0
        
        for i, u in enumerate(all_utterances):
            text = u.text.lower()
            
            is_clarification = any(
                re.search(pattern, text)
                for pattern in self.CLARIFICATION_PATTERNS
            )
            
            if is_clarification:
                total_clarifications += 1
                
                if u.speaker == 'CHI':
                    child_clarifications += 1
                elif u.speaker in adult_codes:
                    adult_clarifications += 1
        
        features['clarification_request_count'] = total_clarifications
        features['clarification_request_ratio'] = total_clarifications / len(all_utterances)
        features['child_clarification_count'] = child_clarifications
        features['adult_clarification_count'] = adult_clarifications
        
        # Determine direction of clarification requests
        for i in range(1, len(all_utterances)):
            curr = all_utterances[i]
            prev = all_utterances[i - 1]
            text = curr.text.lower()
            
            is_clarification = any(
                re.search(pattern, text)
                for pattern in self.CLARIFICATION_PATTERNS
            )
            
            if is_clarification:
                # Who is being asked to clarify?
                if curr.speaker in adult_codes and prev.speaker == 'CHI':
                    features['clarification_to_child_count'] += 1
                elif curr.speaker == 'CHI' and prev.speaker in adult_codes:
                    features['clarification_to_adult_count'] += 1
        
        return features
    
    def _calculate_confirmation_checks(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate confirmation check features."""
        features = {
            'confirmation_check_count': 0,
            'child_confirmation_check_count': 0,
        }
        
        total_confirmations = 0
        child_confirmations = 0
        
        for u in all_utterances:
            text = u.text.lower()
            
            is_confirmation = any(
                re.search(pattern, text)
                for pattern in self.CONFIRMATION_PATTERNS
            )
            
            if is_confirmation:
                total_confirmations += 1
                if u.speaker == 'CHI':
                    child_confirmations += 1
        
        features['confirmation_check_count'] = total_confirmations
        features['child_confirmation_check_count'] = child_confirmations
        
        return features
    
    def _calculate_repetition_repairs(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate repetition-based repair features.
        
        Includes exact repetition, partial repetition, and expansion.
        """
        features = {
            'repetition_repair_count': 0,
            'partial_repetition_count': 0,
            'exact_repetition_count': 0,
            'expansion_repair_count': 0,
        }
        
        if len(utterances) < 2:
            return features
        
        total_repetitions = 0
        partial_reps = 0
        exact_reps = 0
        expansions = 0
        
        for i in range(1, len(utterances)):
            prev = utterances[i - 1]
            curr = utterances[i]
            
            # Skip if same speaker (not a repair situation)
            if prev.speaker == curr.speaker:
                continue
            
            prev_words = set(prev.text.lower().split())
            curr_words = set(curr.text.lower().split())
            
            if not prev_words or not curr_words:
                continue
            
            # Calculate overlap
            overlap = prev_words & curr_words
            overlap_ratio = len(overlap) / len(prev_words)
            
            # Exact repetition
            if prev.text.lower().strip() == curr.text.lower().strip():
                exact_reps += 1
                total_repetitions += 1
            # Partial repetition (significant overlap)
            elif overlap_ratio > 0.5:
                partial_reps += 1
                total_repetitions += 1
                
                # Expansion (current is longer with overlap)
                if len(curr_words) > len(prev_words) and overlap_ratio > 0.3:
                    expansions += 1
        
        features['repetition_repair_count'] = total_repetitions
        features['partial_repetition_count'] = partial_reps
        features['exact_repetition_count'] = exact_reps
        features['expansion_repair_count'] = expansions
        
        return features
    
    def _calculate_repair_success(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """
        Calculate repair success metrics.
        
        Success = repair followed by acknowledgment or topic continuation.
        """
        features = {
            'repair_success_count': 0,
            'repair_success_rate': 0.0,
            'repair_failure_count': 0,
            'repair_attempt_rate': 0.0,
        }
        
        if len(utterances) < 3:
            return features
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        repair_attempts = 0
        successful_repairs = 0
        
        for i in range(1, len(utterances) - 1):
            prev = utterances[i - 1]
            curr = utterances[i]
            next_u = utterances[i + 1]
            
            prev_text = prev.text.lower()
            next_text = next_u.text.lower()
            
            # Check if previous was clarification request
            is_clarification = any(
                re.search(pattern, prev_text)
                for pattern in self.CLARIFICATION_PATTERNS
            )
            
            if is_clarification:
                repair_attempts += 1
                
                # Check if next turn is acknowledgment (success)
                is_acknowledgment = any(
                    re.search(pattern, next_text)
                    for pattern in self.ACKNOWLEDGMENT_PATTERNS
                )
                
                # Or if next turn continues topic (success)
                if is_acknowledgment:
                    successful_repairs += 1
                else:
                    # Check semantic similarity if spaCy available
                    if self._nlp is not None:
                        try:
                            curr_doc = self._nlp(curr.text)
                            next_doc = self._nlp(next_u.text)
                            if curr_doc.vector_norm > 0 and next_doc.vector_norm > 0:
                                sim = curr_doc.similarity(next_doc)
                                if sim > 0.3:  # Topic continues
                                    successful_repairs += 1
                        except Exception:
                            pass
        
        features['repair_success_count'] = successful_repairs
        features['repair_failure_count'] = repair_attempts - successful_repairs
        features['repair_success_rate'] = (
            successful_repairs / repair_attempts if repair_attempts > 0 else 0.0
        )
        features['repair_attempt_rate'] = repair_attempts / len(utterances)
        
        return features
    
    def _calculate_repair_sequences(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate repair sequence features."""
        features = {
            'avg_repair_sequence_length': 0.0,
            'max_repair_sequence_length': 0,
            'extended_repair_count': 0,
            'repair_acknowledgment_count': 0,
            'repair_uptake_ratio': 0.0,
        }
        
        if len(utterances) < 2:
            return features
        
        # Identify repair sequences
        sequence_lengths = []
        current_sequence = 0
        in_repair_sequence = False
        acknowledgments = 0
        repair_occasions = 0
        
        for i, u in enumerate(utterances):
            text = u.text.lower()
            
            # Check for clarification (starts sequence)
            is_clarification = any(
                re.search(pattern, text)
                for pattern in self.CLARIFICATION_PATTERNS
            )
            
            # Check for acknowledgment (ends sequence)
            is_acknowledgment = any(
                re.search(pattern, text)
                for pattern in self.ACKNOWLEDGMENT_PATTERNS
            )
            
            if is_clarification:
                if not in_repair_sequence:
                    in_repair_sequence = True
                    current_sequence = 1
                    repair_occasions += 1
                else:
                    current_sequence += 1
            elif is_acknowledgment and in_repair_sequence:
                current_sequence += 1
                sequence_lengths.append(current_sequence)
                in_repair_sequence = False
                current_sequence = 0
                acknowledgments += 1
            elif in_repair_sequence:
                current_sequence += 1
        
        # Handle unclosed sequences
        if in_repair_sequence and current_sequence > 0:
            sequence_lengths.append(current_sequence)
        
        if sequence_lengths:
            features['avg_repair_sequence_length'] = float(np.mean(sequence_lengths))
            features['max_repair_sequence_length'] = max(sequence_lengths)
            features['extended_repair_count'] = sum(1 for s in sequence_lengths if s > 2)
        
        features['repair_acknowledgment_count'] = acknowledgments
        features['repair_uptake_ratio'] = (
            acknowledgments / repair_occasions if repair_occasions > 0 else 0.0
        )
        
        return features
    
    def _calculate_child_repair_effectiveness(
        self,
        all_utterances: List[Utterance],
        child_utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate child-specific repair effectiveness."""
        features = {
            'child_repair_effectiveness': 0.0,
            'child_needs_repair_ratio': 0.0,
            'child_provides_repair_ratio': 0.0,
        }
        
        if not child_utterances or len(all_utterances) < 2:
            return features
        
        adult_codes = {'MOT', 'FAT', 'INV', 'INV1', 'INV2', 'EXA', 'EXP'}
        
        child_needs_repair = 0  # Adults ask child to clarify
        child_provides_repair = 0  # Child successfully repairs
        child_repair_success = 0
        
        for i in range(1, len(all_utterances)):
            prev = all_utterances[i - 1]
            curr = all_utterances[i]
            prev_text = prev.text.lower()
            
            # Adult requests clarification from child
            if prev.speaker in adult_codes and curr.speaker == 'CHI':
                is_clarification = any(
                    re.search(pattern, prev_text)
                    for pattern in self.CLARIFICATION_PATTERNS
                )
                
                if is_clarification:
                    child_needs_repair += 1
                    child_provides_repair += 1
                    
                    # Check if next turn (after child repair) is acknowledgment
                    if i + 1 < len(all_utterances):
                        next_text = all_utterances[i + 1].text.lower()
                        is_ack = any(
                            re.search(pattern, next_text)
                            for pattern in self.ACKNOWLEDGMENT_PATTERNS
                        )
                        if is_ack:
                            child_repair_success += 1
        
        features['child_needs_repair_ratio'] = (
            child_needs_repair / len(child_utterances) if child_utterances else 0.0
        )
        features['child_provides_repair_ratio'] = (
            child_provides_repair / child_needs_repair if child_needs_repair > 0 else 0.0
        )
        features['child_repair_effectiveness'] = (
            child_repair_success / child_provides_repair if child_provides_repair > 0 else 0.0
        )
        
        return features
    
    def _calculate_repair_strategy_diversity(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate repair strategy diversity."""
        features = {
            'repair_strategy_diversity': 0.0,
            'dominant_repair_strategy': 0,  # 0=none, 1=repetition, 2=reformulation, 3=clarification
        }
        
        strategy_counts = {
            'repetition': 0,
            'reformulation': 0,
            'clarification': 0,
            'expansion': 0,
        }
        
        for u in utterances:
            text = u.text.lower()
            
            # Check for different strategies
            if any(re.search(p, text) for p in self.CLARIFICATION_PATTERNS):
                strategy_counts['clarification'] += 1
            
            if any(re.search(p, text) for p in self.SELF_REPAIR_PATTERNS):
                strategy_counts['reformulation'] += 1
            
            # Check for CHAT markers
            if '[//]' in u.text or '[///]' in u.text:
                strategy_counts['reformulation'] += 1
            elif '[/]' in u.text:
                strategy_counts['repetition'] += 1
        
        # Calculate diversity (number of strategies used / total possible)
        strategies_used = sum(1 for v in strategy_counts.values() if v > 0)
        features['repair_strategy_diversity'] = strategies_used / len(strategy_counts)
        
        # Find dominant strategy
        if strategy_counts:
            dominant = max(strategy_counts.items(), key=lambda x: x[1])
            strategy_map = {'repetition': 1, 'reformulation': 2, 'clarification': 3, 'expansion': 4}
            features['dominant_repair_strategy'] = strategy_map.get(dominant[0], 0)
        
        return features
    
    def _calculate_communication_breakdowns(
        self,
        utterances: List[Utterance]
    ) -> Dict[str, float]:
        """Calculate communication breakdown indicators."""
        features = {
            'breakdown_count': 0,
            'breakdown_resolution_rate': 0.0,
            'unresolved_breakdown_count': 0,
        }
        
        if len(utterances) < 2:
            return features
        
        breakdowns = 0
        resolved = 0
        
        for i in range(len(utterances) - 1):
            curr = utterances[i]
            next_u = utterances[i + 1]
            curr_text = curr.text.lower()
            next_text = next_u.text.lower()
            
            # Breakdown indicators: clarification request
            is_breakdown = any(
                re.search(pattern, next_text)
                for pattern in self.CLARIFICATION_PATTERNS
            )
            
            if is_breakdown:
                breakdowns += 1
                
                # Check for resolution in subsequent turns
                if i + 2 < len(utterances):
                    followup = utterances[i + 2].text.lower()
                    is_resolved = any(
                        re.search(pattern, followup)
                        for pattern in self.ACKNOWLEDGMENT_PATTERNS
                    )
                    if is_resolved:
                        resolved += 1
        
        features['breakdown_count'] = breakdowns
        features['unresolved_breakdown_count'] = breakdowns - resolved
        features['breakdown_resolution_rate'] = (
            resolved / breakdowns if breakdowns > 0 else 0.0
        )
        
        return features


__all__ = ["RepairDetectionFeatures"]


