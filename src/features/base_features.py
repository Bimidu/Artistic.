"""
Base Feature Extractor Class

This module defines the shared contract and helper utilities used by all
feature extractors in the project.

Design goals:
- Keep extractor implementations consistent (same `extract()` interface)
- Centralize common transcript filtering logic
- Reduce repeated utility code across feature modules

Author: Bimidu Gunathilake
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass

from src.parsers.chat_parser import TranscriptData, Utterance
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureResult:
    """
    Container for feature extraction results.
    
    Attributes:
        features: Dictionary of feature names to values
        feature_type: Type of features (e.g., 'turn_taking', 'linguistic')
        metadata: Additional metadata about the extraction
    """
    features: Dict[str, Any]
    feature_type: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Guarantee `metadata` is always a dictionary for downstream code."""
        if self.metadata is None:
            self.metadata = {}


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    
    Subclasses implement domain-specific feature logic, while this base class
    provides a stable API and reusable transcript utilities.
    """
    
    def __init__(self):
        """Initialize the base feature extractor."""
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract features from a transcript.
        
        Subclasses must return a `FeatureResult` with deterministic keys for
        the selected extractor.
        
        Args:
            transcript: Parsed transcript data
            
        Returns:
            FeatureResult containing extracted features
        """
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """
        Get list of feature names this extractor produces.
        
        Returns:
            List of feature names
        """
        pass
    
    def get_child_utterances(self, transcript: TranscriptData) -> List[Utterance]:
        """
        Get only valid child utterances from transcript.
        
        Args:
            transcript: Transcript to extract from
            
        Returns:
            List of valid child utterances
        """
        # Child-side analyses should ignore invalid utterances to avoid
        # propagating parser artifacts into feature values.
        return [
            u for u in transcript.child_utterances
            if u.is_valid
        ]
    
    def get_adult_utterances(self, transcript: TranscriptData) -> List[Utterance]:
        """
        Get adult utterances (MOT, FAT, INV) from transcript.
        
        Args:
            transcript: Transcript to extract from
            
        Returns:
            List of adult utterances
        """
        # Keep adult-role filtering centralized so category extractors use the
        # same speaker definition.
        adult_codes = ['MOT', 'FAT', 'INV', 'INV1', 'INV2']
        
        return [
            u for u in transcript.utterances
            if u.speaker in adult_codes and u.is_valid
        ]
    
    def get_utterance_lengths(
        self,
        utterances: List[Utterance],
        in_words: bool = True
    ) -> List[int]:
        """
        Get lengths of utterances in a consistent numeric format.
        
        Args:
            utterances: List of utterances
            in_words: If True, count words; if False, count characters
            
        Returns:
            List of utterance lengths (including 0 for empty utterances)
        """
        # Keep empty utterances represented as 0 so vector lengths remain
        # aligned with the original utterance list.
        if in_words:
            return [u.word_count for u in utterances]
        else:
            return [len(u.text) for u in utterances]
    
    def extract_timing_gaps(
        self,
        utterances: List[Utterance]
    ) -> List[float]:
        """
        Extract non-negative temporal gaps between consecutive utterances.
        
        Args:
            utterances: List of utterances with timing information
            
        Returns:
            List of time gaps in seconds
        """
        gaps = []
        
        for i in range(1, len(utterances)):
            prev_end_time = utterances[i-1].end_timing
            curr_time = utterances[i].timing
            
            if prev_end_time is not None and curr_time is not None:
                gap = curr_time - prev_end_time
                # Ignore negative gaps produced by imperfect timestamps.
                if gap >= 0:
                    gaps.append(gap)
        
        return gaps


__all__ = ["BaseFeatureExtractor", "FeatureResult"]

