"""
Base Feature Extractor Class

This module provides the base class for all feature extractors,
defining the common interface and utility methods.

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
        if self.metadata is None:
            self.metadata = {}


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.
    
    This class defines the interface that all feature extractors must implement
    and provides common utility methods for feature extraction.
    """
    
    def __init__(self):
        """Initialize the base feature extractor."""
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract features from a transcript.
        
        This method must be implemented by all subclasses.
        
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
        adult_codes = ['MOT', 'FAT', 'INV', 'INV1', 'INV2']
        
        return [
            u for u in transcript.utterances
            if u.speaker in adult_codes and u.is_valid
        ]
    
    def count_pattern(
        self,
        utterances: List[Utterance],
        pattern: str,
        case_sensitive: bool = False
    ) -> int:
        """
        Count occurrences of a pattern in utterances.
        
        Args:
            utterances: List of utterances to search
            pattern: Pattern to search for
            case_sensitive: Whether search is case-sensitive
            
        Returns:
            Count of pattern occurrences
        """
        count = 0
        
        for utterance in utterances:
            text = utterance.text if case_sensitive else utterance.text.lower()
            search_pattern = pattern if case_sensitive else pattern.lower()
            
            if search_pattern in text:
                count += 1
        
        return count
    
    def get_utterance_lengths(
        self,
        utterances: List[Utterance],
        in_words: bool = True
    ) -> List[int]:
        """
        Get lengths of all utterances.
        
        Args:
            utterances: List of utterances
            in_words: If True, count words; if False, count characters
            
        Returns:
            List of utterance lengths
        """
        if in_words:
            return [len(u.tokens) for u in utterances if u.tokens]
        else:
            return [len(u.text) for u in utterances]
    
    def extract_timing_gaps(
        self,
        utterances: List[Utterance]
    ) -> List[float]:
        """
        Extract time gaps between consecutive utterances.
        
        Args:
            utterances: List of utterances with timing information
            
        Returns:
            List of time gaps in seconds
        """
        gaps = []
        
        for i in range(1, len(utterances)):
            prev_time = utterances[i-1].timing
            curr_time = utterances[i].timing
            
            if prev_time is not None and curr_time is not None:
                gap = curr_time - prev_time
                if gap >= 0:  # Ignore negative gaps (timing errors)
                    gaps.append(gap)
        
        return gaps


__all__ = ["BaseFeatureExtractor", "FeatureResult"]

