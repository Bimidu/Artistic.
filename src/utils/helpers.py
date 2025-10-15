"""
Helper Functions and Utilities

This module contains general-purpose utility functions used across
the ASD detection system.

Author: Bimidu Gunathilake
"""

import time
import functools
from typing import Any, Callable, Optional, Union
import numpy as np
from .logger import get_logger

logger = get_logger(__name__)


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log execution time of functions.
    
    This decorator wraps a function and logs how long it took to execute.
    Useful for performance monitoring and optimization.
    
    Args:
        func: Function to be timed
        
    Returns:
        Wrapped function that logs execution time
        
    Example:
        >>> @timing_decorator
        >>> def process_data(data):
        >>>     # ... processing code ...
        >>>     return result
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logger.info(
            f"Function '{func.__name__}' executed in {execution_time:.4f} seconds"
        )
        return result
    
    return wrapper


def safe_divide(
    numerator: Union[int, float],
    denominator: Union[int, float],
    default: float = 0.0
) -> float:
    """
    Safely divide two numbers, handling division by zero.
    
    This function prevents ZeroDivisionError by returning a default value
    when the denominator is zero or very close to zero.
    
    Args:
        numerator: The dividend
        denominator: The divisor
        default: Value to return if division is not possible (default: 0.0)
        
    Returns:
        Result of division or default value
        
    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 0, default=np.nan)
        nan
    """
    # Check for zero or near-zero denominator
    if abs(denominator) < 1e-10:
        return default
    
    return numerator / denominator


def calculate_ratio(
    count: Union[int, float],
    total: Union[int, float],
    percentage: bool = False,
    default: float = 0.0
) -> float:
    """
    Calculate ratio or percentage safely.
    
    Commonly used for calculating feature ratios like:
    - Type-Token Ratio (unique words / total words)
    - Unintelligible ratio (xxx utterances / total utterances)
    - Question ratio (questions / total utterances)
    
    Args:
        count: The count to calculate ratio for
        total: The total count
        percentage: If True, return as percentage (multiply by 100)
        default: Default value if total is zero
        
    Returns:
        Calculated ratio or percentage
        
    Example:
        >>> calculate_ratio(25, 100)
        0.25
        >>> calculate_ratio(25, 100, percentage=True)
        25.0
        >>> calculate_ratio(10, 0)
        0.0
    """
    ratio = safe_divide(count, total, default=default)
    
    if percentage:
        return ratio * 100
    
    return ratio


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.
    
    Performs basic text normalization:
    - Convert to lowercase
    - Strip whitespace
    - Handle None/empty strings
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
        
    Example:
        >>> normalize_text("  Hello World  ")
        'hello world'
        >>> normalize_text(None)
        ''
    """
    if not text:
        return ""
    
    return str(text).strip().lower()


def is_valid_utterance(
    utterance: str,
    min_words: int = 1,
    exclude_markers: bool = True
) -> bool:
    """
    Check if an utterance is valid for analysis.
    
    An utterance is considered invalid if it:
    - Is empty or None
    - Contains only whitespace
    - Contains only non-verbal markers (if exclude_markers=True)
    - Has fewer words than min_words
    
    Args:
        utterance: The utterance text to validate
        min_words: Minimum number of words required
        exclude_markers: Whether to exclude utterances with only markers
        
    Returns:
        True if utterance is valid, False otherwise
        
    Example:
        >>> is_valid_utterance("hello world")
        True
        >>> is_valid_utterance("&=laughs")
        False
        >>> is_valid_utterance("")
        False
    """
    if not utterance or not utterance.strip():
        return False
    
    # Remove CHAT markers for word counting
    if exclude_markers:
        # Remove common CHAT markers: &=, xxx, yyy, www
        clean_text = utterance
        for marker in ['&=', 'xxx', 'yyy', 'www', '@']:
            clean_text = clean_text.replace(marker, '')
        
        # Check if anything remains
        words = clean_text.strip().split()
        if len(words) < min_words:
            return False
    
    return True


def extract_timing_info(timing_str: str) -> Optional[float]:
    """
    Extract timing information from CHAT timing strings.
    
    CHAT files use timing strings like "25:18" for timestamps.
    This function converts them to seconds.
    
    Args:
        timing_str: Timing string in format "MM:SS" or "HH:MM:SS"
        
    Returns:
        Time in seconds, or None if parsing fails
        
    Example:
        >>> extract_timing_info("25:18")
        1518.0
        >>> extract_timing_info("1:30:45")
        5445.0
        >>> extract_timing_info("invalid")
        None
    """
    try:
        parts = timing_str.strip().split(':')
        
        if len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        else:
            logger.warning(f"Invalid timing format: {timing_str}")
            return None
            
    except (ValueError, AttributeError) as e:
        logger.warning(f"Error parsing timing '{timing_str}': {e}")
        return None


def get_age_in_months(age_str: str) -> Optional[int]:
    """
    Convert CHAT age format to months.
    
    CHAT format uses "Y;MM.DD" (e.g., "5;03.10" = 5 years 3 months 10 days)
    
    Args:
        age_str: Age string in CHAT format
        
    Returns:
        Age in months (rounded), or None if parsing fails
        
    Example:
        >>> get_age_in_months("5;03.10")
        63
        >>> get_age_in_months("2;06.00")
        30
    """
    try:
        # Parse format: Y;MM.DD
        parts = age_str.strip().split(';')
        if len(parts) != 2:
            return None
            
        years = int(parts[0])
        months_days = parts[1].split('.')
        months = int(months_days[0])
        
        total_months = years * 12 + months
        return total_months
        
    except (ValueError, IndexError, AttributeError) as e:
        logger.warning(f"Error parsing age '{age_str}': {e}")
        return None


__all__ = [
    "timing_decorator",
    "safe_divide",
    "calculate_ratio",
    "normalize_text",
    "is_valid_utterance",
    "extract_timing_info",
    "get_age_in_months",
]

