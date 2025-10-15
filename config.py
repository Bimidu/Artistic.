"""
Configuration Module for ASD Detection System

This module handles all configuration settings for the ASD detection system,
including paths, feature extraction parameters, and system settings.

Author: Bimidu Gunathilake
Date: 2024
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class PathConfig:
    """
    Configuration for file paths and directories.
    
    Attributes:
        data_dir: Root directory containing all datasets
        output_dir: Directory for processed data and results
        models_dir: Directory for saved ML models
        logs_dir: Directory for log files
        cache_dir: Directory for cached intermediate results
    """
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    output_dir: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "./output")))
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("MODELS_DIR", "./models")))
    logs_dir: Path = field(default_factory=lambda: Path(os.getenv("LOGS_DIR", "./logs")))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", "./cache")))
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.output_dir, self.models_dir, self.logs_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureConfig:
    """
    Configuration for feature extraction parameters.
    
    Attributes:
        min_utterances: Minimum number of utterances required per transcript
        min_words: Minimum number of words for a valid utterance
        window_size: Time window for analysis in seconds
        overlap: Overlap between windows (0.0 to 1.0)
        calculate_mor: Whether to calculate morphological features
        calculate_timing: Whether to extract timing features
    """
    min_utterances: int = int(os.getenv("MIN_UTTERANCES", "10"))
    min_words: int = int(os.getenv("MIN_WORDS", "5"))
    window_size: int = int(os.getenv("WINDOW_SIZE", "300"))
    overlap: float = 0.5
    calculate_mor: bool = True
    calculate_timing: bool = True


@dataclass
class ProcessingConfig:
    """
    Configuration for data processing and system settings.
    
    Attributes:
        n_jobs: Number of parallel jobs for multiprocessing
        batch_size: Batch size for batch processing
        cache_enabled: Whether to enable caching of processed data
        verbose: Verbosity level (0=silent, 1=progress bars, 2=detailed)
    """
    n_jobs: int = int(os.getenv("N_JOBS", "4"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    verbose: int = 1


@dataclass
class LoggingConfig:
    """
    Configuration for logging settings.
    
    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format string
        rotation: Log file rotation settings (e.g., "500 MB")
        retention: Log file retention period (e.g., "10 days")
    """
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format: str = os.getenv(
        "LOG_FORMAT",
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    rotation: str = "500 MB"
    retention: str = "10 days"


@dataclass
class DatasetConfig:
    """
    Configuration for dataset-specific settings.
    
    Attributes:
        datasets: List of dataset names to process
        diagnosis_mapping: Mapping of diagnosis codes to standardized labels
        speaker_roles: Valid speaker role identifiers
    """
    datasets: list = field(default_factory=lambda: [
        "asdbank_aac",
        "asdbank_eigsti", 
        "asdbank_flusberg",
        "asdbank_nadig",
        "asdbank_quigley_mcnalley",
        "asdbank_rollins"
    ])
    
    # Standardize diagnosis labels across datasets
    diagnosis_mapping: Dict[str, str] = field(default_factory=lambda: {
        "ASD": "ASD",
        "TD": "TD",
        "TYP": "TD",  # Typically developing
        "DD": "DD",   # Developmental delay
        "HR": "HR",   # High risk
        "LR": "LR",   # Low risk
    })
    
    # Valid speaker roles for analysis
    speaker_roles: list = field(default_factory=lambda: [
        "CHI",   # Target child
        "MOT",   # Mother
        "FAT",   # Father
        "INV",   # Investigator
        "INV1",  # Investigator 1
        "INV2",  # Investigator 2
        "AAC",   # AAC device
    ])


class Config:
    """
    Main configuration class that aggregates all configuration settings.
    
    This class provides a centralized access point for all configuration
    parameters used throughout the application.
    """
    
    def __init__(self):
        """Initialize all configuration sections."""
        self.paths = PathConfig()
        self.features = FeatureConfig()
        self.processing = ProcessingConfig()
        self.logging = LoggingConfig()
        self.datasets = DatasetConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            "paths": self.paths.__dict__,
            "features": self.features.__dict__,
            "processing": self.processing.__dict__,
            "logging": self.logging.__dict__,
            "datasets": self.datasets.__dict__,
        }
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(paths={self.paths}, features={self.features})"


# Create global configuration instance
config = Config()


# Export commonly used configurations
__all__ = [
    "config",
    "PathConfig",
    "FeatureConfig",
    "ProcessingConfig",
    "LoggingConfig",
    "DatasetConfig",
]

