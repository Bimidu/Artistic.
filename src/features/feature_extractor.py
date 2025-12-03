"""
Main Feature Extraction Orchestrator

This module coordinates all three categories of feature extractors:
1. Acoustic & Prosodic (placeholder - Team Member A)
2. Syntactic & Semantic (placeholder - Team Member B)
3. Pragmatic & Conversational (fully implemented)

Author: Bimidu Gunathilake
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.parsers.chat_parser import TranscriptData
from src.utils.logger import get_logger
from src.utils.helpers import timing_decorator

# Pragmatic & Conversational Features 
from .pragmatic_conversational import (
    TurnTakingFeatures,
    LinguisticFeatures,
    PragmaticFeatures,
    ConversationalFeatures,
)

# Placeholder modules for other team members
# Temporarily disabled: acoustic_prosodic module not fully implemented
# from .acoustic_prosodic import AcousticProsodicFeatures

try:
    from .syntactic_semantic import SyntacticSemanticFeatures
except ImportError:
    SyntacticSemanticFeatures = None

logger = get_logger(__name__)


@dataclass
class FeatureSet:
    """
    Complete feature set extracted from a transcript.
    
    Contains features from all three categories:
    - Acoustic & Prosodic (Team Member A)
    - Syntactic & Semantic (Team Member B)
    - Pragmatic & Conversational (Fully Implemented)
    
    Attributes:
        participant_id: Participant identifier
        file_path: Source transcript file
        diagnosis: Clinical diagnosis
        age_months: Age in months
        features: Dictionary of all extracted features
        metadata: Additional metadata
        feature_categories: Which categories were extracted
    """
    participant_id: str
    file_path: Path
    diagnosis: Optional[str] = None
    age_months: Optional[int] = None
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_categories: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'participant_id': self.participant_id,
            'file_path': str(self.file_path),
            'diagnosis': self.diagnosis,
            'age_months': self.age_months,
            **self.features,
        }
    
    def to_series(self) -> pd.Series:
        """Convert to pandas Series."""
        return pd.Series(self.to_dict())


class FeatureExtractor:
    """
    Main feature extraction orchestrator for all three feature categories.
    
    This class coordinates:
    1. Acoustic & Prosodic extractors (placeholder)
    2. Syntactic & Semantic extractors (placeholder)
    3. Pragmatic & Conversational extractors (fully implemented)
    
    Example:
        >>> # Extract only pragmatic/conversational features
        >>> extractor = FeatureExtractor(
        ...     categories=['pragmatic_conversational']
        ... )
        >>> feature_set = extractor.extract_from_transcript(transcript)
        
        >>> # Extract all features (when all modules are ready)
        >>> extractor = FeatureExtractor(categories='all')
        >>> feature_set = extractor.extract_from_transcript(transcript)
    """
    
    # Define the three main feature categories
    FEATURE_CATEGORIES = {
        'acoustic_prosodic': {
            'description': 'Acoustic and prosodic features from audio',
            'status': 'placeholder',
            'team': 'Team Member A',
        },
        'syntactic_semantic': {
            'description': 'Syntactic and semantic features from text',
            'status': 'placeholder',
            'team': 'Team Member B',
        },
        'pragmatic_conversational': {
            'description': 'Pragmatic and conversational features',
            'status': 'implemented',
            'team': 'Current Implementation',
        },
    }
    
    def __init__(
        self,
        categories: Optional[List[str] | str] = 'pragmatic_conversational',
        include_placeholders: bool = True
    ):
        """
        Initialize feature extractor with specified categories.
        
        Args:
            categories: Which feature categories to extract:
                       - 'all': Extract all three categories
                       - 'pragmatic_conversational': Only pragmatic features (default)
                       - List of category names to extract
            include_placeholders: If True, include placeholder modules (zeros)
                                If False, skip unimplemented modules
        
        Example:
            >>> # Only pragmatic/conversational (recommended for now)
            >>> extractor = FeatureExtractor()
            
            >>> # All categories (with placeholders)
            >>> extractor = FeatureExtractor(categories='all')
            
            >>> # Specific categories
            >>> extractor = FeatureExtractor(
            ...     categories=['pragmatic_conversational', 'syntactic_semantic']
            ... )
        """
        self.include_placeholders = include_placeholders
        
        # Determine which categories to use
        if categories == 'all':
            # Exclude acoustic_prosodic as it's temporarily disabled
            available_categories = [
                cat for cat in self.FEATURE_CATEGORIES.keys()
                if cat != 'acoustic_prosodic'  # Temporarily disabled
            ]
            self.active_categories = available_categories
            logger.info(
                "Acoustic & Prosodic features are temporarily disabled. "
                "Excluded from 'all' categories."
            )
        elif isinstance(categories, str):
            # If specifically requesting acoustic_prosodic, warn but allow (will be skipped)
            if categories == 'acoustic_prosodic':
                logger.warning(
                    "Acoustic & Prosodic features are temporarily disabled. "
                    "This category will be skipped."
                )
            self.active_categories = [categories]
        else:
            # Filter out acoustic_prosodic from list if present
            filtered = []
            for cat in categories:
                if cat == 'acoustic_prosodic':
                    logger.warning(
                        "Acoustic & Prosodic features are temporarily disabled. "
                        "Skipping this category."
                    )
                else:
                    filtered.append(cat)
            self.active_categories = filtered
        
        # Initialize extractors
        self._initialize_extractors()
        
        logger.info(
            f"FeatureExtractor initialized with categories: {self.active_categories}"
        )
        logger.info(
            f"Include placeholders: {include_placeholders}"
        )
    
    def _initialize_extractors(self):
        """Initialize all feature extractors based on active categories."""
        self.extractors = {}
        
        # CATEGORY 1: Acoustic & Prosodic (TEMPORARILY DISABLED)
        # Note: This should not be in active_categories due to filtering in __init__,
        # but keeping as safety check
        if 'acoustic_prosodic' in self.active_categories:
            logger.warning(
                "Acoustic & Prosodic features are temporarily disabled. "
                "Skipping initialization."
            )
            # Temporarily disabled - AcousticProsodicFeatures class not implemented
        
        # CATEGORY 2: Syntactic & Semantic (PLACEHOLDER)
        if 'syntactic_semantic' in self.active_categories:
            if SyntacticSemanticFeatures is not None:
                self.extractors['syntactic_semantic'] = SyntacticSemanticFeatures()
            else:
                logger.warning(
                    "SyntacticSemanticFeatures not available. Skipping this category."
                )
        
        # CATEGORY 3: Pragmatic & Conversational (FULLY IMPLEMENTED)
        if 'pragmatic_conversational' in self.active_categories:
            self.extractors['pragmatic_conversational'] = {
                'turn_taking': TurnTakingFeatures(),
                'linguistic': LinguisticFeatures(),
                'pragmatic': PragmaticFeatures(),
                'conversational': ConversationalFeatures(),
            }
    
    @property
    def all_feature_names(self) -> List[str]:
        """
        Get list of all feature names across active categories.
        
        Returns:
            Complete list of feature names
        """
        feature_names = []
        
        for category, extractors in self.extractors.items():
            if isinstance(extractors, dict):
                # Pragmatic/conversational has sub-extractors
                for extractor in extractors.values():
                    feature_names.extend(extractor.feature_names)
            else:
                # Single extractor
                feature_names.extend(extractors.feature_names)
        
        return feature_names
    
    def extract_from_transcript(
        self,
        transcript: TranscriptData,
        categories: Optional[List[str]] = None
    ) -> FeatureSet:
        """
        Extract features from a single transcript.
        
        Args:
            transcript: Parsed transcript data
            categories: Optional override of categories to extract
        
        Returns:
            FeatureSet with all extracted features
            
        Example:
            >>> features = extractor.extract_from_transcript(transcript)
            >>> print(f"Extracted {len(features.features)} features")
        """
        # Use provided categories or default to active categories
        extract_categories = categories or self.active_categories
        
        all_features = {}
        extraction_metadata = {}
        extracted_categories = []
        
        # Extract from each category
        for category in extract_categories:
            if category not in self.extractors:
                logger.warning(f"Category '{category}' not initialized, skipping")
                continue
            
            try:
                extractors = self.extractors[category]
                
                if isinstance(extractors, dict):
                    # Pragmatic/conversational with sub-extractors
                    for name, extractor in extractors.items():
                        result = extractor.extract(transcript)
                        all_features.update(result.features)
                        extraction_metadata[f"{category}_{name}"] = result.metadata
                else:
                    # Single extractor (acoustic/syntactic)
                    result = extractors.extract(transcript)
                    
                    # Check if it's a placeholder
                    is_placeholder = result.metadata.get('status') == 'placeholder'
                    
                    if is_placeholder and not self.include_placeholders:
                        logger.debug(f"Skipping placeholder category: {category}")
                        continue
                    
                    all_features.update(result.features)
                    extraction_metadata[category] = result.metadata
                
                extracted_categories.append(category)
                
            except Exception as e:
                logger.error(f"Error extracting {category} features: {e}")
                # Continue with other categories
        
        # Create FeatureSet
        feature_set = FeatureSet(
            participant_id=transcript.participant_id,
            file_path=transcript.file_path,
            diagnosis=transcript.diagnosis,
            age_months=transcript.age_months,
            features=all_features,
            metadata={
                'total_utterances': transcript.total_utterances,
                'extraction_metadata': extraction_metadata,
            },
            feature_categories=extracted_categories
        )
        
        logger.debug(
            f"Extracted {len(all_features)} features from {len(extracted_categories)} categories"
        )
        
        return feature_set
    
    def extract_from_files(
        self,
        file_paths: List[Path],
        parser=None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Extract features from multiple transcript files.
        
        Args:
            file_paths: List of paths to .cha files
            parser: CHATParser instance (creates new one if None)
            show_progress: Whether to show progress bar
            
        Returns:
            DataFrame with one row per transcript
            
        Example:
            >>> files = list(Path('data/asdbank_eigsti').rglob('*.cha'))
            >>> df = extractor.extract_from_files(files)
            >>> print(df.head())
        """
        from src.parsers.chat_parser import CHATParser
        
        if parser is None:
            parser = CHATParser()
        
        feature_sets = []
        errors = []
        
        # Process with progress bar
        iterator = tqdm(file_paths, desc="Extracting features") if show_progress else file_paths
        
        for file_path in iterator:
            try:
                # Parse transcript
                transcript = parser.parse_file(file_path)
                
                # Extract features
                feature_set = self.extract_from_transcript(transcript)
                feature_sets.append(feature_set.to_dict())
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                errors.append((file_path, str(e)))
        
        # Convert to DataFrame
        if not feature_sets:
            logger.warning("No features extracted successfully")
            return pd.DataFrame()
        
        df = pd.DataFrame(feature_sets)
        
        logger.info(
            f"Extracted features from {len(feature_sets)}/{len(file_paths)} files "
            f"({len(errors)} errors)"
        )
        
        if errors:
            logger.warning(f"Failed files: {[str(f[0].name) for f in errors[:5]]}")
        
        return df
    
    @timing_decorator
    def extract_from_directory(
        self,
        directory: str | Path,
        pattern: str = "**/*.cha",
        output_file: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Extract features from all transcripts in a directory.
        
        Args:
            directory: Directory containing .cha files
            pattern: Glob pattern for finding files
            output_file: Optional path to save CSV output
            
        Returns:
            DataFrame with all extracted features
            
        Example:
            >>> df = extractor.extract_from_directory(
            ...     'data/asdbank_eigsti',
            ...     output_file='output/eigsti_features.csv'
            ... )
        """
        directory = Path(directory)
        
        # Find all files
        file_paths = list(directory.glob(pattern))
        logger.info(f"Found {len(file_paths)} transcript files in {directory}")
        
        # Extract features
        df = self.extract_from_files(file_paths)
        
        # Save if output file specified
        if output_file and not df.empty:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"Features saved to: {output_file}")
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for extracted features.
        
        Args:
            df: DataFrame with extracted features
            
        Returns:
            Dictionary of summary statistics
        """
        summary = {
            'total_samples': len(df),
            'feature_count': len(self.all_feature_names),
            'diagnosis_counts': {},
            'feature_stats': {},
            'missing_values': {},
            'categories_extracted': self.active_categories,
        }
        
        # Diagnosis distribution
        if 'diagnosis' in df.columns:
            summary['diagnosis_counts'] = df['diagnosis'].value_counts().to_dict()
        
        # Feature statistics
        feature_cols = [col for col in df.columns if col in self.all_feature_names]
        
        for col in feature_cols:
            summary['feature_stats'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
            }
            
            # Count missing values
            missing = df[col].isna().sum()
            if missing > 0:
                summary['missing_values'][col] = int(missing)
        
        return summary
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        method: str = 'zscore'
    ) -> pd.DataFrame:
        """
        Normalize features for machine learning.
        
        Args:
            df: DataFrame with features
            method: Normalization method ('zscore', 'minmax', or 'robust')
            
        Returns:
            DataFrame with normalized features
        """
        df_norm = df.copy()
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in df.columns if col in self.all_feature_names]
        
        for col in feature_cols:
            if method == 'zscore':
                # Z-score normalization
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df_norm[col] = (df[col] - mean) / std
                    
            elif method == 'minmax':
                # Min-max normalization to [0, 1]
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                    
            elif method == 'robust':
                # Robust normalization using median and IQR
                median = df[col].median()
                q75 = df[col].quantile(0.75)
                q25 = df[col].quantile(0.25)
                iqr = q75 - q25
                if iqr > 0:
                    df_norm[col] = (df[col] - median) / iqr
        
        logger.info(f"Normalized {len(feature_cols)} features using {method} method")
        
        return df_norm
    
    def print_category_info(self):
        """Print information about all feature categories."""
        print("\n" + "="*70)
        print("FEATURE EXTRACTION CATEGORIES")
        print("="*70)
        
        for category, info in self.FEATURE_CATEGORIES.items():
            status_icon = "✓" if info['status'] == 'implemented' else "○"
            active_icon = "●" if category in self.active_categories else "○"
            
            print(f"\n{active_icon} {category.upper().replace('_', ' & ')}")
            print(f"   Status: {status_icon} {info['status'].upper()}")
            print(f"   Team: {info['team']}")
            print(f"   Description: {info['description']}")
            
            if category in self.extractors:
                extractors = self.extractors[category]
                if isinstance(extractors, dict):
                    num_features = sum(len(e.feature_names) for e in extractors.values())
                    print(f"   Sub-extractors: {', '.join(extractors.keys())}")
                else:
                    num_features = len(extractors.feature_names)
                print(f"   Features: {num_features}")
        
        print("\n" + "="*70)
        print(f"Total Active Features: {len(self.all_feature_names)}")
        print("="*70 + "\n")


__all__ = ["FeatureExtractor", "FeatureSet"]
