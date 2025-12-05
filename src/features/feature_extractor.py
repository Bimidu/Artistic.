"""
Main Feature Extraction Orchestrator

This module coordinates all feature extractors organized according to
the research methodology:

=== PRIMARY CATEGORIES (Methodology Sections 3.3.1 - 3.3.4) ===

Section 3.3.1 - Turn-Taking Metrics
Section 3.3.2 - Topic Maintenance and Semantic Coherence
Section 3.3.3 - Pause and Latency Analysis
Section 3.3.4 - Conversational Repair Detection

=== SUPPORTING CATEGORY ===

Pragmatic & Linguistic Features
  - MLU, vocabulary diversity, echolalia, pronouns, questions, social language

=== OTHER MODULES (Placeholders) ===

Acoustic & Prosodic Features (Team Member A)
Syntactic & Semantic Features (Team Member B)

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
    TurnTakingFeatures,          # Section 3.3.1
    TopicCoherenceFeatures,      # Section 3.3.2
    PauseLatencyFeatures,        # Section 3.3.3
    RepairDetectionFeatures,     # Section 3.3.4
    PragmaticLinguisticFeatures, # Supporting
)

# Placeholder modules for other team members
try:
    from .syntactic_semantic import SyntacticSemanticFeatures
except ImportError:
    SyntacticSemanticFeatures = None

logger = get_logger(__name__)


@dataclass
class FeatureSet:
    """
    Complete feature set extracted from a transcript.
    
    Contains features from all methodology sections:
    - 3.3.1 Turn-Taking Metrics
    - 3.3.2 Topic Maintenance and Semantic Coherence
    - 3.3.3 Pause and Latency Analysis
    - 3.3.4 Conversational Repair Detection
    - Supporting: Pragmatic & Linguistic Features
    
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
    Main feature extraction orchestrator.
    
    Coordinates extraction from 5 categories:
    
    PRIMARY (Methodology-aligned):
    - Section 3.3.1: Turn-Taking Metrics
    - Section 3.3.2: Topic Maintenance and Semantic Coherence
    - Section 3.3.3: Pause and Latency Analysis
    - Section 3.3.4: Conversational Repair Detection
    
    SUPPORTING:
    - Pragmatic & Linguistic Features (MLU, echolalia, pronouns, etc.)
    
    Example:
        >>> # Extract all features (default)
        >>> extractor = FeatureExtractor()
        >>> feature_set = extractor.extract_from_transcript(transcript)
        
        >>> # Extract only methodology sections
        >>> extractor = FeatureExtractor(categories='methodology')
        >>> feature_set = extractor.extract_from_transcript(transcript)
    """
    
    # Feature categories (5 total: 4 methodology + 1 supporting)
    FEATURE_CATEGORIES = {
        # Primary methodology sections
        'turn_taking': {
            'section': '3.3.1',
            'description': 'Turn-Taking Metrics',
            'status': 'implemented',
        },
        'topic_coherence': {
            'section': '3.3.2',
            'description': 'Topic Maintenance and Semantic Coherence',
            'status': 'implemented',
        },
        'pause_latency': {
            'section': '3.3.3',
            'description': 'Pause and Latency Analysis',
            'status': 'implemented',
        },
        'repair_detection': {
            'section': '3.3.4',
            'description': 'Conversational Repair Detection',
            'status': 'implemented',
        },
        # Supporting category (consolidated)
        'pragmatic_linguistic': {
            'section': 'supporting',
            'description': 'Pragmatic & Linguistic Features (MLU, echolalia, pronouns, etc.)',
            'status': 'implemented',
        },
        # Placeholder categories for other team members
        'acoustic_prosodic': {
            'section': 'audio',
            'description': 'Acoustic and prosodic features from audio',
            'status': 'placeholder',
            'team': 'Team Member A',
        },
        'syntactic_semantic': {
            'section': 'text',
            'description': 'Syntactic and semantic features (POS, dependencies)',
            'status': 'placeholder',
            'team': 'Team Member B',
        },
    }
    
    # Category groupings
    METHODOLOGY_CATEGORIES = [
        'turn_taking', 'topic_coherence', 'pause_latency', 'repair_detection'
    ]
    SUPPORTING_CATEGORIES = ['pragmatic_linguistic']
    ALL_IMPLEMENTED = METHODOLOGY_CATEGORIES + SUPPORTING_CATEGORIES
    
    def __init__(
        self,
        categories: Optional[List[str] | str] = 'all',
        include_supporting: bool = True
    ):
        """
        Initialize feature extractor with specified categories.
        
        Args:
            categories: Which feature categories to extract:
                - 'all': All 5 implemented categories (default)
                - 'methodology': Only sections 3.3.1-3.3.4
                - 'pragmatic_conversational': Alias for 'all' (backward compat)
                - List of specific category names
            include_supporting: If True, include pragmatic_linguistic with methodology
        
        Example:
            >>> # All features (recommended)
            >>> extractor = FeatureExtractor()
            
            >>> # Only methodology sections
            >>> extractor = FeatureExtractor(categories='methodology')
            
            >>> # Specific categories
            >>> extractor = FeatureExtractor(
            ...     categories=['turn_taking', 'topic_coherence']
            ... )
        """
        self.include_supporting = include_supporting
        
        # Determine active categories
        if categories == 'all' or categories == 'pragmatic_conversational':
            self.active_categories = list(self.ALL_IMPLEMENTED)
        elif categories == 'methodology':
            self.active_categories = list(self.METHODOLOGY_CATEGORIES)
            if include_supporting:
                self.active_categories.extend(self.SUPPORTING_CATEGORIES)
        elif isinstance(categories, str):
            self.active_categories = [categories]
        else:
            self.active_categories = list(categories)
        
        # Initialize extractors
        self._initialize_extractors()
        
        logger.info(
            f"FeatureExtractor initialized with {len(self.active_categories)} categories: "
            f"{self.active_categories}"
        )
    
    def _initialize_extractors(self):
        """Initialize all feature extractors based on active categories."""
        self.extractors = {}
        
        # Section 3.3.1: Turn-Taking Metrics
        if 'turn_taking' in self.active_categories:
            self.extractors['turn_taking'] = TurnTakingFeatures()
            logger.debug("Initialized TurnTakingFeatures (Section 3.3.1)")
        
        # Section 3.3.2: Topic Maintenance and Semantic Coherence
        if 'topic_coherence' in self.active_categories:
            self.extractors['topic_coherence'] = TopicCoherenceFeatures()
            logger.debug("Initialized TopicCoherenceFeatures (Section 3.3.2)")
        
        # Section 3.3.3: Pause and Latency Analysis
        if 'pause_latency' in self.active_categories:
            self.extractors['pause_latency'] = PauseLatencyFeatures()
            logger.debug("Initialized PauseLatencyFeatures (Section 3.3.3)")
        
        # Section 3.3.4: Conversational Repair Detection
        if 'repair_detection' in self.active_categories:
            self.extractors['repair_detection'] = RepairDetectionFeatures()
            logger.debug("Initialized RepairDetectionFeatures (Section 3.3.4)")
        
        # Supporting: Pragmatic & Linguistic Features
        if 'pragmatic_linguistic' in self.active_categories:
            self.extractors['pragmatic_linguistic'] = PragmaticLinguisticFeatures()
            logger.debug("Initialized PragmaticLinguisticFeatures (Supporting)")
        
        # Syntactic/Semantic (if available and requested)
        if 'syntactic_semantic' in self.active_categories:
            if SyntacticSemanticFeatures is not None:
                self.extractors['syntactic_semantic'] = SyntacticSemanticFeatures()
                logger.debug("Initialized SyntacticSemanticFeatures")
            else:
                logger.warning("SyntacticSemanticFeatures not available")
    
    @property
    def all_feature_names(self) -> List[str]:
        """
        Get list of all feature names across active categories.
        
        Returns:
            Complete list of feature names
        """
        feature_names = []
        for extractor in self.extractors.values():
            feature_names.extend(extractor.feature_names)
        return feature_names
    
    @property
    def feature_count_by_category(self) -> Dict[str, int]:
        """Get feature count per category."""
        return {
            category: len(extractor.feature_names)
            for category, extractor in self.extractors.items()
        }
    
    @property
    def total_features(self) -> int:
        """Get total number of features."""
        return len(self.all_feature_names)
    
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
        extract_categories = categories or list(self.extractors.keys())
        
        all_features = {}
        extraction_metadata = {}
        extracted_categories = []
        
        logger.debug(f"Extracting features from {transcript.participant_id}")
        
        # Extract from each category
        for category in extract_categories:
            if category not in self.extractors:
                logger.warning(f"Category '{category}' not initialized, skipping")
                continue
            
            try:
                extractor = self.extractors[category]
                result = extractor.extract(transcript)
                
                all_features.update(result.features)
                extraction_metadata[category] = result.metadata
                extracted_categories.append(category)
                
                logger.debug(
                    f"Extracted {len(result.features)} features from {category}"
                )
                
            except Exception as e:
                logger.error(f"Error extracting {category} features: {e}")
        
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
            f"Extracted {len(all_features)} total features from "
            f"{len(extracted_categories)} categories"
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
        """
        from src.parsers.chat_parser import CHATParser
        
        if parser is None:
            parser = CHATParser()
        
        feature_sets = []
        errors = []
        
        iterator = tqdm(file_paths, desc="Extracting features") if show_progress else file_paths
        
        for file_path in iterator:
            try:
                transcript = parser.parse_file(file_path)
                feature_set = self.extract_from_transcript(transcript)
                feature_sets.append(feature_set.to_dict())
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                errors.append((file_path, str(e)))
        
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
        """
        directory = Path(directory)
        file_paths = list(directory.glob(pattern))
        logger.info(f"Found {len(file_paths)} transcript files in {directory}")
        
        df = self.extract_from_files(file_paths)
        
        if output_file and not df.empty:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"Features saved to: {output_file}")
        
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for extracted features."""
        summary = {
            'total_samples': len(df),
            'total_features': self.total_features,
            'feature_count_by_category': self.feature_count_by_category,
            'diagnosis_counts': {},
            'categories_extracted': self.active_categories,
        }
        
        if 'diagnosis' in df.columns:
            summary['diagnosis_counts'] = df['diagnosis'].value_counts().to_dict()
        
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
            method: 'zscore', 'minmax', or 'robust'
            
        Returns:
            DataFrame with normalized features
        """
        df_norm = df.copy()
        feature_cols = [col for col in df.columns if col in self.all_feature_names]
        
        for col in feature_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            if method == 'zscore':
                mean, std = df[col].mean(), df[col].std()
                if std > 0:
                    df_norm[col] = (df[col] - mean) / std
            elif method == 'minmax':
                min_val, max_val = df[col].min(), df[col].max()
                if max_val > min_val:
                    df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'robust':
                median = df[col].median()
                iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                if iqr > 0:
                    df_norm[col] = (df[col] - median) / iqr
        
        logger.info(f"Normalized {len(feature_cols)} features using {method}")
        return df_norm
    
    def print_category_info(self):
        """Print information about all feature categories."""
        print("\n" + "=" * 70)
        print("FEATURE EXTRACTION CATEGORIES")
        print("=" * 70)
        
        # Methodology sections
        print("\n📋 METHODOLOGY SECTIONS (Primary)")
        print("-" * 50)
        
        for category in self.METHODOLOGY_CATEGORIES:
            info = self.FEATURE_CATEGORIES[category]
            active = "●" if category in self.extractors else "○"
            count = len(self.extractors[category].feature_names) if category in self.extractors else 0
            
            print(f"{active} Section {info['section']}: {info['description']}")
            print(f"    Features: {count}")
        
        # Supporting
        print("\n📚 SUPPORTING FEATURES")
        print("-" * 50)
        
        for category in self.SUPPORTING_CATEGORIES:
            info = self.FEATURE_CATEGORIES[category]
            active = "●" if category in self.extractors else "○"
            count = len(self.extractors[category].feature_names) if category in self.extractors else 0
            
            print(f"{active} {info['description']}")
            print(f"    Features: {count}")
        
        # Summary
        print("\n" + "=" * 70)
        print(f"Total Active Categories: {len(self.extractors)}")
        print(f"Total Features: {self.total_features}")
        print("=" * 70 + "\n")


__all__ = ["FeatureExtractor", "FeatureSet"]
