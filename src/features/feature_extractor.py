"""
Main Feature Extraction Orchestrator

This module coordinates all feature extractors organized according to
the research methodology:

Methodology Sections:
  3.3.1 - Turn-Taking Metrics
  3.3.2 - Topic Maintenance and Semantic Coherence
  3.3.3 - Pause and Latency Analysis
  3.3.4 - Conversational Repair Detection

Additional Feature Categories:
  - Linguistic Features (MLU, vocabulary diversity)
  - Pragmatic Features (echolalia, pronouns, social language)
  - Acoustic & Prosodic Features (placeholder - Team Member A)
  - Syntactic & Semantic Features (placeholder - Team Member B)

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

# Pragmatic & Conversational Features - Methodology aligned (Primary)
from .pragmatic_conversational import (
    TurnTakingFeatures,       # Section 3.3.1
    TopicCoherenceFeatures,   # Section 3.3.2
    PauseLatencyFeatures,     # Section 3.3.3
    RepairDetectionFeatures,  # Section 3.3.4
    LinguisticFeatures,       # Supporting
    PragmaticFeatures,        # Supporting
    ConversationalFeatures,   # Supporting (legacy)
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
    
    Plus supporting features:
    - Linguistic (MLU, vocabulary)
    - Pragmatic (echolalia, pronouns)
    
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
    
    Coordinates extraction from all methodology-aligned modules:
    - Section 3.3.1: Turn-Taking Metrics
    - Section 3.3.2: Topic Maintenance and Semantic Coherence
    - Section 3.3.3: Pause and Latency Analysis
    - Section 3.3.4: Conversational Repair Detection
    
    Plus supporting modules:
    - Linguistic Features
    - Pragmatic Features
    
    Example:
        >>> # Extract all methodology-aligned features
        >>> extractor = FeatureExtractor(categories='methodology')
        >>> feature_set = extractor.extract_from_transcript(transcript)
        
        >>> # Extract all features including supporting
        >>> extractor = FeatureExtractor(categories='all')
        >>> feature_set = extractor.extract_from_transcript(transcript)
    """
    
    # Define feature categories aligned with methodology
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
        # Supporting categories
        'linguistic': {
            'section': 'supporting',
            'description': 'Linguistic Features (MLU, vocabulary)',
            'status': 'implemented',
        },
        'pragmatic': {
            'section': 'supporting',
            'description': 'Pragmatic Features (echolalia, pronouns)',
            'status': 'implemented',
        },
        'conversational': {
            'section': 'supporting',
            'description': 'Legacy Conversational Features',
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
            'description': 'Syntactic and semantic features from text',
            'status': 'placeholder',
            'team': 'Team Member B',
        },
    }
    
    # Category groupings
    METHODOLOGY_CATEGORIES = [
        'turn_taking', 'topic_coherence', 'pause_latency', 'repair_detection'
    ]
    SUPPORTING_CATEGORIES = ['linguistic', 'pragmatic']
    LEGACY_CATEGORIES = ['conversational']
    
    def __init__(
        self,
        categories: Optional[List[str] | str] = 'methodology',
        include_legacy: bool = False,
        include_supporting: bool = True
    ):
        """
        Initialize feature extractor with specified categories.
        
        Args:
            categories: Which feature categories to extract:
                - 'methodology': Sections 3.3.1-3.3.4 only (default)
                - 'all': All implemented categories
                - 'pragmatic_conversational': Backward compatible (all pragmatic)
                - List of specific category names
            include_legacy: Whether to include legacy conversational features
            include_supporting: Whether to include supporting features (linguistic, pragmatic)
        
        Example:
            >>> # Methodology sections only
            >>> extractor = FeatureExtractor(categories='methodology')
            
            >>> # All features
            >>> extractor = FeatureExtractor(categories='all')
            
            >>> # Specific categories
            >>> extractor = FeatureExtractor(
            ...     categories=['turn_taking', 'topic_coherence']
            ... )
        """
        self.include_legacy = include_legacy
        self.include_supporting = include_supporting
        
        # Determine which categories to use
        if categories == 'methodology':
            self.active_categories = list(self.METHODOLOGY_CATEGORIES)
            if include_supporting:
                self.active_categories.extend(self.SUPPORTING_CATEGORIES)
        elif categories == 'all':
            self.active_categories = (
                list(self.METHODOLOGY_CATEGORIES) +
                list(self.SUPPORTING_CATEGORIES)
            )
            if include_legacy:
                self.active_categories.extend(self.LEGACY_CATEGORIES)
        elif categories == 'pragmatic_conversational':
            # Backward compatibility
            self.active_categories = (
                list(self.METHODOLOGY_CATEGORIES) +
                list(self.SUPPORTING_CATEGORIES)
            )
        elif isinstance(categories, str):
            self.active_categories = [categories]
        else:
            self.active_categories = list(categories)
        
        # Initialize extractors
        self._initialize_extractors()
        
        logger.info(
            f"FeatureExtractor initialized with categories: {self.active_categories}"
        )
    
    def _initialize_extractors(self):
        """Initialize all feature extractors based on active categories."""
        self.extractors = {}
        
        # Methodology-aligned extractors (Sections 3.3.1 - 3.3.4)
        if 'turn_taking' in self.active_categories:
            self.extractors['turn_taking'] = TurnTakingFeatures()
            logger.debug("Initialized TurnTakingFeatures (Section 3.3.1)")
        
        if 'topic_coherence' in self.active_categories:
            self.extractors['topic_coherence'] = TopicCoherenceFeatures()
            logger.debug("Initialized TopicCoherenceFeatures (Section 3.3.2)")
        
        if 'pause_latency' in self.active_categories:
            self.extractors['pause_latency'] = PauseLatencyFeatures()
            logger.debug("Initialized PauseLatencyFeatures (Section 3.3.3)")
        
        if 'repair_detection' in self.active_categories:
            self.extractors['repair_detection'] = RepairDetectionFeatures()
            logger.debug("Initialized RepairDetectionFeatures (Section 3.3.4)")
        
        # Supporting extractors
        if 'linguistic' in self.active_categories:
            self.extractors['linguistic'] = LinguisticFeatures()
            logger.debug("Initialized LinguisticFeatures")
        
        if 'pragmatic' in self.active_categories:
            self.extractors['pragmatic'] = PragmaticFeatures()
            logger.debug("Initialized PragmaticFeatures")
        
        # Legacy extractors
        if 'conversational' in self.active_categories:
            self.extractors['conversational'] = ConversationalFeatures()
            logger.debug("Initialized ConversationalFeatures (legacy)")
        
        # Syntactic/Semantic (if available)
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
        
        for category, extractor in self.extractors.items():
            feature_names.extend(extractor.feature_names)
        
        return feature_names
    
    @property
    def feature_count_by_category(self) -> Dict[str, int]:
        """Get feature count per category."""
        return {
            category: len(extractor.feature_names)
            for category, extractor in self.extractors.items()
        }
    
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
            f"Extracted {len(all_features)} features from "
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
            'feature_count_by_category': self.feature_count_by_category,
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
            if pd.api.types.is_numeric_dtype(df[col]):
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
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df_norm[col] = (df[col] - mean) / std
                    
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df_norm[col] = (df[col] - min_val) / (max_val - min_val)
                    
            elif method == 'robust':
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
        
        # Methodology sections
        print("\n📋 METHODOLOGY SECTIONS (Primary)")
        print("-" * 50)
        
        for category in self.METHODOLOGY_CATEGORIES:
            info = self.FEATURE_CATEGORIES[category]
            active_icon = "●" if category in self.active_categories else "○"
            count = len(self.extractors[category].feature_names) if category in self.extractors else 0
            
            print(f"{active_icon} Section {info['section']}: {info['description']}")
            print(f"    Features: {count}")
        
        # Supporting sections
        print("\n📚 SUPPORTING FEATURES")
        print("-" * 50)
        
        for category in self.SUPPORTING_CATEGORIES:
            info = self.FEATURE_CATEGORIES[category]
            active_icon = "●" if category in self.active_categories else "○"
            count = len(self.extractors[category].feature_names) if category in self.extractors else 0
            
            print(f"{active_icon} {info['description']}")
            print(f"    Features: {count}")
        
        # Summary
        print("\n" + "="*70)
        print(f"Total Active Features: {len(self.all_feature_names)}")
        print(f"Active Categories: {len(self.active_categories)}")
        print("="*70 + "\n")


__all__ = ["FeatureExtractor", "FeatureSet"]
