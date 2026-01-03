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
    PragmaticAudioFeatures,      # Audio-derived features
)

# Placeholder modules for other team members
try:
    from .syntactic_semantic import SyntacticSemanticFeatures
except ImportError:
    SyntacticSemanticFeatures = None

try:
    from .acoustic_prosodic.audio_features import AcousticAudioFeatures
except ImportError:
    AcousticAudioFeatures = None

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
    
    PRIMARY:
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
    
    # Feature categories
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
        'pragmatic_linguistic': {
            'section': 'supporting',
            'description': 'Pragmatic & Linguistic Features (MLU, echolalia, pronouns, etc.)',
            'status': 'implemented',
        },
        'pragmatic_audio': {
            'section': 'supporting',
            'description': 'Audio-derived pragmatic features (pauses, timing)',
            'status': 'implemented',
        },
        # Acoustic & Prosodic features (implemented)
        'acoustic_prosodic': {
            'section': 'audio',
            'description': 'Acoustic and prosodic features from audio',
            'status': 'implemented',
        },
        'acoustic_audio': {
            'section': 'audio',
            'description': 'Acoustic and prosodic features from audio (alias)',
            'status': 'implemented',
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
    SUPPORTING_CATEGORIES = ['pragmatic_linguistic', 'pragmatic_audio']
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
        
        # Supporting: Audio-derived pragmatic features
        if 'pragmatic_audio' in self.active_categories:
            self.extractors['pragmatic_audio'] = PragmaticAudioFeatures()
            logger.debug("Initialized PragmaticAudioFeatures (Audio)")
        
        # Acoustic & Prosodic audio features (if available and requested)
        if 'acoustic_audio' in self.active_categories or 'acoustic_prosodic' in self.active_categories:
            if AcousticAudioFeatures is not None:
                # Use 'acoustic_audio' as the key for consistency
                # extract_child_only=True to extract only child speech segments
                self.extractors['acoustic_audio'] = AcousticAudioFeatures(extract_child_only=True)
                # Also map 'acoustic_prosodic' to the same extractor for API compatibility
                if 'acoustic_prosodic' in self.active_categories:
                    self.extractors['acoustic_prosodic'] = self.extractors['acoustic_audio']
                logger.debug("Initialized AcousticAudioFeatures (child-only extraction enabled)")
            else:
                logger.warning("AcousticAudioFeatures not available")
        
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
    
    def extract_with_audio(
        self,
        transcript: TranscriptData,
        audio_path: Optional[Path] = None,
        transcription_result: Optional[Any] = None,
        categories: Optional[List[str]] = None
    ) -> FeatureSet:
        """
        Extract features with audio support.
        
        This method extracts both text-based and audio-based features.
        Audio features are extracted by the pragmatic_audio extractor
        when audio_path or transcription_result is provided.
        
        Args:
            transcript: Parsed transcript data
            audio_path: Optional path to audio file
            transcription_result: Optional TranscriptionResult with timing
            categories: Optional override of categories to extract
            
        Returns:
            FeatureSet with all extracted features including audio features
        """
        extract_categories = categories or list(self.extractors.keys())
        
        all_features = {}
        extraction_metadata = {}
        extracted_categories = []
        
        logger.debug(f"Extracting features with audio from {transcript.participant_id}")
        
        # Extract from each category
        for category in extract_categories:
            if category not in self.extractors:
                logger.warning(f"Category '{category}' not initialized, skipping")
                continue
            
            try:
                extractor = self.extractors[category]
                
                # Special handling for audio extractors
                if category == 'pragmatic_audio':
                    result = extractor.extract(
                        transcript,
                        audio_path=audio_path,
                        transcription_result=transcription_result
                    )
                elif category in ('acoustic_audio', 'acoustic_prosodic'):
                    result = extractor.extract(
                        transcript,
                        audio_path=audio_path,
                        transcription_result=transcription_result
                    )
                else:
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
                'audio_path': str(audio_path) if audio_path else None,
                'has_audio': audio_path is not None,
            },
            feature_categories=extracted_categories
        )
        
        logger.debug(
            f"Extracted {len(all_features)} total features from "
            f"{len(extracted_categories)} categories (with audio)"
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
        
        Supports both:
        - CHAT files (.cha) - parsed directly
        - Audio files (.wav, .mp3, etc.) - transcribed with Whisper first
        
        Args:
            directory: Directory containing .cha files or audio files
            pattern: Glob pattern for finding files (default: "**/*.cha")
            output_file: Optional path to save CSV output
            
        Returns:
            DataFrame with all extracted features
        """
        directory = Path(directory)
        
        # First, try to find .cha files
        cha_files = list(directory.glob("**/*.cha"))
        
        # If no .cha files, look for audio files
        audio_files = []
        if not cha_files:
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
            audio_files = [
                f for f in directory.rglob("*")
                if f.is_file() and f.suffix.lower() in audio_extensions
            ]
            logger.info(f"Found {len(audio_files)} audio files in {directory} (no .cha files)")
        else:
            logger.info(f"Found {len(cha_files)} transcript files in {directory}")
        
        # Process .cha files if available
        if cha_files:
            df = self.extract_from_files(cha_files)
        elif audio_files:
            # Process audio files - transcribe and extract features
            # Pass directory name for diagnosis detection
            # Save transcriptions next to audio files
            df = self.extract_from_audio_files(
                audio_files, 
                dataset_dir=directory,
                save_transcriptions=True,
                transcription_output_dir=directory  # Save .cha files in same directory as audio
            )
        else:
            logger.warning(f"No .cha or audio files found in {directory}")
            return pd.DataFrame()
        
        if output_file and not df.empty:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_file, index=False)
            logger.info(f"Features saved to: {output_file}")
        
        return df
    
    def extract_from_audio_files(
        self,
        audio_paths: List[Path],
        dataset_dir: Optional[Path] = None,
        show_progress: bool = True,
        save_transcriptions: bool = True,
        transcription_output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Extract features from audio files by transcribing them first.
        
        This method:
        1. Transcribes each audio file using faster-whisper/Whisper/Google
        2. Converts transcription to TranscriptData format
        3. Optionally saves transcriptions as .cha files
        4. Extracts features using extract_with_audio (includes audio-based features)
        
        Args:
            audio_paths: List of paths to audio files
            dataset_dir: Optional dataset directory path (for diagnosis detection)
            show_progress: Whether to show progress bar
            save_transcriptions: If True, save transcriptions as .cha files
            transcription_output_dir: Directory to save .cha files (default: same as audio files)
            
        Returns:
            DataFrame with one row per audio file
        """
        from src.audio.audio_processor import AudioProcessor
        
        # Initialize audio processor with error handling
        # On macOS, Whisper often crashes with PyTorch segfault
        import platform
        is_macos = platform.system() == 'Darwin'
        
        logger.info("Initializing audio processor for transcription...")
        audio_processor = None
        
        # Priority order: faster-whisper > vosk > google > whisper (last due to crashes)
        # faster-whisper uses CTranslate2 instead of PyTorch, avoids macOS crashes
        backends_to_try = []
        
        if is_macos:
            logger.info("macOS detected: Preferring faster-whisper (avoids PyTorch crashes)")
            backends_to_try = [
                ('faster-whisper', 'tiny'),
                ('vosk', None),
                ('google', None)
            ]
        else:
            backends_to_try = [
                ('faster-whisper', 'tiny'),
                ('whisper', 'tiny'),
                ('vosk', None),
                ('google', None)
            ]
        
        for backend, model_size in backends_to_try:
            try:
                logger.info(f"Trying {backend} backend...")
                if backend in ('faster-whisper', 'whisper'):
                    audio_processor = AudioProcessor(
                        transcriber_backend=backend,
                        whisper_model_size=model_size or 'tiny',
                        device='cpu',
                        language='en'
                    )
                else:
                    audio_processor = AudioProcessor(
                        transcriber_backend=backend,
                        language='en'
                    )
                
                if audio_processor.transcriber_available:
                    logger.info(f"✓ Audio processor initialized with {backend}")
                    if backend == 'google':
                        logger.info("Note: Google Speech-to-Text requires internet connection")
                    break
            except (RuntimeError, OSError, ImportError) as e:
                logger.debug(f"{backend} failed: {e}")
                continue
            except Exception as e:
                logger.debug(f"{backend} error: {e}")
                continue
        
        if audio_processor is None or not audio_processor.transcriber_available:
            logger.error("All transcription backends failed!")
            logger.error(
                "Install one of:\n"
                "1. faster-whisper (recommended for macOS): pip install faster-whisper\n"
                "2. Vosk: pip install vosk (also download model from https://alphacephei.com/vosk/models)\n"
                "3. Google: pip install SpeechRecognition\n"
                "4. Whisper: pip install openai-whisper (may crash on macOS)"
            )
            return pd.DataFrame()
        
        feature_sets = []
        errors = []
        
        iterator = tqdm(audio_paths, desc="Transcribing and extracting features") if show_progress else audio_paths
        
        for audio_path in iterator:
            try:
                # Extract participant ID and diagnosis from path if possible
                # Try to infer from directory structure (e.g., data/td/ASD/audio.wav)
                participant_id = "CHI"
                diagnosis = None
                
                # First, check if dataset directory name is a diagnosis label
                if dataset_dir:
                    dataset_name = dataset_dir.name.upper()
                    if dataset_name in ['ASD', 'TD', 'DD', 'HR', 'LR', 'TYP']:
                        diagnosis = dataset_name
                        if diagnosis == 'TYP':
                            diagnosis = 'TD'
                    elif dataset_dir.name.lower() in ['td', 'asd', 'dd', 'hr', 'lr']:
                        diagnosis = dataset_dir.name.upper()
                
                # If not found, check parent directories for diagnosis labels
                if not diagnosis:
                    path_parts = audio_path.parts
                    for part in reversed(path_parts):
                        part_upper = part.upper()
                        # Check if directory name is a diagnosis label
                        if part_upper in ['ASD', 'TD', 'DD', 'HR', 'LR', 'TYP']:
                            diagnosis = part_upper
                            if diagnosis == 'TYP':
                                diagnosis = 'TD'
                            break
                        # Also check lowercase directory names
                        elif part.lower() in ['td', 'asd', 'dd', 'hr', 'lr']:
                            diagnosis = part.upper()
                            break
                
                # Use filename as participant ID if no better option
                if not participant_id or participant_id == "CHI":
                    participant_id = audio_path.stem
                
                # Step 1: Transcribe audio
                logger.debug(f"Transcribing {audio_path.name}...")
                audio_result = None
                transcription_failed = False
                
                try:
                    audio_result = audio_processor.process(
                        audio_path,
                        participant_id=participant_id,
                        diagnosis=diagnosis
                    )
                    
                    if not audio_result.success:
                        logger.warning(f"Transcription failed for {audio_path.name}")
                        transcription_failed = True
                except (RuntimeError, OSError) as transcribe_error:
                    # If Whisper crashes during model loading (segfault), try Google fallback
                    error_msg = str(transcribe_error).lower()
                    if 'whisper' in error_msg or 'model' in error_msg or 'pytorch' in error_msg:
                        logger.warning(
                            f"Whisper failed for {audio_path.name}: {transcribe_error}. "
                            f"Switching to Google Speech-to-Text fallback..."
                        )
                        # Try Google Speech-to-Text as fallback
                        try:
                            google_processor = AudioProcessor(
                                transcriber_backend='google',
                                language='en'
                            )
                            audio_result = google_processor.process(
                                audio_path,
                                participant_id=participant_id,
                                diagnosis=diagnosis
                            )
                            logger.info(f"✓ Google Speech-to-Text succeeded for {audio_path.name}")
                            # Update audio_processor for future files
                            audio_processor = google_processor
                        except Exception as google_error:
                            logger.error(f"Google fallback also failed: {google_error}")
                            errors.append((audio_path, f"Both Whisper and Google failed"))
                            continue
                    else:
                        logger.error(f"Transcription error for {audio_path.name}: {transcribe_error}")
                        errors.append((audio_path, f"Transcription error: {transcribe_error}"))
                        continue
                except Exception as transcribe_error:
                    # Other unexpected errors
                    logger.error(f"Unexpected transcription error for {audio_path.name}: {transcribe_error}")
                    errors.append((audio_path, f"Unexpected error: {transcribe_error}"))
                    continue
                
                if transcription_failed or audio_result is None:
                    continue
                
                # Step 2: Save transcription as .cha file (optional)
                if save_transcriptions:
                    try:
                        # Determine output path for transcription
                        if transcription_output_dir:
                            output_dir = Path(transcription_output_dir)
                        elif dataset_dir:
                            # Save in same directory as audio files
                            output_dir = audio_path.parent
                        else:
                            # Save next to audio file
                            output_dir = audio_path.parent
                        
                        output_dir.mkdir(parents=True, exist_ok=True)
                        chat_file_path = output_dir / f"{audio_path.stem}.cha"
                        audio_processor.save_chat_file(audio_result, chat_file_path)
                        logger.debug(f"Saved transcription to {chat_file_path}")
                    except Exception as save_error:
                        logger.warning(f"Failed to save transcription for {audio_path.name}: {save_error}")
                
                # Step 3: Extract features with audio
                feature_set = self.extract_with_audio(
                    transcript=audio_result.transcript_data,
                    audio_path=audio_path,
                    transcription_result=audio_result.transcription
                )
                
                feature_sets.append(feature_set.to_dict())
                
                logger.debug(
                    f"Extracted {len(feature_set.features)} features from {audio_path.name}"
                )
                
            except Exception as e:
                logger.error(f"Error processing {audio_path.name}: {e}")
                errors.append((audio_path, str(e)))
        
        if not feature_sets:
            logger.warning("No features extracted successfully from audio files")
            return pd.DataFrame()
        
        df = pd.DataFrame(feature_sets)
        
        logger.info(
            f"Extracted features from {len(feature_sets)}/{len(audio_paths)} audio files "
            f"({len(errors)} errors)"
        )
        
        if errors:
            logger.warning(f"Failed files: {[str(f[0].name) for f in errors[:5]]}")
        
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
