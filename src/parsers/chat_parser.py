"""
CHAT File Parser Module

This module provides comprehensive parsing functionality for CHAT-formatted
transcript files from the TalkBank/CHILDES database. It uses the pylangacq
library for core parsing and adds additional extraction capabilities.

Key Features:
- Parse .cha files with full metadata extraction
- Extract utterances with morphological and grammatical information
- Calculate timing information
- Handle multi-speaker conversations
- Extract behavioral markers and annotations

Author: Bimidu Gunathilake
"""

import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import pylangacq
import pandas as pd
from src.utils.logger import get_logger
from src.utils.helpers import extract_timing_info, get_age_in_months, is_valid_utterance
from src.parsers.diagnosis_mapper import DiagnosisMapper

logger = get_logger(__name__)


@dataclass
class Utterance:
    """
    Represents a single utterance from a transcript.
    
    Attributes:
        speaker: Speaker code (e.g., 'CHI', 'MOT', 'INV')
        text: The utterance text
        tokens: List of word tokens
        morphology: Morphological analysis (%mor tier)
        grammar: Grammatical relations (%gra tier)
        timing: Timestamp in seconds
        actions: Non-verbal actions (%act tier)
        comments: Comments about the utterance (%com tier)
        is_valid: Whether utterance meets validity criteria
    """
    speaker: str
    text: str
    tokens: List[str] = field(default_factory=list)
    morphology: Optional[str] = None
    grammar: Optional[str] = None
    timing: Optional[float] = None
    end_timing: Optional[float] = None
    actions: Optional[str] = None
    comments: Optional[str] = None
    is_valid: bool = True
    
    @property
    def word_count(self) -> int:
        """Get the number of words in the utterance."""
        if self.tokens:
            # Count non-empty tokens (filter out punctuation-only tokens)
            return len([token for token in self.tokens if hasattr(token, 'word') and token.word and token.word.strip()])
        elif self.text:
            # Fallback to text splitting
            return len(self.text.split())
        else:
            return 0
    
    @property
    def morpheme_count(self) -> int:
        """Get the number of morphemes in the utterance."""
        if self.morphology:
            # Count morphemes by splitting on spaces and counting non-empty elements
            morphemes = [m.strip() for m in self.morphology.split() if m.strip()]
            return len(morphemes)
        return self.word_count


@dataclass
class TranscriptData:
    """
    Complete data structure for a parsed CHAT transcript.
    
    This class holds all information extracted from a .cha file including
    metadata, participant information, and all utterances.
    
    Attributes:
        file_path: Path to the .cha file
        participant_id: Unique identifier for the target child
        diagnosis: Clinical diagnosis (ASD, TD, DD, etc.)
        age_months: Child's age in months
        gender: Child's gender
        session_date: Date of recording session
        session_type: Type of session (e.g., 'clinical', 'naturalistic')
        languages: List of languages used
        utterances: List of all utterances in the transcript
        metadata: Additional metadata dictionary
        speakers: Information about all speakers in the transcript
    """
    file_path: Path
    participant_id: str
    diagnosis: Optional[str] = None
    age_months: Optional[int] = None
    gender: Optional[str] = None
    session_date: Optional[datetime] = None
    session_type: Optional[str] = None
    languages: List[str] = field(default_factory=list)
    utterances: List[Utterance] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    speakers: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    @property
    def total_utterances(self) -> int:
        """Total number of utterances in transcript."""
        return len(self.utterances)
    
    @property
    def child_utterances(self) -> List[Utterance]:
        """Get only the target child's utterances."""
        return [u for u in self.utterances if u.speaker == 'CHI']
    
    @property
    def valid_utterances(self) -> List[Utterance]:
        """Get only valid utterances."""
        return [u for u in self.utterances if u.is_valid]
    
    def get_utterances_by_speaker(self, speaker_code: str) -> List[Utterance]:
        """
        Get utterances from a specific speaker.
        
        Args:
            speaker_code: Speaker identifier (e.g., 'MOT', 'INV')
            
        Returns:
            List of utterances from that speaker
        """
        return [u for u in self.utterances if u.speaker == speaker_code]


class CHATParser:
    """
    Comprehensive parser for CHAT-formatted transcript files.
    
    This class provides methods to parse .cha files and extract all relevant
    information including metadata, utterances, morphological analysis, and
    timing information.
    
    Example:
        >>> parser = CHATParser()
        >>> transcript = parser.parse_file("path/to/file.cha")
        >>> print(f"Parsed {transcript.total_utterances} utterances")
        >>> print(f"Child age: {transcript.age_months} months")
    """
    
    def __init__(self, min_words: int = 1):
        """
        Initialize the CHAT parser.
        
        Args:
            min_words: Minimum words required for valid utterance (default: 1)
        """
        self.min_words = min_words
        self.diagnosis_mapper = DiagnosisMapper()
        logger.info(f"CHATParser initialized with min_words={min_words}")
    
    def _preprocess_chat_file(self, file_path: Path) -> Path:
        """
        Preprocess CHAT file to handle tokens that cause pylangacq alignment issues.
        
        The '0' prefix in CHAT format marks omitted words (e.g., '0do' means 'do' is omitted).
        These tokens are correctly excluded from the %mor tier, but pylangacq's alignment
        algorithm fails when it sees them in the utterance. We remove them to match the %mor tier.
        
        Args:
            file_path: Path to original CHAT file
            
        Returns:
            Path to preprocessed temporary file
        """
        import tempfile
        import re
        
        # Read original file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Track preprocessing statistics
        tokens_removed = 0
        lines_modified = 0
        
        # Process line by line
        processed_lines = []
        
        for line in lines:
            # Only process utterance lines (start with *)
            if line.startswith('*'):
                original_line = line
                # Remove '0' prefix tokens entirely (e.g., '0do you' -> 'you')
                # This matches what the %mor tier already has
                modified_line = re.sub(r'\s*\b0[a-zA-Z]+\b', '', line)
                # Clean up any double spaces
                modified_line = re.sub(r'  +', ' ', modified_line)
                
                if modified_line != original_line:
                    lines_modified += 1
                    # Count how many tokens were removed
                    tokens_removed += len(re.findall(r'\b0[a-zA-Z]+\b', original_line))
                
                processed_lines.append(modified_line)
            else:
                processed_lines.append(line)
        
        # Log preprocessing statistics
        if tokens_removed > 0:
            logger.info(
                f"Preprocessed {file_path.name}: removed {tokens_removed} '0' prefix tokens "
                f"from {lines_modified} utterances"
            )
        
        # Write to temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.cha',
            delete=False,
            encoding='utf-8'
        )
        temp_file.writelines(processed_lines)
        temp_file.close()
        
        return Path(temp_file.name)
    
    def parse_file(self, file_path: str | Path) -> TranscriptData:
        """
        Parse a CHAT file and extract all information.
        
        This is the main entry point for parsing. It coordinates all
        parsing operations and returns a complete TranscriptData object.
        
        Args:
            file_path: Path to the .cha file
            
        Returns:
            TranscriptData object containing all parsed information
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be parsed
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Parsing CHAT file: {file_path.name}")
        
        # Preprocess file to handle '0' prefix tokens
        preprocessed_path = self._preprocess_chat_file(file_path)
        
        try:
            # Use pylangacq to read the preprocessed file
            reader = pylangacq.read_chat(str(preprocessed_path))
            
            # Extract metadata
            metadata = self._extract_metadata(reader, file_path)
            
            # Extract participant information
            participants = self._extract_participants(reader)
            
            # Resolve diagnosis using the diagnosis mapper
            diagnosis = metadata.get('diagnosis')
            if not diagnosis and '_diagnosis' in participants:
                diagnosis = participants['_diagnosis']
            
            # Use diagnosis mapper to infer/normalize diagnosis
            diagnosis = self.diagnosis_mapper.infer_diagnosis(file_path, diagnosis)

            # Create TranscriptData object
            transcript = TranscriptData(
                file_path=file_path,
                participant_id=metadata.get('participant_id', file_path.stem),
                diagnosis=diagnosis,
                age_months=metadata.get('age_months'),
                gender=metadata.get('gender'),
                session_date=metadata.get('session_date'),
                session_type=metadata.get('session_type'),
                languages=metadata.get('languages', []),
                metadata=metadata,
                speakers=participants,
            )
            
            # Extract all utterances
            transcript.utterances = self._extract_utterances(reader)
            
            logger.info(
                f"Successfully parsed {transcript.total_utterances} utterances "
                f"from {file_path.name}"
            )
            
            return transcript
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            raise ValueError(f"Failed to parse {file_path}: {e}")
        finally:
            # Clean up temporary preprocessed file
            if preprocessed_path and preprocessed_path.exists():
                try:
                    preprocessed_path.unlink()
                except Exception:
                    pass  # Ignore cleanup errors
    
    def _extract_metadata(
        self,
        reader: pylangacq.Reader,
        file_path: Path
    ) -> Dict[str, Any]:
        """
        Extract metadata from CHAT file headers.
        
        Parses header lines (@PID, @Date, @Languages, etc.) to extract
        session and participant metadata.
        
        Args:
            reader: pylangacq Reader object
            file_path: Path to the file being parsed
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Get headers from first file in reader
        headers = reader.headers()
        
        if not headers:
            logger.warning(f"No headers found in {file_path.name}")
            return metadata
        
        # Extract from first file (usually only one file per reader)
        # Handle both dict and list formats from pylangacq
        if isinstance(headers, dict):
            file_headers = list(headers.values())[0] if headers else {}
        elif isinstance(headers, list) and headers:
            file_headers = headers[0] if headers[0] else {}
        else:
            file_headers = {}
        
        # Extract participant ID
        pid = file_headers.get('PID', '')
        if pid:
            # PIDs are often in format "corpus/id"
            metadata['pid'] = pid
            parts = pid.split('/')
            if len(parts) > 1:
                metadata['participant_id'] = parts[-1]
        
        # Extract date
        date_str = file_headers.get('Date', '')
        if date_str:
            # Handle different date formats from pylangacq
            if isinstance(date_str, date):
                # Already a date object
                metadata['session_date'] = date_str
                metadata['date_str'] = date_str.strftime('%d-%b-%Y')
            elif isinstance(date_str, (set, list)):
                # Handle set/list format
                date_str = list(date_str)[0] if date_str else ''
                if date_str and isinstance(date_str, str):
                    metadata['date_str'] = date_str
                    try:
                        metadata['session_date'] = datetime.strptime(date_str, '%d-%b-%Y')
                    except ValueError:
                        logger.warning(f"Could not parse date: {date_str}")
            elif isinstance(date_str, str):
                # String format
                metadata['date_str'] = date_str
                try:
                    metadata['session_date'] = datetime.strptime(date_str, '%d-%b-%Y')
                except ValueError:
                    logger.warning(f"Could not parse date: {date_str}")
            else:
                # Convert to string for storage
                metadata['date_str'] = str(date_str)
        
        # Extract languages
        langs = file_headers.get('Languages', '')
        if langs:
            # Handle case where langs might be a set or other type
            if isinstance(langs, (set, list)):
                langs = list(langs)[0] if langs else ''
            elif not isinstance(langs, str):
                langs = str(langs)
            
            if langs:
                metadata['languages'] = [l.strip() for l in langs.split(',')]
        
        # Extract other headers
        for key in ['Media', 'Location', 'Situation', 'Comment', 'Tape Location']:
            if key in file_headers:
                metadata[key.lower().replace(' ', '_')] = file_headers[key]
        
        return metadata
    
    def _extract_participants(
        self,
        reader: pylangacq.Reader
    ) -> Dict[str, Dict[str, str]]:
        """
        Extract participant information from @ID headers.
        
        @ID format: lang|corpus|code|age|sex|group|SES|role|education|custom
        
        Args:
            reader: pylangacq Reader object
            
        Returns:
            Dictionary mapping speaker codes to participant info
        """
        participants = {}
        
        # Get headers
        headers = reader.headers()
        if not headers:
            return participants
            
        # Get first file headers
        if isinstance(headers, list) and headers:
            file_headers = headers[0]
        elif isinstance(headers, dict):
            file_headers = list(headers.values())[0] if headers else {}
        else:
            file_headers = {}
            
        # Extract participants dict from headers
        # key 'Participants' usually corresponds to @ID fields in pylangacq
        participant_data = file_headers.get('Participants', {})
        
        if not participant_data:
            # Fallback to reader.participants() which might just be codes
            # But we can't get metadata from that
            logger.debug("No detailed participant data in headers")
            return participants
            
        for speaker_code, info in participant_data.items():
            if not isinstance(info, dict):
                continue
                
            participants[speaker_code] = {
                'code': speaker_code,
                'language': info.get('language', ''),
                'corpus': info.get('corpus', ''),
                'age': info.get('age', ''),
                'sex': info.get('sex', ''),
                'group': info.get('group', ''),
                'SES': info.get('ses', ''),  # Note: lower case in dict
                'role': info.get('role', ''),
                'education': info.get('education', ''),
                'custom': info.get('custom', ''),
            }
            
            # Extract diagnosis from group field (ASD, TD, DD, etc.)
            group = info.get('group', '').upper()
            if speaker_code == 'CHI' and group:
                # Store diagnosis in metadata (will be used in TranscriptData)
                participants['_diagnosis'] = group
            
            # Extract age in months from age field (format: Y;MM.DD)
            age_str = info.get('age', '')
            if speaker_code == 'CHI' and age_str:
                age_months = get_age_in_months(age_str)
                if age_months:
                    participants['_age_months'] = age_months
            
            # Extract gender
            sex = info.get('sex', '').lower()
            if speaker_code == 'CHI' and sex:
                participants['_gender'] = sex
        
        return participants
    
    def _extract_utterances(
        self,
        reader: pylangacq.Reader
    ) -> List[Utterance]:
        """
        Extract all utterances from the transcript.
        
        This method processes each utterance and extracts:
        - Speaker code
        - Utterance text
        - Tokens
        - Morphological analysis (%mor)
        - Grammatical relations (%gra)
        - Timing information (%tim)
        - Actions (%act)
        - Comments (%com)
        
        Args:
            reader: pylangacq Reader object
            
        Returns:
            List of Utterance objects
        """
        utterances = []
        skipped_count = 0
        
        # Get utterances from reader
        for utterance in reader.utterances():
            try:
                # Extract speaker and text
                speaker = utterance.participant
                text = utterance.tiers.get('utterance', '')
                
                # Get tokens (words)
                tokens = utterance.tokens or []
                
                # If text is empty but we have tokens, reconstruct text from tokens
                if not text and tokens:
                    text = ' '.join(token.word for token in tokens if token.word)
                
                # Get tiers (morphology, grammar, timing, etc.)
                tiers = utterance.tiers or {}
                
                # Extract morphology (%mor) - try from tiers first, then from tokens
                morphology = tiers.get('mor')
                if not morphology and tokens:
                    # Extract morphology from token morpheme information
                    mor_list = []
                    for token in tokens:
                        if hasattr(token, 'mor') and token.mor:
                            mor_list.append(token.mor)
                    morphology = ' '.join(mor_list) if mor_list else None
                
                # Extract grammar (%gra)
                grammar = tiers.get('gra')
                
                # Extract timing
                timing = None
                end_timing = None
                
                # Try getting timing from time_marks first (most reliable for bullets)
                if hasattr(utterance, 'time_marks') and utterance.time_marks:
                    start_ms, end_ms = utterance.time_marks
                    timing = float(start_ms) / 1000.0
                    end_timing = float(end_ms) / 1000.0
                
                # Fallback to %tim tier if time_marks not available
                if timing is None:
                    timing_str = tiers.get('tim')
                    if timing_str:
                        timing = extract_timing_info(timing_str)
                
                # Extract actions (%act)
                actions = tiers.get('act')
                
                # Extract comments (%com)
                comments = tiers.get('com')
                
                # Create Utterance object first
                utt = Utterance(
                    speaker=speaker,
                    text=text,
                    tokens=tokens,
                    morphology=morphology,
                    grammar=grammar,
                    timing=timing,
                    end_timing=end_timing,
                    actions=actions,
                    comments=comments,
                    is_valid=True,  # Will be validated below
                )
                
                # Validate utterance after creation (can use word_count property)
                if utt.word_count < self.min_words:
                    utt.is_valid = False
                
                utterances.append(utt)
                
            except Exception as e:
                # Skip utterances that cause alignment or other errors
                # This commonly happens with '0' prefix tokens in some CHAT files
                skipped_count += 1
                logger.debug(f"Skipped utterance due to error: {str(e)[:100]}")
                continue
        
        # Log if any utterances were skipped
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} utterances due to alignment or parsing errors")
        
        return utterances
    
    def parse_directory(
        self,
        directory: str | Path,
        pattern: str = "**/*.cha",
        recursive: bool = True
    ) -> List[TranscriptData]:
        """
        Parse all CHAT files in a directory.
        
        Args:
            directory: Path to directory containing .cha files
            pattern: Glob pattern for finding files (default: "**/*.cha")
            recursive: Whether to search recursively
            
        Returns:
            List of TranscriptData objects
            
        Example:
            >>> parser = CHATParser()
            >>> transcripts = parser.parse_directory("data/asdbank_eigsti")
            >>> print(f"Parsed {len(transcripts)} transcripts")
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all .cha files
        if recursive:
            files = list(directory.glob(pattern))
        else:
            files = list(directory.glob("*.cha"))
        
        logger.info(f"Found {len(files)} .cha files in {directory}")
        
        transcripts = []
        errors = []
        
        for file_path in files:
            try:
                transcript = self.parse_file(file_path)
                transcripts.append(transcript)
            except Exception as e:
                logger.error(f"Failed to parse {file_path.name}: {e}")
                errors.append((file_path, str(e)))
        
        logger.info(
            f"Successfully parsed {len(transcripts)}/{len(files)} files "
            f"({len(errors)} errors)"
        )
        
        if errors:
            logger.warning(f"Errors occurred in {len(errors)} files")
        
        return transcripts


__all__ = ["CHATParser", "TranscriptData", "Utterance"]

