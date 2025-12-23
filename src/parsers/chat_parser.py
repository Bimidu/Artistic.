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
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

import pylangacq
import pandas as pd
from src.utils.logger import get_logger
from src.utils.helpers import extract_timing_info, get_age_in_months, is_valid_utterance

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
        logger.info(f"CHATParser initialized with min_words={min_words}")
    
    def parse_file(self, file_path: Union[str, Path]) -> TranscriptData:
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

        # Initialize variables for exception handling
        metadata = {}
        participants = {}

        try:
            # Use pylangacq to read the file
            reader = pylangacq.read_chat(str(file_path))

            # Extract metadata
            metadata = self._extract_metadata(reader, file_path)

            # Extract participant information
            participants = self._extract_participants(reader)

            # Extract diagnosis, age, and gender from participants
            if '_diagnosis' in participants:
                metadata['diagnosis'] = participants['_diagnosis']
            if '_age_months' in participants:
                metadata['age_months'] = participants['_age_months']
            if '_gender' in participants:
                metadata['gender'] = participants['_gender']

            # Create TranscriptData object
            transcript = TranscriptData(
                file_path=file_path,
                participant_id=metadata.get('participant_id', file_path.stem),
                diagnosis=metadata.get('diagnosis'),
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

        except ValueError as e:
            # Catch alignment errors from pylangacq and handle gracefully
            error_msg = str(e)
            if "align" in error_msg.lower():
                logger.warning(
                    f"Skipping file {file_path.name} due to alignment error "
                    f"(likely contains 0-prefixed words without morphology): {error_msg[:100]}"
                )
                # Return a minimal transcript with metadata but no utterances
                return TranscriptData(
                    file_path=file_path,
                    participant_id=metadata.get('participant_id', file_path.stem),
                    diagnosis=metadata.get('diagnosis'),
                    age_months=metadata.get('age_months'),
                    gender=metadata.get('gender'),
                    session_date=metadata.get('session_date'),
                    session_type=metadata.get('session_type'),
                    languages=metadata.get('languages', []),
                    metadata=metadata,
                    speakers=participants,
                )
            else:
                logger.error(f"Error parsing file {file_path}: {e}")
                raise ValueError(f"Failed to parse {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            raise ValueError(f"Failed to parse {file_path}: {e}")
    
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

        # Access participant data from the internal file structure
        if not reader._files:
            logger.warning("No files in reader")
            return participants

        first_file = reader._files[0]

        # Get participant data from header
        if 'Participants' not in first_file.header:
            logger.warning("No Participants in file header")
            return participants

        file_participants = first_file.header['Participants']

        if not isinstance(file_participants, dict):
            logger.warning(f"Participants data is not a dict: {type(file_participants)}")
            return participants

        for speaker_code, info in file_participants.items():
            participants[speaker_code] = {
                'code': speaker_code,
                'language': info.get('language', ''),
                'corpus': info.get('corpus', ''),
                'age': info.get('age', ''),
                'sex': info.get('sex', ''),
                'group': info.get('group', ''),
                'SES': info.get('SES', ''),
                'role': info.get('role', ''),
                'education': info.get('education', ''),
                'custom': info.get('custom', ''),
            }
            
            # Extract diagnosis from group field (ASD, TD, DD, etc.)
            group = info.get('group', '').upper()
            if speaker_code == 'CHI' and group:
                # Validate that group is a diagnosis label, not an age value
                # Age values typically match pattern: X;YY. or similar
                # Valid diagnosis labels: ASD, TD, TYP, DD, ADHD, SLI, etc.
                if not re.match(r'^\d+;\d+\.?$', group):  # Reject age-like patterns
                    # Store diagnosis in metadata (will be used in TranscriptData)
                    participants['_diagnosis'] = group
                else:
                    logger.warning(f"Rejected age-like value '{group}' as diagnosis for {speaker_code}")
            
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
                
                # Extract timing (%tim)
                timing = None
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
                    actions=actions,
                    comments=comments,
                    is_valid=True,  # Will be validated below
                )
                
                # Validate utterance after creation (can use word_count property)
                if utt.word_count < self.min_words:
                    utt.is_valid = False
                
                utterances.append(utt)
                
            except Exception as e:
                logger.warning(f"Error extracting utterance: {e}")
                continue
        
        return utterances
    
    def parse_directory(
        self,
        directory: Union[str, Path],
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

