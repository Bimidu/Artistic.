"""
Unified Input Handler Module

This module provides a unified interface for processing both audio and text inputs
through the ASD detection pipeline. It automatically determines the input type
and routes it through the appropriate processing path.

Author: Bimidu Gunathilake
"""

from pathlib import Path
from typing import Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger import get_logger
from src.parsers.chat_parser import CHATParser, TranscriptData
from src.audio.audio_processor import AudioProcessor, AudioProcessingResult

logger = get_logger(__name__)


class InputType(Enum):
    """Enumeration of supported input types."""
    AUDIO = "audio"
    CHAT_FILE = "chat_file"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class ProcessedInput:
    """
    Container for processed input data.
    
    Provides a unified format regardless of whether input was
    audio or text based.
    
    Attributes:
        input_type: Type of original input
        transcript_data: Parsed transcript data
        audio_path: Path to audio file (for feature extractors to use)
        transcription_result: TranscriptionResult with timing (for audio inputs)
        raw_text: Raw text content
        source_path: Path to source file if applicable
        metadata: Additional processing metadata
        
    NOTE: Audio features are NOT extracted here. Each feature module
    extracts its own audio features using the audio_path.
    """
    input_type: InputType
    transcript_data: TranscriptData
    audio_path: Optional[Path] = None
    transcription_result: Optional[Any] = None
    raw_text: str = ""
    source_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_audio(self) -> bool:
        """Check if audio file is available for feature extraction."""
        return self.audio_path is not None and self.audio_path.exists()
    
    @property
    def is_from_audio(self) -> bool:
        """Check if input originated from audio."""
        return self.input_type == InputType.AUDIO


class InputHandler:
    """
    Unified handler for audio and text inputs.
    
    Automatically determines input type and routes through appropriate
    processing pipeline:
    
    - Audio files (.wav, .mp3, etc.) → Audio Processor → Transcript
    - CHAT files (.cha) → CHAT Parser → Transcript
    - Raw text → Direct text processing → Transcript
    
    Example:
        >>> handler = InputHandler()
        >>> result = handler.process("recording.wav")
        >>> print(f"Type: {result.input_type}")
        >>> print(f"Utterances: {result.transcript_data.total_utterances}")
        
        >>> result = handler.process("transcript.cha")
        >>> print(f"Type: {result.input_type}")
    """
    
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    CHAT_EXTENSIONS = {'.cha'}
    TEXT_EXTENSIONS = {'.txt'}
    
    def __init__(
        self,
        transcriber_backend: str = 'whisper',
        whisper_model_size: str = 'base',
        device: str = 'cpu',
        language: str = 'en'
    ):
        """
        Initialize the input handler.
        
        Args:
            transcriber_backend: Backend for audio transcription
            whisper_model_size: Whisper model size if using whisper
            device: Device for processing
            language: Target language
        """
        # Initialize parsers
        self.chat_parser = CHATParser()
        
        # Initialize audio processor (lazy loading)
        self._audio_processor = None
        self._audio_config = {
            'transcriber_backend': transcriber_backend,
            'whisper_model_size': whisper_model_size,
            'device': device,
            'language': language
        }
        
        logger.info("InputHandler initialized")
    
    @property
    def audio_processor(self) -> AudioProcessor:
        """Lazy-load audio processor."""
        if self._audio_processor is None:
            self._audio_processor = AudioProcessor(**self._audio_config)
        return self._audio_processor
    
    def determine_input_type(self, input_path: str | Path) -> InputType:
        """
        Determine the type of input file.
        
        Args:
            input_path: Path to input file
            
        Returns:
            InputType enum value
        """
        path = Path(input_path)
        suffix = path.suffix.lower()
        
        if suffix in self.AUDIO_EXTENSIONS:
            return InputType.AUDIO
        elif suffix in self.CHAT_EXTENSIONS:
            return InputType.CHAT_FILE
        elif suffix in self.TEXT_EXTENSIONS:
            return InputType.TEXT
        else:
            return InputType.UNKNOWN
    
    def process(
        self,
        input_source: Union[str, Path],
        participant_id: Optional[str] = None,
        diagnosis: Optional[str] = None,
        **kwargs
    ) -> ProcessedInput:
        """
        Process any supported input type.
        
        Args:
            input_source: Path to input file or raw text
            participant_id: Optional participant ID
            diagnosis: Optional diagnosis label
            **kwargs: Additional processing arguments
            
        Returns:
            ProcessedInput with unified format
        """
        # Determine if input is a file or raw text
        if isinstance(input_source, str) and not Path(input_source).exists():
            # Treat as raw text
            return self._process_raw_text(input_source, participant_id, diagnosis)
        
        input_path = Path(input_source)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        input_type = self.determine_input_type(input_path)
        
        if input_type == InputType.AUDIO:
            return self._process_audio(input_path, participant_id, diagnosis, **kwargs)
        elif input_type == InputType.CHAT_FILE:
            return self._process_chat_file(input_path, participant_id, diagnosis)
        elif input_type == InputType.TEXT:
            return self._process_text_file(input_path, participant_id, diagnosis)
        else:
            raise ValueError(f"Unsupported input type: {input_path.suffix}")
    
    def _process_audio(
        self,
        audio_path: Path,
        participant_id: Optional[str],
        diagnosis: Optional[str],
        **kwargs
    ) -> ProcessedInput:
        """Process audio input - transcription only, features extracted later."""
        logger.info(f"Processing audio input: {audio_path.name}")
        
        pid = participant_id or audio_path.stem
        
        # Process audio (transcription only)
        result: AudioProcessingResult = self.audio_processor.process(
            audio_path,
            participant_id=pid,
            diagnosis=diagnosis,
            **kwargs
        )
        
        return ProcessedInput(
            input_type=InputType.AUDIO,
            transcript_data=result.transcript_data,
            audio_path=audio_path,
            transcription_result=result.transcription,
            raw_text=result.transcription.text,
            source_path=audio_path,
            metadata={
                'transcription_segments': len(result.transcription.segments),
                'duration': result.transcription.duration,
            }
        )
    
    def _process_chat_file(
        self,
        chat_path: Path,
        participant_id: Optional[str],
        diagnosis: Optional[str]
    ) -> ProcessedInput:
        """Process CHAT file input."""
        logger.info(f"Processing CHAT file: {chat_path.name}")
        
        # Parse CHAT file
        transcript = self.chat_parser.parse_file(chat_path)
        
        # Override participant_id and diagnosis if provided
        if participant_id:
            transcript.participant_id = participant_id
        if diagnosis:
            transcript.diagnosis = diagnosis
        
        # Combine all utterance text
        raw_text = " ".join(u.text for u in transcript.utterances)
        
        return ProcessedInput(
            input_type=InputType.CHAT_FILE,
            transcript_data=transcript,
            raw_text=raw_text,
            source_path=chat_path,
            metadata={
                'total_utterances': transcript.total_utterances,
                'child_utterances': len(transcript.child_utterances),
            }
        )
    
    def _process_text_file(
        self,
        text_path: Path,
        participant_id: Optional[str],
        diagnosis: Optional[str]
    ) -> ProcessedInput:
        """Process plain text file input."""
        logger.info(f"Processing text file: {text_path.name}")
        
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self._process_raw_text(text, participant_id, diagnosis, source_path=text_path)
    
    def _process_raw_text(
        self,
        text: str,
        participant_id: Optional[str],
        diagnosis: Optional[str],
        source_path: Optional[Path] = None
    ) -> ProcessedInput:
        """Process raw text input."""
        logger.info("Processing raw text input")
        
        from src.parsers.chat_parser import Utterance
        
        pid = participant_id or "CHI"
        
        # Split text into sentences/utterances
        sentences = self._split_into_sentences(text)
        
        utterances = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                utterance = Utterance(
                    speaker=pid,
                    text=sentence.strip(),
                    tokens=sentence.strip().split(),
                    is_valid=len(sentence.strip().split()) >= 1
                )
                utterances.append(utterance)
        
        # Create TranscriptData
        transcript = TranscriptData(
            file_path=source_path or Path("raw_text"),
            participant_id=pid,
            diagnosis=diagnosis,
            utterances=utterances,
            metadata={'source': 'raw_text'}
        )
        
        return ProcessedInput(
            input_type=InputType.TEXT,
            transcript_data=transcript,
            raw_text=text,
            source_path=source_path,
            metadata={'sentence_count': len(sentences)}
        )
    
    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting on punctuation
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def process_batch(
        self,
        input_sources: list,
        participant_ids: Optional[list] = None,
        diagnoses: Optional[list] = None,
        **kwargs
    ) -> list:
        """
        Process multiple inputs.
        
        Args:
            input_sources: List of input paths or texts
            participant_ids: Optional list of participant IDs
            diagnoses: Optional list of diagnoses
            **kwargs: Additional processing arguments
            
        Returns:
            List of ProcessedInput objects
        """
        results = []
        
        for i, source in enumerate(input_sources):
            pid = participant_ids[i] if participant_ids else None
            diag = diagnoses[i] if diagnoses else None
            
            try:
                result = self.process(source, participant_id=pid, diagnosis=diag, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {source}: {e}")
                results.append(None)
        
        return results


__all__ = ["InputHandler", "ProcessedInput", "InputType"]

