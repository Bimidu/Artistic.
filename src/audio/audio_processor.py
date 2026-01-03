"""
Audio Processor Module

Common audio preprocessing for the ASD detection pipeline.
This module handles:
1. Audio loading and preprocessing
2. Speech-to-text transcription
3. Conversion to transcript format for downstream processing

NOTE: Audio FEATURE EXTRACTION is done by each feature module separately.
This module only provides the common preprocessing step.

Author: Bimidu Gunathilake
"""

from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

from src.utils.logger import get_logger
from src.parsers.chat_parser import TranscriptData, Utterance
from .transcriber import AudioTranscriber, TranscriptionResult

logger = get_logger(__name__)


@dataclass
class AudioProcessingResult:
    """
    Result from common audio preprocessing.
    
    Contains:
    - Original transcription with timestamps
    - Transcript data in CHAT-compatible format
    - Audio file path (for feature extractors to use)
    - Processing metadata
    
    NOTE: Audio features are extracted by individual feature modules,
    not in this common preprocessing step.
    """
    transcription: TranscriptionResult
    transcript_data: TranscriptData
    audio_path: Path
    chat_file_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if processing was successful."""
        return bool(self.transcription.text)
    
    @property
    def has_timing(self) -> bool:
        """Check if transcription has timing information."""
        return len(self.transcription.segments) > 0


class AudioProcessor:
    """
    Common audio preprocessing for ASD detection.
    
    This module handles the COMMON preprocessing steps:
    1. Load and validate audio file
    2. Transcribe using speech recognition
    3. Convert to TranscriptData format
    
    Audio FEATURE EXTRACTION is done separately by each feature module:
    - src/features/pragmatic_conversational/ handles pause/timing features
    - src/features/acoustic_prosodic/ handles pitch/prosody features
    - src/features/syntactic_semantic/ handles text-derived features
    
    Example:
        >>> processor = AudioProcessor()
        >>> result = processor.process("audio.wav")
        >>> print(result.transcript_data.total_utterances)
        >>> # Then pass result to feature extractors
    """
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    def __init__(
        self,
        transcriber_backend: str = 'whisper',
        whisper_model_size: str = 'base',
        device: str = 'cpu',
        language: str = 'en',
    ):
        """
        Initialize the audio processor.
        
        Args:
            transcriber_backend: Backend for transcription ('faster-whisper', 'whisper', 'vosk', 'google')
            whisper_model_size: Model size if using whisper/faster-whisper
            device: Device for processing ('cpu' or 'cuda')
            language: Target language for transcription
        """
        self.language = language
        
        # Initialize transcriber
        try:
            self.transcriber = AudioTranscriber(
                backend=transcriber_backend,
                model_size=whisper_model_size,
                device=device,
                language=language
            )
            self.transcriber_available = True
        except ImportError as e:
            logger.warning(f"Transcriber not available: {e}")
            self.transcriber = None
            self.transcriber_available = False
        
        logger.info(f"AudioProcessor initialized (transcriber_available={self.transcriber_available})")
    
    def process(
        self,
        audio_path: str | Path,
        participant_id: str = "CHI",
        diagnosis: Optional[str] = None,
        use_diarization: bool = False,
        num_speakers: int = 2
    ) -> AudioProcessingResult:
        """
        Process an audio file - transcription only.
        
        Args:
            audio_path: Path to audio file
            participant_id: Default participant ID for the child
            diagnosis: Diagnosis label if known
            use_diarization: Whether to attempt speaker diarization
            num_speakers: Number of expected speakers if using diarization
            
        Returns:
            AudioProcessingResult with transcription and transcript data
        """
        audio_path = Path(audio_path)
        
        # Validate file
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if audio_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {audio_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
        
        logger.info(f"Processing audio file: {audio_path.name}")
        
        # Step 1: Transcribe audio
        if self.transcriber_available:
            if use_diarization:
                transcription = self.transcriber.transcribe_with_diarization(
                    audio_path,
                    num_speakers=num_speakers
                )
            else:
                transcription = self.transcriber.transcribe(audio_path)
        else:
            # Return empty transcription if no transcriber
            transcription = TranscriptionResult(
                text="",
                segments=[],
                language=self.language,
                metadata={'error': 'Transcriber not available'}
            )
        
        # Step 2: Convert to TranscriptData format
        transcript_data = self._transcription_to_transcript_data(
            transcription,
            audio_path,
            participant_id,
            diagnosis
        )
        
        # Step 3: Generate CHAT file content
        chat_content = transcription.to_chat_format(participant_id)
        
        logger.info(
            f"Audio preprocessing complete: {len(transcription.segments)} segments"
        )
        
        return AudioProcessingResult(
            transcription=transcription,
            transcript_data=transcript_data,
            audio_path=audio_path,
            chat_file_content=chat_content,
            metadata={
                'file_path': str(audio_path),
                'transcriber_backend': self.transcriber.backend if self.transcriber else None,
                'participant_id': participant_id,
                'diagnosis': diagnosis,
                'duration': transcription.duration,
            }
        )
    
    def _transcription_to_transcript_data(
        self,
        transcription: TranscriptionResult,
        file_path: Path,
        participant_id: str,
        diagnosis: Optional[str]
    ) -> TranscriptData:
        """Convert TranscriptionResult to TranscriptData format."""
        
        # Create utterances from segments
        utterances = []
        for i, segment in enumerate(transcription.segments):
            utterance = Utterance(
                speaker=segment.speaker or participant_id,
                text=segment.text,
                tokens=segment.text.split(),  # Simple tokenization
                timing=segment.start_time,
                end_timing=segment.end_time,
                is_valid=len(segment.text.split()) >= 1
            )
            utterances.append(utterance)
        
        # Create TranscriptData
        transcript = TranscriptData(
            file_path=file_path,
            participant_id=participant_id,
            diagnosis=diagnosis,
            utterances=utterances,
            metadata={
                'source': 'audio_transcription',
                'transcription_backend': transcription.metadata.get('backend'),
                'duration': transcription.duration,
                'language': transcription.language,
            }
        )
        
        return transcript
    
    def process_batch(
        self,
        audio_paths: list,
        participant_ids: Optional[list] = None,
        diagnoses: Optional[list] = None,
        **kwargs
    ) -> list:
        """
        Process multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            participant_ids: Optional list of participant IDs
            diagnoses: Optional list of diagnoses
            **kwargs: Additional arguments for processing
            
        Returns:
            List of AudioProcessingResult objects
        """
        results = []
        
        for i, path in enumerate(audio_paths):
            pid = participant_ids[i] if participant_ids else f"P{i+1:03d}"
            diag = diagnoses[i] if diagnoses else None
            
            try:
                result = self.process(path, participant_id=pid, diagnosis=diag, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                results.append(None)
        
        return results
    
    def save_chat_file(
        self,
        result: AudioProcessingResult,
        output_path: str | Path
    ):
        """
        Save transcription as a CHAT format file.
        
        Args:
            result: Processing result
            output_path: Path to save the .cha file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.chat_file_content)
        
        logger.info(f"CHAT file saved to: {output_path}")


__all__ = ["AudioProcessor", "AudioProcessingResult"]
