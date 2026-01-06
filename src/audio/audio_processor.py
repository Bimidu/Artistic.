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
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field
import numpy as np

from src.utils.logger import get_logger
from src.parsers.chat_parser import TranscriptData, Utterance
from .transcriber import AudioTranscriber, TranscriptionResult

logger = get_logger(__name__)

# Try to import librosa for pitch analysis
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available - speaker identification will be limited")


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
        
        # Step 2: Identify speakers using pitch analysis (if audio available)
        if transcription.segments and LIBROSA_AVAILABLE:
            transcription = self._identify_speakers_by_pitch(
                transcription,
                audio_path
            )
        
        # Step 3: Convert to TranscriptData format
        transcript_data = self._transcription_to_transcript_data(
            transcription,
            audio_path,
            participant_id,
            diagnosis
        )
        
        # Step 4: Generate CHAT file content
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
    
    def _identify_speakers_by_pitch(
        self,
        transcription: TranscriptionResult,
        audio_path: Path
    ) -> TranscriptionResult:
        """
        Identify child vs adult speakers using pitch (F0) analysis.
        
        Children typically have higher pitch (F0 > 250 Hz) than adults.
        This method extracts pitch from each audio segment and classifies
        speakers accordingly.
        
        Args:
            transcription: Transcription result with segments
            audio_path: Path to audio file
            
        Returns:
            TranscriptionResult with updated speaker labels
        """
        if not LIBROSA_AVAILABLE:
            logger.warning("Librosa not available - cannot identify speakers by pitch")
            return transcription
        
        if not transcription.segments:
            return transcription
        
        try:
            # Load audio file
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            duration = len(audio) / sr
            
            logger.info(f"Analyzing pitch for {len(transcription.segments)} segments to identify speakers")
            
            # Extract pitch for each segment
            segment_pitches = []
            for segment in transcription.segments:
                start_time = segment.start_time
                end_time = segment.end_time
                
                # Convert time to sample indices
                start_sample = int(start_time * sr)
                end_sample = int(min(end_time * sr, len(audio)))
                
                if end_sample <= start_sample or start_sample >= len(audio):
                    segment_pitches.append(None)
                    continue
                
                # Extract audio segment
                segment_audio = audio[start_sample:end_sample]
                
                if len(segment_audio) < 1024:  # Too short for pitch analysis
                    segment_pitches.append(None)
                    continue
                
                try:
                    # Extract pitch using librosa's pyin algorithm
                    f0, _, _ = librosa.pyin(
                        segment_audio,
                        fmin=librosa.note_to_hz('C2'),  # ~65 Hz (lowest adult male)
                        fmax=librosa.note_to_hz('C7'),  # ~2093 Hz (highest child)
                        frame_length=2048,
                        hop_length=512
                    )
                    
                    # Filter out unvoiced frames (NaN values)
                    f0_voiced = f0[~np.isnan(f0)]
                    
                    if len(f0_voiced) > 0:
                        mean_pitch = np.mean(f0_voiced)
                        segment_pitches.append(mean_pitch)
                    else:
                        segment_pitches.append(None)
                        
                except Exception as e:
                    logger.debug(f"Error extracting pitch for segment {start_time}-{end_time}: {e}")
                    segment_pitches.append(None)
            
            # Classify speakers based on pitch
            # Children typically have F0 > 250 Hz, adults < 250 Hz
            # But we'll use a clustering approach: find the pitch distribution
            # and assign higher-pitched segments as child
            
            valid_pitches = [p for p in segment_pitches if p is not None]
            
            if len(valid_pitches) < 2:
                # Not enough data for classification, use default
                logger.warning("Not enough pitch data for speaker identification")
                return transcription
            
            # Determine threshold for child vs adult classification
            # Children typically have F0 > 250 Hz, adults < 250 Hz
            # But we'll use a relative approach based on the distribution
            
            mean_pitch = np.mean(valid_pitches)
            std_pitch = np.std(valid_pitches)
            
            # If there's a clear bimodal distribution (high std), use clustering
            # Otherwise, use absolute threshold (250 Hz) or relative threshold
            if len(valid_pitches) >= 4 and std_pitch > 50:  # Significant variation suggests multiple speakers
                # Use 75th percentile as threshold (top 25% are likely children)
                pitch_threshold = np.percentile(valid_pitches, 75)
                logger.debug(f"Using percentile-based threshold (std={std_pitch:.1f}): {pitch_threshold:.1f} Hz")
            elif mean_pitch > 250:
                # High overall pitch suggests mostly child speech
                # Use a threshold slightly below mean to catch any adult segments
                pitch_threshold = mean_pitch - std_pitch * 0.5
                logger.debug(f"Using mean-based threshold (high pitch): {pitch_threshold:.1f} Hz")
            else:
                # Lower overall pitch - use absolute threshold
                pitch_threshold = 250.0  # Standard threshold for child vs adult
                logger.debug(f"Using absolute threshold: {pitch_threshold:.1f} Hz")
            
            # Update speaker labels
            child_count = 0
            adult_count = 0
            
            for i, segment in enumerate(transcription.segments):
                pitch = segment_pitches[i]
                
                if pitch is None:
                    # If we can't determine pitch, keep existing speaker or default to participant_id
                    if not segment.speaker:
                        segment.speaker = "CHI"  # Default to child
                    continue
                
                # Classify based on pitch threshold
                # Children typically have higher pitch
                if pitch > pitch_threshold:
                    segment.speaker = "CHI"
                    child_count += 1
                else:
                    # Assign adult speaker code (MOT for mother, or INV for investigator)
                    # Default to MOT (mother) as most common adult in child speech studies
                    segment.speaker = "MOT"
                    adult_count += 1
            
            logger.info(
                f"Speaker identification complete: {child_count} child segments, "
                f"{adult_count} adult segments (pitch threshold: {pitch_threshold:.1f} Hz)"
            )
            
            # Update metadata
            transcription.metadata['speaker_identification'] = 'pitch_based'
            transcription.metadata['pitch_threshold'] = float(pitch_threshold)
            transcription.metadata['child_segments'] = child_count
            transcription.metadata['adult_segments'] = adult_count
            
        except Exception as e:
            logger.warning(f"Error identifying speakers by pitch: {e}. Using default speaker labels.")
            # If pitch analysis fails, keep existing speaker labels or use defaults
        
        return transcription
    
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
