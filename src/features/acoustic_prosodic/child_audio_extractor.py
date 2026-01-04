"""
Child Audio Segment Extractor

This module extracts child-only speech segments from audio files using
transcript timing information. It creates a temporary audio file containing
only the child's speech for acoustic/prosodic feature extraction.

For datasets:
- With transcripts (.cha files): Uses *CHI: timestamps from transcript
- Without transcripts (Whisper-generated): Uses speaker-tagged segments from transcription

Author: Implementation for child-specific acoustic analysis
"""

import re
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from src.utils.logger import get_logger
from src.parsers.chat_parser import TranscriptData, Utterance

logger = get_logger(__name__)

# Try to import audio processing libraries
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("PyDub not available - child audio extraction will be limited")

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class ChildAudioExtractor:
    """
    Extract child-only speech segments from audio files.
    
    This class provides functionality to:
    1. Extract child utterances from transcripts with timing
    2. Concatenate child speech segments into a single audio
    3. Handle both .cha transcripts and Whisper-generated transcriptions
    
    Example:
        >>> extractor = ChildAudioExtractor()
        >>> child_audio_path = extractor.extract_child_audio(
        ...     audio_path="audio.wav",
        ...     transcript=transcript_data
        ... )
    """
    
    def __init__(self):
        """Initialize the child audio extractor."""
        if not PYDUB_AVAILABLE and not LIBROSA_AVAILABLE:
            logger.warning(
                "Neither PyDub nor librosa available. "
                "Install with: pip install pydub librosa soundfile"
            )
        
        logger.debug("ChildAudioExtractor initialized")
    
    def extract_child_audio(
        self,
        audio_path: Path,
        transcript: Optional[TranscriptData] = None,
        transcription_result: Optional[any] = None,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Extract child-only audio segments and save to a file.
        
        Args:
            audio_path: Path to full audio file
            transcript: Parsed transcript data with timing (optional)
            transcription_result: TranscriptionResult from Whisper (optional)
            output_path: Optional output path (if None, creates temp file)
        
        Returns:
            Path to child-only audio file, or None if extraction failed
        """
        if not audio_path or not Path(audio_path).exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return None
        
        # Get child segments from transcript or transcription result
        child_segments = self._get_child_segments(transcript, transcription_result)
        
        if not child_segments:
            logger.warning(
                f"No child segments found in transcript/transcription for {audio_path.name}. "
                f"Using full audio instead."
            )
            return audio_path  # Return original audio if no child segments found
        
        # Extract and concatenate child audio segments
        if PYDUB_AVAILABLE:
            return self._extract_with_pydub(audio_path, child_segments, output_path)
        elif LIBROSA_AVAILABLE:
            return self._extract_with_librosa(audio_path, child_segments, output_path)
        else:
            logger.error("No audio library available for child audio extraction")
            return audio_path  # Return original audio as fallback
    
    def _get_child_segments(
        self,
        transcript: Optional[TranscriptData],
        transcription_result: Optional[any]
    ) -> List[Tuple[float, float]]:
        """
        Get child speech segments (start_time, end_time) from transcript.
        
        Args:
            transcript: Parsed transcript data
            transcription_result: Whisper transcription result
        
        Returns:
            List of (start_time, end_time) tuples in seconds
        """
        segments = []
        
        # Method 1: From CHAT transcript with timing
        if transcript and transcript.utterances:
            child_utterances = [
                u for u in transcript.utterances
                if u.speaker == 'CHI' and u.timing is not None
            ]
            
            for utterance in child_utterances:
                start = utterance.timing
                end = utterance.end_timing if utterance.end_timing else start + 1.0
                
                # Validate segment
                if end > start and (end - start) < 60:  # Max 60 seconds per utterance
                    segments.append((start, end))
            
            if segments:
                logger.debug(
                    f"Found {len(segments)} child utterances from CHAT transcript "
                    f"(total duration: {sum(e-s for s,e in segments):.1f}s)"
                )
                return segments
        
        # Method 2: From Whisper transcription result with speaker diarization
        if transcription_result and hasattr(transcription_result, 'segments'):
            # First, try to find segments explicitly labeled as CHI (from diarization)
            for segment in transcription_result.segments:
                if hasattr(segment, 'speaker') and segment.speaker == 'CHI':
                    start = segment.start_time
                    end = segment.end_time
                    
                    if end > start and (end - start) < 60:
                        segments.append((start, end))
            
            if segments:
                logger.debug(
                    f"Found {len(segments)} child segments from Whisper transcription (with diarization) "
                    f"(total duration: {sum(e-s for s,e in segments):.1f}s)"
                )
                return segments
            
            # Fallback: If no speaker labels but we have a transcript with participant_id='CHI',
            # use all segments (assumes all speech is from child in user mode)
            if transcript and transcript.participant_id == 'CHI':
                for segment in transcription_result.segments:
                    if hasattr(segment, 'start_time') and hasattr(segment, 'end_time'):
                        start = segment.start_time
                        end = segment.end_time
                        
                        if end > start and (end - start) < 60:
                            segments.append((start, end))
                
                if segments:
                    logger.info(
                        f"Using all {len(segments)} transcription segments as child speech "
                        f"(participant_id=CHI, no speaker diarization available). "
                        f"Total duration: {sum(e-s for s,e in segments):.1f}s"
                    )
                    return segments
        
        # Method 3: Parse timing from text (fallback for some CHAT files)
        if transcript and transcript.utterances:
            segments_from_text = self._parse_timing_from_text(transcript)
            if segments_from_text:
                logger.debug(
                    f"Extracted {len(segments_from_text)} child segments from text timestamps"
                )
                return segments_from_text
        
        return segments
    
    def _parse_timing_from_text(self, transcript: TranscriptData) -> List[Tuple[float, float]]:
        """
        Parse timing information from utterance text (e.g., "text . 1000_2000").
        
        Some CHAT files include timing in the utterance text itself.
        """
        segments = []
        
        for utterance in transcript.utterances:
            if utterance.speaker != 'CHI':
                continue
            
            # Look for timing pattern: "text . start_end" or "text start_end"
            text = utterance.text
            
            # Pattern: digits_digits (e.g., "1000_2000" for milliseconds)
            timing_pattern = r'(\d+)_(\d+)'
            matches = re.findall(timing_pattern, text)
            
            for match in matches:
                start_ms = int(match[0])
                end_ms = int(match[1])
                
                # Convert milliseconds to seconds
                start_sec = start_ms / 1000.0
                end_sec = end_ms / 1000.0
                
                if end_sec > start_sec and (end_sec - start_sec) < 60:
                    segments.append((start_sec, end_sec))
        
        return segments
    
    def _extract_with_pydub(
        self,
        audio_path: Path,
        segments: List[Tuple[float, float]],
        output_path: Optional[Path] = None
    ) -> Path:
        """Extract child audio using PyDub."""
        try:
            # Load full audio
            audio = AudioSegment.from_wav(str(audio_path))
            
            # Concatenate child segments
            child_audio = AudioSegment.empty()
            total_duration = 0
            
            for start, end in segments:
                # Convert to milliseconds for pydub
                start_ms = int(start * 1000)
                end_ms = int(end * 1000)
                
                # Extract segment
                segment = audio[start_ms:end_ms]
                child_audio += segment
                total_duration += (end - start)
            
            if len(child_audio) == 0:
                logger.warning("No child audio extracted")
                return audio_path
            
            # Create output file
            if output_path is None:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.wav',
                    delete=False,
                    prefix='child_audio_'
                )
                output_path = Path(temp_file.name)
            
            # Export child audio
            child_audio.export(str(output_path), format="wav")
            
            logger.info(
                f"Extracted {len(segments)} child segments "
                f"({total_duration:.1f}s total) to {output_path.name}"
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error extracting child audio with PyDub: {e}")
            return audio_path
    
    def _extract_with_librosa(
        self,
        audio_path: Path,
        segments: List[Tuple[float, float]],
        output_path: Optional[Path] = None
    ) -> Path:
        """Extract child audio using librosa."""
        try:
            # Load full audio
            audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
            
            # Concatenate child segments
            child_segments_audio = []
            total_duration = 0
            
            for start, end in segments:
                # Convert to sample indices
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                
                # Extract segment
                segment = audio[start_sample:end_sample]
                child_segments_audio.append(segment)
                total_duration += (end - start)
            
            if not child_segments_audio:
                logger.warning("No child audio extracted")
                return audio_path
            
            # Concatenate all segments
            child_audio = np.concatenate(child_segments_audio)
            
            # Create output file
            if output_path is None:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.wav',
                    delete=False,
                    prefix='child_audio_'
                )
                output_path = Path(temp_file.name)
            
            # Save child audio
            sf.write(str(output_path), child_audio, sr)
            
            logger.info(
                f"Extracted {len(segments)} child segments "
                f"({total_duration:.1f}s total) to {output_path.name}"
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error extracting child audio with librosa: {e}")
            return audio_path


__all__ = ["ChildAudioExtractor"]

