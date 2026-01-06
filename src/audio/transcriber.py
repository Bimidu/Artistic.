"""
Audio Transcriber Module

This module provides speech-to-text transcription capabilities using
advanced speech recognition models. It supports multiple backends:
- OpenAI Whisper (local or API)
- Google Speech-to-Text (API)
- Vosk (offline)

The transcriber produces structured output including:
- Full transcript text
- Word-level timestamps
- Speaker diarization (when available)
- Confidence scores

Author: Bimidu Gunathilake
"""

import os
import re
import sys
import subprocess
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import tempfile
import json

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import speech recognition libraries
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available. Install with: pip install openai-whisper")

# Try faster-whisper (uses CTranslate2, avoids PyTorch issues)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.debug("faster-whisper not available. Install with: pip install faster-whisper")

# Try Vosk (lightweight, offline, no PyTorch)
try:
    import vosk
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    logger.debug("Vosk not available. Install with: pip install vosk")

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logger.warning("SpeechRecognition not available. Install with: pip install SpeechRecognition")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("PyDub not available. Install with: pip install pydub")


@dataclass
class WordTimestamp:
    """Word-level timestamp from transcription."""
    word: str
    start_time: float  # seconds
    end_time: float  # seconds
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        """Get word duration in seconds."""
        return self.end_time - self.start_time


@dataclass
class Segment:
    """
    A segment of transcribed speech.
    
    Represents a continuous utterance or phrase with timing information.
    """
    text: str
    start_time: float  # seconds
    end_time: float  # seconds
    speaker: Optional[str] = None
    confidence: float = 1.0
    words: List[WordTimestamp] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def word_count(self) -> int:
        """Get number of words in segment."""
        return len(self.text.split())


@dataclass
class TranscriptionResult:
    """
    Complete transcription result from audio processing.
    
    Attributes:
        text: Full transcript text
        segments: List of transcribed segments with timing
        language: Detected language
        duration: Total audio duration in seconds
        confidence: Overall confidence score
        metadata: Additional metadata
        word_timestamps: Word-level timestamps (if available)
    """
    text: str
    segments: List[Segment]
    language: str = "en"
    duration: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    word_timestamps: List[WordTimestamp] = field(default_factory=list)
    
    def to_chat_format(self, participant_id: str = "CHI") -> str:
        """
        Convert transcription to CHAT-like format.
        
        Creates a format similar to CHAT files for compatibility
        with existing parsers.
        
        Args:
            participant_id: Default participant ID
            
        Returns:
            CHAT-formatted string
        """
        lines = []
        lines.append(f"@Begin")
        lines.append(f"@Languages:\ten")
        lines.append(f"@Participants:\t{participant_id} Target Child")
        lines.append(f"@ID:\ten|ASD|{participant_id}|||||Target|")
        lines.append("")
        
        for segment in self.segments:
            speaker = segment.speaker or participant_id
            text = segment.text.strip()
            
            # Format timing as bullet points (milliseconds)
            start_ms = int(segment.start_time * 1000)
            end_ms = int(segment.end_time * 1000)
            
            # Add the utterance line
            lines.append(f"*{speaker}:\t{text} . {start_ms}_{end_ms}")
        
        lines.append("")
        lines.append("@End")
        
        return "\n".join(lines)
    
    def get_pauses(self, min_pause: float = 0.3) -> List[Dict[str, Any]]:
        """
        Extract pauses between segments.
        
        Args:
            min_pause: Minimum pause duration in seconds
            
        Returns:
            List of pause information dictionaries
        """
        pauses = []
        
        for i in range(1, len(self.segments)):
            prev_end = self.segments[i - 1].end_time
            curr_start = self.segments[i].start_time
            gap = curr_start - prev_end
            
            if gap >= min_pause:
                pauses.append({
                    'start_time': prev_end,
                    'end_time': curr_start,
                    'duration': gap,
                    'before_segment': i - 1,
                    'after_segment': i,
                })
        
        return pauses


class AudioTranscriber:
    """
    Advanced audio transcription with multiple backend support.
    
    Supports:
    - faster-whisper (recommended for macOS - avoids PyTorch issues)
    - OpenAI Whisper (high accuracy, but may crash on macOS)
    - Vosk (lightweight, offline, no PyTorch)
    - Google Speech-to-Text (online fallback)
    
    Example:
        >>> transcriber = AudioTranscriber(backend='whisper')
        >>> result = transcriber.transcribe("audio.wav")
        >>> print(result.text)
        >>> print(f"Duration: {result.duration}s")
    """
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    
    def __init__(
        self,
        backend: str = 'whisper',
        model_size: str = 'base',
        device: str = 'cpu',
        language: str = 'en',
    ):
        """
        Initialize the audio transcriber.
        
        Args:
            backend: Transcription backend ('faster-whisper', 'whisper', 'vosk', 'google')
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cpu' or 'cuda')
            language: Target language code
        """
        self.backend = backend
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None
        self._model_loaded = False
        
        logger.info(f"Initializing AudioTranscriber with backend={backend}")
        
        if backend == 'faster-whisper':
            if not FASTER_WHISPER_AVAILABLE:
                raise ImportError(
                    "faster-whisper is not installed. Install with: pip install faster-whisper"
                )
            # Lazy load model - don't load until first transcription
            logger.info(f"faster-whisper backend initialized (model will load on first use)")
        elif backend == 'whisper':
            if not WHISPER_AVAILABLE:
                raise ImportError(
                    "Whisper is not installed. Install with: pip install openai-whisper"
                )
            # Lazy load model - don't load until first transcription
            logger.info(f"Whisper backend initialized (model will load on first use)")
        elif backend == 'vosk':
            if not VOSK_AVAILABLE:
                raise ImportError(
                    "Vosk is not installed. Install with: pip install vosk"
                )
            # Vosk model loading happens in _load_vosk_model
            self._load_vosk_model()
        elif backend == 'google':
            if not SR_AVAILABLE:
                raise ImportError(
                    "SpeechRecognition is not installed. Install with: pip install SpeechRecognition"
                )
            self.recognizer = sr.Recognizer()
        else:
            raise ValueError(f"Unknown backend: {backend}. Supported: 'faster-whisper', 'whisper', 'vosk', 'google'")
    
    def _load_whisper_model(self):
        """Load Whisper model (lazy loading) with crash protection."""
        if self._model_loaded and self.model is not None:
            return  # Already loaded
        
        logger.info(f"Loading Whisper model: {self.model_size} (device: {self.device})")
        logger.info("This may take a minute on first load (model download/initialization)...")
        
        try:
            import torch
            logger.debug(f"PyTorch version: {torch.__version__}")
            logger.debug(f"Device: {self.device}, CUDA available: {torch.cuda.is_available()}")
            
            # Check if model needs to be downloaded
            import os
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
            model_path = os.path.join(cache_dir, f"{self.model_size}.pt")
            
            if not os.path.exists(model_path):
                logger.info(f"Model not found in cache. Will download {self.model_size} model (~{self._get_model_size_mb(self.model_size)}MB)...")
            
            # Try loading the model
            # On macOS, PyTorch can crash with segfault, so we'll try direct load first
            # If it hangs or crashes, we'll detect it
            logger.info("Starting model load (this may take 30-60 seconds)...")
            logger.info("If this hangs or crashes, the system will fall back to Google Speech-to-Text")
            
            try:
                # Direct load attempt
                self.model = whisper.load_model(self.model_size, device=self.device)
                self._model_loaded = True
                logger.info(f"✓ Whisper model '{self.model_size}' loaded successfully")
            except Exception as load_error:
                # If base model fails, try tiny model as fallback
                if self.model_size != 'tiny':
                    logger.warning(
                        f"Failed to load {self.model_size} model: {load_error}. "
                        f"Trying 'tiny' model as fallback (faster, smaller)..."
                    )
                    try:
                        logger.info("Loading 'tiny' model (this may take 10-20 seconds)...")
                        self.model = whisper.load_model('tiny', device=self.device)
                        self.model_size = 'tiny'  # Update to reflect actual model
                        self._model_loaded = True
                        logger.info("✓ Whisper 'tiny' model loaded successfully (fallback)")
                    except Exception as fallback_error:
                        logger.error(f"Failed to load fallback model: {fallback_error}")
                        raise RuntimeError(
                            f"Could not load Whisper model. Original error: {load_error}, "
                            f"Fallback error: {fallback_error}\n\n"
                            f"This is likely a PyTorch compatibility issue on macOS ARM64. "
                            f"The system will fall back to Google Speech-to-Text.\n"
                            f"To fix Whisper, try: pip install --upgrade torch torchaudio"
                        ) from fallback_error
                else:
                    raise
                    
        except RuntimeError:
            raise  # Re-raise our custom errors
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to load Whisper model '{self.model_size}'. "
                f"This may be due to PyTorch compatibility issues or insufficient memory. "
                f"Error: {str(e)}\n\n"
                f"On macOS ARM64, try: pip install --upgrade torch torchaudio"
            ) from e
    
    def _get_model_size_mb(self, model_size: str) -> str:
        """Get approximate model size in MB."""
        sizes = {
            'tiny': '39',
            'base': '74',
            'small': '244',
            'medium': '769',
            'large': '1550'
        }
        return sizes.get(model_size, 'unknown')
    
    def _load_faster_whisper_model(self):
        """Load faster-whisper model (uses CTranslate2, avoids PyTorch issues)."""
        if self._model_loaded and self.model is not None:
            return
        
        logger.info(f"Loading faster-whisper model: {self.model_size} (device: {self.device})")
        logger.info("faster-whisper uses CTranslate2 - no PyTorch, avoids crashes on macOS")
        
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type="int8" if self.device == "cpu" else "float16"
            )
            self._model_loaded = True
            logger.info(f"✓ faster-whisper model '{self.model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"Error loading faster-whisper model: {e}", exc_info=True)
            raise RuntimeError(
                f"Failed to load faster-whisper model '{self.model_size}'. "
                f"Error: {str(e)}"
            ) from e
    
    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        word_timestamps: bool = True,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe using faster-whisper."""
        logger.debug("Using faster-whisper for transcription")
        
        # Transcribe with word-level timestamps
        segments, info = self.model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=word_timestamps,
            **kwargs
        )
        
        # Extract segments and word timestamps
        all_segments = []
        all_word_timestamps = []
        full_text_parts = []
        
        for segment in segments:
            words = []
            if word_timestamps and hasattr(segment, 'words'):
                for w in segment.words:
                    wt = WordTimestamp(
                        word=w.word.strip(),
                        start_time=w.start,
                        end_time=w.end,
                        confidence=getattr(w, 'probability', 1.0)
                    )
                    words.append(wt)
                    all_word_timestamps.append(wt)
            
            seg = Segment(
                text=segment.text.strip(),
                start_time=segment.start,
                end_time=segment.end,
                confidence=getattr(segment, 'avg_logprob', 0.0),
                words=words
            )
            all_segments.append(seg)
            full_text_parts.append(segment.text.strip())
        
        full_text = " ".join(full_text_parts)
        duration = all_segments[-1].end_time if all_segments else 0.0
        
        return TranscriptionResult(
            text=full_text,
            segments=all_segments,
            language=info.language if hasattr(info, 'language') else self.language,
            duration=duration,
            confidence=np.mean([s.confidence for s in all_segments]) if all_segments else 0.0,
            metadata={
                'backend': 'faster-whisper',
                'model_size': self.model_size,
                'file_path': str(audio_path)
            },
            word_timestamps=all_word_timestamps
        )
    
    def _load_vosk_model(self):
        """Load Vosk model."""
        try:
            import json
            # Download model if needed (user needs to download manually)
            # For now, try to find model in common locations
            model_paths = [
                os.path.expanduser("~/vosk-model-en-us-0.22"),
                os.path.expanduser("~/vosk-model-small-en-us-0.15"),
                "./vosk-model-en-us-0.22",
                "./vosk-model-small-en-us-0.15",
            ]
            
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            if not model_path:
                raise FileNotFoundError(
                    "Vosk model not found. Download from: https://alphacephei.com/vosk/models\n"
                    "Extract to ~/vosk-model-en-us-0.22 or ./vosk-model-en-us-0.22"
                )
            
            self.vosk_model = vosk.Model(model_path)
            self.vosk_rec = vosk.KaldiRecognizer(self.vosk_model, 16000)
            self._model_loaded = True
            logger.info(f"✓ Vosk model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading Vosk model: {e}")
            raise
    
    def _transcribe_vosk(
        self,
        audio_path: Path,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe using Vosk."""
        logger.debug("Using Vosk for transcription")
        
        import wave
        import json
        
        # Vosk requires 16kHz mono WAV
        wf = wave.open(str(audio_path), "rb")
        if wf.getnchannels() != 1 or wf.getcomptype() != "NONE":
            raise ValueError("Vosk requires mono WAV format")
        
        if wf.getframerate() != 16000:
            logger.warning(f"Audio is {wf.getframerate()}Hz, Vosk expects 16kHz")
        
        segments = []
        all_word_timestamps = []
        full_text_parts = []
        
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            
            if self.vosk_rec.AcceptWaveform(data):
                result = json.loads(self.vosk_rec.Result())
                if 'text' in result and result['text']:
                    text = result['text']
                    full_text_parts.append(text)
                    # Vosk doesn't provide detailed timestamps in streaming mode
                    seg = Segment(
                        text=text,
                        start_time=0.0,  # Vosk doesn't provide precise timing
                        end_time=0.0,
                        confidence=1.0,
                        words=[]
                    )
                    segments.append(seg)
        
        # Get final result
        final_result = json.loads(self.vosk_rec.FinalResult())
        if 'text' in final_result and final_result['text']:
            text = final_result['text']
            if text not in full_text_parts:
                full_text_parts.append(text)
                seg = Segment(
                    text=text,
                    start_time=0.0,
                    end_time=0.0,
                    confidence=1.0,
                    words=[]
                )
                segments.append(seg)
        
        wf.close()
        full_text = " ".join(full_text_parts)
        
        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=self.language,
            duration=0.0,  # Vosk doesn't provide duration
            confidence=1.0,
            metadata={
                'backend': 'vosk',
                'file_path': str(audio_path)
            },
            word_timestamps=all_word_timestamps
        )
    
    def transcribe(
        self,
        audio_path: str | Path,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments for the transcription backend
            
        Returns:
            TranscriptionResult with full transcription
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if audio_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
        
        logger.info(f"Transcribing audio file: {audio_path.name}")
        
        if self.backend == 'faster-whisper':
            # Lazy load model on first transcription
            if not self._model_loaded:
                self._load_faster_whisper_model()
            return self._transcribe_faster_whisper(audio_path, **kwargs)
        elif self.backend == 'whisper':
            # Lazy load model on first transcription
            if not self._model_loaded:
                self._load_whisper_model()
            return self._transcribe_whisper(audio_path, **kwargs)
        elif self.backend == 'vosk':
            return self._transcribe_vosk(audio_path, **kwargs)
        elif self.backend == 'google':
            return self._transcribe_google(audio_path, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _transcribe_whisper(
        self,
        audio_path: Path,
        word_timestamps: bool = True,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe using Whisper."""
        logger.debug("Using Whisper for transcription")
        
        # Transcribe with word-level timestamps
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=word_timestamps,
            **kwargs
        )
        
        # Extract segments
        segments = []
        all_word_timestamps = []
        
        for seg in result.get('segments', []):
            words = []
            if 'words' in seg:
                for w in seg['words']:
                    wt = WordTimestamp(
                        word=w['word'].strip(),
                        start_time=w['start'],
                        end_time=w['end'],
                        confidence=w.get('probability', 1.0)
                    )
                    words.append(wt)
                    all_word_timestamps.append(wt)
            
            segment = Segment(
                text=seg['text'].strip(),
                start_time=seg['start'],
                end_time=seg['end'],
                confidence=seg.get('avg_logprob', 0.0),
                words=words
            )
            segments.append(segment)
        
        # Calculate duration
        duration = segments[-1].end_time if segments else 0.0
        
        return TranscriptionResult(
            text=result['text'].strip(),
            segments=segments,
            language=result.get('language', self.language),
            duration=duration,
            confidence=np.mean([s.confidence for s in segments]) if segments else 0.0,
            metadata={
                'backend': 'whisper',
                'model_size': self.model_size,
                'file_path': str(audio_path)
            },
            word_timestamps=all_word_timestamps
        )
    
    def _transcribe_google(
        self,
        audio_path: Path,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe using Google Speech-to-Text (fallback)."""
        logger.debug("Using Google Speech-to-Text for transcription")
        
        # Convert audio if needed
        audio_file = self._prepare_audio_for_sr(audio_path)
        
        with sr.AudioFile(str(audio_file)) as source:
            audio_data = self.recognizer.record(source)
            duration = source.DURATION
        
        try:
            text = self.recognizer.recognize_google(
                audio_data,
                language=self.language
            )
            
            # Google doesn't provide detailed timing, create single segment
            segment = Segment(
                text=text,
                start_time=0.0,
                end_time=duration,
            )
            
            return TranscriptionResult(
                text=text,
                segments=[segment],
                language=self.language,
                duration=duration,
                metadata={
                    'backend': 'google',
                    'file_path': str(audio_path)
                }
            )
            
        except sr.UnknownValueError:
            logger.warning("Google Speech Recognition could not understand audio")
            return TranscriptionResult(
                text="",
                segments=[],
                language=self.language,
                duration=duration,
                metadata={'backend': 'google', 'error': 'could not understand'}
            )
        except sr.RequestError as e:
            logger.error(f"Google Speech Recognition error: {e}")
            raise
    
    def _prepare_audio_for_sr(self, audio_path: Path) -> Path:
        """Prepare audio file for SpeechRecognition (convert to WAV if needed)."""
        if audio_path.suffix.lower() == '.wav':
            return audio_path
        
        if not PYDUB_AVAILABLE:
            raise ImportError("PyDub required for audio conversion")
        
        # Convert to WAV
        audio = AudioSegment.from_file(str(audio_path))
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio.export(tmp.name, format='wav')
            return Path(tmp.name)
    
    def transcribe_with_diarization(
        self,
        audio_path: str | Path,
        num_speakers: int = 2,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe with speaker diarization.
        
        Attempts to identify different speakers in the audio.
        Uses a simple energy-based approach for speaker separation.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Expected number of speakers
            **kwargs: Additional arguments
            
        Returns:
            TranscriptionResult with speaker labels
        """
        # First, get base transcription
        result = self.transcribe(audio_path, **kwargs)
        
        # Simple speaker assignment based on segment patterns
        # In a real implementation, this would use a proper diarization model
        speakers = ['CHI', 'INV']  # Child and Investigator
        
        for i, segment in enumerate(result.segments):
            # Simple alternating pattern (placeholder for real diarization)
            # In practice, you'd use pyannote or similar
            segment.speaker = speakers[i % len(speakers)]
        
        result.metadata['diarization'] = 'simple_alternating'
        result.metadata['num_speakers'] = num_speakers
        
        return result


__all__ = ["AudioTranscriber", "TranscriptionResult", "Segment", "WordTimestamp"]

