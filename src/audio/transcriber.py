"""
Audio Transcriber Module

This module provides speech-to-text transcription via AssemblyAI.
AssemblyAI is the sole active backend — it provides speaker diarization,
word-level timestamps, and high-accuracy transcription out of the box.

The transcriber produces structured output including:
- Full transcript text
- Word-level timestamps
- Speaker diarization (utterance-level)
- Confidence scores

Author: Bimidu Gunathilake
"""

import os
import re
import sys
import subprocess
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import tempfile
import json

import numpy as np
import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Unused local / OSS transcription backends — kept for reference only.
# Only AssemblyAI is used in production.
# ---------------------------------------------------------------------------
# def _configure_torch_safe_globals_for_whisperx() -> None: ...
# try: import whisper; WHISPER_AVAILABLE = True ...
# try: from faster_whisper import WhisperModel; FASTER_WHISPER_AVAILABLE = True ...
# try: import vosk; VOSK_AVAILABLE = True ...
# try: import speech_recognition as sr; SR_AVAILABLE = True ...
# try: from pydub import AudioSegment; PYDUB_AVAILABLE = True ...
# try: import whisperx; WHISPERX_AVAILABLE = True ...
# ---------------------------------------------------------------------------


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
    Audio transcription via AssemblyAI.

    AssemblyAI provides speaker diarization, word-level timestamps, and
    high-accuracy transcription.  It is the sole active backend.

    Example:
        >>> transcriber = AudioTranscriber(backend='assemblyai')
        >>> result = transcriber.transcribe("audio.wav")
        >>> print(result.text)
    """

    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

    def __init__(
        self,
        backend: str = 'assemblyai',
        model_size: str = 'base',
        device: str = 'cpu',
        language: str = 'en',
    ):
        """
        Initialize the audio transcriber.

        Args:
            backend: Must be 'assemblyai' (other values accepted for compatibility
                     but will raise at transcription time if not assemblyai).
            model_size: Ignored for AssemblyAI; retained for API compatibility.
            device: Ignored for AssemblyAI; retained for API compatibility.
            language: Target language code passed to AssemblyAI.
        """
        self.backend = backend
        self.model_size = model_size
        self.device = device
        self.language = language
        self.model = None
        self._model_loaded = False

        logger.info(f"Initializing AudioTranscriber with backend={backend}")

        if backend == 'assemblyai':
            self.assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
            if not self.assemblyai_api_key:
                raise ImportError(
                    "ASSEMBLYAI_API_KEY is not configured. Set it in your .env file."
                )
        else:
            # Deprecated backends — kept for signature compatibility only.
            # Attempting to transcribe with them will raise ValueError.
            logger.warning(
                f"Backend '{backend}' is deprecated. Only 'assemblyai' is supported. "
                "Transcription calls will fail unless backend is assemblyai."
            )
    
    # -----------------------------------------------------------------------
    # Deprecated local / OSS backend methods — commented out.
    # Only AssemblyAI (_transcribe_assemblyai) is active.
    # -----------------------------------------------------------------------
    # def _load_whisper_model(self): ...
    # def _get_model_size_mb(self, model_size): ...
    # def _load_faster_whisper_model(self): ...
    # def _transcribe_faster_whisper(self, audio_path, ...): ...
    # def _load_vosk_model(self): ...
    # def _transcribe_vosk(self, audio_path, ...): ...
    # -----------------------------------------------------------------------
    
    def transcribe(
        self,
        audio_path: str | Path,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe an audio file using AssemblyAI.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments forwarded to _transcribe_assemblyai

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

        if self.backend != 'assemblyai':
            raise ValueError(
                f"Backend '{self.backend}' is not supported. Only 'assemblyai' is active."
            )

        logger.info(f"Transcribing audio file: {audio_path.name} via AssemblyAI")
        return self._transcribe_assemblyai(audio_path, **kwargs)
    
    # -----------------------------------------------------------------------
    # Deprecated backend implementations — kept for reference, not called.
    # -----------------------------------------------------------------------
    # def _transcribe_whisper(self, audio_path, ...): ...
    # def _transcribe_google(self, audio_path, ...): ...
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Deprecated Deepgram backend — kept for reference only, not called.
    # def _transcribe_deepgram(self, audio_path, num_speakers=2, **kwargs): ...
    # -----------------------------------------------------------------------

    def _transcribe_assemblyai(
        self,
        audio_path: Path,
        num_speakers: int = 2,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe using AssemblyAI async API with diarization enabled."""
        headers = {"authorization": self.assemblyai_api_key}
        upload_headers = {**headers, "content-type": "application/octet-stream"}

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        with httpx.Client(timeout=300.0) as client:
            upload_response = client.post(
                "https://api.assemblyai.com/v2/upload",
                headers=upload_headers,
                content=audio_bytes,
            )
            if upload_response.status_code >= 400:
                raise RuntimeError(
                    f"AssemblyAI upload failed ({upload_response.status_code}): {upload_response.text}"
                )
            upload_payload = upload_response.json()
            upload_url = str(upload_payload.get("upload_url", "")).strip()
            if not upload_url:
                raise RuntimeError("AssemblyAI upload did not return upload_url")

            speech_models_raw = os.getenv(
                "ASSEMBLYAI_SPEECH_MODELS",
                "universal-3-pro,universal-2",
            )
            speech_models = [
                model.strip()
                for model in speech_models_raw.split(",")
                if model.strip()
            ]

            transcript_request = {
                "audio_url": upload_url,
                "speech_models": speech_models,
                "language_detection": True,
                "speaker_labels": True,
            }
            # Optional diarization hint; some accounts/plans may reject this field.
            if kwargs.get("use_speakers_expected", False):
                transcript_request["speakers_expected"] = max(1, int(num_speakers or 2))

            create_response = client.post(
                "https://api.assemblyai.com/v2/transcript",
                headers=headers,
                json=transcript_request,
            )
            if create_response.status_code >= 400:
                raise RuntimeError(
                    f"AssemblyAI transcript create failed ({create_response.status_code}): "
                    f"{create_response.text}. Request payload: {json.dumps(transcript_request)}"
                )
            create_payload = create_response.json()
            transcript_id = str(create_payload.get("id", "")).strip()
            if not transcript_id:
                raise RuntimeError("AssemblyAI transcript creation did not return id")

            logger.info(
                f"AssemblyAI request submitted — transcript_id={transcript_id}, "
                f"speech_models={transcript_request.get('speech_models')}"
            )

            poll_interval = float(kwargs.get("poll_interval_seconds", 2.0))
            max_wait_seconds = float(kwargs.get("max_wait_seconds", 600.0))
            started_at = time.time()
            status_payload: Dict[str, Any] = {}

            while True:
                status_response = client.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}",
                    headers=headers,
                )
                status_response.raise_for_status()
                status_payload = status_response.json()
                status_value = str(status_payload.get("status", "")).lower()

                if status_value == "completed":
                    break
                if status_value == "error":
                    error_text = status_payload.get("error") or "unknown AssemblyAI error"
                    raise RuntimeError(f"AssemblyAI transcription failed: {error_text}")
                if time.time() - started_at > max_wait_seconds:
                    raise TimeoutError(
                        f"AssemblyAI transcription timed out after {max_wait_seconds:.0f}s"
                    )
                time.sleep(max(0.5, poll_interval))

        full_text = str(status_payload.get("text", "") or "").strip()
        utterances = status_payload.get("utterances", []) or []
        raw_words = status_payload.get("words", []) or []

        def _to_seconds(value: Any) -> float:
            try:
                return float(value) / 1000.0
            except (TypeError, ValueError):
                return 0.0

        all_word_timestamps: List[WordTimestamp] = []
        all_segments: List[Segment] = []

        speaker_alias_map: Dict[str, str] = {}
        speaker_counter = 0

        def _speaker_to_label(speaker_raw: Any) -> Optional[str]:
            nonlocal speaker_counter
            if speaker_raw is None:
                return None
            speaker_key = str(speaker_raw).strip()
            if not speaker_key:
                return None
            if speaker_key not in speaker_alias_map:
                speaker_alias_map[speaker_key] = f"spk_{speaker_counter}"
                speaker_counter += 1
            return speaker_alias_map[speaker_key]

        if utterances:
            for utt in utterances:
                utt_text = str(utt.get("text", "") or "").strip()
                utt_start = _to_seconds(utt.get("start"))
                utt_end = _to_seconds(utt.get("end"))
                utt_conf = float(utt.get("confidence", 1.0) or 1.0)

                seg_words: List[WordTimestamp] = []
                word_speaker_counts: Dict[str, int] = {}
                for w in utt.get("words", []) or []:
                    wt = WordTimestamp(
                        word=str(w.get("text", "") or "").strip(),
                        start_time=_to_seconds(w.get("start")),
                        end_time=_to_seconds(w.get("end")),
                        confidence=float(w.get("confidence", 1.0) or 1.0),
                    )
                    seg_words.append(wt)
                    all_word_timestamps.append(wt)
                    w_spk = w.get("speaker")
                    if w_spk is not None:
                        key = str(w_spk).strip()
                        if key:
                            word_speaker_counts[key] = word_speaker_counts.get(key, 0) + 1

                speaker_raw = None
                if word_speaker_counts:
                    speaker_raw = max(word_speaker_counts.items(), key=lambda kv: kv[1])[0]
                if speaker_raw is None:
                    speaker_raw = utt.get("speaker")

                all_segments.append(
                    Segment(
                        text=utt_text,
                        start_time=utt_start,
                        end_time=utt_end,
                        speaker=_speaker_to_label(speaker_raw),
                        confidence=utt_conf,
                        words=seg_words,
                    )
                )
        else:
            # Fallback if utterances are missing: create one segment and still collect words.
            seg_words: List[WordTimestamp] = []
            for w in raw_words:
                wt = WordTimestamp(
                    word=str(w.get("text", "") or "").strip(),
                    start_time=_to_seconds(w.get("start")),
                    end_time=_to_seconds(w.get("end")),
                    confidence=float(w.get("confidence", 1.0) or 1.0),
                )
                seg_words.append(wt)
                all_word_timestamps.append(wt)

            segment_end = seg_words[-1].end_time if seg_words else 0.0
            all_segments.append(
                Segment(
                    text=full_text,
                    start_time=0.0,
                    end_time=segment_end,
                    speaker=None,
                    confidence=float(status_payload.get("confidence", 0.0) or 0.0),
                    words=seg_words,
                )
            )

        duration = all_segments[-1].end_time if all_segments else 0.0
        return TranscriptionResult(
            text=full_text,
            segments=all_segments,
            language=self.language,
            duration=duration,
            confidence=float(np.mean([s.confidence for s in all_segments])) if all_segments else 0.0,
            metadata={
                "backend": "assemblyai",
                "engine": "assemblyai",
                "file_path": str(audio_path),
                "transcript_id": status_payload.get("id"),
            },
            word_timestamps=all_word_timestamps,
        )

    # -----------------------------------------------------------------------
    # Deprecated backend implementations — kept for reference, not called.
    # -----------------------------------------------------------------------
    # def _transcribe_whisperx(self, audio_path, ...): ...
    # def _prepare_audio_for_sr(self, audio_path): ...
    # -----------------------------------------------------------------------

    def transcribe_with_diarization(
        self,
        audio_path: str | Path,
        num_speakers: int = 2,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe with speaker diarization via AssemblyAI.

        AssemblyAI performs model-based diarization natively.

        Args:
            audio_path: Path to audio file
            num_speakers: Expected number of speakers (hint for AssemblyAI)
            **kwargs: Additional arguments forwarded to transcribe()

        Returns:
            TranscriptionResult with speaker labels
        """
        result = self.transcribe(audio_path, num_speakers=num_speakers, **kwargs)
        result.metadata['num_speakers'] = num_speakers
        result.metadata['diarization'] = 'model_based'
        return result


__all__ = ["AudioTranscriber", "TranscriptionResult", "Segment", "WordTimestamp"]

