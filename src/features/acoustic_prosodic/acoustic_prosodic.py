"""
Enhanced Acoustic & Prosodic Feature Extractor
Author: Team Member A (Sanuthi)

Implemented Features (per child-only .wav):
 - Pitch mean, std, range, slope (F0)
 - Energy mean, std, IQR, max (RMS)
 - Tempo (approximate speech rate proxy)
 - Speaking rate (onset count / total duration)
 - Articulation rate (onset count / voiced duration)
 - Speech vs silence time & ratio (via non-silent detection)
 - Rhythm score (PVI over RMS envelope)
 - MFCC mean & std (13 coefficients)
 - Pause features from .cha:
    - pause_rate_per_min
    - mean_pause_duration_sec
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import librosa
import numpy as np

from src.features.base_features import BaseFeatureExtractor, FeatureResult
from src.parsers.chat_parser import TranscriptData


class AcousticProsodicFeatures(BaseFeatureExtractor):
    """Integrated acoustic & prosodic feature extractor."""

    # ---------- Feature names exposed to the rest of the system ----------
    @property
    def feature_names(self) -> List[str]:
        names = [
            "duration_sec",
            # Pitch
            "pitch_mean",
            "pitch_std",
            "pitch_range",
            "pitch_slope",
            # Energy
            "energy_mean",
            "energy_std",
            "energy_iqr",
            "energy_max",
            # Tempo / rate
            "tempo",
            "speaking_rate",
            "articulation_rate",
            # Speech vs silence
            "speech_time_sec",
            "silence_time_sec",
            "speech_ratio",
            # Rhythm
            "rhythm_score",
            # Pause features
            "pause_rate_per_min",
            "mean_pause_duration_sec",
        ]

        # MFCC stats (13 × mean, 13 × std)
        for i in range(1, 14):
            names.append(f"mfcc{i}_mean")
        for i in range(1, 14):
            names.append(f"mfcc{i}_std")

        return names

    # ---------- Internal helpers ----------

    def _get_paths(self, transcript: TranscriptData) -> tuple[Path, Path]:
        """
        Derive .cha path and child-only .wav path from transcript.file_path.

        Assumes:
          - original: data/asdbank_aac/AAC/01_T1_1.cha
          - child audio: data/asdbank_aac/AAC/child_only/01_T1_1_child.wav
        """
        cha_path = Path(transcript.file_path)
        base_dir = cha_path.parent              # .../AAC
        child_dir = base_dir / "child_only"     # .../AAC/child_only

        wav_path = child_dir / f"{cha_path.stem}_child.wav"
        return cha_path, wav_path

    @staticmethod
    def _pairwise_variability_index(values: np.ndarray) -> float:
        """Simple PVI (Pairwise Variability Index) over a 1D sequence."""
        if values.size < 2:
            return 0.0
        diffs = np.abs(np.diff(values))
        return float(np.mean(diffs))

    @staticmethod
    def _extract_pause_features(cha_path: Path, min_pause_ms: int = 200) -> tuple[float, float]:
        """
        Extract pause features from .cha using CHI timecodes.

        - pause_rate_per_min: pauses per minute between CHI turns
        - mean_pause_duration_sec: average gap duration in seconds
        """
        try:
            text = cha_path.read_text(encoding="utf-8", errors="ignore")
        except FileNotFoundError:
            return 0.0, 0.0

        matches = re.findall(r"\*CHI:.*?(\d+)_(\d+)", text)
        if len(matches) < 2:
            return 0.0, 0.0

        timestamps = [(int(s), int(e)) for s, e in matches]
        timestamps.sort()

        pauses = []
        for i in range(1, len(timestamps)):
            prev_end = timestamps[i - 1][1]
            next_start = timestamps[i][0]
            gap = next_start - prev_end
            if gap >= min_pause_ms:
                pauses.append(gap)

        if not pauses:
            return 0.0, 0.0

        pauses = np.array(pauses, dtype=float)
        mean_pause_sec = float(np.mean(pauses) / 1000.0)  # ms → sec

        # Time span for CHI speech (ms)
        first_start = timestamps[0][0]
        last_end = timestamps[-1][1]
        total_span_ms = max(last_end - first_start, 1)
        span_minutes = total_span_ms / 60000.0

        pause_rate_per_min = float(len(pauses) / span_minutes) if span_minutes > 0 else 0.0
        return pause_rate_per_min, mean_pause_sec

    # ---------- Main extraction method required by BaseFeatureExtractor ----------

    def extract(self, transcript: TranscriptData) -> FeatureResult:
        """
        Extract all acoustic & prosodic features for a single transcript.

        Uses:
         - child-only .wav (from AAC/child_only)
         - corresponding .cha for pause features
        """
        cha_path, wav_path = self._get_paths(transcript)

        # Default output (all zeros) in case of any failure
        zero_features = {name: 0.0 for name in self.feature_names}

        try:
            if not wav_path.exists():
                raise FileNotFoundError(f"Child audio not found: {wav_path}")

            # Load audio
            y, sr = librosa.load(wav_path, sr=16000)
            duration_sec = len(y) / sr if len(y) else 0.0

            # ----- Pitch (F0) -----
            f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
            if f0 is None:
                f0 = np.array([])
            else:
                f0 = f0[~np.isnan(f0)]

            pitch_mean = float(np.mean(f0)) if f0.size else 0.0
            pitch_std = float(np.std(f0)) if f0.size else 0.0
            pitch_range = float(np.max(f0) - np.min(f0)) if f0.size else 0.0

            if f0.size > 1:
                x = np.arange(len(f0))
                pitch_slope = float(np.polyfit(x, f0, 1)[0])
            else:
                pitch_slope = 0.0

            # ----- Energy / RMS -----
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = float(np.mean(rms))
            energy_std = float(np.std(rms))
            energy_iqr = float(np.subtract(*np.percentile(rms, [75, 25])))
            energy_max = float(np.max(rms))

            # ----- Tempo / speech rate proxy -----
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo = float(tempo) if np.isfinite(tempo) else 0.0

            # ----- Speech / Silence -----
            intervals = librosa.effects.split(y, top_db=30)  # non-silent regions
            speech_time = float(sum((e - s) for s, e in intervals) / sr)
            silence_time = max(duration_sec - speech_time, 0.0)
            speech_ratio = float(speech_time / duration_sec) if duration_sec > 0 else 0.0

            # ----- Speaking & articulation rate -----
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            onset_count = len(onsets)

            speaking_rate = float(onset_count / duration_sec) if duration_sec > 0 else 0.0
            articulation_rate = float(onset_count / speech_time) if speech_time > 0 else 0.0

            # ----- Rhythm (PVI over RMS) -----
            rhythm_score = self._pairwise_variability_index(rms)

            # ----- MFCCs -----
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfcc, axis=1)
            mfcc_stds = np.std(mfcc, axis=1)

            # ----- Pause features from .cha -----
            pause_rate, mean_pause_sec = self._extract_pause_features(cha_path)

            features = {
                "duration_sec": duration_sec,
                # Pitch
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "pitch_range": pitch_range,
                "pitch_slope": pitch_slope,
                # Energy
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "energy_iqr": energy_iqr,
                "energy_max": energy_max,
                # Tempo / rate
                "tempo": tempo,
                "speaking_rate": speaking_rate,
                "articulation_rate": articulation_rate,
                # Speech vs silence
                "speech_time_sec": speech_time,
                "silence_time_sec": silence_time,
                "speech_ratio": speech_ratio,
                # Rhythm
                "rhythm_score": rhythm_score,
                # Pauses
                "pause_rate_per_min": pause_rate,
                "mean_pause_duration_sec": mean_pause_sec,
            }

            for i, (m, s) in enumerate(zip(mfcc_means, mfcc_stds), start=1):
                features[f"mfcc{i}_mean"] = float(m)
                features[f"mfcc{i}_std"] = float(s)

            return FeatureResult(
                features=features,
                feature_type="acoustic_prosodic",
                metadata={
                    "status": "implemented",
                    "wav_path": str(wav_path),
                    "cha_path": str(cha_path),
                },
            )

        except Exception as e:
            print(f"[Acoustic Error] {cha_path.name}: {e}")
            # Fall back to zeros so the pipeline doesn’t crash
            return FeatureResult(
                features=zero_features,
                feature_type="acoustic_prosodic",
                metadata={
                    "status": "error",
                    "error": str(e),
                    "wav_exists": wav_path.exists(),
                    "cha_path": str(cha_path),
                    "wav_path": str(wav_path),
                },
            )


__all__ = ["AcousticProsodicFeatures"]
