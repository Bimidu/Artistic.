"""
Audio-only Acoustic & Prosodic Feature Extractor

This module extracts ONLY audio-derived features.
NO transcript (.cha) features are used.

Reason:
TD dataset contains only .wav files. Using transcript-based pause features
would cause dataset leakage and unfair ASD vs TD classification.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import librosa


class AcousticProsodicFeatures:
    """Extracts acoustic & prosodic features from a single .wav file."""

    @property
    def feature_names(self) -> List[str]:
        names = [
            "duration_sec",
            "pitch_mean",
            "pitch_std",
            "pitch_range",
            "pitch_slope",
            "energy_mean",
            "energy_std",
            "energy_iqr",
            "energy_max",
            "tempo",
            "speaking_rate",
            "articulation_rate",
            "speech_time_sec",
            "silence_time_sec",
            "speech_ratio",
            "rhythm_score",
        ]
        for i in range(1, 14):
            names.append(f"mfcc{i}_mean")
        for i in range(1, 14):
            names.append(f"mfcc{i}_std")
        return names

    @staticmethod
    def _safe(x: float) -> float:
        return float(x) if x is not None and np.isfinite(x) else 0.0

    @staticmethod
    def _pvi(values: np.ndarray) -> float:
        if values.size < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(values))))

    def extract_from_wav(self, wav_path: Path, sr: int = 16000) -> Optional[Dict[str, float]]:
        """
        Extract audio-only acoustic prosodic features from a wav file.
        Returns None if extraction fails.
        """
        try:
            y, sr = librosa.load(str(wav_path), sr=sr)
            if y is None or len(y) == 0:
                return None

            y = y / (np.max(np.abs(y)) + 1e-9)
            duration = len(y) / sr

            # Pitch
            f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
            f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

            pitch_mean = self._safe(np.mean(f0)) if f0.size else 0.0
            pitch_std = self._safe(np.std(f0)) if f0.size else 0.0
            pitch_range = self._safe(np.max(f0) - np.min(f0)) if f0.size else 0.0
            pitch_slope = self._safe(np.polyfit(np.arange(len(f0)), f0, 1)[0]) if f0.size > 1 else 0.0

            # Energy
            rms = librosa.feature.rms(y=y)[0]
            energy_mean = self._safe(np.mean(rms))
            energy_std = self._safe(np.std(rms))
            energy_iqr = self._safe(np.subtract(*np.percentile(rms, [75, 25])))
            energy_max = self._safe(np.max(rms))

            # Tempo & rate
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

            intervals = librosa.effects.split(y, top_db=30)
            speech_time = sum((e - s) for s, e in intervals) / sr if intervals.any() else 0.0
            silence_time = max(duration - speech_time, 0.0)

            speaking_rate = self._safe(len(onsets) / duration) if duration > 0 else 0.0
            articulation_rate = self._safe(len(onsets) / speech_time) if speech_time > 0 else 0.0
            speech_ratio = self._safe(speech_time / duration) if duration > 0 else 0.0

            # Rhythm
            rhythm_score = self._safe(self._pvi(rms))

            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_means = np.mean(mfcc, axis=1)
            mfcc_stds = np.std(mfcc, axis=1)

            features = {
                "duration_sec": duration,
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "pitch_range": pitch_range,
                "pitch_slope": pitch_slope,
                "energy_mean": energy_mean,
                "energy_std": energy_std,
                "energy_iqr": energy_iqr,
                "energy_max": energy_max,
                "tempo": self._safe(tempo),
                "speaking_rate": speaking_rate,
                "articulation_rate": articulation_rate,
                "speech_time_sec": speech_time,
                "silence_time_sec": silence_time,
                "speech_ratio": speech_ratio,
                "rhythm_score": rhythm_score,
            }

            for i, (m, s) in enumerate(zip(mfcc_means, mfcc_stds), start=1):
                features[f"mfcc{i}_mean"] = self._safe(m)
                features[f"mfcc{i}_std"] = self._safe(s)

            return features

        except Exception:
            return None
