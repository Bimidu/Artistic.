"""
Basic Acoustic & Prosodic Feature Extractor
Author: Team Member A

Current Status:
---------------
Implemented Features:
 - Pitch mean & standard deviation (fundamental frequency, F0)
 - Energy mean & standard deviation (RMS)
 - Tempo (approximate speaking rate proxy)
 - MFCC mean & standard deviation (13 coefficients)

To Be Implemented (Next Phase):
 - Pitch range and slope
 - Speaking rate, articulation rate, and pause rate
 - Intonation, stress, and rhythm measures
 - Pause duration and filled pause ratio
 - Advanced prosodic & temporal dynamics
"""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path


def extract_basic_features(wav_path: Path):
    """Extract basic acoustic & prosodic features from one audio file."""
    try:
        y, sr = librosa.load(wav_path, sr=16000)

        # --- Pitch (fundamental frequency) ---
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        f0 = f0[~np.isnan(f0)]
        pitch_mean = np.mean(f0) if len(f0) else 0
        pitch_std = np.std(f0) if len(f0) else 0

        # --- Energy / RMS ---
        rms = librosa.feature.rms(y=y)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)

        # --- Tempo (rough speech rate proxy) ---
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = tempo if np.isfinite(tempo) else 0

        # --- MFCCs (Mel-Frequency Cepstral Coefficients) ---
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_stds = np.std(mfcc, axis=1)

        # Combine everything
        features = {
            "file": wav_path.name,
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "tempo": tempo,
        }

        for i, (m, s) in enumerate(zip(mfcc_means, mfcc_stds), start=1):
            features[f"mfcc{i}_mean"] = m
            features[f"mfcc{i}_std"] = s

        return features

    except Exception as e:
        print(f"Error processing {wav_path.name}: {e}")
        return {}


if __name__ == "__main__":
    input_dir = Path("data/asdbank_aac/AAC/child_only")
    output_csv = Path("output/audio/acoustic_features.csv")

    rows = []
    wav_files = list(input_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} child audio files to process.\n")

    for wav in wav_files:
        feats = extract_basic_features(wav)
        if feats:
            rows.append(feats)
            print(f"Extracted: {wav.name}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"\nAll features saved to → {output_csv}")
    else:
        print("No valid features extracted.")
