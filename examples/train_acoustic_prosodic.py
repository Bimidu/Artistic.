"""
Train Acoustic–Prosodic ASD vs TD Model (AUDIO-ONLY)

This script orchestrates:
- Feature extraction
- Dataset balancing (undersampling)
- Preprocessing
- Model training

Why NO transcripts?
TD dataset has only .wav files. Transcript-based pause features would cause
dataset leakage. Therefore ONLY audio-derived features are used.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---- Import YOUR existing components ----
from src.features.acoustic_prosodic.acoustic_prosodic import AcousticProsodicFeatures
from src.models.acoustic_prosodic.preprocessor import AcousticProsodicPreprocessor
from src.models.acoustic_prosodic.model_trainer import AcousticProsodicTrainer


# =====================
# CONFIG
# =====================
ASD_DIR = Path("data/asdbank_aac/AAC/child_only")
TD_DIR = Path("data/td")

OUTPUT_DIR = Path("output")
OUTPUT_CSV = OUTPUT_DIR / "asd_td_acoustic_features.csv"

OUTPUT_DIR.mkdir(exist_ok=True)


# =====================
# UTIL
# =====================
def list_wavs(folder: Path):
    return sorted(folder.glob("*.wav")) if folder.exists() else []


def balance_dataset_undersample(df, label_col="label", random_state=42):
    """
    Balance dataset by undersampling the majority class (TD).
    Keeps all ASD samples and randomly samples TD to match ASD count.
    """
    counts = df[label_col].value_counts()
    print("[INFO] Before balancing:", counts.to_dict())

    if len(counts) != 2:
        print("[WARN] Dataset does not have exactly 2 classes. Skipping balancing.")
        return df

    minority_label = counts.idxmin()
    majority_label = counts.idxmax()
    n_minority = counts.min()

    df_min = df[df[label_col] == minority_label]
    df_maj = df[df[label_col] == majority_label].sample(
        n=n_minority, random_state=random_state
    )

    df_balanced = (
        pd.concat([df_min, df_maj])
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )

    print("[INFO] After balancing:", df_balanced[label_col].value_counts().to_dict())
    return df_balanced


# =====================
# MAIN
# =====================
def main():
    print("\n===== TRAIN ACOUSTIC PROSODIC MODEL (ASD vs TD) =====\n")

    extractor = AcousticProsodicFeatures()
    preprocessor = AcousticProsodicPreprocessor()
    trainer = AcousticProsodicTrainer()

    rows = []

    # -------- ASD --------
    asd_files = list_wavs(ASD_DIR)
    print(f"[INFO] ASD wavs found: {len(asd_files)}")

    for wav in tqdm(asd_files, desc="Extracting ASD features"):
        feats = extractor.extract_from_wav(wav)
        if feats is not None:
            feats["label"] = "ASD"
            feats["file_path"] = str(wav)
            rows.append(feats)

    # -------- TD --------
    td_files = list_wavs(TD_DIR)
    print(f"[INFO] TD wavs found: {len(td_files)}")

    for wav in tqdm(td_files, desc="Extracting TD features"):
        feats = extractor.extract_from_wav(wav)
        if feats is not None:
            feats["label"] = "TD"
            feats["file_path"] = str(wav)
            rows.append(feats)

    if not rows:
        raise RuntimeError("No features extracted")

    df = pd.DataFrame(rows)

    # -------- BALANCE DATASET --------
    df = balance_dataset_undersample(df, label_col="label")

    # Save features
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Features saved to {OUTPUT_CSV}")

    # -------- PREPROCESS --------
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)

    # -------- TRAIN --------
    results = trainer.train_multiple_models(
        X_train, y_train, X_test, y_test
    )

    # -------- REPORT --------
    for name, res in results.items():
        print(f"\nModel: {name}")
        print(f"Accuracy: {res['accuracy']:.4f}")
        print(f"F1 score: {res['f1']:.4f}")
        print(res["report"])

    print("\n✅ Training complete.")


if __name__ == "__main__":
    main()
