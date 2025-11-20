"""
Kaggle ASD Acoustic Pipeline

Dataset layout (relative to project root):

data/
  kaggle-testdataset/
    autism_dataset_index.csv
    autism_dataset_metadata.json
    voice/
      child_001.wav
      child_002.wav
      ...

This script:
  1) Loads autism_dataset_index.csv
  2) Extracts acoustic features from each voice/*.wav
  3) Saves features -> output/kaggle_acoustic_features.csv
  4) Preprocesses (clean, scale, split)
  5) Trains models (LogReg, RandomForest, XGBoost)
  6) Picks best model (by F1)
  7) Saves it -> models/kaggle_acoustic_best.pkl
  8) Runs a demo prediction on one audio file
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import noisereduce as nr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib


# ==========================================================
#                    CONFIGURATION
# ==========================================================

DATASET_DIR = Path("data/kaggle-testdataset")
INDEX_CSV = DATASET_DIR / "autism_dataset_index.csv"
AUDIO_ROOT = DATASET_DIR

OUTPUT_DIR = Path("output")
MODEL_DIR = Path("models")

FEATURE_CSV = OUTPUT_DIR / "kaggle_acoustic_features.csv"
MODEL_PATH = MODEL_DIR / "kaggle_acoustic_best.pkl"


# ==========================================================
#                 LABEL MAPPING
# ==========================================================

def map_label_to_diagnosis(label: str) -> str:
    """
    Convert Kaggle labels to ASD / TYPICAL
    """
    label = str(label).lower()
    if "asd" in label:
        return "ASD"
    return "TYPICAL"


# ==========================================================
#           FEATURE EXTRACTION FOR ONE FILE
# ==========================================================

def extract_features_for_file(path: str) -> dict:
    """
    Extract MFCC, RMS, Pitch, Duration features.
    """
    path = str(path)

    # Load audio
    y, sr = librosa.load(path, sr=16000)

    # Noise reduction
    y = nr.reduce_noise(y=y, sr=sr)

    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-9)

    duration_sec = len(y) / sr

    # RMS
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(rms.mean())

    # Pitch
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7")
    )
    pitch_mean = float(np.nanmean(f0)) if f0 is not None else 0.0

    # MFCC (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)

    features = {
        "duration_sec": duration_sec,
        "rms_mean": rms_mean,
        "pitch_mean": pitch_mean,
    }

    for i, val in enumerate(mfcc_means, start=1):
        features[f"mfcc{i}_mean"] = float(val)

    return features


# ==========================================================
#           BUILD COMPLETE FEATURE DATASET
# ==========================================================

def build_feature_dataset() -> pd.DataFrame:
    """
    Loop through all Kaggle audio files and extract features.
    """
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Index CSV missing: {INDEX_CSV}")

    df_index = pd.read_csv(INDEX_CSV)

    required = {"sample_id", "voice", "label"}
    if not required.issubset(df_index.columns):
        raise ValueError(f"CSV must contain {required}")

    df_index["file_path"] = df_index["voice"].apply(lambda p: str(AUDIO_ROOT / p))
    df_index["diagnosis"] = df_index["label"].apply(map_label_to_diagnosis)

    print(f"[INFO] Found {len(df_index)} samples.")
    print(df_index.head())

    rows = []

    for _, row in df_index.iterrows():
        audio_path = row["file_path"]

        if not os.path.exists(audio_path):
            print(f"[WARN] Missing audio: {audio_path}")
            continue

        feats = extract_features_for_file(audio_path)
        feats["sample_id"] = row["sample_id"]
        feats["label_raw"] = row["label"]
        feats["diagnosis"] = row["diagnosis"]

        rows.append(feats)

    df = pd.DataFrame(rows)
    return df


def get_or_create_feature_csv() -> pd.DataFrame:
    """
    Load existing CSV or extract new features.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    if FEATURE_CSV.exists():
        print(f"[INFO] Loading features from {FEATURE_CSV}")
        return pd.read_csv(FEATURE_CSV)

    print("[INFO] Extracting features from audio...")
    df_features = build_feature_dataset()
    df_features.to_csv(FEATURE_CSV, index=False)

    print(f"[CHECK] Extracted features for {len(df_features)} samples.")
    return df_features


# ==========================================================
#                  PREPROCESSING
# ==========================================================

def preprocess_data(df_features: pd.DataFrame):
    """
    Clean, scale and split features.
    """
    y = df_features["diagnosis"]

    X = df_features.drop(columns=["sample_id", "label_raw", "diagnosis"], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[INFO] Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    print(f"[INFO] y_train: {y_train.value_counts().to_dict()}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


# ==========================================================
#                 MODEL TRAINING
# ==========================================================

def train_models(X_train, y_train, X_test, y_test):
    """
    Train LogReg, RandomForest, XGBoost (with numeric labels).
    """
    # Convert to numeric for XGBoost
    y_train_num = y_train.map({"ASD": 1, "TYPICAL": 0})
    y_test_num = y_test.map({"ASD": 1, "TYPICAL": 0})

    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n[TRAIN] {name}")

        if name == "XGBoost":
            model.fit(X_train, y_train_num)
            y_pred_num = model.predict(X_test)
            y_pred = np.where(y_pred_num == 1, "ASD", "TYPICAL")
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, digits=4)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "report": report,
            "cm": cm,
        }

        print(f"[RESULT] {name}: Accuracy={acc:.4f}, F1={f1:.4f}")
        print(report)
        print(cm)

    return results


# ==========================================================
#                 SAVE BEST MODEL
# ==========================================================

def pick_and_save_best_model(results, scaler, feature_names):
    MODEL_DIR.mkdir(exist_ok=True)

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    best_f1 = results[best_name]["f1"]

    payload = {
        "model_name": best_name,
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names
    }

    joblib.dump(payload, MODEL_PATH)

    print(f"\n[SAVED] Best model: {best_name} (F1={best_f1:.4f})")
    print(f"[DISK] {MODEL_PATH}")


# ==========================================================
#                DEMO PREDICTION
# ==========================================================

def demo_predict_on_file(audio_path: str):
    if not MODEL_PATH.exists():
        print("[X] Train a model first!")
        return

    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    scaler = payload["scaler"]
    feature_names = payload["feature_names"]

    feats = extract_features_for_file(audio_path)
    x = np.array([[feats.get(f, 0.0) for f in feature_names]])
    x_scaled = scaler.transform(x)

    pred = model.predict(x_scaled)[0]
    print(f"\n[DEMO] File: {audio_path}")
    print(f"Prediction: {pred}")


# ==========================================================
#                        MAIN
# ==========================================================

def main():
    print("\n========== KAGGLE ASD ACOUSTIC PIPELINE ==========\n")

    df_features = get_or_create_feature_csv()

    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df_features)

    results = train_models(X_train, y_train, X_test, y_test)

    pick_and_save_best_model(results, scaler, feature_names)

    # Demo run
    first_id = df_features.iloc[0]["sample_id"]
    audio_path = DATASET_DIR / "voice" / f"{first_id}.wav"
    if audio_path.exists():
        demo_predict_on_file(str(audio_path))


if __name__ == "__main__":
    main()
