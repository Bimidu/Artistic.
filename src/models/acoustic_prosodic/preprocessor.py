"""
Acoustic & Prosodic Preprocessor - FULL IMPLEMENTATION
Author: Team Member A (Sanuthi)
Status: IMPLEMENTED
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from src.utils.logger import get_logger
logger = get_logger(__name__)


class AcousticProsodicPreprocessor:
    """Cleans, scales, and prepares prosodic features for model training."""

    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.var_thresh = VarianceThreshold(threshold=0.001)

    def fit_transform(self, df: pd.DataFrame, label_col="label") -> Tuple:
        logger.info("Starting AcousticProsodicPreprocessor...")

        # --------- 0. Drop non-numeric columns ----------
        non_numeric_cols = [c for c in df.columns if df[c].dtype == "object"]
        non_numeric_cols = [c for c in non_numeric_cols if c != label_col]

        if non_numeric_cols:
            logger.info(f"Dropping non-numeric columns: {non_numeric_cols}")
            df = df.drop(columns=non_numeric_cols)

        # --------- 1. Drop metadata columns if still present -----------
        if "file_path" in df.columns:
            df = df.drop(columns=["file_path"])

        # --------- 2. Separate labels ----------
        y = df[label_col]
        X = df.drop(columns=[label_col])

        # --------- 3. Replace NaN and infinite values ----------
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # --------- 4. Remove near-zero variance features ----------
        X = pd.DataFrame(
            self.var_thresh.fit_transform(X),
            columns=[
                c for i, c in enumerate(X.columns)
                if self.var_thresh.get_support()[i]
            ]
        )

        # --------- 5. Standardize features ----------
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )

        # --------- 6. Train-test split ----------
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        logger.info("AcousticProsodicPreprocessor: Completed successfully.")
        return X_train, X_test, y_train, y_test
