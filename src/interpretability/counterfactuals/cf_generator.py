import torch
import numpy as np
import pandas as pd
from .autoencoder import FeatureAutoencoder

class CounterfactualGenerator:
    def __init__(
        self,
        model,
        autoencoder_path,
        feature_names,
        decision_threshold=0.5
    ):
        self.model = model
        self.feature_names = feature_names
        self.threshold = decision_threshold

        self.ae = FeatureAutoencoder(len(feature_names))
        self.ae.load_state_dict(torch.load(autoencoder_path))
        self.ae.eval()

    def _realism_score(self, x):
        """Autoencoder reconstruction error"""
        x_t = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            z = self.ae.encode(x_t)
            recon = self.ae.decode(z)

        return torch.mean((recon - x_t) ** 2).item()

    def generate(
        self,
        x: np.ndarray,
        target_class: int,
        max_iter: int = 3000,
        step_scale: float = 0.3
    ):
        """
        Gradient-free counterfactual search
        """

        x = x.copy()

        best_cf = x.copy()
        best_score = -np.inf

        original = x.copy()

        for _ in range(max_iter):

            # random perturbation
            candidate = x + np.random.normal(
                loc=0,
                scale=step_scale,
                size=x.shape
            )

            # classifier probability
            X_df = pd.DataFrame(
                candidate.reshape(1, -1),
                columns=self.feature_names
            )

            proba = self.model.predict_proba(X_df)[0][target_class]

            # proximity penalty (prefer minimal changes)
            proximity = np.linalg.norm(candidate - original)

            # realism penalty
            realism = self._realism_score(candidate)

            # final score
            score = proba - 0.2 * proximity - 0.1 * realism

            if score > best_score:
                best_score = score
                best_cf = candidate

            # stop if prediction flipped
            if proba > self.threshold:
                return candidate

        return best_cf


