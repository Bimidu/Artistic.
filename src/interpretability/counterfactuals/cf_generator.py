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

    def generate(
        self,
        x: np.ndarray,
        target_class: int,
        max_steps: int = 500,
        step_size: float = 0.01
    ):
        x_cf = torch.tensor(
            x,
            dtype=torch.float32,
            requires_grad=True
        )

        optimizer = torch.optim.Adam([x_cf], lr=step_size)

        for _ in range(max_steps):
            optimizer.zero_grad()

            # Classifier prediction
            X_df = pd.DataFrame(
                x_cf.detach().numpy().reshape(1, -1),
                columns=self.feature_names
            )

            proba = self.model.predict_proba(X_df)[0][target_class]

            loss_pred = -torch.tensor(proba)

            # Realism loss (autoencoder reconstruction)
            z_cf = self.ae.encode(x_cf)
            x_recon = self.ae.decode(z_cf)

            loss_realism = torch.mean((x_recon - x_cf) ** 2)

            loss = loss_pred + 0.1 * loss_realism
            loss.backward()
            optimizer.step()

            if proba > self.threshold:
                break

        return x_cf.detach().numpy()


