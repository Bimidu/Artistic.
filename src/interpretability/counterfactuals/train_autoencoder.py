import torch
import numpy as np
from pathlib import Path

from .autoencoder import FeatureAutoencoder
from src.utils.logger import get_logger


logger = get_logger(__name__)


def train_autoencoder(
    X_train: np.ndarray,
    component: str,
    save_dir: Path,
    epochs: int = 100,
    lr: float = 1e-3,
    log_every: int = 10,
    device: str = "cpu"
):
    logger.info(
        f"[AE:{component}] Training autoencoder | "
        f"samples={X_train.shape[0]}, features={X_train.shape[1]}"
    )

    input_dim = X_train.shape[1]
    model = FeatureAutoencoder(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    ).to(device)

    # ======================
    # Training loop
    # ======================
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        recon = model(X_tensor)
        loss = loss_fn(recon, X_tensor)

        loss.backward()
        optimizer.step()

        # Log reconstruction loss
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            logger.info(
                f"[AE:{component}] Epoch {epoch:03d}/{epochs} | "
                f"Reconstruction MSE={loss.item():.6f}"
            )

    # ======================
    # Post-training diagnostics
    # ======================
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        final_loss = loss_fn(recon, X_tensor).item()

        # If your autoencoder returns latent Z separately
        if hasattr(model, "encode"):
            Z = model.encode(X_tensor)
        else:
            # Fallback if AE returns (recon, z)
            _, Z = model(X_tensor)

    logger.info(
        f"[AE:{component}] Final reconstruction MSE={final_loss:.6f}"
    )
    logger.info(
        f"[AE:{component}] Latent stats | "
        f"mean={Z.mean().item():.3f}, std={Z.std().item():.3f}"
    )

    # ======================
    # Save model
    # ======================
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{component}_ae.pt"

    torch.save(model.state_dict(), path)

    logger.info(
        f"[AE:{component}] Autoencoder saved to {path}"
    )

    return path
