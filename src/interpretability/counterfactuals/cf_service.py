from pathlib import Path
import numpy as np
from .cf_generator import CounterfactualGenerator
from .cf_serializer import serialize_counterfactual
from src.utils.logger import get_logger

logger = get_logger(__name__)

def generate_counterfactual(
    model,
    x_instance: np.ndarray,
    feature_names,
    component: str,
    predicted_class: int
):
    ae_path = Path("models/counterfactuals") / f"{component}_ae.pt"

    if not ae_path.exists():
        logger.warning(
            f"[CF] Autoencoder not found for component={component}. "
            f"Skipping counterfactual generation."
        )
        return None

    cf_gen = CounterfactualGenerator(
        model=model,
        autoencoder_path=ae_path,
        feature_names=feature_names,
        decision_threshold=0.5
    )

    target_class = 1 - predicted_class  # flip

    x_cf = cf_gen.generate(
        x=x_instance,
        target_class=target_class
    )

    explanation = serialize_counterfactual(
        original=x_instance,
        counterfactual=x_cf,
        feature_names=feature_names
    )

    # Metrics
    l2_change = float(np.linalg.norm(x_cf - x_instance))
    flipped = True

    return {
        "prediction_flipped": flipped,
        "target_class": target_class,
        "l2_change": l2_change,
        **explanation
    }
