import numpy as np

def serialize_counterfactual(
        original: np.ndarray,
        counterfactual: np.ndarray,
        feature_names,
        top_k: int = 8
):
    deltas = counterfactual - original
    changes = []

    for f, d, o, c in zip(feature_names, deltas, original, counterfactual):
        if abs(d) > 1e-4:
            changes.append({
                "feature": f,
                "from": float(o),
                "to": float(c),
                "change": float(d)
            })

    changes.sort(
        key=lambda x: abs(x["change"]),
        reverse=True
    )

    return {
        "total_features_changed": len(changes),
        "top_changes": changes[:top_k]
    }