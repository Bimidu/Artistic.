import numpy as np

class CounterfactualMetrics:

    def __init__(self):
        self.total = 0
        self.valid = 0
        self.l2_values = []
        self.changed_features = []
        self.normalized_l2 = []

    def update(self, cf_result, n_features):

        if cf_result is None:
            return

        self.total += 1

        if cf_result.get("prediction_flipped"):
            self.valid += 1

        l2 = cf_result.get("l2_change", 0)
        self.l2_values.append(l2)

        changed = cf_result.get("total_features_changed", 0)
        self.changed_features.append(changed)

        if n_features > 0:
            norm_l2 = l2 / np.sqrt(n_features)
            self.normalized_l2.append(norm_l2)

    def summary(self):

        if self.total == 0:
            return {}

        return {
            "cf_validity": self.valid / self.total,
            "avg_l2": float(np.mean(self.l2_values)),
            "avg_changed_features": float(np.mean(self.changed_features)),
            "normalized_l2": float(np.mean(self.normalized_l2)),
        }