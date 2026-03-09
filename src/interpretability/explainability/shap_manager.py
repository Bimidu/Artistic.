import matplotlib
matplotlib.use("Agg")

import shap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SHAPManager:

    def __init__(self, model, background_data, feature_names, model_type):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type

        # Kernel SHAP must use SMALL background
        background_data = np.asarray(background_data)
        if len(background_data) > 100:
            background_data = shap.sample(background_data, 100, random_state=42)

        self.background_data = background_data.astype("float32")

    # --------------------------------------------------
    # unwrap sklearn pipeline
    # --------------------------------------------------
    def _unwrap_model(self):
        if isinstance(self.model, Pipeline):
            return list(self.model.named_steps.values())[-1]
        return self.model

    # --------------------------------------------------
    # numeric safety
    # --------------------------------------------------
    def _ensure_numeric(self, X):
        X = np.asarray(X)
        if X.dtype == object:
            X = X.astype(float)
        return X

    # --------------------------------------------------
    # unified explainer selection
    # --------------------------------------------------
    def _get_explainer(self):

        model_for_shap = self._unwrap_model()

        # ---------- TREE MODELS ----------
        if self.model_type in [
            "random_forest",
            "xgboost",
            "lightgbm",
            "gradient_boosting",
        ]:
            return shap.TreeExplainer(model_for_shap)

        # ---------- LINEAR MODELS ----------
        elif self.model_type == "logistic":
            return shap.LinearExplainer(model_for_shap, self.background_data)

        # ---------- SVM / OTHER ----------
        else:
            # IMPORTANT: wrap prediction safely
            def predict_fn(X):
                return self.model.decision_function(X)

            if not hasattr(self.model, "decision_function"):
                def predict_fn(X):
                    return self.model.predict_proba(X)

            return shap.KernelExplainer(predict_fn, self.background_data)

    # --------------------------------------------------
    # GLOBAL EXPLANATIONS
    # --------------------------------------------------
    def generate_global_explanations(self, X_train, save_dir):

        logger.info(f"[SHAP] Generating global explanations at {save_dir}")

        save_dir.mkdir(parents=True, exist_ok=True)
        X_train = self._ensure_numeric(X_train)

        explainer = self._get_explainer()
        shap_values = explainer.shap_values(X_train)

        # handle binary outputs safely
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Beeswarm
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_train,
            feature_names=self.feature_names,
            show=False,
        )
        plt.savefig(save_dir / "global_beeswarm.png", bbox_inches="tight")
        plt.close()

        # Bar plot
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_train,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False,
        )
        plt.savefig(save_dir / "global_bar.png", bbox_inches="tight")
        plt.close()

        # save sampled background
        np.save(save_dir / "background.npy", self.background_data)

        logger.info("[SHAP] Global explanations saved successfully")

    # --------------------------------------------------
    # LOCAL WATERFALL
    # --------------------------------------------------
    def generate_local_waterfall(
        self,
        X_instance,
        save_dir: Path,
        predicted_class: int = 1,
    ):

        save_dir.mkdir(parents=True, exist_ok=True)

        X_instance = np.asarray(X_instance)
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)

        explainer = self._get_explainer()
        shap_values = explainer.shap_values(X_instance)

        # ------------------------------
        # Handle different SHAP formats
        # ------------------------------
        if isinstance(shap_values, list):
            # TreeExplainer sometimes returns list[class]
            shap_vals = shap_values[predicted_class][0]

        elif isinstance(shap_values, np.ndarray):

            if shap_values.ndim == 3:
                # shape = (samples, features, classes)
                shap_vals = shap_values[0, :, predicted_class]

            elif shap_values.ndim == 2:
                # shape = (samples, features)
                shap_vals = shap_values[0]

            else:
                raise ValueError(f"Unexpected SHAP shape {shap_values.shape}")

        else:
            raise ValueError(f"Unknown SHAP output type: {type(shap_values)}")

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[predicted_class]

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=X_instance[0],
                feature_names=self.feature_names,
            ),
            show=False,
        )

        plt.savefig(save_dir / "waterfall.png", bbox_inches="tight")
        plt.close()

        logger.info("[SHAP] Local waterfall generated")