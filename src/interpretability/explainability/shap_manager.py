import matplotlib
# Set non-interactive backend before importing pyplot to prevent GUI issues in background threads
matplotlib.use('Agg')  # Use Agg backend (no GUI) - required for background tasks on macOS

import shap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SHAPManager:

    def __init__(self, model, background_data, feature_names, model_type):
        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.model_type = model_type

    #  MOVE THIS INSIDE THE CLASS
    def _ensure_numeric(self, X):
        """
        Ensure input is numeric ndarray (fixes XGBoost string issues)
        """
        X = np.asarray(X)

        if X.dtype == object:
            X = X.astype(float)

        return X

    def _get_explainer(self, background):
        """Select correct SHAP explainer."""
        if self.model_type in ["random_forest", "xgboost", "lightgbm", "gradient_boosting"]:
            return shap.TreeExplainer(self.model)
        elif self.model_type in ["logistic"]:
            return shap.LinearExplainer(self.model, background)
        else:
            # Fallback (slow but safe)
            return shap.KernelExplainer(self.model.predict_proba, background)

    def generate_global_explanations(self, X_train, save_dir):
        logger.info(f"[SHAP] Generating global explanations at {save_dir}")

        save_dir.mkdir(parents=True, exist_ok=True)

        # 🔧 FIX: enforce numeric input
        X_train = self._ensure_numeric(X_train)

        # XGBoost-specific handling
        if self.model_type == "xgboost":
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_train)

            # Binary classifier → list output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        else:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_train)

        # ============================
        # Global Beeswarm
        # ============================
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_train,
            feature_names=self.feature_names,
            show=False
        )
        plt.savefig(save_dir / "global_beeswarm.png", bbox_inches="tight")
        plt.close()

        # ============================
        # Global Bar Plot
        # ============================
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_train,
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.savefig(save_dir / "global_bar.png", bbox_inches="tight")
        plt.close()

        # Save background data
        np.save(save_dir / "background.npy", X_train)

        logger.info("[SHAP] Global explanations saved successfully")

    def select_shap_values(shap_values):
        """
        Robust SHAP output selector for:
        - Tree models
        - KernelExplainer
        - Binary & single-output models
        """
        if isinstance(shap_values, list):
            # Binary classifier → prefer positive class if available
            return shap_values[1] if len(shap_values) > 1 else shap_values[0]

        # Single output (most SVM / kernel cases)
        return shap_values

    def _select_shap_output(self, shap_values, predicted_class):
        """
        Safely select SHAP values for binary / single-output models
        """
        if isinstance(shap_values, list):
            # Binary classifier → prefer positive class if available
            return shap_values[predicted_class] if len(shap_values) > predicted_class else shap_values[0]

        # Single-output (SVM, kernel, some linear models)
        return shap_values

    def generate_local_waterfall(
            self,
            X_instance,
            save_dir: Path,
            predicted_class: int = 1
    ):
        save_dir.mkdir(parents=True, exist_ok=True)

        X_instance = np.asarray(X_instance)
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)

        # =========================
        # Select explainer
        # =========================
        if self.model_type in ["xgboost", "random_forest", "gradient_boosting", "lightgbm"]:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_instance)

        elif self.model_type == "logistic":
            explainer = shap.LinearExplainer(self.model, self.background_data)
            shap_values = explainer.shap_values(X_instance)

        else:  # SVM and others
            explainer = shap.KernelExplainer(
                self.model.predict_proba,
                self.background_data
            )
            shap_values = explainer.shap_values(X_instance)

        # =========================
        # FIX: select ONE class
        # =========================
        shap_values = np.array(shap_values)

        if shap_values.ndim == 3:
            shap_vals = shap_values[0, :, predicted_class]
        elif shap_values.ndim == 2:
            shap_vals = shap_values[:, predicted_class]
        else:
            shap_vals = shap_values[0]

        # Base value
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[predicted_class]

        # =========================
        # Waterfall plot
        # =========================
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=X_instance[0],
                feature_names=self.feature_names
            ),
            show=False
        )

        plt.savefig(save_dir / "waterfall.png", bbox_inches="tight")
        plt.close()

