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

    # ✅ MOVE THIS INSIDE THE CLASS
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
