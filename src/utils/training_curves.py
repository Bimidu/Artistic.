import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import learning_curve


def generate_training_curve(model, X, y, model_name, save_dir):
    """
    Generates train vs validation accuracy curve and saves it.
    Also returns learning curve values so they can be reused later.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    train_scores_mean = train_scores.mean(axis=1)
    val_scores_mean = val_scores.mean(axis=1)

    # Individual plot (unchanged)
    plt.figure(figsize=(8,6))

    plt.plot(train_sizes, train_scores_mean, marker="o", label="Train Accuracy")
    plt.plot(train_sizes, val_scores_mean, marker="o", label="Validation Accuracy")

    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.title(f"Training Curve - {model_name}")
    plt.legend()
    plt.grid(True)

    save_path = save_dir / f"{model_name}_training_curve.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path, train_sizes, val_scores_mean

def generate_component_comparison_plot(results, save_dir):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10,6))

    for component, data in results.items():

        label = f"{component} ({data['model']})"

        plt.plot(
            data["sizes"],
            data["val"],
            marker="o",
            linewidth=2,
            label=label
        )

    plt.xlabel("Training Samples")
    plt.ylabel("Validation Accuracy")
    plt.title("Best Model Validation Curves per Component")

    plt.legend()
    plt.grid(True)

    save_path = save_dir / "component_model_comparison.png"

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path