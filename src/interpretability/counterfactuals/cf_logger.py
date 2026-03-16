import csv
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("assets/counterfactual_metrics.csv")


def log_cf_metrics(summary):

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    write_header = not LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "timestamp",
                "cf_validity",
                "avg_l2",
                "avg_changed_features",
                "normalized_l2"
            ])

        writer.writerow([
            datetime.utcnow().isoformat(),
            summary.get("cf_validity", 0),
            summary.get("avg_l2", 0),
            summary.get("avg_changed_features", 0),
            summary.get("normalized_l2", 0)
        ])