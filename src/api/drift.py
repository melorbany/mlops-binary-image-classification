"""
Post-deployment model performance drift tracker (M5).

Collects prediction requests + true labels, evaluates accuracy over
time, and logs a drift report to MLflow.

Usage:
    python -m src.api.drift --service-url http://localhost:8000
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def simulate_prediction_batch(
    service_url: str,
    image_dir: Path,
    true_labels: List[int],
) -> Tuple[List[int], List[float]]:
    """Send images to /predict, collect predicted labels and probabilities."""
    import requests

    predicted = []
    confidences = []

    images = list(image_dir.rglob("*.jpg"))[:len(true_labels)]

    for img_path in images:
        with open(img_path, "rb") as f:
            resp = requests.post(
                f"{service_url}/predict",
                files={"file": (img_path.name, f, "image/jpeg")},
                timeout=10,
            )
        if resp.status_code == 200:
            data = resp.json()
            label_idx = 0 if data["label"] == "cat" else 1
            predicted.append(label_idx)
            confidences.append(data["probability"])
        else:
            logger.warning("Predict failed for %s: %s", img_path.name, resp.text)

    return predicted, confidences


def compute_drift_report(
    true_labels: List[int],
    predicted: List[int],
    confidences: List[float],
    window_size: int = 100,
) -> dict:
    """Compute accuracy and average confidence; flag drift if accuracy drops."""
    if not true_labels or not predicted:
        return {"error": "No predictions collected"}

    n = min(len(true_labels), len(predicted))
    correct = sum(t == p for t, p in zip(true_labels[:n], predicted[:n]))
    accuracy = correct / n
    avg_confidence = sum(confidences[:n]) / len(confidences[:n])

    # Baseline accuracy threshold (could be loaded from a registry)
    ACCURACY_THRESHOLD = 0.75
    drift_detected = accuracy < ACCURACY_THRESHOLD

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "avg_confidence": round(avg_confidence, 4),
        "drift_detected": drift_detected,
        "threshold": ACCURACY_THRESHOLD,
    }

    if drift_detected:
        logger.warning(
            "DRIFT DETECTED: accuracy=%.4f below threshold=%.4f",
            accuracy, ACCURACY_THRESHOLD,
        )
    else:
        logger.info(
            "No drift: accuracy=%.4f (threshold=%.4f)",
            accuracy, ACCURACY_THRESHOLD,
        )

    return report


def log_drift_to_mlflow(report: dict) -> None:
    """Log drift report to MLflow."""
    try:
        import mlflow
        mlflow.set_experiment("cats-vs-dogs-drift")
        with mlflow.start_run(run_name="post-deploy-drift"):
            mlflow.log_metrics({
                "post_deploy_accuracy": report.get("accuracy", 0),
                "post_deploy_avg_confidence": report.get("avg_confidence", 0),
                "drift_detected": int(report.get("drift_detected", False)),
            })
            mlflow.log_dict(report, "drift_report.json")
        logger.info("Drift report logged to MLflow")
    except Exception as exc:
        logger.warning("Could not log to MLflow: %s", exc)


def save_report(report: dict, output_path: Path = Path("reports/drift_report.json")) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    logger.info("Drift report saved to %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-deploy drift tracker")
    parser.add_argument("--service-url", default="http://localhost:8000")
    parser.add_argument("--test-dir", default="data/processed/test")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    # Simulate true labels from directory structure
    cat_images = list((test_dir / "cat").glob("*.jpg")) if (test_dir / "cat").exists() else []
    dog_images = list((test_dir / "dog").glob("*.jpg")) if (test_dir / "dog").exists() else []

    # Use up to 50 images per class
    sample_images = cat_images[:50] + dog_images[:50]
    true_labels = [0] * min(50, len(cat_images)) + [1] * min(50, len(dog_images))

    if not sample_images:
        logger.error("No test images found in %s. Run preprocessing first.", test_dir)
        raise SystemExit(1)

    predicted, confidences = simulate_prediction_batch(
        args.service_url,
        test_dir,
        true_labels,
    )

    report = compute_drift_report(true_labels, predicted, confidences)
    log_drift_to_mlflow(report)
    save_report(report)

    print(json.dumps(report, indent=2))
