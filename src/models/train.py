"""
Train SimpleCNN on Cats vs Dogs and log everything to MLflow.

Usage:
    python -m src.models.train
    # or
    bash scripts/train.sh
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.models.cnn import get_model
from src.models.dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
PARAMS_FILE = Path("params.yaml")


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_params() -> Dict:
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == labels).float().mean().item()


# ── One epoch ─────────────────────────────────────────────────────────────────


def _run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
    desc: str = "",
) -> Tuple[float, float]:
    """Run one epoch with a tqdm progress bar; return (avg_loss, accuracy)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    bar = tqdm(
        loader,
        desc=desc,
        unit="batch",
        leave=False,
        file=sys.stdout,
        dynamic_ncols=True,
    )

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_acc += _accuracy(logits, labels)
            n_batches += 1

            bar.set_postfix(
                loss=f"{total_loss / n_batches:.4f}", acc=f"{total_acc / n_batches:.4f}"
            )

    return total_loss / n_batches, total_acc / n_batches


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray]:
    """Return (loss, accuracy, confusion_matrix) in a single pass."""
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    all_preds, all_labels = [], []

    bar = tqdm(
        loader,
        desc="  test ",
        unit="batch",
        leave=False,
        file=sys.stdout,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_acc += _accuracy(logits, labels)
            n_batches += 1

            preds = (torch.sigmoid(logits) >= 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            bar.set_postfix(
                loss=f"{total_loss / n_batches:.4f}", acc=f"{total_acc / n_batches:.4f}"
            )

    cm = confusion_matrix(all_labels, all_preds)
    return total_loss / n_batches, total_acc / n_batches, cm


# ── Training loop ─────────────────────────────────────────────────────────────


def train(params: Dict | None = None) -> Path:
    """Full training pipeline. Returns path to saved model."""
    if params is None:
        params = load_params()

    tp = params["train"]
    pp = params["preprocess"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Windows does not support fork-based multiprocessing for DataLoader;
    # force num_workers=0 to avoid silent hangs on startup.
    num_workers = 0 if sys.platform == "win32" else tp["num_workers"]
    if num_workers != tp["num_workers"]:
        logger.info(
            "Windows detected — setting num_workers=0 (was %d)", tp["num_workers"]
        )

    train_loader, val_loader, test_loader = get_dataloaders(
        processed_dir=PROCESSED_DIR,
        batch_size=tp["batch_size"],
        num_workers=num_workers,
        image_size=pp["image_size"][0],
    )

    model = get_model(tp["model_arch"], tp["dropout"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=tp["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "model.pt"

    mlflow.set_experiment("cats-vs-dogs")

    with mlflow.start_run() as run:
        logger.info("MLflow run id: %s", run.info.run_id)

        # Log all params
        mlflow.log_params(
            {
                "epochs": tp["epochs"],
                "batch_size": tp["batch_size"],
                "learning_rate": tp["learning_rate"],
                "model_arch": tp["model_arch"],
                "dropout": tp["dropout"],
                "image_size": pp["image_size"][0],
                "train_ratio": pp["train_ratio"],
                "val_ratio": pp["val_ratio"],
            }
        )

        best_val_loss = float("inf")
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(1, tp["epochs"] + 1):
            t0 = time.time()
            train_loss, train_acc = _run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                desc=f"Epoch {epoch}/{tp['epochs']} train",
            )
            val_loss, val_acc = _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                desc=f"Epoch {epoch}/{tp['epochs']}   val",
            )
            elapsed = time.time() - t0

            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                    "epoch_time_s": elapsed,
                },
                step=epoch,
            )

            logger.info(
                "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  "
                "train_acc=%.4f  val_acc=%.4f  (%.1fs)",
                epoch,
                tp["epochs"],
                train_loss,
                val_loss,
                train_acc,
                val_acc,
                elapsed,
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_path)
                logger.info("  → Saved best model (val_loss=%.4f)", best_val_loss)

        # ── Final test evaluation ─────────────────────────────────────────
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_loss, test_acc, cm = evaluate(model, test_loader, criterion, device)

        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        logger.info("Test — loss=%.4f  acc=%.4f", test_loss, test_acc)
        logger.info("Confusion matrix:\n%s", cm)

        # ── Log artifacts ─────────────────────────────────────────────────
        _log_loss_curve(train_losses, val_losses, tp["epochs"])
        _log_confusion_matrix(cm)

        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(str(model_path))

    logger.info("Training complete. Model saved to %s", model_path)
    return model_path


# ── Artifact helpers ──────────────────────────────────────────────────────────


def _log_loss_curve(train_losses, val_losses, epochs):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        xs = range(1, epochs + 1)
        ax.plot(xs, train_losses, label="train")
        ax.plot(xs, val_losses, label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        fig.tight_layout()
        path = "/tmp/loss_curve.png"
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, artifact_path="plots")
    except Exception as exc:
        logger.warning("Could not log loss curve: %s", exc)


def _log_confusion_matrix(cm):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Cat", "Dog"])
        ax.set_yticklabels(["Cat", "Dog"])
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
        fig.tight_layout()
        path = "/tmp/confusion_matrix.png"
        fig.savefig(path)
        plt.close(fig)
        mlflow.log_artifact(path, artifact_path="plots")
    except Exception as exc:
        logger.warning("Could not log confusion matrix: %s", exc)


if __name__ == "__main__":
    train()
