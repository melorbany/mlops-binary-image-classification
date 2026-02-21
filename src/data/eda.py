"""
Exploratory Data Analysis — generate sample plots for the processed dataset.

Usage:
    python -m src.data.eda
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports/eda")


def run_eda(
    processed_dir: Path = PROCESSED_DIR, reports_dir: Path = REPORTS_DIR
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as e:
        logger.warning("EDA skipped: %s", e)
        return

    reports_dir.mkdir(parents=True, exist_ok=True)

    # Count images per split/class
    splits = ["train", "val", "test"]
    classes = ["cat", "dog"]
    counts = {}
    for split in splits:
        counts[split] = {}
        for cls in classes:
            d = processed_dir / split / cls
            counts[split][cls] = len(list(d.glob("*.jpg"))) if d.exists() else 0

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(splits))
    width = 0.35
    cat_counts = [counts[s]["cat"] for s in splits]
    dog_counts = [counts[s]["dog"] for s in splits]
    ax.bar([i - width / 2 for i in x], cat_counts, width, label="Cat")
    ax.bar([i + width / 2 for i in x], dog_counts, width, label="Dog")
    ax.set_xticks(list(x))
    ax.set_xticklabels(splits)
    ax.set_title("Dataset Split Distribution")
    ax.set_ylabel("Image Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "split_distribution.png")
    plt.close(fig)
    logger.info("Saved split_distribution.png")

    # Sample grid (4 cats + 4 dogs from train)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for row, cls in enumerate(classes):
        cls_dir = processed_dir / "train" / cls
        if not cls_dir.exists():
            continue
        imgs = list(cls_dir.glob("*.jpg"))[:4]
        for col, img_path in enumerate(imgs):
            img = Image.open(img_path)
            axes[row][col].imshow(img)
            axes[row][col].set_title(cls)
            axes[row][col].axis("off")
    fig.suptitle("Sample Training Images")
    fig.tight_layout()
    fig.savefig(reports_dir / "sample_images.png")
    plt.close(fig)
    logger.info("Saved sample_images.png")

    logger.info("EDA complete. Reports at %s", reports_dir)


if __name__ == "__main__":
    run_eda()
