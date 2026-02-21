"""
Preprocess raw Cats vs Dogs images.

Steps:
  1. Scan raw directory for cat/dog images
  2. Convert to RGB, resize to 224×224
  3. Split into train / val / test (80 / 10 / 10)
  4. Apply augmentation metadata (actual augmentation done in DataLoader)
  5. Save split file lists to data/processed/

Usage:
    python -m src.data.preprocess
    # or
    bash scripts/preprocess.sh
"""

import logging
import random
import shutil
from pathlib import Path
from typing import List, Tuple

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
IMAGE_SIZE = (64, 64)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO = 0.10 (remainder)

RANDOM_SEED = 42


# ── Core transformation ──────────────────────────────────────────────────────

def resize_and_convert(src: Path, dst: Path) -> None:
    """Open an image, convert to RGB, resize to IMAGE_SIZE, and save."""
    img = Image.open(src).convert("RGB")
    img = img.resize(IMAGE_SIZE, Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


# ── Splitting ────────────────────────────────────────────────────────────────

def split_files(
    files: List[Path],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Randomly split a file list into train / val / test."""
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = files[:n_train]
    val = files[n_train : n_train + n_val]
    test = files[n_train + n_val :]

    return train, val, test


# ── Main pipeline ────────────────────────────────────────────────────────────

def _collect_images(raw_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Return (cat_images, dog_images) lists from raw_dir."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    cats, dogs = [], []

    for p in raw_dir.rglob("*"):
        if p.suffix.lower() not in exts:
            continue
        name = p.stem.lower()
        if "cat" in name:
            cats.append(p)
        elif "dog" in name:
            dogs.append(p)

    logger.info("Found %d cat images and %d dog images", len(cats), len(dogs))
    return cats, dogs


def preprocess(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> None:
    """Full preprocessing pipeline."""
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory '{raw_dir}' not found. "
            "Run `make download` first."
        )

    cats, dogs = _collect_images(raw_dir)

    if not cats and not dogs:
        raise RuntimeError(
            f"No images found in '{raw_dir}'. "
            "Ensure the dataset was downloaded correctly."
        )

    for label, images in [("cat", cats), ("dog", dogs)]:
        train_imgs, val_imgs, test_imgs = split_files(images)
        logger.info(
            "[%s] train=%d  val=%d  test=%d",
            label, len(train_imgs), len(val_imgs), len(test_imgs),
        )

        for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            split_dir = processed_dir / split_name / label
            split_dir.mkdir(parents=True, exist_ok=True)

            for src in split_imgs:
                dst = split_dir / src.name
                try:
                    resize_and_convert(src, dst)
                except Exception as exc:
                    logger.warning("Skipping %s: %s", src, exc)

    logger.info("Preprocessing complete. Output at '%s'", processed_dir)


if __name__ == "__main__":
    preprocess()
