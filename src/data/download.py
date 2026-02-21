"""
Download the Cats vs Dogs dataset from Hugging Face (microsoft/cats_vs_dogs).

No account, credentials, or tokens required — the dataset is public.

Source: https://huggingface.co/datasets/microsoft/cats_vs_dogs

Images are saved to dest_dir/{cat,dog}/ so the preprocessing
pipeline can consume them directly.

Usage:
    python -m src.data.download
    python scripts/download.py
"""

import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
HF_DATASET = "microsoft/cats_vs_dogs"
LABEL_MAP = {0: "cat", 1: "dog"}
MAX_PER_LABEL = 100  # cap at 100 images per class (200 total)
CHUNK_SIZE = 10  # log progress and flush every 10 images per label


def download_dataset(
    dest_dir: Path = RAW_DIR,
    max_per_label: int = MAX_PER_LABEL,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """
    Stream the dataset from Hugging Face in chunks of chunk_size images per
    label, saving to dest_dir/{cat,dog}/ until max_per_label is reached.
    Safe to re-run — skips if folders already contain enough images.
    """
    from datasets import load_dataset  # huggingface datasets

    # Check if already downloaded
    cat_dir = dest_dir / "cat"
    dog_dir = dest_dir / "dog"
    cat_count = len(list(cat_dir.glob("*.jpg"))) if cat_dir.exists() else 0
    dog_count = len(list(dog_dir.glob("*.jpg"))) if dog_dir.exists() else 0
    if cat_count >= max_per_label and dog_count >= max_per_label:
        logger.info(
            "Dataset already present (%d cats, %d dogs) — skipping download.",
            cat_count,
            dog_count,
        )
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    cat_dir.mkdir(exist_ok=True)
    dog_dir.mkdir(exist_ok=True)

    logger.info(
        "Downloading '%s' — %d images per label in chunks of %d ...",
        HF_DATASET,
        max_per_label,
        chunk_size,
    )
    ds = load_dataset(HF_DATASET, split="train", trust_remote_code=True)

    counts = {"cat": 0, "dog": 0}
    # Track the start of the current chunk per label for progress reporting
    chunk_start = {"cat": 0, "dog": 0}

    for example in ds:
        label = LABEL_MAP.get(example["labels"], "unknown")
        if label not in counts:
            continue
        if counts[label] >= max_per_label:
            if all(v >= max_per_label for v in counts.values()):
                break
            continue

        idx = counts[label]
        out_path = dest_dir / label / f"{label}_{idx:05d}.jpg"
        example["image"].convert("RGB").save(out_path, format="JPEG")
        counts[label] += 1

        # Log at the end of each chunk
        if counts[label] - chunk_start[label] >= chunk_size:
            logger.info(
                "  [%s] chunk done — %d / %d saved",
                label,
                counts[label],
                max_per_label,
            )
            chunk_start[label] = counts[label]

    logger.info(
        "Download complete — cat: %d images, dog: %d images → %s",
        counts["cat"],
        counts["dog"],
        dest_dir,
    )


if __name__ == "__main__":
    download_dataset()
