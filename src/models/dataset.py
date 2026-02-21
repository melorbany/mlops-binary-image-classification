"""
PyTorch Dataset for the preprocessed Cats vs Dogs splits.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Transforms ───────────────────────────────────────────────────────────────


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """Training transforms with data augmentation."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transform(image_size: int = 224) -> transforms.Compose:
    """Validation / test transforms (no augmentation)."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# ── Dataset ───────────────────────────────────────────────────────────────────


class CatsDogsDataset(Dataset):
    """
    Loads images from data/processed/{split}/{cat|dog}/.

    Label mapping: cat → 0, dog → 1
    """

    LABEL_MAP = {"cat": 0, "dog": 1}

    def __init__(
        self,
        root: Path,
        split: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.transform = transform
        self.samples: list[Tuple[Path, int]] = []

        split_dir = root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory '{split_dir}' not found. "
                "Run `make preprocess` first."
            )

        for label_name, label_idx in self.LABEL_MAP.items():
            label_dir = split_dir / label_name
            if not label_dir.exists():
                continue
            for img_path in label_dir.iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.samples.append((img_path, label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


# ── DataLoader factory ────────────────────────────────────────────────────────


def get_dataloaders(
    processed_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_ds = CatsDogsDataset(processed_dir, "train", get_train_transform(image_size))
    val_ds = CatsDogsDataset(processed_dir, "val", get_eval_transform(image_size))
    test_ds = CatsDogsDataset(processed_dir, "test", get_eval_transform(image_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
