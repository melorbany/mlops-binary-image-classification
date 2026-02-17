"""PyTorch Dataset for Cats vs Dogs."""
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CatsDogsDataset(Dataset):
    """Custom Dataset for Cats vs Dogs classification."""
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = "train",
        transform: Optional[transforms.Compose] = None
    ):
        self.data_dir = Path(data_dir) / split
        self.transform = transform or self._default_transform(split)
        self.classes = ["cat", "dog"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            cls_dir = self.data_dir / cls
            if cls_dir.exists():
                for img_path in cls_dir.glob("*.jpg"):
                    self.samples.append((img_path, self.class_to_idx[cls]))
    
    def _default_transform(self, split: str) -> transforms.Compose:
        if split == "train":
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    train_dataset = CatsDogsDataset(data_dir, split="train")
    val_dataset = CatsDogsDataset(data_dir, split="val")
    test_dataset = CatsDogsDataset(data_dir, split="test")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
