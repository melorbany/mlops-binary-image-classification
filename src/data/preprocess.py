"""Preprocess images for training."""
import os
import shutil
import yaml
from pathlib import Path
from PIL import Image
from torchvision import transforms
import random

def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def preprocess_image(image_path: Path, output_path: Path, image_size: int):
    """Resize and save a single image."""
    try:
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])
        img = transform(img)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def split_dataset(raw_dir: Path, processed_dir: Path, params: dict):
    """Split dataset into train/val/test sets."""
    image_size = params["preprocess"]["image_size"]
    train_split = params["preprocess"]["train_split"]
    val_split = params["preprocess"]["val_split"]
    
    # Create directories
    for split in ["train", "val", "test"]:
        for cls in ["cat", "dog"]:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    for cls_name, folder_name in [("cat", "Cat"), ("dog", "Dog")]:
        source_dir = raw_dir / folder_name
        if not source_dir.exists():
            # Try alternate structure
            source_dir = raw_dir
        
        images = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))
        images = [img for img in images if cls_name in img.name.lower()]
        
        random.seed(42)
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }
        
        for split_name, split_images in splits.items():
            for i, img_path in enumerate(split_images):
                output_path = processed_dir / split_name / cls_name / f"{cls_name}_{i}.jpg"
                preprocess_image(img_path, output_path, image_size)
        
        print(f"{cls_name}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

def main():
    params = load_params()
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    # Clean processed directory
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    
    split_dataset(raw_dir, processed_dir, params)
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
