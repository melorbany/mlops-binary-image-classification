# scripts/python/preprocess.py
#!/usr/bin/env python3
"""Cross-platform data preprocessing script."""
import os
import sys
import shutil
import random
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 50)
    print("  Preprocessing Dataset")
    print("=" * 50)
    
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    
    # Check for raw data
    if not raw_dir.exists():
        print(f"Error: Raw data not found at {raw_dir}")
        print("Run create_sample_data.py first.")
        sys.exit(1)
    
    # Find images
    cat_images = list((raw_dir / "Cat").glob("*.jpg")) if (raw_dir / "Cat").exists() else []
    dog_images = list((raw_dir / "Dog").glob("*.jpg")) if (raw_dir / "Dog").exists() else []
    
    # Also check lowercase
    if not cat_images:
        cat_images = list((raw_dir / "cat").glob("*.jpg")) if (raw_dir / "cat").exists() else []
    if not dog_images:
        dog_images = list((raw_dir / "dog").glob("*.jpg")) if (raw_dir / "dog").exists() else []
    
    print(f"\nFound {len(cat_images)} cat images")
    print(f"Found {len(dog_images)} dog images")
    
    if not cat_images or not dog_images:
        print("Error: No images found. Check data/raw/ directory.")
        sys.exit(1)
    
    # Shuffle
    random.seed(42)
    random.shuffle(cat_images)
    random.shuffle(dog_images)
    
    # Split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    # test_ratio = 0.1 (remaining)
    
    def split_data(images):
        n = len(images)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }
    
    cat_splits = split_data(cat_images)
    dog_splits = split_data(dog_images)
    
    # Create directories and copy files
    for split in ['train', 'val', 'test']:
        for label in ['cat', 'dog']:
            split_dir = processed_dir / split / label
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Clear existing files
            for f in split_dir.glob("*.jpg"):
                f.unlink()
    
    # Copy files
    print("\nCopying files...")
    for split in ['train', 'val', 'test']:
        # Cats
        for i, src in enumerate(cat_splits[split]):
            dst = processed_dir / split / "cat" / f"cat_{i}.jpg"
            shutil.copy2(src, dst)
        
        # Dogs
        for i, src in enumerate(dog_splits[split]):
            dst = processed_dir / split / "dog" / f"dog_{i}.jpg"
            shutil.copy2(src, dst)
    
    print("\n" + "=" * 50)
    print("  Preprocessing completed!")
    print("=" * 50)
    print("\nData splits:")
    for split in ['train', 'val', 'test']:
        cat_count = len(list((processed_dir / split / "cat").glob("*.jpg")))
        dog_count = len(list((processed_dir / split / "dog").glob("*.jpg")))
        print(f"  {split}: {cat_count} cats, {dog_count} dogs")

if __name__ == "__main__":
    main()