"""Download Cats vs Dogs dataset from Kaggle."""
import os
import zipfile
import shutil
from pathlib import Path

def download_dataset():
    """Download and extract the Cats vs Dogs dataset."""
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if (raw_dir / "Cat").exists() and (raw_dir / "Dog").exists():
        print("Dataset already exists.")
        return
    
    # Download using kaggle API
    os.system("kaggle datasets download -d salader/dogs-vs-cats -p data/raw")
    
    # Extract
    zip_path = raw_dir / "dogs-vs-cats.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        zip_path.unlink()
    
    print(f"Dataset downloaded to {raw_dir}")

if __name__ == "__main__":
    download_dataset()
