# scripts/python/create_sample_data.py
#!/usr/bin/env python3
"""Create synthetic sample data for testing."""
import os
import sys
from pathlib import Path
import random

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def create_cat_image(size=(224, 224)):
    """Create a synthetic 'cat' image."""
    from PIL import Image, ImageDraw
    
    img = Image.new('RGB', size, color=(
        random.randint(200, 255),
        random.randint(180, 220),
        random.randint(150, 200)
    ))
    draw = ImageDraw.Draw(img)
    cx, cy = size[0] // 2, size[1] // 2
    
    # Face
    face_color = (random.randint(150, 200), random.randint(130, 180), random.randint(100, 150))
    draw.ellipse([cx-50, cy-30, cx+50, cy+50], fill=face_color)
    
    # Ears (triangles)
    draw.polygon([(cx-40, cy-20), (cx-50, cy-60), (cx-20, cy-30)], fill=face_color)
    draw.polygon([(cx+40, cy-20), (cx+50, cy-60), (cx+20, cy-30)], fill=face_color)
    
    # Eyes
    draw.ellipse([cx-30, cy-10, cx-15, cy+10], fill='green')
    draw.ellipse([cx+15, cy-10, cx+30, cy+10], fill='green')
    
    # Nose
    draw.polygon([(cx, cy+15), (cx-8, cy+25), (cx+8, cy+25)], fill='pink')
    
    return img

def create_dog_image(size=(224, 224)):
    """Create a synthetic 'dog' image."""
    from PIL import Image, ImageDraw
    
    img = Image.new('RGB', size, color=(
        random.randint(150, 200),
        random.randint(200, 255),
        random.randint(150, 200)
    ))
    draw = ImageDraw.Draw(img)
    cx, cy = size[0] // 2, size[1] // 2
    
    # Face (rounder)
    face_color = (random.randint(139, 180), random.randint(90, 130), random.randint(43, 80))
    draw.ellipse([cx-60, cy-40, cx+60, cy+60], fill=face_color)
    
    # Floppy ears
    ear_color = (random.randint(120, 160), random.randint(70, 110), random.randint(30, 60))
    draw.ellipse([cx-80, cy-30, cx-40, cy+40], fill=ear_color)
    draw.ellipse([cx+40, cy-30, cx+80, cy+40], fill=ear_color)
    
    # Eyes
    draw.ellipse([cx-25, cy-15, cx-10, cy+5], fill='brown')
    draw.ellipse([cx+10, cy-15, cx+25, cy+5], fill='brown')
    
    # Nose
    draw.ellipse([cx-10, cy+10, cx+10, cy+30], fill='black')
    
    return img

def main():
    print("=" * 50)
    print("  Creating Synthetic Dataset")
    print("=" * 50)
    
    # Check for PIL
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow not installed. Run: pip install Pillow")
        sys.exit(1)
    
    base_dir = project_root / "data" / "raw"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    n_images = 100  # Images per class
    
    # Create cats
    print(f"\nCreating {n_images} cat images...")
    cat_dir = base_dir / "Cat"
    cat_dir.mkdir(exist_ok=True)
    for i in range(n_images):
        img = create_cat_image()
        img.save(cat_dir / f"cat.{i}.jpg")
        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{n_images} cats")
    
    # Create dogs
    print(f"\nCreating {n_images} dog images...")
    dog_dir = base_dir / "Dog"
    dog_dir.mkdir(exist_ok=True)
    for i in range(n_images):
        img = create_dog_image()
        img.save(dog_dir / f"dog.{i}.jpg")
        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{n_images} dogs")
    
    print("\n" + "=" * 50)
    print("  Dataset created successfully!")
    print("=" * 50)
    print(f"\nLocation: {base_dir}")
    print(f"  Cats: {n_images} images")
    print(f"  Dogs: {n_images} images")
    print("\nNote: This is synthetic data for testing.")

if __name__ == "__main__":
    main()