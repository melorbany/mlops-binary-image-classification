"""Tests for data preprocessing functions."""
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import os

from src.data.preprocess import preprocess_image

class TestPreprocess:
    """Test suite for preprocessing functions."""
    
    def test_preprocess_image_resize(self):
        """Test that images are resized correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            input_path = Path(tmpdir) / "test_input.jpg"
            output_path = Path(tmpdir) / "test_output.jpg"
            
            # Create a 500x500 test image
            img = Image.new("RGB", (500, 500), color="red")
            img.save(input_path)
            
            # Preprocess
            result = preprocess_image(input_path, output_path, image_size=224)
            
            assert result is True
            assert output_path.exists()
            
            # Check output size
            output_img = Image.open(output_path)
            assert output_img.size == (224, 224)
    
    def test_preprocess_image_rgb_conversion(self):
        """Test that grayscale images are converted to RGB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "gray_input.jpg"
            output_path = Path(tmpdir) / "gray_output.jpg"
            
            # Create grayscale image
            img = Image.new("L", (100, 100), color=128)
            img.save(input_path)
            
            result = preprocess_image(input_path, output_path, image_size=224)
            
            assert result is True
            output_img = Image.open(output_path)
            assert output_img.mode == "RGB"
    
    def test_preprocess_invalid_image(self):
        """Test handling of invalid image files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "invalid.jpg"
            output_path = Path(tmpdir) / "output.jpg"
            
            # Create invalid file
            with open(input_path, "w") as f:
                f.write("not an image")
            
            result = preprocess_image(input_path, output_path, image_size=224)
            assert result is False
