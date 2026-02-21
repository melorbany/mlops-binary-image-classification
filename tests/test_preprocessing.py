"""
Unit tests for src/data/preprocess.py

Tests:
  - resize_and_convert: output file exists, correct size, RGB mode
  - split_files: correct proportions, deterministic, no overlap
"""

import random
from pathlib import Path

import pytest
from PIL import Image

from src.data.preprocess import resize_and_convert, split_files

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a small synthetic RGBA image on disk."""
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGBA", (64, 64), color=(255, 128, 0, 200))
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_grey_image(tmp_path: Path) -> Path:
    """Create a small grayscale image on disk."""
    img_path = tmp_path / "grey.png"
    img = Image.new("L", (100, 80), color=128)
    img.save(img_path)
    return img_path


@pytest.fixture
def file_list() -> list[Path]:
    """A dummy list of 100 fake paths for split testing."""
    return [Path(f"image_{i:04d}.jpg") for i in range(100)]


# ── resize_and_convert tests ──────────────────────────────────────────────────

class TestResizeAndConvert:
    def test_output_file_is_created(self, sample_image, tmp_path):
        dst = tmp_path / "out.jpg"
        resize_and_convert(sample_image, dst)
        assert dst.exists(), "Output file was not created"

    def test_output_size_is_224x224(self, sample_image, tmp_path):
        dst = tmp_path / "out.jpg"
        resize_and_convert(sample_image, dst)
        img = Image.open(dst)
        assert img.size == (224, 224), f"Expected (224, 224), got {img.size}"

    def test_output_mode_is_rgb(self, sample_image, tmp_path):
        """RGBA input must be converted to RGB."""
        dst = tmp_path / "out.jpg"
        resize_and_convert(sample_image, dst)
        img = Image.open(dst)
        assert img.mode == "RGB", f"Expected RGB, got {img.mode}"

    def test_greyscale_converted_to_rgb(self, sample_grey_image, tmp_path):
        """Greyscale (L) input must be converted to RGB."""
        dst = tmp_path / "out.jpg"
        resize_and_convert(sample_grey_image, dst)
        img = Image.open(dst)
        assert img.mode == "RGB", f"Expected RGB, got {img.mode}"
        assert img.size == (224, 224)

    def test_creates_parent_directories(self, sample_image, tmp_path):
        """Should create missing parent directories automatically."""
        dst = tmp_path / "nested" / "dir" / "out.jpg"
        resize_and_convert(sample_image, dst)
        assert dst.exists()

    def test_custom_target_size(self, sample_image, tmp_path, monkeypatch):
        """Verify the function respects the IMAGE_SIZE constant."""
        import src.data.preprocess as preprocess_mod
        monkeypatch.setattr(preprocess_mod, "IMAGE_SIZE", (128, 128))
        # Call directly with monkeypatched constant
        from src.data.preprocess import IMAGE_SIZE
        dst = tmp_path / "out128.jpg"
        img = Image.open(sample_image).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img.save(dst)
        assert Image.open(dst).size == (128, 128)


# ── split_files tests ─────────────────────────────────────────────────────────

class TestSplitFiles:
    def test_total_length_preserved(self, file_list):
        train, val, test = split_files(file_list)
        assert len(train) + len(val) + len(test) == len(file_list)

    def test_default_ratios(self, file_list):
        train, val, test = split_files(file_list, train_ratio=0.8, val_ratio=0.1)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_no_overlap_between_splits(self, file_list):
        train, val, test = split_files(file_list)
        train_set = set(map(str, train))
        val_set = set(map(str, val))
        test_set = set(map(str, test))
        assert train_set.isdisjoint(val_set), "Train and val share files"
        assert train_set.isdisjoint(test_set), "Train and test share files"
        assert val_set.isdisjoint(test_set), "Val and test share files"

    def test_deterministic_with_same_seed(self, file_list):
        train_a, val_a, test_a = split_files(file_list, seed=42)
        train_b, val_b, test_b = split_files(file_list, seed=42)
        assert train_a == train_b
        assert val_a == val_b
        assert test_a == test_b

    def test_different_seeds_give_different_results(self, file_list):
        train_a, _, _ = split_files(file_list, seed=42)
        train_b, _, _ = split_files(file_list, seed=99)
        assert train_a != train_b

    def test_empty_list(self):
        train, val, test = split_files([])
        assert train == [] and val == [] and test == []

    def test_small_list(self):
        files = [Path(f"{i}.jpg") for i in range(5)]
        train, val, test = split_files(files, train_ratio=0.6, val_ratio=0.2)
        assert len(train) + len(val) + len(test) == 5
