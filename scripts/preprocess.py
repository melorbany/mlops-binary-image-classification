#!/usr/bin/env python
"""
Preprocess raw images: resize to 224×224 RGB, split train/val/test.

Cross-platform entry point — delegates to src.data.preprocess.

Usage:
    python scripts/preprocess.py
    python -m src.data.preprocess   # equivalent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import preprocess

if __name__ == "__main__":
    preprocess()
