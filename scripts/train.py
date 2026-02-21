#!/usr/bin/env python
"""
Train the CNN model and log experiments to MLflow.

Cross-platform entry point — delegates to src.models.train.

Usage:
    python scripts/train.py
    python -m src.models.train   # equivalent
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.train import train

if __name__ == "__main__":
    train()
