#!/usr/bin/env python
"""
Download the Cats vs Dogs dataset (Microsoft public mirror — no credentials).

This is the original Microsoft Cats vs Dogs dataset, the same data that
is hosted on Kaggle as a public dataset. No account or token required.

Usage:
    python scripts/download.py
    python scripts/download.py --dest data/raw
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.download import download_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Cats vs Dogs dataset")
    parser.add_argument("--dest", default="data/raw", help="Destination directory")
    args = parser.parse_args()

    download_dataset(Path(args.dest))
