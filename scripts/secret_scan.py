#!/usr/bin/env python
"""
Basic secret scanner — detect hardcoded credentials in Python source files.

Cross-platform replacement for the grep-based secret scan in CI.

Usage:
    python scripts/secret_scan.py [--dirs src tests]
"""

import argparse
import re
import sys
from pathlib import Path

# Pattern: assignment of a suspicious key name to a string value ≥8 chars
SECRET_PATTERN = re.compile(
    r'(password|secret|token|api_key)\s*=\s*["\'][^"\']{8,}["\']',
    re.IGNORECASE,
)

# Known-safe false-positive strings to ignore
ALLOWLIST = [
    "your_password_here",
    "changeme",
    "<your_token>",
    "placeholder",
]


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return list of (line_number, line_content) for matches."""
    hits = []
    try:
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if SECRET_PATTERN.search(line):
                lower = line.lower()
                if not any(safe in lower for safe in ALLOWLIST):
                    hits.append((lineno, line.strip()))
    except Exception as exc:
        print(f"[secret_scan] Warning: could not read {path}: {exc}")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan source files for hardcoded secrets")
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["src", "tests"],
        help="Directories to scan (default: src tests)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    total_hits = 0

    for dir_name in args.dirs:
        scan_dir = root / dir_name
        if not scan_dir.exists():
            continue
        for py_file in scan_dir.rglob("*.py"):
            hits = scan_file(py_file)
            for lineno, line in hits:
                print(f"[secret_scan] POTENTIAL SECRET — {py_file}:{lineno}: {line}")
                total_hits += 1

    if total_hits > 0:
        print(f"\n[secret_scan] ERROR: {total_hits} potential hardcoded secret(s) found.")
        sys.exit(1)

    print("[secret_scan] No hardcoded secrets found.")


if __name__ == "__main__":
    main()
