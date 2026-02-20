# scripts/python/run_tests.py
#!/usr/bin/env python3
"""Cross-platform test runner."""
import os
import sys
import subprocess
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)

def main():
    parser = argparse.ArgumentParser(description='Run tests')
    parser.add_argument('--coverage', action='store_true', default=True)
    parser.add_argument('--verbose', '-v', action='store_true', default=True)
    parser.add_argument('--path', default='tests/')
    args = parser.parse_args()
    
    print("=" * 50)
    print("  Running Tests")
    print("=" * 50)
    
    cmd = [sys.executable, '-m', 'pytest', args.path]
    
    if args.verbose:
        cmd.append('-v')
    
    if args.coverage:
        cmd.extend(['--cov=src', '--cov-report=term-missing', '--cov-report=html'])
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if args.coverage:
        print(f"\nCoverage report: {project_root / 'htmlcov' / 'index.html'}")
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()