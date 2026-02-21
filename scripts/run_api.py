#!/usr/bin/env python
"""
Start the FastAPI inference service locally using uvicorn.

Cross-platform — uses subprocess so it works on Windows and Linux.

Usage:
    python scripts/run_api.py [--port 8000] [--host 0.0.0.0] [--no-reload]
"""

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Cats vs Dogs API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", default=False,
                        help="Enable hot reload (disabled by default on Windows to avoid "
                             "SpawnProcess traceback on Ctrl+C)")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.api.app:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    print(f"[run_api] Starting API on {args.host}:{args.port} ...")
    print("[run_api] Press Ctrl+C to stop.")
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        pass
    print("[run_api] Server stopped.")


if __name__ == "__main__":
    main()
