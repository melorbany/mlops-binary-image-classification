#!/usr/bin/env python
"""
Deploy the inference service locally using Docker Compose.

Cross-platform — uses subprocess; no bash required.

Usage:
    python scripts/deploy.py [--image cats-vs-dogs:latest] [--smoke-test]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, inheriting the current environment plus any extras."""
    merged_env = {**os.environ, **(env or {})}
    print(f"[deploy] Running: {' '.join(cmd)}")
    return subprocess.run(cmd, env=merged_env, check=check, cwd=str(ROOT))


def pull_image(image: str) -> None:
    result = _run(["docker", "pull", image], check=False)
    if result.returncode != 0:
        print(f"[deploy] Pull failed — will use local image '{image}' if available.")


def compose_up(image: str, container_name: str) -> None:
    _run(
        ["docker", "compose", "up", "-d"],
        env={
            "DOCKER_IMAGE": image,
            "APP_CONTAINER_NAME": container_name,
        },
    )


def compose_down() -> None:
    _run(["docker", "compose", "down"], check=False)


def run_smoke_tests(url: str, wait: int = 10) -> None:
    smoke_script = ROOT / "scripts" / "smoke_test.py"
    _run([sys.executable, str(smoke_script), "--url", url, "--wait", str(wait)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy Cats vs Dogs API via Docker Compose")
    parser.add_argument(
        "--image",
        default=os.environ.get("DOCKER_IMAGE", "cats-vs-dogs:latest"),
        help="Docker image to deploy (default: cats-vs-dogs:latest)",
    )
    parser.add_argument(
        "--container-name",
        default=os.environ.get("APP_CONTAINER_NAME", "cats-vs-dogs-api"),
        help="Container name (default: cats-vs-dogs-api)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port the API listens on (default: 8000)",
    )
    parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip docker pull (use local image)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=True,
        help="Run smoke tests after deployment (default: True)",
    )
    parser.add_argument(
        "--no-smoke-test",
        dest="smoke_test",
        action="store_false",
        help="Skip smoke tests",
    )
    args = parser.parse_args()

    print(f"[deploy] Deploying image: {args.image}")

    if not args.no_pull:
        pull_image(args.image)

    compose_down()
    compose_up(args.image, args.container_name)

    if args.smoke_test:
        service_url = f"http://localhost:{args.port}"
        run_smoke_tests(service_url, wait=15)

    print("[deploy] Deployment complete.")


if __name__ == "__main__":
    main()
