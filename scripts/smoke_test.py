#!/usr/bin/env python
"""
Post-deployment smoke tests: health check + predict check.

Cross-platform — uses only stdlib (urllib, zlib, struct); no curl/bash, no Pillow.

Usage:
    # Local dev — skips predict if model not yet loaded:
    python scripts/smoke_test.py

    # CI / Docker — fails hard if model not loaded:
    python scripts/smoke_test.py --require-model

    # Point at a remote server:
    python scripts/smoke_test.py --url http://my-server:8000 --require-model
"""

import argparse
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib


# ── Health check ──────────────────────────────────────────────────────────────

def _check_health(base_url: str, require_model: bool) -> bool:
    """
    Call GET /health.

    Returns True if model is loaded, False if not.
    Exits on any HTTP error or connection failure.
    """
    url = f"{base_url}/health"
    print(f"[smoke] GET {url} ...")
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            status = resp.status
            body = json.loads(resp.read())
    except Exception as exc:
        print(f"[smoke] FAIL: /health raised {exc}")
        sys.exit(1)

    if status != 200:
        print(f"[smoke] FAIL: /health returned HTTP {status}")
        sys.exit(1)

    model_loaded = body.get("model_loaded", False)

    if not model_loaded:
        msg = f"[smoke] /health → model_loaded=false  (response: {body})"
        if require_model:
            print(f"{msg}")
            print("[smoke] FAIL: model must be loaded in CI/Docker mode (--require-model).")
            print("[smoke]       Ensure the image was built with a trained model.pt baked in.")
            sys.exit(1)
        else:
            print(f"[smoke] WARN: {msg}")
            print("[smoke] WARN: Skipping /predict — train the model first: python -m src.models.train")
            return False

    print(f"[smoke] PASS: /health → {body}")
    return True


# ── PNG helper ────────────────────────────────────────────────────────────────

def _make_png_bytes() -> bytes:
    """Create a minimal 1×1 RGB PNG using only stdlib (zlib + struct)."""
    def png_chunk(name: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)

    # IHDR: width=1, height=1, bit_depth=8, color_type=2 (RGB), compression/filter/interlace=0
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw_row = b"\x00" + bytes([120, 80, 40])  # filter byte + RGB pixel
    idat = zlib.compress(raw_row)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", idat)
        + png_chunk(b"IEND", b"")
    )


# ── Predict check ─────────────────────────────────────────────────────────────

def _check_predict(base_url: str) -> None:
    """POST a synthetic PNG to /predict and verify the response shape."""
    url = f"{base_url}/predict"
    print(f"[smoke] POST {url} ...")

    image_bytes = _make_png_bytes()

    boundary = "----SmokeTestBoundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="smoke.png"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body_txt = exc.read().decode(errors="replace")
        print(f"[smoke] FAIL: /predict returned HTTP {exc.code}: {body_txt}")
        sys.exit(1)
    except Exception as exc:
        print(f"[smoke] FAIL: /predict raised {exc}")
        sys.exit(1)

    label = data.get("label", "")
    if label not in ("cat", "dog"):
        print(f"[smoke] FAIL: unexpected label '{label}' in response {data}")
        sys.exit(1)

    print(f"[smoke] PASS: /predict → label='{label}', probability={data.get('probability')}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke tests for the inference API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the service (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Seconds to wait before running tests (for container startup)",
    )
    parser.add_argument(
        "--require-model",
        action="store_true",
        default=False,
        help=(
            "Fail if the model is not loaded (use in CI/Docker). "
            "Without this flag, /predict is skipped when model_loaded=false "
            "so local dev can test the API without a trained model."
        ),
    )
    args = parser.parse_args()

    if args.wait > 0:
        print(f"[smoke] Waiting {args.wait}s for service to start ...")
        time.sleep(args.wait)

    model_ready = _check_health(args.url, require_model=args.require_model)

    if model_ready:
        _check_predict(args.url)
        print("[smoke] All smoke tests PASSED.")
    else:
        print("[smoke] Health check PASSED (model not loaded — predict skipped).")


if __name__ == "__main__":
    main()
