"""
FastAPI inference service for Cats vs Dogs binary classifier.

Endpoints:
  GET  /health   — liveness check
  POST /predict  — upload an image, get class label + probability
"""

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from src.models.cnn import get_model
from src.models.dataset import get_eval_transform

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pt"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "cat", 1: "dog"}

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "predict_requests_total",
    "Total number of prediction requests",
    ["status"],
)
REQUEST_LATENCY = Histogram(
    "predict_latency_seconds",
    "Prediction request latency in seconds",
)

# ── Global model state ────────────────────────────────────────────────────────
_model: Optional[torch.nn.Module] = None
_transform = get_eval_transform(IMAGE_SIZE)


def _load_model() -> torch.nn.Module:
    """Load model from MODEL_PATH."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Run `make train` to train the model first."
        )
    model = get_model("SimpleCNN").to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded from %s on %s", MODEL_PATH, DEVICE)
    return model


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logger.info("Starting up — loading model …")
    try:
        _model = _load_model()
    except Exception as exc:
        logger.warning("Model not loaded at startup: %s", exc)
        logger.warning("Start the service after training: python -m src.models.train")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="Binary image classification service (M2 — MLOps Assignment)",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
async def health() -> JSONResponse:
    """Liveness check."""
    model_ok = _model is not None
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "model_loaded": model_ok,
            "version": app.version,
        },
    )


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Classify an uploaded image as cat or dog.

    Returns:
        label: 'cat' or 'dog'
        probability: confidence in the predicted class
        raw_logit: model raw output
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate content type
    if file.content_type not in {"image/jpeg", "image/png", "image/bmp", "image/webp"}:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    t0 = time.perf_counter()
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = _transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

        with torch.no_grad():
            logit = _model(tensor).squeeze().item()

        prob_dog = float(torch.sigmoid(torch.tensor(logit)).item())
        label_idx = int(prob_dog >= 0.5)
        label = LABELS[label_idx]
        prob = prob_dog if label_idx == 1 else 1.0 - prob_dog

        elapsed = time.perf_counter() - t0
        REQUEST_LATENCY.observe(elapsed)
        REQUEST_COUNT.labels(status="ok").inc()

        logger.info(
            "predict: file=%s label=%s prob=%.4f latency=%.3fs",
            file.filename,
            label,
            prob,
            elapsed,
        )

        return JSONResponse(
            content={
                "label": label,
                "probability": round(prob, 4),
                "raw_logit": round(logit, 6),
                "latency_ms": round(elapsed * 1000, 2),
            }
        )

    except HTTPException:
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(status="error").inc()
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Internal prediction error")


@app.get("/metrics", tags=["Observability"])
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")
