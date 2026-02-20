"""FastAPI application for model serving."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.api.schemas import HealthResponse, PredictionResponse, MetricsResponse
from src.api.inference import get_inference_service

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Prometheus metrics
# NOTE: Prometheus label name is "class" (string). When incrementing, use the
# same label key. Since "class" is a Python keyword, we pass it via **dict.
# -----------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["class"],
)
REQUEST_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
HEALTH_CHECKS = Counter("health_check_total", "Total health checks")

# -----------------------------------------------------------------------------
# Application metrics (kept simple, in-memory)
# Do NOT hardcode class keys; allow whatever the model returns (or normalize).
# -----------------------------------------------------------------------------
app_metrics = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "predictions_by_class": {},  # dynamically populated
}

# -----------------------------------------------------------------------------
# Label normalization
# If you want to strictly expose only "cat" / "dog", normalize & validate here.
# Adjust mapping to match whatever your model returns.
# -----------------------------------------------------------------------------
_ALLOWED_CLASSES = ("cat", "dog")
_LABEL_MAP = {
    # common variants
    "cat": "cat",
    "cats": "cat",
    "feline": "cat",
    "0": "cat",
    0: "cat",
    "dog": "dog",
    "dogs": "dog",
    "canine": "dog",
    "1": "dog",
    1: "dog",
}


def _normalize_label(label) -> str:
    """
    Normalize a predicted label into 'cat'/'dog'.

    Raises:
        ValueError: if the label cannot be normalized to allowed classes.
    """
    # Try exact mapping first (handles ints too)
    if label in _LABEL_MAP:
        normalized = _LABEL_MAP[label]
    else:
        normalized = str(label).strip().lower()
        normalized = _LABEL_MAP.get(normalized, normalized)

    if normalized not in _ALLOWED_CLASSES:
        raise ValueError(f"Incorrect label names: {label!r}")

    return normalized


# -----------------------------------------------------------------------------
# Lifespan
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting up inference service...")
    get_inference_service()
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down inference service...")


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="API for binary image classification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    HEALTH_CHECKS.inc()
    inference_service = get_inference_service()

    return HealthResponse(
        status="healthy",
        model_loaded=inference_service.is_loaded(),
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict whether image contains a cat or dog.

    Accepts image file and returns prediction with confidence scores.
    """
    inference_service = get_inference_service()

    if not inference_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not (file.content_type and file.content_type.startswith("image/")):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()

        with REQUEST_LATENCY.time():
            predicted_class, confidence, probabilities, inference_time = (
                inference_service.predict(image_bytes)
            )

        # Normalize/validate label to avoid downstream "incorrect label names"
        predicted_class = _normalize_label(predicted_class)

        # Update Prometheus metrics
        REQUEST_COUNT.labels(**{"class": predicted_class}).inc()

        # Update app metrics
        app_metrics["total_requests"] += 1
        app_metrics["total_latency_ms"] += float(inference_time)
        app_metrics["predictions_by_class"].setdefault(predicted_class, 0)
        app_metrics["predictions_by_class"][predicted_class] += 1

        # Log prediction
        logger.info(
            "Prediction: %s (confidence: %.4f) - latency: %.2fms - file: %s",
            predicted_class,
            float(confidence),
            float(inference_time),
            file.filename,
        )

        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            inference_time_ms=inference_time,
        )

    except ValueError as e:
        # For label normalization / known validation errors
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/app-metrics", response_model=MetricsResponse)
async def app_metrics_endpoint():
    """Application-specific metrics."""
    total = app_metrics["total_requests"]
    avg_latency = (app_metrics["total_latency_ms"] / total) if total > 0 else 0.0

    return MetricsResponse(
        total_requests=total,
        avg_latency_ms=avg_latency,
        predictions_by_class=app_metrics["predictions_by_class"],
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)