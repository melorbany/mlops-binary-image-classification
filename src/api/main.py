"""FastAPI application for model serving."""
import os
import time
import logging
from datetime import datetime
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.api.schemas import HealthResponse, PredictionResponse, MetricsResponse
from src.api.inference import get_inference_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests', ['class'])
REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
HEALTH_CHECKS = Counter('health_check_total', 'Total health checks')

# Application metrics
app_metrics = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "predictions_by_class": {"cat": 0, "dog": 0}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting up inference service...")
    get_inference_service()
    logger.info("Model loaded successfully")
    yield
    logger.info("Shutting down inference service...")

app = FastAPI(
    title="Cats vs Dogs Classifier",
    description="API for binary image classification",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    HEALTH_CHECKS.inc()
    inference_service = get_inference_service()
    
    return HealthResponse(
        status="healthy",
        model_loaded=inference_service.is_loaded(),
        version="1.0.0"
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
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_bytes = await file.read()
        
        with REQUEST_LATENCY.time():
            predicted_class, confidence, probabilities, inference_time = inference_service.predict(image_bytes)
        
        # Update metrics
        REQUEST_COUNT.labels(class_=predicted_class).inc()
        app_metrics["total_requests"] += 1
        app_metrics["total_latency_ms"] += inference_time
        app_metrics["predictions_by_class"][predicted_class] += 1
        
        # Log prediction
        logger.info(
            f"Prediction: {predicted_class} (confidence: {confidence:.4f}) - "
            f"latency: {inference_time:.2f}ms - file: {file.filename}"
        )
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            inference_time_ms=inference_time
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/app-metrics", response_model=MetricsResponse)
async def app_metrics_endpoint():
    """Application-specific metrics."""
    avg_latency = 0.0
    if app_metrics["total_requests"] > 0:
        avg_latency = app_metrics["total_latency_ms"] / app_metrics["total_requests"]
    
    return MetricsResponse(
        total_requests=app_metrics["total_requests"],
        avg_latency_ms=avg_latency,
        predictions_by_class=app_metrics["predictions_by_class"]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
