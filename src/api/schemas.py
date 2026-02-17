"""Pydantic schemas for API."""
from pydantic import BaseModel
from typing import Dict, List, Optional

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float

class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    predictions_by_class: Dict[str, int]
