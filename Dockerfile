# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements-serve.txt \
    && find /install -name "*.pyc" -delete \
    && find /install -name "*.pyo" -delete \
    && find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="MLOps Student" \
      description="Cats vs Dogs binary classifier inference service" \
      version="1.0.0"

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ ./src/
COPY models/ ./models/
COPY params.yaml .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/model.pt \
    IMAGE_SIZE=64

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
