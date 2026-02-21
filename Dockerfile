# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only the packages needed for inference (CPU-only torch)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements-serve.txt \
    # Remove compiled test files, __pycache__, and dist-info bloat
    && find /install -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /install -type d -name "*.dist-info" -exec rm -rf {}/RECORD {} + 2>/dev/null || true \
    && find /install -name "*.pyc" -delete \
    && find /install -name "*.pyo" -delete

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="MLOps Student" \
      description="Cats vs Dogs binary classifier inference service" \
      version="1.0.0"

WORKDIR /app

# Copy only installed inference packages from builder
COPY --from=builder /install /usr/local

# Copy application source (only the api + models modules needed at runtime)
COPY src/ ./src/
COPY models/ ./models/
COPY params.yaml .

# Non-root user for security
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
