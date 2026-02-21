# Cats vs Dogs — MLOps Pipeline

Binary image classification pipeline built for the MLOps Assignment (S1-25_AIMLCZG523).

## Project Structure

```
image-class/
├── .github/workflows/ci_cd.yml   # Single CI/CD workflow
├── data/
│   ├── raw/                      # Raw Kaggle dataset (DVC tracked)
│   └── processed/                # Preprocessed splits (DVC tracked)
├── models/                       # Saved model artifacts
├── src/
│   ├── data/
│   │   ├── download_kaggle.py    # Dataset download from Kaggle
│   │   └── preprocess.py         # Preprocessing & augmentation
│   ├── models/
│   │   └── train.py              # CNN training + MLflow logging
│   └── api/
│       └── app.py                # FastAPI inference service
├── tests/                        # Pytest unit tests
├── scripts/                      # Local runner scripts
├── Dockerfile                    # Container image
├── docker-compose.yml            # Deployment spec
├── Makefile                      # Convenience targets
└── requirements.txt              # Pinned dependencies
```

## Milestones

| # | Milestone | Description |
|---|-----------|-------------|
| M1 | Model Development | CNN baseline + MLflow tracking + DVC versioning |
| M2 | Packaging & Containerization | FastAPI + Dockerfile |
| M3 | CI Pipeline | GitHub Actions: lint, test, build, push |
| M4 | CD Pipeline | Auto-deploy on master via SSH + Docker Compose |
| M5 | Monitoring & Logging | Prometheus metrics + structured logging |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
make download

# 3. Preprocess data
make preprocess

# 4. Train model
make train

# 5. Run API locally
make serve

# 6. Run tests
make test

# 7. Build Docker image
make docker-build

# 8. Deploy with Docker Compose
make deploy
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `KAGGLE_USERNAME` | Kaggle API username |
| `KAGGLE_KEY` | Kaggle API key |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI (default: `./mlruns`) |

## CI/CD

The pipeline runs on every push/PR to `master`:

1. Lint (flake8 + black + isort)
2. Security scan (bandit + safety)
3. Download & preprocess data
4. Train model (MLflow)
5. Run tests + coverage
6. Build & validate Docker image
7. Push image to Docker Hub
8. Deploy to cloud server (SSH)
9. Smoke tests (health + predict)
