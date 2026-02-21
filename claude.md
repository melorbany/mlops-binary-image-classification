# MLOps (S1-25_AIMLCZG523) — Assignment 2 (Total Marks: 50)

## Problem Statement
Design and implement an end-to-end **MLOps pipeline** for **model building**, **artifact/image creation**, **packaging**, **containerization**, and **CI/CD-based deployment** using **open-source tools**.

---

## Use Case
**Binary image classification (Cats vs Dogs)** for a pet adoption platform.

---

## Dataset (Mandatory: Download from Kaggle)
You must **download the Cats vs Dogs dataset from Kaggle** as stated in the assignment.

- **CATS and Dogs binary classification dataset from Kaggle**
- The Kaggle dataset is the source of truth and must be tracked/versioned using **DVC (recommended)** or **Git‑LFS**.

### Pre-processing Requirements
- Convert all images to **RGB (3 channels)**
- Resize to **224×224**
- Apply **data augmentation** on training data

### Data Splitting
Split into:
- **Train / Validation / Test** = **80% / 10% / 10%** (or similar)

---

# Milestones

## M1: Model Development & Experiment Tracking — 10M
**Objective:** Build a baseline model, track experiments, and version all artifacts.

### Tasks
1. **Data & Code Versioning**
   - Use **Git** for source code versioning (project structure, scripts, notebooks).
   - Use **DVC (or Git‑LFS)** for dataset versioning and for tracking pre-processed data.

2. **Model Building**
   - Implement at least one baseline model (e.g., simple CNN or logistic regression on flattened pixels).
   - Save the trained model in a standard format (e.g., `.pkl`, `.pt`, `.h5`).

3. **Experiment Tracking**
   - Use **MLflow** or **Neptune** to log:
     - runs, parameters, metrics
     - artifacts such as confusion matrix and loss curves

---

## M2: Model Packaging & Containerization — 10M
**Objective:** Package the trained model into a reproducible, containerized inference service.

### Tasks
1. **Inference Service**
   - Build a REST API using **FastAPI/Flask**.
   - Include at least:
     - `GET /health`
     - `POST /predict` (returns class label and probabilities)

2. **Environment Specification**
   - Provide `requirements.txt`
   - Pin versions of key libraries (reproducibility)

3. **Containerization**
   - Create a `Dockerfile`
   - Build and run locally; verify via curl/Postman

---

## M3: CI Pipeline for Build, Test & Image Creation — 10M
**Objective:** Add CI that tests, packages, and builds Docker images automatically.

### Tasks
1. **Automated Testing**
   - Unit tests (pytest) for:
     - one preprocessing function (e.g., image resize/normalize)
     - one inference utility function (e.g., model load / prediction wrapper)

2. **CI Setup (Single workflow file)**
   - Use **one single GitHub Actions workflow** (example: `ci_cd.yml`) that runs on:
     - push and PRs to `master`
   - Pipeline must:
     - checkout code
     - install dependencies
     - run linting + security checks
     - run unit tests
     - build Docker image

3. **Artifact Publishing**
   - Push Docker image to a registry (Docker Hub / GitHub Container Registry)

---

## M4: CD Pipeline & Deployment — 10M
**Objective:** Automatically deploy the containerized inference service after CI passes.

### Tasks
1. **Deployment Target**
   Choose one:
   - Kubernetes (kind/minikube/microk8s) **or**
   - Docker Compose **or**
   - VM/cloud server using Docker

2. **CD / GitOps Flow (Single workflow file)**
   - Use the same GitHub Actions workflow file (`ci_cd.yml`) to:
     - pull the latest image from the registry
     - deploy/update the running service automatically on `master`

3. **Smoke Tests / Health Check**
   - After deployment:
     - call `/health`
     - make at least one `/predict` call
   - Fail pipeline if smoke tests fail

---

## M5: Monitoring, Logs & Final Submission — 10M
**Objective:** Add basic monitoring/logging and prepare final submission artifacts.

### Tasks
1. **Basic Monitoring & Logging**
   - Enable request/response logging (no sensitive data)
   - Track basic metrics:
     - request count
     - latency  
     (via logs, Prometheus, or in-app counters)

2. **Model Performance Tracking (Post‑Deployment)**
   - Collect a small batch of real/simulated prediction requests + true labels
   - Evaluate and report post-deployment performance drift (basic)

---

# GitHub Actions (Single CI/CD Workflow Requirement)

Your repository must include **one single workflow file** (e.g., `.github/workflows/ci_cd.yml`) that combines **CI + CD** similar to the provided reference workflow.

### Required Characteristics
- Runs on:
  - `push` to `master`
  - `pull_request` to `master`
- Uses environment variables similar to the reference:
  - `PYTHON_VERSION`
  - `DOCKER_IMAGE`
  - `APP_CONTAINER_NAME`
- Includes steps comparable to the attached workflow (can be added/removed to match assignment), typically:
  1. **Linting** (flake8/pylint/black/isort)
  2. **Security checks** (bandit/safety/pip-audit and basic secret scanning)
  3. **Data fetch & preprocessing**  
     - Must download **Cats vs Dogs from Kaggle**
     - Preprocess into **224×224 RGB**
     - Split train/val/test
     - Upload processed dataset as workflow artifact (optional but recommended)
  4. **(Optional) EDA artifacts** (plots/images uploaded as workflow artifacts)
  5. **Train model** (log metrics/artifacts to MLflow/Neptune; upload trained model)
  6. **Tests & Coverage** (pytest + coverage threshold)
  7. **Docker build & local validation**  
     - Build image
     - Run container
     - Validate `GET /health`
  8. **Push image to registry**
  9. **Deploy on `master` only**
  10. **Post-deploy smoke tests**
      - health check + prediction check

### Secrets / Variables Naming (Must Match Reference)
Use the **same secret/variable names** as in the attached workflow (do not rename them):
- **Repository Variables**
  - `DOCKERHUB_USERNAME` (as `vars.DOCKERHUB_USERNAME`)
- **Repository Secrets**
  - `DOCKERHUB_TOKEN`
  - `CLOUD_SSH_KEY`
  - `CLOUD_HOST`
  - `CLOUD_SSH_USER`

> Note: These names must be used exactly in the workflow to align with the provided reference file.

---

# Local Scripts / Commands (Run Everything Easily)

To make testing simple, the project must provide **scripts/commands** to run each stage locally (same stages as CI/CD).  
Below is the expected set of scripts/commands (names can match exactly for consistency).

## Recommended Project Commands
> Implement these as either:
> - `Makefile` targets, **or**
> - shell scripts under `scripts/`, **or**
> - Python module commands (`python -m ...`)

### 1) Download Dataset (Kaggle)
Downloads the Kaggle cats vs dogs dataset into `data/raw/`.

```bash
# Example (Python module)
python -m src.data.download_kaggle

# OR script
bash scripts/download.sh