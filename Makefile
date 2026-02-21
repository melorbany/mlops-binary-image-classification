PYTHON         := python
PIP            := pip
DOCKER_IMAGE   := cats-vs-dogs
CONTAINER_NAME := cats-vs-dogs-api
PORT           := 8000

.PHONY: install download preprocess train test lint format security \
        docker-build docker-run docker-stop docker-validate \
        deploy smoke-test clean

## ── Dependencies ────────────────────────────────────────────────
install:
	$(PIP) install -r requirements.txt

## ── Data ────────────────────────────────────────────────────────
download:
	$(PYTHON) scripts/download.py

preprocess:
	$(PYTHON) scripts/preprocess.py

## ── Training ────────────────────────────────────────────────────
train:
	$(PYTHON) scripts/train.py

## ── API ─────────────────────────────────────────────────────────
serve:
	$(PYTHON) scripts/run_api.py --port $(PORT)

## ── Quality ─────────────────────────────────────────────────────
lint:
	flake8 src/ tests/ --max-line-length=120 --statistics
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

security:
	bandit -r src/ -ll
	safety check -r requirements.txt
	$(PYTHON) scripts/secret_scan.py

## ── Tests ───────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=70

## ── Docker ──────────────────────────────────────────────────────
docker-build:
	docker build -t $(DOCKER_IMAGE):latest .

docker-run:
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):$(PORT) $(DOCKER_IMAGE):latest

docker-stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

docker-validate:
	$(PYTHON) scripts/smoke_test.py --url http://localhost:$(PORT)

## ── Deployment ──────────────────────────────────────────────────
deploy:
	$(PYTHON) scripts/deploy.py --image $(DOCKER_IMAGE):latest --port $(PORT)

smoke-test:
	$(PYTHON) scripts/smoke_test.py --url http://localhost:$(PORT)

## ── Cleanup ─────────────────────────────────────────────────────
clean:
	$(PYTHON) -c "\
import shutil, pathlib; \
[p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]; \
[shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; \
[shutil.rmtree(d, ignore_errors=True) for d in ['.pytest_cache','htmlcov'] if pathlib.Path(d).exists()]; \
[pathlib.Path(f).unlink() for f in ['.coverage','coverage.xml'] if pathlib.Path(f).exists()]; \
print('Cleaned.')"
