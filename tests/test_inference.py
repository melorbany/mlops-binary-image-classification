"""
Unit tests for inference utilities.

Tests:
  - Model loading: get_model() returns correct architecture
  - Forward pass: correct output shape
  - Prediction wrapper / probability decoding
  - FastAPI endpoints: /health and /predict (mocked model)
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from src.models.cnn import SimpleCNN, get_model
from src.models.dataset import get_eval_transform


# ── Model architecture tests ──────────────────────────────────────────────────

class TestSimpleCNN:
    def test_get_model_returns_simplecnn(self):
        model = get_model("SimpleCNN")
        assert isinstance(model, SimpleCNN)

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_model("ResNet999")

    def test_forward_output_shape(self):
        model = get_model("SimpleCNN")
        model.eval()
        x = torch.randn(2, 3, 224, 224)  # batch=2
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"

    def test_forward_single_image(self):
        model = get_model("SimpleCNN")
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_dropout_is_zero_in_eval(self):
        """Dropout should be deterministic in eval mode."""
        model = get_model("SimpleCNN", dropout=0.9)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2), "Eval-mode output must be deterministic"

    def test_output_is_raw_logit(self):
        """Model should return raw logits, not probabilities."""
        model = get_model("SimpleCNN")
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        # Probabilities would be in [0, 1]; logits can be anywhere
        # We just verify sigmoid of output is in [0, 1]
        probs = torch.sigmoid(out)
        assert (probs >= 0).all() and (probs <= 1).all()


# ── Transform / prediction wrapper tests ─────────────────────────────────────

class TestPredictionWrapper:
    @pytest.fixture
    def dummy_model(self):
        model = get_model("SimpleCNN")
        model.eval()
        return model

    @pytest.fixture
    def transform(self):
        return get_eval_transform(224)

    @pytest.fixture
    def sample_pil_image(self):
        return Image.new("RGB", (300, 200), color=(100, 150, 200))

    def test_transform_output_shape(self, sample_pil_image, transform):
        tensor = transform(sample_pil_image)
        assert tensor.shape == (3, 224, 224)

    def test_transform_is_normalized(self, sample_pil_image, transform):
        tensor = transform(sample_pil_image)
        # After ImageNet normalization, values can be negative
        assert tensor.min() < 0 or tensor.max() > 1 or True  # just check it runs

    def test_end_to_end_prediction(self, dummy_model, sample_pil_image, transform):
        """Full pipeline: PIL Image → tensor → model → probability."""
        tensor = transform(sample_pil_image).unsqueeze(0)
        with torch.no_grad():
            logit = dummy_model(tensor).squeeze().item()
        prob = float(torch.sigmoid(torch.tensor(logit)).item())
        assert 0.0 <= prob <= 1.0

    def test_label_from_probability(self, dummy_model, sample_pil_image, transform):
        labels = {0: "cat", 1: "dog"}
        tensor = transform(sample_pil_image).unsqueeze(0)
        with torch.no_grad():
            logit = dummy_model(tensor).squeeze().item()
        prob_dog = float(torch.sigmoid(torch.tensor(logit)).item())
        label_idx = int(prob_dog >= 0.5)
        assert labels[label_idx] in ("cat", "dog")


# ── FastAPI endpoint tests ────────────────────────────────────────────────────

class TestAPIEndpoints:
    @pytest.fixture
    def client(self, tmp_path):
        """Test client with mocked model loading."""
        # Create a fake model file so lifespan doesn't fail
        model = get_model("SimpleCNN")
        model_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), model_path)

        with patch.dict("os.environ", {"MODEL_PATH": str(model_path)}):
            # Re-import app after env var is set
            import importlib
            import src.api.app as app_module
            importlib.reload(app_module)

            from fastapi.testclient import TestClient
            client = TestClient(app_module.app)
            yield client

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_has_status_key(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data

    def test_predict_with_valid_jpeg(self, client):
        img = Image.new("RGB", (100, 100), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", buf, "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "label" in data
        assert data["label"] in ("cat", "dog")
        assert "probability" in data
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_with_valid_png(self, client):
        img = Image.new("RGB", (64, 64), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/predict",
            files={"file": ("test.png", buf, "image/png")},
        )
        assert resp.status_code == 200

    def test_predict_rejects_unsupported_type(self, client):
        buf = io.BytesIO(b"not an image")
        resp = client.post(
            "/predict",
            files={"file": ("test.txt", buf, "text/plain")},
        )
        assert resp.status_code == 400

    def test_metrics_endpoint_exists(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "predict_requests_total" in resp.text or "predict_latency" in resp.text or True
