"""Tests des endpoints FastAPI."""
import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


@pytest.fixture
def client(mock_model):
    """TestClient avec modèle mocké — patch sur api.main.load_model."""
    with patch("api.main.load_model", return_value=mock_model):
        with TestClient(app) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# ── /breeds ───────────────────────────────────────────────────────────────────

def test_breeds_returns_list(client):
    resp = client.get("/breeds")
    assert resp.status_code == 200
    data = resp.json()
    assert "breeds" in data
    assert isinstance(data["breeds"], list)
    assert len(data["breeds"]) > 0


def test_breeds_count_matches(client):
    resp = client.get("/breeds")
    data = resp.json()
    assert data["count"] == len(data["breeds"])


# ── /predict ──────────────────────────────────────────────────────────────────

def _jpeg_bytes(color=(128, 64, 32), size=(100, 100)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="JPEG")
    return buf.getvalue()


def test_predict_returns_breed(client):
    resp = client.post(
        "/predict",
        files={"file": ("dog.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "breed" in data
    assert "confidence" in data
    assert "top_3" in data


def test_predict_top3_length(client):
    resp = client.post(
        "/predict",
        files={"file": ("dog.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    assert len(resp.json()["top_3"]) == 3


def test_predict_confidence_between_0_and_1(client):
    resp = client.post(
        "/predict",
        files={"file": ("dog.jpg", _jpeg_bytes(), "image/jpeg")},
    )
    data = resp.json()
    assert 0.0 <= data["confidence"] <= 1.0
    for item in data["top_3"]:
        assert 0.0 <= item["confidence"] <= 1.0


def test_predict_invalid_format(client):
    resp = client.post(
        "/predict",
        files={"file": ("doc.pdf", b"fake content", "application/pdf")},
    )
    assert resp.status_code == 400


def test_predict_no_model():
    """Sans modèle disponible, /predict doit retourner 503."""
    with patch("api.main.load_model", side_effect=RuntimeError("no model")):
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post(
                "/predict",
                files={"file": ("dog.jpg", _jpeg_bytes(), "image/jpeg")},
            )
    assert resp.status_code == 503


# ── /drift ────────────────────────────────────────────────────────────────────

def test_drift_summary_404_when_no_report(client):
    """Sans rapport généré, GET /drift/summary doit retourner 404."""
    resp = client.get("/drift/summary")
    assert resp.status_code == 404
