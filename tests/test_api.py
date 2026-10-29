"""Tests for ALPR REST API endpoints."""

from __future__ import annotations

import io
from pathlib import Path
import pytest

from vehicle_intelligence.api import app as api_module


def test_api_health_endpoint():
    if not api_module.HAS_FASTAPI:
        pytest.skip("FastAPI not installed in environment (per 'do not reinstall dependencies' constraint).")

    from fastapi.testclient import TestClient

    client = TestClient(api_module.app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "vehicle-intelligence-alpr"


def test_api_read_plate_endpoint():
    if not api_module.HAS_FASTAPI:
        pytest.skip("FastAPI not installed in environment (per 'do not reinstall dependencies' constraint).")

    from fastapi.testclient import TestClient

    client = TestClient(api_module.app)
    golden_img = Path("tests/golden/images/0010.jpg")
    if not golden_img.exists():
        pytest.skip("Golden sample image missing.")

    with open(golden_img, "rb") as f:
        response = client.post(
            "/v1/read-plate",
            files={"file": ("0010.jpg", f.read(), "image/jpeg")},
        )

    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) > 0
    plate = results[0]
    assert "plate_text" in plate
    assert "digits" in plate
    assert "letters" in plate
    assert "plate_confidence" in plate
