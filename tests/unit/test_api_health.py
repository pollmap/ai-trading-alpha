"""Tests for health check route."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture()
def client() -> TestClient:
    """Create a test client with mocked DB engine."""
    with patch("src.api.main.get_engine", new_callable=lambda: lambda: MagicMock()):
        with patch("src.api.main.close_engine", new_callable=lambda: lambda: MagicMock()):
            app = create_app()
            return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client: TestClient) -> None:
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_health_env(self, client: TestClient) -> None:
        response = client.get("/api/health")
        data = response.json()
        assert "environment" in data
