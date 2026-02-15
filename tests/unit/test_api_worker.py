"""Tests for simulation worker — status updates and config fetching."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.worker import _update_simulation_status, _get_simulation_config


def _mock_engine(mock_conn: AsyncMock) -> MagicMock:
    """Create a mock engine with proper async context manager for begin()."""
    engine = MagicMock()

    @asynccontextmanager
    async def _begin() -> AsyncIterator[AsyncMock]:
        yield mock_conn

    engine.begin = _begin
    return engine


class TestUpdateSimulationStatus:
    @pytest.mark.asyncio
    async def test_running(self) -> None:
        mock_conn = AsyncMock()
        engine = _mock_engine(mock_conn)

        with patch("src.api.worker.get_engine", AsyncMock(return_value=engine)):
            await _update_simulation_status("sim-1", "running")
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_completed(self) -> None:
        mock_conn = AsyncMock()
        engine = _mock_engine(mock_conn)

        with patch("src.api.worker.get_engine", AsyncMock(return_value=engine)):
            await _update_simulation_status(
                "sim-1", "completed", total_cycles=10
            )
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed(self) -> None:
        mock_conn = AsyncMock()
        engine = _mock_engine(mock_conn)

        with patch("src.api.worker.get_engine", AsyncMock(return_value=engine)):
            await _update_simulation_status(
                "sim-1", "failed", error_message="Something broke"
            )
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_queued(self) -> None:
        mock_conn = AsyncMock()
        engine = _mock_engine(mock_conn)

        with patch("src.api.worker.get_engine", AsyncMock(return_value=engine)):
            await _update_simulation_status("sim-1", "queued")
        mock_conn.execute.assert_called_once()


class TestGetSimulationConfig:
    @pytest.mark.asyncio
    async def test_found(self) -> None:
        config = {"markets": ["US"], "cycles": 10}
        mock_result = MagicMock()
        mock_result.mappings.return_value.first.return_value = {
            "config_json": config
        }

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        engine = _mock_engine(mock_conn)

        with patch("src.api.worker.get_engine", AsyncMock(return_value=engine)):
            result = await _get_simulation_config("sim-1")
            assert result is not None
            assert result["markets"] == ["US"]

    @pytest.mark.asyncio
    async def test_not_found(self) -> None:
        mock_result = MagicMock()
        mock_result.mappings.return_value.first.return_value = None

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        engine = _mock_engine(mock_conn)

        with patch("src.api.worker.get_engine", AsyncMock(return_value=engine)):
            result = await _get_simulation_config("no-such")
            assert result is None

    @pytest.mark.asyncio
    async def test_string_json(self) -> None:
        config_str = '{"markets": ["CRYPTO"], "cycles": 5}'
        mock_result = MagicMock()
        mock_result.mappings.return_value.first.return_value = {
            "config_json": config_str
        }

        mock_conn = AsyncMock()
        mock_conn.execute.return_value = mock_result
        engine = _mock_engine(mock_conn)

        with patch("src.api.worker.get_engine", AsyncMock(return_value=engine)):
            result = await _get_simulation_config("sim-2")
            assert result is not None
            assert result["markets"] == ["CRYPTO"]
