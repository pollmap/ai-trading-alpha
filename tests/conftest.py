"""Pytest configuration and compatibility helpers.

This project includes async tests marked with ``@pytest.mark.asyncio``.
Some environments run unit tests without ``pytest-asyncio`` installed, which
would otherwise make those tests fail at collection/runtime.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register local markers used in the suite."""
    config.addinivalue_line("markers", "asyncio: mark test as asyncio-compatible")


def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    """Run ``@pytest.mark.asyncio`` tests without external plugins.

    If pytest-asyncio (or another async plugin) is installed, this hook may be
    bypassed by that plugin depending on hook ordering. In plugin-less
    environments, this fallback executes coroutine tests via ``asyncio.run``.
    """
    if pyfuncitem.get_closest_marker("asyncio") is None:
        return None

    test_func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_func):
        return None

    kwargs: dict[str, Any] = {
        arg: pyfuncitem.funcargs[arg]
        for arg in pyfuncitem._fixtureinfo.argnames
    }
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(test_func(**kwargs))
    finally:
        loop.close()
        # Keep a default loop available for sync tests that call
        # ``asyncio.get_event_loop()`` directly.
        asyncio.set_event_loop(asyncio.new_event_loop())
    return True
