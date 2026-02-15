"""Tests for the shared YFinanceBaseAdapter and all yfinance-backed adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import DataFetchError
from src.core.types import Market, SymbolData
from src.data.adapters.yfinance_base import (
    YFinanceAdapterConfig,
    YFinanceBaseAdapter,
    _load_symbols_from_yaml,
    _yfinance_fetch_batch,
)


# ── Config fixture ──────────────────────────────────────────────

_TEST_CONFIG = YFinanceAdapterConfig(
    market=Market.JPX,
    currency="JPY",
    yaml_section="JPX",
    default_symbols=["7203.T", "6758.T"],
    info_extra_fields={
        "dividend_yield": "dividendYield",
        "fifty_two_week_high": "fiftyTwoWeekHigh",
    },
)

_BOND_CONFIG = YFinanceAdapterConfig(
    market=Market.BOND,
    currency="USD",
    yaml_section="BOND",
    default_symbols=["TLT"],
    per_field=None,
    pbr_field=None,
    market_cap_field="totalAssets",
    info_extra_fields={"yield": "yield"},
)

_COMMODITY_CONFIG = YFinanceAdapterConfig(
    market=Market.COMMODITIES,
    currency="USD",
    yaml_section="COMMODITIES",
    default_symbols=["GC=F"],
    use_info=False,
    per_field=None,
    pbr_field=None,
    market_cap_field=None,
    static_extra={"contract_type": "futures"},
)


# ── _load_symbols_from_yaml ─────────────────────────────────────

def test_load_symbols_fallback_on_missing_file() -> None:
    """Falls back to defaults when markets.yaml is missing."""
    defaults = ["A", "B"]
    with patch("src.data.adapters.yfinance_base._MARKETS_YAML", Path("/nonexistent")):
        result = _load_symbols_from_yaml("JPX", defaults)
    assert result == defaults


def test_load_symbols_fallback_on_empty_section(tmp_path: Path) -> None:
    """Falls back to defaults when the yaml section has no symbols."""
    yaml_file = tmp_path / "markets.yaml"
    yaml_file.write_text("JPX:\n  something_else: true\n")
    with patch("src.data.adapters.yfinance_base._MARKETS_YAML", yaml_file):
        result = _load_symbols_from_yaml("JPX", ["X"])
    assert result == ["X"]


def test_load_symbols_from_yaml_success(tmp_path: Path) -> None:
    """Reads symbols from yaml when present."""
    yaml_file = tmp_path / "markets.yaml"
    yaml_file.write_text("JPX:\n  symbols:\n    - '7203.T'\n    - '9984.T'\n")
    with patch("src.data.adapters.yfinance_base._MARKETS_YAML", yaml_file):
        result = _load_symbols_from_yaml("JPX", ["fallback"])
    assert result == ["7203.T", "9984.T"]


# ── _yfinance_fetch_batch ───────────────────────────────────────

def _make_mock_ticker(
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
    volume: float = 1000.0,
    info: dict | None = None,
) -> MagicMock:
    """Create a mock yfinance ticker with history."""
    ticker = MagicMock()
    hist = MagicMock()
    hist.empty = False
    row = MagicMock()
    row.__getitem__ = lambda _, key: {
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }[key]
    hist.iloc.__getitem__ = lambda _, idx: row
    ticker.history.return_value = hist
    ticker.info = info or {
        "trailingPE": 15.0,
        "priceToBook": 2.5,
        "marketCap": 1_000_000,
        "dividendYield": 0.02,
        "fiftyTwoWeekHigh": 110.0,
    }
    return ticker


def test_fetch_batch_stock_adapter() -> None:
    """Standard stock adapter extracts per/pbr/market_cap + info extras."""
    mock_tickers = MagicMock()
    mock_tickers.tickers = {"7203.T": _make_mock_ticker()}

    with patch.dict("sys.modules", {"yfinance": MagicMock(Tickers=lambda x: mock_tickers)}):
        result = _yfinance_fetch_batch(["7203.T"], _TEST_CONFIG)

    assert "7203.T" in result
    data = result["7203.T"]
    assert data.market == Market.JPX
    assert data.currency == "JPY"
    assert data.close == 102.0
    assert data.per == 15.0
    assert data.pbr == 2.5
    assert data.extra["dividend_yield"] == 0.02
    assert data.extra["fifty_two_week_high"] == 110.0


def test_fetch_batch_commodity_adapter() -> None:
    """Commodity adapter skips info and uses static_extra."""
    mock_tickers = MagicMock()
    mock_tickers.tickers = {"GC=F": _make_mock_ticker()}

    with patch.dict("sys.modules", {"yfinance": MagicMock(Tickers=lambda x: mock_tickers)}):
        result = _yfinance_fetch_batch(["GC=F"], _COMMODITY_CONFIG)

    assert "GC=F" in result
    data = result["GC=F"]
    assert data.market == Market.COMMODITIES
    assert data.per is None
    assert data.pbr is None
    assert data.market_cap is None
    assert data.extra == {"contract_type": "futures"}


def test_fetch_batch_skips_missing_ticker() -> None:
    """Symbols not found in batch are skipped gracefully."""
    mock_tickers = MagicMock()
    mock_tickers.tickers = {}  # no tickers found

    with patch.dict("sys.modules", {"yfinance": MagicMock(Tickers=lambda x: mock_tickers)}):
        result = _yfinance_fetch_batch(["MISSING.T"], _TEST_CONFIG)

    assert result == {}


def test_fetch_batch_skips_empty_history() -> None:
    """Symbols with empty history are skipped."""
    ticker = MagicMock()
    hist = MagicMock()
    hist.empty = True
    ticker.history.return_value = hist

    mock_tickers = MagicMock()
    mock_tickers.tickers = {"7203.T": ticker}

    with patch.dict("sys.modules", {"yfinance": MagicMock(Tickers=lambda x: mock_tickers)}):
        result = _yfinance_fetch_batch(["7203.T"], _TEST_CONFIG)

    assert result == {}


# ── YFinanceBaseAdapter.fetch_latest ────────────────────────────

@pytest.fixture
def adapter() -> YFinanceBaseAdapter:
    """Create a test adapter with mock settings."""
    with patch("src.data.adapters.yfinance_base._load_symbols_from_yaml", return_value=["7203.T"]):
        with patch("src.data.adapters.yfinance_base.get_settings"):
            return YFinanceBaseAdapter(config=_TEST_CONFIG)


@pytest.mark.asyncio
async def test_fetch_latest_success(adapter: YFinanceBaseAdapter) -> None:
    """fetch_latest returns data on first attempt."""
    mock_data = {"7203.T": SymbolData(
        symbol="7203.T", market=Market.JPX,
        open=100.0, high=105.0, low=95.0, close=102.0, volume=1000.0,
        currency="JPY",
    )}
    with patch("src.data.adapters.yfinance_base._yfinance_fetch_batch", return_value=mock_data):
        result = await adapter.fetch_latest()
    assert result == mock_data


@pytest.mark.asyncio
async def test_fetch_latest_raises_on_empty(adapter: YFinanceBaseAdapter) -> None:
    """fetch_latest raises DataFetchError when batch returns nothing."""
    with patch("src.data.adapters.yfinance_base._yfinance_fetch_batch", return_value={}):
        with pytest.raises(DataFetchError):
            await adapter.fetch_latest()


@pytest.mark.asyncio
async def test_fetch_latest_retries_on_error(adapter: YFinanceBaseAdapter) -> None:
    """fetch_latest retries transient errors before raising DataFetchError."""
    with patch(
        "src.data.adapters.yfinance_base._yfinance_fetch_batch",
        side_effect=RuntimeError("network error"),
    ):
        with pytest.raises(DataFetchError, match="fetch failed"):
            await adapter.fetch_latest()


# ── Subclass smoke tests ────────────────────────────────────────

@pytest.mark.parametrize("adapter_cls,module_path,market", [
    ("JPXAdapter", "src.data.adapters.jpx_adapter", Market.JPX),
    ("SSEAdapter", "src.data.adapters.sse_adapter", Market.SSE),
    ("HKEXAdapter", "src.data.adapters.hkex_adapter", Market.HKEX),
    ("EuronextAdapter", "src.data.adapters.euronext_adapter", Market.EURONEXT),
    ("LSEAdapter", "src.data.adapters.lse_adapter", Market.LSE),
    ("BondAdapter", "src.data.adapters.bond_adapter", Market.BOND),
    ("CommoditiesAdapter", "src.data.adapters.commodities_adapter", Market.COMMODITIES),
])
def test_subclass_instantiation(adapter_cls: str, module_path: str, market: Market) -> None:
    """All yfinance subclasses can be instantiated with mock settings."""
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, adapter_cls)

    with patch("src.data.adapters.yfinance_base._load_symbols_from_yaml", return_value=["TEST"]):
        with patch("src.data.adapters.yfinance_base.get_settings"):
            instance = cls()
    assert instance._config.market == market
    assert instance.symbols == ["TEST"]
