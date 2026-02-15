"""Tests for core type definitions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.core.types import (
    Action,
    AgentArchitecture,
    MacroData,
    Market,
    MarketSnapshot,
    ModelProvider,
    NewsItem,
    PortfolioState,
    Position,
    SymbolData,
    TradingSignal,
)


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_symbol() -> SymbolData:
    return SymbolData(
        symbol="005930",
        market=Market.KRX,
        open=71000.0,
        high=72000.0,
        low=70500.0,
        close=71500.0,
        volume=15_000_000.0,
        currency="KRW",
        per=12.5,
        pbr=1.3,
        market_cap=426_000_000_000_000.0,
    )


@pytest.fixture
def sample_macro() -> MacroData:
    return MacroData(
        kr_base_rate=3.50,
        us_fed_rate=5.25,
        usdkrw=1320.5,
        vix=18.2,
        fear_greed_index=55.0,
    )


@pytest.fixture
def sample_snapshot(sample_symbol: SymbolData, sample_macro: MacroData) -> MarketSnapshot:
    return MarketSnapshot(
        snapshot_id="test-snap-001",
        timestamp=datetime(2025, 6, 15, 3, 0, 0, tzinfo=timezone.utc),
        market=Market.KRX,
        symbols={"005930": sample_symbol},
        macro=sample_macro,
        news=[
            NewsItem(
                timestamp=datetime(2025, 6, 15, 2, 30, 0, tzinfo=timezone.utc),
                title="Samsung Q2 earnings beat estimates",
                summary="Samsung Electronics reported Q2 profit above consensus.",
                source="Reuters",
                relevance_score=0.9,
                sentiment=0.6,
            )
        ],
    )


@pytest.fixture
def sample_position() -> Position:
    return Position(
        symbol="005930",
        quantity=100.0,
        avg_entry_price=70000.0,
        current_price=71500.0,
    )


@pytest.fixture
def sample_portfolio(sample_position: Position) -> PortfolioState:
    return PortfolioState(
        portfolio_id="pf-test-001",
        model=ModelProvider.DEEPSEEK,
        architecture=AgentArchitecture.SINGLE,
        market=Market.KRX,
        cash=90_000_000.0,
        positions={"005930": sample_position},
        initial_capital=100_000_000.0,
    )


# ── Enum Tests ───────────────────────────────────────────────────

class TestEnums:
    def test_market_values(self) -> None:
        assert Market.KRX.value == "KRX"
        assert Market.US.value == "US"
        assert Market.CRYPTO.value == "CRYPTO"

    def test_action_values(self) -> None:
        assert Action.BUY.value == "BUY"
        assert Action.SELL.value == "SELL"
        assert Action.HOLD.value == "HOLD"

    def test_model_provider_values(self) -> None:
        assert ModelProvider.DEEPSEEK.value == "deepseek"
        assert ModelProvider.GEMINI.value == "gemini"
        assert ModelProvider.CLAUDE.value == "claude"
        assert ModelProvider.GPT.value == "gpt"

    def test_architecture_values(self) -> None:
        assert AgentArchitecture.SINGLE.value == "single"
        assert AgentArchitecture.MULTI.value == "multi"


# ── SymbolData Tests ─────────────────────────────────────────────

class TestSymbolData:
    def test_creation(self, sample_symbol: SymbolData) -> None:
        assert sample_symbol.symbol == "005930"
        assert sample_symbol.close == 71500.0
        assert sample_symbol.per == 12.5

    def test_negative_close_raises(self) -> None:
        with pytest.raises(ValueError, match="close price must be positive"):
            SymbolData(
                symbol="TEST", market=Market.KRX,
                open=100.0, high=100.0, low=100.0, close=-1.0,
                volume=1000.0, currency="KRW",
            )

    def test_negative_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="volume cannot be negative"):
            SymbolData(
                symbol="TEST", market=Market.KRX,
                open=100.0, high=100.0, low=100.0, close=100.0,
                volume=-1.0, currency="KRW",
            )

    def test_extra_field(self) -> None:
        sd = SymbolData(
            symbol="TEST", market=Market.US,
            open=100.0, high=105.0, low=99.0, close=103.0,
            volume=5000.0, currency="USD",
            extra={"sector": "Technology"},
        )
        assert sd.extra["sector"] == "Technology"


# ── NewsItem Tests ───────────────────────────────────────────────

class TestNewsItem:
    def test_clamping(self) -> None:
        item = NewsItem(
            timestamp=datetime.now(timezone.utc),
            title="Test",
            summary="Test",
            source="Test",
            relevance_score=1.5,
            sentiment=-2.0,
        )
        assert item.relevance_score == 1.0
        assert item.sentiment == -1.0


# ── MarketSnapshot Tests ─────────────────────────────────────────

class TestMarketSnapshot:
    def test_creation(self, sample_snapshot: MarketSnapshot) -> None:
        assert sample_snapshot.snapshot_id == "test-snap-001"
        assert sample_snapshot.market == Market.KRX
        assert len(sample_snapshot.symbols) == 1
        assert len(sample_snapshot.news) == 1

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            MarketSnapshot(
                snapshot_id="bad",
                timestamp=datetime(2025, 1, 1),  # naive, no tzinfo
                market=Market.KRX,
                symbols={},
                macro=MacroData(),
            )

    def test_to_prompt_summary(self, sample_snapshot: MarketSnapshot) -> None:
        summary = sample_snapshot.to_prompt_summary()
        assert "KRX" in summary
        assert "005930" in summary
        assert "71500.00" in summary
        assert "VIX=18.2" in summary
        assert "Samsung" in summary

    def test_to_prompt_summary_empty_news(self) -> None:
        snap = MarketSnapshot(
            snapshot_id="empty",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            market=Market.CRYPTO,
            symbols={},
            macro=MacroData(),
            news=[],
        )
        summary = snap.to_prompt_summary()
        assert "CRYPTO" in summary
        assert "Recent News" not in summary


# ── TradingSignal Tests ──────────────────────────────────────────

class TestTradingSignal:
    def test_creation(self) -> None:
        signal = TradingSignal(
            signal_id="sig-001",
            snapshot_id="snap-001",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            market=Market.CRYPTO,
            action=Action.BUY,
            weight=0.15,
            confidence=0.8,
            reasoning="Strong upward momentum with bullish RSI divergence.",
            model=ModelProvider.DEEPSEEK,
            architecture=AgentArchitecture.SINGLE,
            latency_ms=1500.0,
        )
        assert signal.action == Action.BUY
        assert signal.weight == 0.15

    def test_weight_clamping(self) -> None:
        signal = TradingSignal(
            signal_id="sig-002",
            snapshot_id="snap-001",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            market=Market.CRYPTO,
            action=Action.HOLD,
            weight=1.5,
            confidence=-0.5,
            reasoning="Clamping test",
            model=ModelProvider.GEMINI,
            architecture=AgentArchitecture.MULTI,
        )
        assert signal.weight == 1.0
        assert signal.confidence == 0.0

    def test_empty_reasoning_raises(self) -> None:
        with pytest.raises(ValueError, match="reasoning must not be empty"):
            TradingSignal(
                signal_id="sig-003",
                snapshot_id="snap-001",
                timestamp=datetime.now(timezone.utc),
                symbol="AAPL",
                market=Market.US,
                action=Action.SELL,
                weight=0.1,
                confidence=0.5,
                reasoning="   ",
                model=ModelProvider.CLAUDE,
                architecture=AgentArchitecture.SINGLE,
            )


# ── Position Tests ───────────────────────────────────────────────

class TestPosition:
    def test_unrealized_pnl_calculation(self, sample_position: Position) -> None:
        expected_pnl = (71500.0 - 70000.0) * 100.0
        assert sample_position.unrealized_pnl == expected_pnl

    def test_zero_quantity(self) -> None:
        pos = Position(
            symbol="TEST", quantity=0.0,
            avg_entry_price=100.0, current_price=110.0,
        )
        assert pos.unrealized_pnl == 0.0


# ── PortfolioState Tests ─────────────────────────────────────────

class TestPortfolioState:
    def test_total_value(self, sample_portfolio: PortfolioState) -> None:
        expected = 90_000_000.0 + (71500.0 * 100.0)
        assert sample_portfolio.total_value == expected

    def test_cash_ratio(self, sample_portfolio: PortfolioState) -> None:
        total = sample_portfolio.total_value
        expected_ratio = 90_000_000.0 / total
        assert abs(sample_portfolio.cash_ratio - expected_ratio) < 1e-6

    def test_to_prompt_summary(self, sample_portfolio: PortfolioState) -> None:
        summary = sample_portfolio.to_prompt_summary()
        assert "deepseek" in summary
        assert "single" in summary
        assert "005930" in summary
        assert "Cash:" in summary

    def test_empty_portfolio(self) -> None:
        pf = PortfolioState(
            portfolio_id="empty",
            model=ModelProvider.GPT,
            architecture=AgentArchitecture.SINGLE,
            market=Market.US,
            cash=100_000.0,
            positions={},
            initial_capital=100_000.0,
        )
        assert pf.total_value == 100_000.0
        assert pf.cash_ratio == 1.0
        assert "None" in pf.to_prompt_summary()
