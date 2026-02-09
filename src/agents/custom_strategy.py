"""Custom strategy system — user-defined LLM trading strategies.

Users can create custom strategies by:
1. Defining a prompt template with variables
2. Setting risk parameters (max position, stop loss, etc.)
3. Choosing which markets and LLM to use
4. The system compiles the template into a full agent prompt

Variables available in templates:
- {market_data} — current market snapshot summary
- {portfolio_state} — current portfolio summary
- {regime} — current market regime
- {news} — recent news summary
- {macro} — macroeconomic data
- {custom_indicators} — any user-defined indicator values
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from uuid_extensions import uuid7

from src.core.logging import get_logger
from src.core.types import Market, ModelProvider, MarketSnapshot, PortfolioState

log = get_logger(__name__)


class StrategyStatus(str, Enum):
    """Lifecycle status of a custom strategy."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


@dataclass
class RiskParameters:
    """Risk management parameters for a custom strategy."""
    max_position_weight: float = 0.30
    min_cash_ratio: float = 0.20
    stop_loss_pct: float = 0.05       # 5% stop loss
    take_profit_pct: float = 0.15     # 15% take profit
    max_trades_per_day: int = 10
    max_drawdown_limit: float = 0.20  # 20% max portfolio drawdown
    position_size_mode: str = "fixed"  # "fixed" | "volatility_scaled" | "rl_optimized"

    def validate(self) -> list[str]:
        """Validate parameters and return list of errors."""
        errors: list[str] = []
        if not 0.0 < self.max_position_weight <= 1.0:
            errors.append(f"max_position_weight must be (0,1], got {self.max_position_weight}")
        if not 0.0 <= self.min_cash_ratio < 1.0:
            errors.append(f"min_cash_ratio must be [0,1), got {self.min_cash_ratio}")
        if self.stop_loss_pct <= 0:
            errors.append(f"stop_loss_pct must be positive, got {self.stop_loss_pct}")
        if self.take_profit_pct <= 0:
            errors.append(f"take_profit_pct must be positive, got {self.take_profit_pct}")
        if self.max_trades_per_day < 1:
            errors.append(f"max_trades_per_day must be >= 1, got {self.max_trades_per_day}")
        if self.position_size_mode not in ("fixed", "volatility_scaled", "rl_optimized"):
            errors.append(f"Invalid position_size_mode: {self.position_size_mode}")
        return errors


TEMPLATE_VARIABLES: list[str] = [
    "market_data", "portfolio_state", "regime", "news", "macro", "custom_indicators"
]


@dataclass
class StrategyTemplate:
    """User-defined trading strategy template."""
    strategy_id: str
    name: str
    description: str
    owner_id: str  # tenant/user ID for SaaS
    prompt_template: str  # template with {variable} placeholders
    model: ModelProvider = ModelProvider.CLAUDE
    markets: list[Market] = field(default_factory=lambda: [Market.US])
    risk_params: RiskParameters = field(default_factory=RiskParameters)
    status: StrategyStatus = StrategyStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    tags: list[str] = field(default_factory=list)
    backtest_results: dict[str, object] = field(default_factory=dict)


class StrategyBuilder:
    """Validates and compiles custom strategy templates."""

    # Strict allowed variable pattern
    _VAR_PATTERN: re.Pattern[str] = re.compile(r"\{(\w+)\}")

    def validate_template(self, template: StrategyTemplate) -> list[str]:
        """Validate a strategy template. Returns list of error messages."""
        errors: list[str] = []

        if not template.name.strip():
            errors.append("Strategy name cannot be empty")
        if len(template.name) > 100:
            errors.append("Strategy name must be <= 100 characters")
        if not template.prompt_template.strip():
            errors.append("Prompt template cannot be empty")
        if len(template.prompt_template) > 10_000:
            errors.append("Prompt template must be <= 10,000 characters")
        if not template.markets:
            errors.append("At least one market must be selected")

        # Check for unknown variables
        used_vars = set(self._VAR_PATTERN.findall(template.prompt_template))
        unknown = used_vars - set(TEMPLATE_VARIABLES)
        if unknown:
            errors.append(f"Unknown template variables: {', '.join(sorted(unknown))}")

        # Validate risk parameters
        errors.extend(template.risk_params.validate())

        log.debug("strategy_validated", strategy_id=template.strategy_id,
                  errors=len(errors))
        return errors

    def compile_prompt(
        self,
        template: StrategyTemplate,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
        regime: str = "unknown",
        custom_indicators: dict[str, float] | None = None,
    ) -> str:
        """Compile template into a full prompt with actual data."""
        variables: dict[str, str] = {
            "market_data": snapshot.to_prompt_summary(),
            "portfolio_state": portfolio.to_prompt_summary(),
            "regime": regime,
            "news": self._format_news(snapshot),
            "macro": self._format_macro(snapshot),
            "custom_indicators": self._format_indicators(custom_indicators or {}),
        }

        prompt = template.prompt_template
        for var_name, var_value in variables.items():
            prompt = prompt.replace(f"{{{var_name}}}", str(var_value))

        # Wrap with system instructions
        full_prompt = (
            f"You are executing the trading strategy '{template.name}'.\n"
            f"Risk Parameters: max_position={template.risk_params.max_position_weight:.0%}, "
            f"stop_loss={template.risk_params.stop_loss_pct:.1%}, "
            f"take_profit={template.risk_params.take_profit_pct:.1%}\n\n"
            f"{prompt}\n\n"
            f"Respond with a JSON object: "
            f'{{"action": "BUY"|"SELL"|"HOLD", "symbol": "...", '
            f'"weight": 0.0-1.0, "confidence": 0.0-1.0, "reasoning": "..."}}'
        )

        log.debug("strategy_prompt_compiled", strategy_id=template.strategy_id,
                  prompt_length=len(full_prompt))
        return full_prompt

    @staticmethod
    def _format_news(snapshot: MarketSnapshot) -> str:
        """Format news items from snapshot for template injection."""
        if not snapshot.news:
            return "No recent news available."
        lines: list[str] = []
        for item in sorted(snapshot.news, key=lambda n: n.relevance_score, reverse=True)[:5]:
            lines.append(f"- [{item.source}] {item.title} (sentiment: {item.sentiment:+.2f})")
        return "\n".join(lines)

    @staticmethod
    def _format_macro(snapshot: MarketSnapshot) -> str:
        """Format macroeconomic data from snapshot for template injection."""
        macro = snapshot.macro
        items: list[str] = []
        for field_name in ["kr_base_rate", "us_fed_rate", "usdkrw", "vix", "fear_greed_index",
                          "jp_base_rate", "cn_base_rate", "ecb_rate", "gold_price", "oil_wti_price",
                          "us_10y_yield"]:
            val = getattr(macro, field_name, None)
            if val is not None:
                items.append(f"{field_name}={val}")
        return ", ".join(items) if items else "No macro data"

    @staticmethod
    def _format_indicators(indicators: dict[str, float]) -> str:
        """Format custom indicators for template injection."""
        if not indicators:
            return "No custom indicators."
        return ", ".join(f"{k}={v:.4f}" for k, v in indicators.items())


class StrategyStore:
    """In-memory storage for custom strategies (SaaS-aware).

    For production, replace with database-backed implementation.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, StrategyTemplate] = {}

    def create(self, template: StrategyTemplate) -> StrategyTemplate:
        """Store a new strategy."""
        if not template.strategy_id:
            template.strategy_id = str(uuid7())
        self._strategies[template.strategy_id] = template
        log.info("strategy_created", strategy_id=template.strategy_id,
                 name=template.name, owner=template.owner_id)
        return template

    def get(self, strategy_id: str) -> StrategyTemplate | None:
        """Retrieve a strategy by ID."""
        return self._strategies.get(strategy_id)

    def list_by_owner(self, owner_id: str) -> list[StrategyTemplate]:
        """List all strategies belonging to an owner."""
        return [s for s in self._strategies.values() if s.owner_id == owner_id]

    def update(self, template: StrategyTemplate) -> StrategyTemplate:
        """Update an existing strategy, bumping version."""
        template.updated_at = datetime.now(timezone.utc)
        template.version += 1
        self._strategies[template.strategy_id] = template
        log.info("strategy_updated", strategy_id=template.strategy_id, version=template.version)
        return template

    def delete(self, strategy_id: str) -> bool:
        """Delete a strategy by ID. Returns True if found and deleted."""
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            log.info("strategy_deleted", strategy_id=strategy_id)
            return True
        return False

    def activate(self, strategy_id: str) -> bool:
        """Set a strategy's status to ACTIVE. Returns True if found."""
        s = self._strategies.get(strategy_id)
        if s:
            s.status = StrategyStatus.ACTIVE
            s.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def archive(self, strategy_id: str) -> bool:
        """Set a strategy's status to ARCHIVED. Returns True if found."""
        s = self._strategies.get(strategy_id)
        if s:
            s.status = StrategyStatus.ARCHIVED
            s.updated_at = datetime.now(timezone.utc)
            return True
        return False
