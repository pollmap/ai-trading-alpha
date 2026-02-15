"""Pydantic V2 request/response schemas for ATLAS API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from typing import Literal

from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    """Current authenticated user info."""

    tenant_id: str
    name: str
    email: str
    avatar_url: str = ""
    plan: str
    provider: str = ""


# ── Portfolios ───────────────────────────────────────────────────

class PositionOut(BaseModel):
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0


class PortfolioOut(BaseModel):
    portfolio_id: str
    model: str
    architecture: str
    market: str
    cash: float
    total_value: float
    positions: list[PositionOut] = Field(default_factory=list)


# ── Simulations ──────────────────────────────────────────────────

class SimulationStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SimulationCreate(BaseModel):
    """Request body for creating a new simulation."""

    name: str = Field(..., min_length=1, max_length=200)
    markets: list[str] = Field(default_factory=lambda: ["US"])
    models: list[str] = Field(default_factory=lambda: ["claude"])
    architectures: list[str] = Field(default_factory=lambda: ["single"])
    cycles: int = Field(default=10, ge=1, le=1000)


class SimulationOut(BaseModel):
    simulation_id: str
    tenant_id: str
    name: str
    config_json: dict[str, object]
    status: SimulationStatus = SimulationStatus.CREATED
    started_at: datetime | None = None
    completed_at: datetime | None = None
    total_cycles: int = 0
    error_message: str | None = None
    created_at: datetime


# ── Strategies ───────────────────────────────────────────────────

class StrategyCreate(BaseModel):
    """Request body for creating a custom strategy."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    prompt_template: str = Field(..., min_length=10)
    model: str = "claude"
    markets: list[str] = Field(default_factory=lambda: ["US"])
    risk_params: dict[str, object] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class StrategyUpdate(BaseModel):
    """Request body for updating a strategy."""

    name: str | None = None
    description: str | None = None
    prompt_template: str | None = None
    model: str | None = None
    markets: list[str] | None = None
    risk_params: dict[str, object] | None = None
    tags: list[str] | None = None
    status: Literal["draft", "active", "paused", "archived"] | None = None


class StrategyOut(BaseModel):
    strategy_id: str
    tenant_id: str
    name: str
    description: str = ""
    prompt_template: str
    model: str
    markets: list[str]
    risk_params: dict[str, object]
    status: str = "draft"
    version: int = 1
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


# ── Usage ─────────────────────────────────────────────────────────

class UsageOut(BaseModel):
    tenant_id: str
    plan: str
    limits: dict[str, int]
    current: dict[str, int] = Field(default_factory=dict)


# ── Generic ───────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    environment: str = "dev"


class ErrorResponse(BaseModel):
    detail: str
