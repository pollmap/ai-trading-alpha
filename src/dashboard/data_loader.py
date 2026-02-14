"""Dashboard data loader — reads from ResultsStore JSONL files.

Provides cached data loading for Streamlit pages. Falls back to demo data
when no results are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.results_store import ResultsStore, DEFAULT_RESULTS_DIR


def get_results_store() -> ResultsStore:
    """Get a ResultsStore instance pointing to the default results directory."""
    return ResultsStore(DEFAULT_RESULTS_DIR)


def has_real_data() -> bool:
    """Check if real benchmark results exist."""
    return get_results_store().has_data()


def load_status() -> dict[str, Any] | None:
    """Load current benchmark status."""
    return get_results_store().load_status()


def load_equity_curves_df() -> pd.DataFrame | None:
    """Load equity curves as a DataFrame.

    Returns:
        DataFrame with columns: cycle, timestamp, portfolio_id, model,
        architecture, market, cash, total_value, initial_capital,
        return_pct, n_positions.
        Returns None if no data available.
    """
    records = get_results_store().load_equity_curves()
    if not records:
        return None
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_signals_df() -> pd.DataFrame | None:
    """Load all signals as a DataFrame."""
    records = get_results_store().load_signals()
    if not records:
        return None
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_trades_df() -> pd.DataFrame | None:
    """Load all trades as a DataFrame."""
    records = get_results_store().load_trades()
    if not records:
        return None
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_costs_df() -> pd.DataFrame | None:
    """Load all cost records as a DataFrame."""
    records = get_results_store().load_costs()
    if not records:
        return None
    df = pd.DataFrame(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def build_portfolio_label(model: str, arch: str) -> str:
    """Create a human-readable label for a model/architecture pair."""
    return f"{model.capitalize()} ({arch.capitalize()})"


def get_latest_portfolios(equity_df: pd.DataFrame) -> pd.DataFrame:
    """Get the latest portfolio snapshot for each agent.

    Returns DataFrame with one row per portfolio, sorted by return_pct desc.
    """
    if equity_df.empty:
        return pd.DataFrame()

    latest_cycle = equity_df["cycle"].max()
    latest = equity_df[equity_df["cycle"] == latest_cycle].copy()
    latest["portfolio"] = latest.apply(
        lambda r: build_portfolio_label(r["model"], r["architecture"]),
        axis=1,
    )
    return latest.sort_values("return_pct", ascending=False)
