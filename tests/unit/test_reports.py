"""Tests for report generation (Excel, Word, PDF)."""

from __future__ import annotations

import pytest

from src.reports.excel_report import (
    ExcelReportGenerator,
    PortfolioReportData,
    ReportConfig,
)
from src.reports.word_report import WordReportGenerator
from src.reports.pdf_report import PDFReportGenerator
from src.reports.report_factory import ReportFactory, ReportFormat


def _sample_portfolios() -> list[PortfolioReportData]:
    return [
        PortfolioReportData(
            model="claude",
            architecture="single",
            market="US",
            initial_capital=100_000.0,
            final_value=112_500.0,
            total_return_pct=12.5,
            sharpe_ratio=1.85,
            max_drawdown=0.065,
            win_rate=0.62,
            total_trades=48,
            total_api_cost=3.25,
            cost_adjusted_alpha=0.0385,
            equity_curve=[100000, 101000, 102500, 105000, 108000, 112500],
            trade_log=[
                {"timestamp": "2026-01-15T10:00:00Z", "symbol": "AAPL",
                 "action": "BUY", "quantity": 10, "price": 185.50,
                 "commission": 0.93, "realized_pnl": 0.0},
                {"timestamp": "2026-01-20T14:30:00Z", "symbol": "AAPL",
                 "action": "SELL", "quantity": 10, "price": 192.30,
                 "commission": 1.06, "realized_pnl": 66.01},
            ],
        ),
        PortfolioReportData(
            model="deepseek",
            architecture="multi",
            market="US",
            initial_capital=100_000.0,
            final_value=108_200.0,
            total_return_pct=8.2,
            sharpe_ratio=1.22,
            max_drawdown=0.085,
            win_rate=0.55,
            total_trades=62,
            total_api_cost=1.80,
            cost_adjusted_alpha=0.0456,
            equity_curve=[100000, 100500, 103000, 106000, 107000, 108200],
        ),
        PortfolioReportData(
            model="buy_hold",
            architecture="baseline",
            market="US",
            initial_capital=100_000.0,
            final_value=105_000.0,
            total_return_pct=5.0,
            sharpe_ratio=0.90,
            max_drawdown=0.12,
            win_rate=0.0,
            total_trades=1,
            total_api_cost=0.0,
            cost_adjusted_alpha=0.0,
        ),
    ]


class TestExcelReport:
    """Tests for ExcelReportGenerator."""

    def test_generate_returns_bytes(self) -> None:
        gen = ExcelReportGenerator()
        result = gen.generate(_sample_portfolios())
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_generate_with_config(self) -> None:
        config = ReportConfig(
            title="Custom Report",
            author="Test Suite",
            include_trade_log=True,
            include_equity_curves=True,
        )
        gen = ExcelReportGenerator(config)
        result = gen.generate(_sample_portfolios())
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_generate_empty_portfolios(self) -> None:
        gen = ExcelReportGenerator()
        result = gen.generate([])
        assert isinstance(result, bytes)

    def test_generate_no_equity_curves(self) -> None:
        config = ReportConfig(include_equity_curves=False, include_trade_log=False)
        gen = ExcelReportGenerator(config)
        result = gen.generate(_sample_portfolios())
        assert isinstance(result, bytes)


class TestWordReport:
    """Tests for WordReportGenerator."""

    def test_generate_returns_bytes(self) -> None:
        gen = WordReportGenerator()
        result = gen.generate(_sample_portfolios())
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_generate_empty(self) -> None:
        gen = WordReportGenerator()
        result = gen.generate([])
        assert isinstance(result, bytes)


class TestPDFReport:
    """Tests for PDFReportGenerator."""

    def test_generate_returns_bytes(self) -> None:
        gen = PDFReportGenerator()
        result = gen.generate(_sample_portfolios())
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_generate_empty(self) -> None:
        gen = PDFReportGenerator()
        result = gen.generate([])
        assert isinstance(result, bytes)


class TestReportFactory:
    """Tests for ReportFactory."""

    def test_generate_excel(self) -> None:
        result = ReportFactory.generate(ReportFormat.EXCEL, _sample_portfolios())
        assert isinstance(result, bytes)

    def test_generate_word(self) -> None:
        result = ReportFactory.generate(ReportFormat.WORD, _sample_portfolios())
        assert isinstance(result, bytes)

    def test_generate_pdf(self) -> None:
        result = ReportFactory.generate(ReportFormat.PDF, _sample_portfolios())
        assert isinstance(result, bytes)

    def test_content_types(self) -> None:
        assert "spreadsheet" in ReportFactory.get_content_type(ReportFormat.EXCEL)
        assert "word" in ReportFactory.get_content_type(ReportFormat.WORD) or \
               "document" in ReportFactory.get_content_type(ReportFormat.WORD)
        assert "pdf" in ReportFactory.get_content_type(ReportFormat.PDF)

    def test_extensions(self) -> None:
        assert ReportFactory.get_extension(ReportFormat.EXCEL) == ".xlsx"
        assert ReportFactory.get_extension(ReportFormat.WORD) == ".docx"
        assert ReportFactory.get_extension(ReportFormat.PDF) == ".pdf"
