"""PDF report generator — compact summary PDF with key metrics.

Uses reportlab for PDF generation. Creates:
- Header with title and date
- Summary table of all models
- Key metrics highlighted

Falls back to plain text if reportlab is not installed.
"""

from __future__ import annotations

import io

from src.core.logging import get_logger
from src.reports.excel_report import PortfolioReportData, ReportConfig

log = get_logger(__name__)


class PDFReportGenerator:
    """Generate compact PDF summary reports."""

    def __init__(self, config: ReportConfig | None = None) -> None:
        self._config = config or ReportConfig()

    def generate(self, portfolios: list[PortfolioReportData]) -> bytes:
        """Generate PDF document as bytes.

        Returns bytes that can be written to file or sent as HTTP response.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                SimpleDocTemplate, Table, TableStyle, Paragraph,
                Spacer, PageBreak,
            )
        except ImportError:
            log.warning("reportlab_not_installed", msg="Using fallback plain text output")
            return self._generate_text_fallback(portfolios)

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            topMargin=20 * mm,
            bottomMargin=20 * mm,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
        )

        styles = getSampleStyleSheet()
        elements: list[object] = []

        # Custom styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontSize=22,
            spaceAfter=6 * mm,
            textColor=colors.HexColor("#1F4E79"),
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            spaceAfter=3 * mm,
            textColor=colors.gray,
        )
        heading_style = ParagraphStyle(
            "ReportHeading",
            parent=styles["Heading2"],
            fontSize=14,
            spaceBefore=8 * mm,
            spaceAfter=4 * mm,
            textColor=colors.HexColor("#1F4E79"),
        )
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontSize=10,
            spaceAfter=3 * mm,
        )

        # Header
        elements.append(Paragraph(self._config.title, title_style))
        elements.append(Paragraph(
            f"Generated: {self._config.generated_at.strftime('%Y-%m-%d %H:%M UTC')} "
            f"| Author: {self._config.author}",
            subtitle_style,
        ))
        elements.append(Spacer(1, 5 * mm))

        # Key highlights
        if portfolios:
            elements.append(Paragraph("Key Highlights", heading_style))
            self._add_highlights(elements, portfolios, body_style, colors)
            elements.append(Spacer(1, 5 * mm))

        # Summary table
        elements.append(Paragraph("Performance Summary", heading_style))
        if portfolios:
            table = self._build_summary_table(portfolios, colors, mm)
            elements.append(table)
        else:
            elements.append(Paragraph("No portfolio data available.", body_style))

        elements.append(Spacer(1, 8 * mm))

        # Architecture comparison
        if portfolios:
            elements.append(Paragraph("Architecture Comparison", heading_style))
            arch_table = self._build_architecture_table(portfolios, colors, mm)
            if arch_table is not None:
                elements.append(arch_table)
            elements.append(Spacer(1, 5 * mm))

        # Cost analysis
        if portfolios:
            elements.append(Paragraph("Cost Analysis", heading_style))
            self._add_cost_analysis(elements, portfolios, body_style)

        # Footer note
        elements.append(Spacer(1, 10 * mm))
        footer_style = ParagraphStyle(
            "ReportFooter",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.gray,
            alignment=1,  # center
        )
        elements.append(Paragraph(
            f"ATLAS Benchmark System | {self._config.generated_at.strftime('%Y-%m-%d')}",
            footer_style,
        ))

        doc.build(elements)
        buf.seek(0)
        log.info("pdf_report_generated", portfolios=len(portfolios))
        return buf.getvalue()

    def _add_highlights(self, elements: list[object],
                        portfolios: list[PortfolioReportData],
                        body_style: object, colors: object) -> None:
        """Add key metric highlights."""
        best_return = max(portfolios, key=lambda p: p.total_return_pct)
        best_sharpe = max(portfolios, key=lambda p: p.sharpe_ratio)
        best_caa = max(portfolios, key=lambda p: p.cost_adjusted_alpha)
        lowest_dd = min(portfolios, key=lambda p: p.max_drawdown)

        from reportlab.lib.styles import ParagraphStyle

        highlight_style = ParagraphStyle(
            "Highlight",
            parent=body_style,
            fontSize=10,
            leftIndent=10,
            bulletIndent=0,
        )

        highlights = [
            f"<b>Best Return:</b> {best_return.model}/{best_return.architecture} "
            f"({best_return.total_return_pct:+.2f}%) in {best_return.market}",
            f"<b>Best Sharpe:</b> {best_sharpe.model}/{best_sharpe.architecture} "
            f"({best_sharpe.sharpe_ratio:.4f})",
            f"<b>Best CAA:</b> {best_caa.model}/{best_caa.architecture} "
            f"({best_caa.cost_adjusted_alpha:.4f})",
            f"<b>Lowest Drawdown:</b> {lowest_dd.model}/{lowest_dd.architecture} "
            f"({lowest_dd.max_drawdown*100:.2f}%)",
        ]

        from reportlab.platypus import Paragraph as P

        for h in highlights:
            elements.append(P(f"&bull; {h}", highlight_style))

    def _build_summary_table(self, portfolios: list[PortfolioReportData],
                             colors: object, mm: float) -> object:
        """Build the main summary table."""
        from reportlab.platypus import Table, TableStyle

        headers = ["Model", "Arch", "Market", "Return %", "Sharpe",
                   "Max DD %", "Win Rate %", "Trades", "API Cost", "CAA"]
        data: list[list[str]] = [headers]

        for p in portfolios:
            data.append([
                p.model,
                p.architecture,
                p.market,
                f"{p.total_return_pct:+.2f}",
                f"{p.sharpe_ratio:.4f}",
                f"{p.max_drawdown * 100:.2f}",
                f"{p.win_rate * 100:.1f}",
                str(p.total_trades),
                f"${p.total_api_cost:.4f}",
                f"{p.cost_adjusted_alpha:.4f}",
            ])

        col_widths = [55, 35, 45, 42, 40, 42, 45, 35, 50, 42]
        table = Table(data, colWidths=col_widths)

        style_commands: list[tuple[str, tuple[int, int], tuple[int, int], object]] = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]

        # Alternate row colors
        for i in range(1, len(data)):
            if i % 2 == 0:
                style_commands.append(
                    ("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F2F7FB"))
                )

        # Highlight best return row
        best_idx = max(range(len(portfolios)),
                       key=lambda i: portfolios[i].total_return_pct) + 1
        style_commands.append(
            ("TEXTCOLOR", (3, best_idx), (3, best_idx), colors.HexColor("#006600"))
        )

        table.setStyle(TableStyle(style_commands))
        return table

    def _build_architecture_table(self, portfolios: list[PortfolioReportData],
                                  colors: object, mm: float) -> object | None:
        """Build architecture comparison table."""
        from reportlab.platypus import Table, TableStyle

        single = [p for p in portfolios if p.architecture == "single"]
        multi = [p for p in portfolios if p.architecture == "multi"]

        if not single or not multi:
            return None

        avg_single_ret = sum(p.total_return_pct for p in single) / len(single)
        avg_multi_ret = sum(p.total_return_pct for p in multi) / len(multi)
        avg_single_sharpe = sum(p.sharpe_ratio for p in single) / len(single)
        avg_multi_sharpe = sum(p.sharpe_ratio for p in multi) / len(multi)
        avg_single_cost = sum(p.total_api_cost for p in single) / len(single)
        avg_multi_cost = sum(p.total_api_cost for p in multi) / len(multi)
        avg_single_caa = sum(p.cost_adjusted_alpha for p in single) / len(single)
        avg_multi_caa = sum(p.cost_adjusted_alpha for p in multi) / len(multi)

        data = [
            ["Metric", "Single Agent", "Multi Agent"],
            ["Avg Return %", f"{avg_single_ret:+.2f}", f"{avg_multi_ret:+.2f}"],
            ["Avg Sharpe", f"{avg_single_sharpe:.4f}", f"{avg_multi_sharpe:.4f}"],
            ["Avg API Cost", f"${avg_single_cost:.4f}", f"${avg_multi_cost:.4f}"],
            ["Avg CAA", f"{avg_single_caa:.4f}", f"{avg_multi_caa:.4f}"],
        ]

        table = Table(data, colWidths=[100, 100, 100])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return table

    def _add_cost_analysis(self, elements: list[object],
                           portfolios: list[PortfolioReportData],
                           body_style: object) -> None:
        """Add cost analysis section."""
        from reportlab.platypus import Paragraph

        total_cost = sum(p.total_api_cost for p in portfolios)
        avg_cost = total_cost / len(portfolios) if portfolios else 0.0
        total_trades = sum(p.total_trades for p in portfolios)
        cost_per_trade = total_cost / total_trades if total_trades > 0 else 0.0

        elements.append(Paragraph(
            f"<b>Total API Cost:</b> ${total_cost:.4f} across {len(portfolios)} portfolios",
            body_style,
        ))
        elements.append(Paragraph(
            f"<b>Average Cost per Portfolio:</b> ${avg_cost:.4f}",
            body_style,
        ))
        elements.append(Paragraph(
            f"<b>Total Trades:</b> {total_trades} | "
            f"<b>Cost per Trade:</b> ${cost_per_trade:.6f}",
            body_style,
        ))

        # Most cost-efficient
        if portfolios:
            best_efficiency = max(portfolios,
                                  key=lambda p: p.total_return_pct / max(p.total_api_cost, 0.0001))
            elements.append(Paragraph(
                f"<b>Most Cost-Efficient:</b> {best_efficiency.model}/"
                f"{best_efficiency.architecture} "
                f"({best_efficiency.total_return_pct:+.2f}% return / "
                f"${best_efficiency.total_api_cost:.4f} cost)",
                body_style,
            ))

    def _generate_text_fallback(self, portfolios: list[PortfolioReportData]) -> bytes:
        """Plain text fallback when reportlab is not installed."""
        lines: list[str] = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  {self._config.title} (PDF Fallback)")
        lines.append(f"  Generated: {self._config.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append(f"  Author: {self._config.author}")
        lines.append(f"{'=' * 60}")
        lines.append("")

        if not portfolios:
            lines.append("No portfolio data available.")
            return "\n".join(lines).encode("utf-8")

        # Key highlights
        lines.append("KEY HIGHLIGHTS")
        lines.append("-" * 40)
        best_return = max(portfolios, key=lambda p: p.total_return_pct)
        best_sharpe = max(portfolios, key=lambda p: p.sharpe_ratio)
        lines.append(
            f"  Best Return: {best_return.model}/{best_return.architecture} "
            f"({best_return.total_return_pct:+.2f}%)"
        )
        lines.append(
            f"  Best Sharpe: {best_sharpe.model}/{best_sharpe.architecture} "
            f"({best_sharpe.sharpe_ratio:.4f})"
        )
        lines.append("")

        # Summary table
        lines.append("PERFORMANCE SUMMARY")
        lines.append("-" * 40)
        header = (f"{'Model':<12} {'Arch':<8} {'Market':<8} {'Ret%':>7} "
                  f"{'Sharpe':>8} {'DD%':>7} {'WR%':>6} {'Trades':>6}")
        lines.append(header)
        lines.append("-" * len(header))
        for p in portfolios:
            lines.append(
                f"{p.model:<12} {p.architecture:<8} {p.market:<8} "
                f"{p.total_return_pct:>7.2f} {p.sharpe_ratio:>8.4f} "
                f"{p.max_drawdown * 100:>7.2f} {p.win_rate * 100:>6.1f} "
                f"{p.total_trades:>6}"
            )
        lines.append("")

        # Cost analysis
        total_cost = sum(p.total_api_cost for p in portfolios)
        lines.append(f"Total API Cost: ${total_cost:.4f}")

        return "\n".join(lines).encode("utf-8")
