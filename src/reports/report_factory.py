"""Report factory — unified interface for generating reports in any format.

Supports Excel (.xlsx), Word (.docx), and PDF (.pdf) output formats.
Each format gracefully degrades if the required library is not installed.
"""

from __future__ import annotations

from enum import Enum

from src.core.logging import get_logger
from src.reports.excel_report import ExcelReportGenerator, PortfolioReportData, ReportConfig
from src.reports.pdf_report import PDFReportGenerator
from src.reports.word_report import WordReportGenerator

log = get_logger(__name__)


class ReportFormat(str, Enum):
    """Supported report output formats."""
    EXCEL = "excel"
    WORD = "word"
    PDF = "pdf"


class ReportFactory:
    """Factory for generating reports in various formats."""

    @staticmethod
    def generate(
        format: ReportFormat,
        portfolios: list[PortfolioReportData],
        config: ReportConfig | None = None,
    ) -> bytes:
        """Generate a report in the specified format.

        Args:
            format: The desired output format (excel, word, pdf).
            portfolios: List of portfolio data to include in the report.
            config: Optional report configuration (title, author, etc.).

        Returns:
            Raw bytes of the generated report file.

        Raises:
            ValueError: If the format is not recognized.
        """
        log.info("report_generation_started", format=format.value,
                 portfolios=len(portfolios))

        if format == ReportFormat.EXCEL:
            generator = ExcelReportGenerator(config)
        elif format == ReportFormat.WORD:
            generator = WordReportGenerator(config)
        elif format == ReportFormat.PDF:
            generator = PDFReportGenerator(config)
        else:
            msg = f"Unsupported report format: {format}"
            raise ValueError(msg)

        result = generator.generate(portfolios)
        log.info("report_generation_complete", format=format.value,
                 size_bytes=len(result))
        return result

    @staticmethod
    def get_content_type(format: ReportFormat) -> str:
        """Return the MIME content type for the given format.

        Args:
            format: The report format.

        Returns:
            MIME content type string.
        """
        content_types: dict[ReportFormat, str] = {
            ReportFormat.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ReportFormat.WORD: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ReportFormat.PDF: "application/pdf",
        }
        return content_types[format]

    @staticmethod
    def get_extension(format: ReportFormat) -> str:
        """Return the file extension for the given format.

        Args:
            format: The report format.

        Returns:
            File extension string including the dot (e.g. '.xlsx').
        """
        extensions: dict[ReportFormat, str] = {
            ReportFormat.EXCEL: ".xlsx",
            ReportFormat.WORD: ".docx",
            ReportFormat.PDF: ".pdf",
        }
        return extensions[format]
