"""Report exporter — CSV, JSON, and Markdown output."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

from src.core.logging import get_logger

log = get_logger(__name__)


class ReportExporter:
    """Export benchmark results in multiple formats."""

    def __init__(self, output_dir: str = "reports") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(self, data: dict, filename: str = "report.json") -> Path:
        """Export report as JSON."""
        path = self._output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info("report_exported", format="json", path=str(path))
        return path

    def export_csv(
        self,
        rows: list[dict],
        filename: str = "report.csv",
    ) -> Path:
        """Export tabular data as CSV."""
        if not rows:
            return self._output_dir / filename

        path = self._output_dir / filename
        fieldnames = list(rows[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        log.info("report_exported", format="csv", path=str(path), rows=len(rows))
        return path

    def export_markdown(
        self,
        title: str,
        sections: list[tuple[str, str]],
        filename: str = "report.md",
    ) -> Path:
        """Export report as Markdown.

        Args:
            title: Report title.
            sections: List of (heading, content) pairs.
        """
        path = self._output_dir / filename
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            f"# {title}",
            f"",
            f"*Generated: {now}*",
            f"",
        ]

        for heading, content in sections:
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(content)
            lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        log.info("report_exported", format="markdown", path=str(path))
        return path
