from __future__ import annotations

from pathlib import Path


def write_html_report(summary: dict, path: str | Path) -> None:
    path = Path(path)
    path.write_text(
        "<html><body><h1>ASFuzz Report</h1><pre>"
        + __import__("json").dumps(summary, indent=2, sort_keys=True)
        + "</pre></body></html>"
    )

