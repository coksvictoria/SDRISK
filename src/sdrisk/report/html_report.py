"""Disclosure-safe HTML report rendering."""

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from sdrisk.report.json_report import build_report_payload
from sdrisk.results import RunResult


def render_html_report(result: RunResult) -> str:
    """Render the same output-filtered aggregate payload used by JSON."""
    environment = Environment(
        loader=PackageLoader("sdrisk.report", "templates"),
        autoescape=select_autoescape(("html", "xml")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = environment.get_template("report.html")
    return template.render(report=build_report_payload(result))


def write_html_report(result: RunResult, path: str | Path) -> Path:
    """Write a disclosure-safe HTML report to `path`."""
    output_path = Path(path)
    output_path.write_text(render_html_report(result), encoding="utf-8")
    return output_path
