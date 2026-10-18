"""Reporting package."""

from ai_inference_bench.reporting.comparator import ComparisonMetrics, compare_runs
from ai_inference_bench.reporting.html_report import generate_html_report
from ai_inference_bench.reporting.markdown_report import generate_markdown_report

__all__ = [
    "ComparisonMetrics",
    "compare_runs",
    "generate_markdown_report",
    "generate_html_report",
]
