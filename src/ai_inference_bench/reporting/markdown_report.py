"""Markdown Benchmark Report Generator."""

from pathlib import Path
from typing import Optional
from tabulate import tabulate

from ai_inference_bench.core.result import BenchmarkResult
from ai_inference_bench.reporting.comparator import ComparisonMetrics


def generate_markdown_report(
    result: BenchmarkResult,
    comparison: Optional[ComparisonMetrics] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Formats benchmark results and optional comparative analysis into GitHub Markdown."""
    cfg = result.config
    sys_info = result.system_info
    metrics = result.metrics
    lat = metrics.latency

    lines = [
        f"# Inference Benchmark Report: `{cfg.model_name}`",
        "",
        f"**Run ID:** `{result.run_id}`  ",
        f"**Timestamp (UTC):** `{result.timestamp}`  ",
        f"**Backend:** `{cfg.backend}` | **Precision:** `{cfg.precision}` | **Device:** `{cfg.device}`  ",
        f"**Batch Size:** `{cfg.batch_size}` | **Warm-up Iterations:** `{cfg.warm_up_iterations}` | **Measured Iterations:** `{cfg.iterations}`  ",
        "",
        "## Hardware & Environment",
        "",
        f"- **CPU Model:** {sys_info.cpu_model}",
        f"- **Cores:** {sys_info.physical_cores} Physical / {sys_info.logical_cores} Logical",
        f"- **System RAM:** {sys_info.total_ram_gb} GB ({sys_info.available_ram_gb} GB available)",
        f"- **OS:** {sys_info.os_name} {sys_info.os_release} ({sys_info.os_architecture})",
        f"- **Runtimes:** Python `{sys_info.python_version}` | PyTorch `{sys_info.pytorch_version}` | ONNX Runtime `{sys_info.onnxruntime_version or 'N/A'}`",
        f"- **Accelerator Detected:** `{sys_info.accelerator_type.upper()}` (CUDA Available: `{sys_info.cuda_available}`)",
        "",
        "## Measured Performance Metrics",
        "",
        "### 1. Latency Percentiles (Milliseconds)",
        "",
    ]

    lat_table = [
        ["Metric", "Value (ms)"],
        ["Mean", f"{lat.mean_ms:.3f}"],
        ["Median (p50)", f"{lat.median_p50_ms:.3f}"],
        ["p90", f"{lat.p90_ms:.3f}"],
        ["p95", f"{lat.p95_ms:.3f}"],
        ["p99", f"{lat.p99_ms:.3f}"],
        ["Min", f"{lat.min_ms:.3f}"],
        ["Max", f"{lat.max_ms:.3f}"],
        ["Std Deviation", f"{lat.std_dev_ms:.3f}"],
    ]
    lines.append(tabulate(lat_table, headers="firstrow", tablefmt="github"))
    lines.append("")

    if metrics.notes:
        lines.append(f"> [!NOTE]\n> {metrics.notes}\n")

    lines.extend([
        "### 2. Throughput & Memory Consumption",
        "",
        f"- **Throughput (Samples / sec):** **`{metrics.throughput.samples_per_second:.2f}`** items/sec",
        f"- **Throughput (Queries / sec):** `{metrics.throughput.queries_per_second:.2f}` batches/sec",
        f"- **Peak Process RAM:** `{metrics.memory.peak_ram_mb:.2f}` MB",
        f"- **Net RAM Delta:** `{metrics.memory.delta_ram_mb:.2f}` MB",
    ])

    if metrics.memory.peak_vram_mb is not None:
        lines.append(f"- **Peak GPU VRAM:** `{metrics.memory.peak_vram_mb:.2f}` MB")

    if comparison:
        lines.extend([
            "",
            "## Baseline Comparison & Optimization Gains",
            "",
            f"Comparing target **`{comparison.target_backend}`** against baseline **`{comparison.baseline_backend}`**:",
            "",
        ])

        if not comparison.comparable:
            lines.append(f"> [!WARNING]\n> Baseline comparison invalid: {comparison.incomparable_reason}\n")
        else:
            comp_table = [
                ["Metric", "Baseline", "Optimized", "Delta / Gain"],
                [
                    "Mean Latency",
                    f"{comparison.baseline_latency_mean_ms:.3f} ms",
                    f"{comparison.target_latency_mean_ms:.3f} ms",
                    f"-{comparison.latency_reduction_pct:.2f}% (Speedup: {comparison.speedup_ratio}x)",
                ],
                [
                    "Throughput",
                    f"{comparison.baseline_throughput_fps:.2f} samples/s",
                    f"{comparison.target_throughput_fps:.2f} samples/s",
                    f"+{comparison.throughput_gain_pct:.2f}%",
                ],
                [
                    "Peak RAM",
                    f"{comparison.baseline_ram_mb:.2f} MB",
                    f"{comparison.target_ram_mb:.2f} MB",
                    f"{comparison.target_ram_mb - comparison.baseline_ram_mb:+.2f} MB",
                ],
            ]
            lines.append(tabulate(comp_table, headers="firstrow", tablefmt="github"))
            lines.append("")

    content = "\n".join(lines)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(content)

    return content
