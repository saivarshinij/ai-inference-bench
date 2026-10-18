"""End-to-End CPU Smoke Benchmark Test."""

from pathlib import Path
from ai_inference_bench.core.config import BenchmarkConfig
from ai_inference_bench.engine import BenchmarkRunner
from ai_inference_bench.reporting.html_report import generate_html_report
from ai_inference_bench.reporting.markdown_report import generate_markdown_report


def test_cpu_smoke_benchmark_mlp(temp_results_dir: Path):
    config = BenchmarkConfig(
        model_name="mlp_classifier",
        batch_size=4,
        warm_up_iterations=2,
        iterations=10,
        backend="pytorch_eager",
        precision="fp32",
        device="cpu",
    )

    runner = BenchmarkRunner(config, results_dir=temp_results_dir)
    result = runner.run()

    # Verify result object properties
    assert result.run_id.startswith("run_")
    assert result.metrics.sample_size == 10
    lat = result.metrics.latency
    assert lat.mean_ms > 0
    assert lat.median_p50_ms > 0
    assert lat.min_ms <= lat.median_p50_ms <= lat.max_ms
    assert result.metrics.throughput.samples_per_second > 0

    # Verify JSON persistence
    json_file = temp_results_dir / f"{result.run_id}.json"
    assert json_file.exists()
    assert json_file.stat().st_size > 0

    # Verify CSV persistence
    csv_file = temp_results_dir / f"{result.run_id}.csv"
    assert csv_file.exists()
    with open(csv_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 11  # 1 header + 10 iterations

    # Verify Markdown & HTML report generation
    md_file = temp_results_dir / f"{result.run_id}.md"
    html_file = temp_results_dir / f"{result.run_id}.html"
    generate_markdown_report(result, output_path=md_file)
    generate_html_report(result, output_path=html_file)

    assert md_file.exists() and md_file.stat().st_size > 0
    assert html_file.exists() and html_file.stat().st_size > 0


def test_cpu_smoke_benchmark_quantized(temp_results_dir: Path):
    config = BenchmarkConfig(
        model_name="mlp_classifier",
        batch_size=2,
        warm_up_iterations=2,
        iterations=5,
        backend="torch_quantized_int8",
        precision="int8",
        device="cpu",
    )

    runner = BenchmarkRunner(config, results_dir=temp_results_dir)
    result = runner.run()

    assert result.metrics.sample_size == 5
    assert result.metrics.latency.mean_ms > 0
