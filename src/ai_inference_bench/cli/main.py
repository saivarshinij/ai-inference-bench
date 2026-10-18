"""AI Inference Optimization & Benchmarking CLI Tool."""

import json
import sys
from pathlib import Path

# Enforce UTF-8 on Windows terminal streams
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import click
from tabulate import tabulate

from ai_inference_bench.core.config import BenchmarkConfig
from ai_inference_bench.core.result import BenchmarkResult
from ai_inference_bench.core.system_info import collect_system_info
from ai_inference_bench.engine import BenchmarkRunner
from ai_inference_bench.models.registry import list_registered_models
from ai_inference_bench.reporting.comparator import compare_runs
from ai_inference_bench.reporting.html_report import generate_html_report
from ai_inference_bench.reporting.markdown_report import generate_markdown_report


@click.group()
def cli():
    """AI Inference Optimization & Benchmarking Lab CLI."""
    pass


@cli.command("run")
@click.option("--model", "-m", default="mobilenet_v3_small", help="Model name from registry")
@click.option("--batch-size", "-b", default=1, type=int, help="Inference batch size")
@click.option("--warm-up", default=10, type=int, help="Warm-up iterations")
@click.option("--iterations", "-n", default=50, type=int, help="Measured iterations")
@click.option(
    "--backend",
    default="pytorch_eager",
    type=click.Choice(["pytorch_eager", "torchscript", "torch_quantized_int8", "onnxruntime"]),
    help="Inference runtime backend",
)
@click.option(
    "--precision",
    default="fp32",
    type=click.Choice(["fp32", "fp16", "int8"]),
    help="Numerical precision",
)
@click.option(
    "--device",
    default="cpu",
    type=click.Choice(["cpu", "cuda", "mps", "rocm", "auto"]),
    help="Target compute device",
)
@click.option("--out-dir", default="results", help="Directory for benchmark results")
def run_benchmark(model, batch_size, warm_up, iterations, backend, precision, device, out_dir):
    """Executes an inference benchmark run and saves reproducible results."""
    click.echo(click.style(f"\n[*] Initializing Benchmark for '{model}'...", fg="cyan", bold=True))
    click.echo(f"     Backend: {backend} | Precision: {precision} | Batch: {batch_size} | Device: {device}")

    config = BenchmarkConfig(
        model_name=model,
        batch_size=batch_size,
        warm_up_iterations=warm_up,
        iterations=iterations,
        backend=backend,
        precision=precision,
        device=device,
    )

    runner = BenchmarkRunner(config, results_dir=Path(out_dir))
    result = runner.run()

    # Generate Markdown and HTML reports
    md_path = Path(out_dir) / f"{result.run_id}.md"
    html_path = Path(out_dir) / f"{result.run_id}.html"
    generate_markdown_report(result, output_path=md_path)
    generate_html_report(result, output_path=html_path)

    lat = result.metrics.latency
    click.echo(click.style("\n[OK] Benchmark Complete!", fg="green", bold=True))
    click.echo(f"    Run ID: {result.run_id}")
    click.echo(f"    Mean Latency:    {lat.mean_ms:.2f} ms")
    click.echo(f"    Median (p50):    {lat.median_p50_ms:.2f} ms")
    click.echo(f"    p95 Latency:     {lat.p95_ms:.2f} ms")
    click.echo(f"    p99 Latency:     {lat.p99_ms:.2f} ms")
    click.echo(f"    Throughput:      {result.metrics.throughput.samples_per_second:.1f} samples/sec")
    click.echo(f"    Peak RAM:        {result.metrics.memory.peak_ram_mb:.1f} MB")
    click.echo(f"\n    Saved JSON:   {out_dir}/{result.run_id}.json")
    click.echo(f"    Saved CSV:    {out_dir}/{result.run_id}.csv")
    click.echo(f"    Saved Report: {md_path}")
    click.echo(f"    HTML Visual:  {html_path}\n")


@cli.command("rerun")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--out-dir", default="results", help="Directory for benchmark results")
def rerun_benchmark(config_file, out_dir):
    """Reproduces a benchmark using a saved configuration or result JSON."""
    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Allow passing either raw config JSON or full result JSON
    if "config" in data:
        cfg_dict = data["config"]
    else:
        cfg_dict = data

    config = BenchmarkConfig(**cfg_dict)
    click.echo(click.style(f"[*] Reproducing benchmark from '{config_file}'...", fg="yellow"))
    runner = BenchmarkRunner(config, results_dir=Path(out_dir))
    res = runner.run()
    click.echo(click.style(f"[OK] Reproduced as Run ID: {res.run_id}", fg="green"))


@cli.command("compare")
@click.argument("baseline_json", type=click.Path(exists=True))
@click.argument("target_json", type=click.Path(exists=True))
def compare_benchmarks(baseline_json, target_json):
    """Compares two benchmark runs side-by-side (Speedup, Latency delta, Memory)."""
    base = BenchmarkResult.load_json(Path(baseline_json))
    target = BenchmarkResult.load_json(Path(target_json))

    comp = compare_runs(base, target)

    click.echo(click.style("\n=== Benchmark Optimization Comparison ===", fg="cyan", bold=True))
    if not comp.comparable:
        click.echo(click.style(f"Warning: {comp.incomparable_reason}", fg="yellow"))

    table = [
        ["Metric", "Baseline", "Target", "Delta"],
        ["Run ID", base.run_id, target.run_id, ""],
        ["Backend", comp.baseline_backend, comp.target_backend, ""],
        ["Mean Latency", f"{comp.baseline_latency_mean_ms:.2f} ms", f"{comp.target_latency_mean_ms:.2f} ms", f"-{comp.latency_reduction_pct:.1f}% (Speedup: {comp.speedup_ratio}x)"],
        ["Throughput", f"{comp.baseline_throughput_fps:.1f} fps", f"{comp.target_throughput_fps:.1f} fps", f"+{comp.throughput_gain_pct:.1f}%"],
        ["Peak RAM", f"{comp.baseline_ram_mb:.1f} MB", f"{comp.target_ram_mb:.1f} MB", f"{comp.target_ram_mb - comp.baseline_ram_mb:+.1f} MB"],
    ]
    click.echo(tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
    click.echo("")


@cli.command("list-models")
def list_models():
    """Lists registered benchmark models."""
    models = list_registered_models()
    table = [[m["name"], m["domain"], str(m["default_shape"]), m["description"]] for m in models]
    click.echo(tabulate(table, headers=["Model", "Domain", "Input Shape", "Description"], tablefmt="github"))


@cli.command("system-info")
def system_info_cmd():
    """Displays hardware, CPU, RAM, accelerator, and runtime details."""
    info = collect_system_info()
    table = [
        ["CPU Model", info.cpu_model],
        ["Cores", f"{info.physical_cores} Physical / {info.logical_cores} Logical"],
        ["Total RAM", f"{info.total_ram_gb} GB ({info.available_ram_gb} GB free)"],
        ["OS", f"{info.os_name} {info.os_release} ({info.os_architecture})"],
        ["Python", info.python_version],
        ["PyTorch", info.pytorch_version],
        ["ONNX Runtime", info.onnxruntime_version or "Not installed"],
        ["CUDA Available", f"{info.cuda_available} ({info.cuda_device_name or 'None'})"],
        ["ROCm Available", info.rocm_available],
        ["Active Accelerator", info.accelerator_type.upper()],
    ]
    click.echo(tabulate(table, headers=["System Property", "Value"], tablefmt="github"))


@cli.command("serve-dashboard")
@click.option("--port", default=8080, help="Port to host interactive dashboard")
def serve_dashboard(port):
    """Launches the interactive benchmark results dashboard."""
    import uvicorn
    from ai_inference_bench.dashboard.app import app
    click.echo(f"Starting Benchmark Explorer at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    cli()
