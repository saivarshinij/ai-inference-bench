"""Unit tests for run comparison and delta metrics."""

from ai_inference_bench.core.config import BenchmarkConfig
from ai_inference_bench.core.metrics import compute_metrics
from ai_inference_bench.core.result import BenchmarkResult
from ai_inference_bench.core.system_info import collect_system_info
from ai_inference_bench.reporting.comparator import compare_runs


def test_compare_runs_metrics():
    cfg = BenchmarkConfig(model_name="tiny_transformer", batch_size=4)
    sys_info = collect_system_info()

    # Baseline: 10 ms mean latency
    base_metrics = compute_metrics(
        latencies_ms=[10.0] * 20,
        batch_size=4,
        warm_up_iters=5,
        measured_iters=20,
        peak_ram_mb=100.0,
        delta_ram_mb=5.0,
    )
    base = BenchmarkResult(
        run_id="run_baseline",
        config=cfg,
        system_info=sys_info,
        metrics=base_metrics,
        raw_latencies_ms=[10.0] * 20,
    )

    # Optimized: 5 ms mean latency (2x speedup, 50% reduction)
    opt_cfg = BenchmarkConfig(model_name="tiny_transformer", batch_size=4, backend="torchscript")
    opt_metrics = compute_metrics(
        latencies_ms=[5.0] * 20,
        batch_size=4,
        warm_up_iters=5,
        measured_iters=20,
        peak_ram_mb=90.0,
        delta_ram_mb=2.0,
    )
    opt = BenchmarkResult(
        run_id="run_opt",
        config=opt_cfg,
        system_info=sys_info,
        metrics=opt_metrics,
        raw_latencies_ms=[5.0] * 20,
    )

    comp = compare_runs(base, opt)
    assert comp.comparable is True
    assert comp.speedup_ratio == 2.0
    assert comp.latency_reduction_pct == 50.0
    assert comp.throughput_gain_pct == 100.0


def test_incomparable_different_models():
    sys_info = collect_system_info()
    m = compute_metrics([10.0] * 5, 1, 0, 5, 50.0, 0.0)

    r1 = BenchmarkResult(
        run_id="r1",
        config=BenchmarkConfig(model_name="mobilenet_v3_small", batch_size=1),
        system_info=sys_info,
        metrics=m,
        raw_latencies_ms=[10.0] * 5,
    )
    r2 = BenchmarkResult(
        run_id="r2",
        config=BenchmarkConfig(model_name="resnet18", batch_size=1),
        system_info=sys_info,
        metrics=m,
        raw_latencies_ms=[10.0] * 5,
    )

    comp = compare_runs(r1, r2)
    assert comp.comparable is False
    assert "Models differ" in comp.incomparable_reason
