"""Run Comparison Engine for Measuring Optimization Gains."""

from typing import Any, Dict, Optional
from pydantic import BaseModel
from ai_inference_bench.core.result import BenchmarkResult


class ComparisonMetrics(BaseModel):
    baseline_run_id: str
    target_run_id: str
    model_name: str
    batch_size: int
    baseline_backend: str
    target_backend: str
    baseline_latency_mean_ms: float
    target_latency_mean_ms: float
    latency_reduction_pct: float
    speedup_ratio: float
    baseline_throughput_fps: float
    target_throughput_fps: float
    throughput_gain_pct: float
    baseline_ram_mb: float
    target_ram_mb: float
    comparable: bool = True
    incomparable_reason: Optional[str] = None


def compare_runs(baseline: BenchmarkResult, target: BenchmarkResult) -> ComparisonMetrics:
    """Compares baseline run with target run to evaluate optimization performance."""
    b_conf = baseline.config
    t_conf = target.config

    comparable = True
    reason = None

    if b_conf.model_name != t_conf.model_name:
        comparable = False
        reason = f"Models differ: '{b_conf.model_name}' vs '{t_conf.model_name}'."
    elif b_conf.batch_size != t_conf.batch_size:
        comparable = False
        reason = f"Batch sizes differ: {b_conf.batch_size} vs {t_conf.batch_size}."

    b_mean = baseline.metrics.latency.mean_ms
    t_mean = target.metrics.latency.mean_ms
    b_tp = baseline.metrics.throughput.samples_per_second
    t_tp = target.metrics.throughput.samples_per_second

    # Speedup ratio = Baseline time / Target time
    speedup = round(b_mean / t_mean, 2) if t_mean > 0 else 0.0
    latency_reduction = round(((b_mean - t_mean) / b_mean) * 100, 2) if b_mean > 0 else 0.0
    tp_gain = round(((t_tp - b_tp) / b_tp) * 100, 2) if b_tp > 0 else 0.0

    return ComparisonMetrics(
        baseline_run_id=baseline.run_id,
        target_run_id=target.run_id,
        model_name=b_conf.model_name,
        batch_size=b_conf.batch_size,
        baseline_backend=f"{b_conf.backend} ({b_conf.precision})",
        target_backend=f"{t_conf.backend} ({t_conf.precision})",
        baseline_latency_mean_ms=b_mean,
        target_latency_mean_ms=t_mean,
        latency_reduction_pct=latency_reduction,
        speedup_ratio=speedup,
        baseline_throughput_fps=b_tp,
        target_throughput_fps=t_tp,
        throughput_gain_pct=tp_gain,
        baseline_ram_mb=baseline.metrics.memory.peak_ram_mb,
        target_ram_mb=target.metrics.memory.peak_ram_mb,
        comparable=comparable,
        incomparable_reason=reason,
    )
