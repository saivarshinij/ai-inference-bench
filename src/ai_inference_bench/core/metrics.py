"""Metrics Computation: Latency Percentiles, Throughput, and Memory Tracking."""

import math
from typing import List, Optional
import numpy as np
from pydantic import BaseModel, Field


class LatencyMetrics(BaseModel):
    mean_ms: float = Field(..., description="Mean latency per inference call in milliseconds")
    median_p50_ms: float = Field(..., description="50th percentile (median) latency in ms")
    p90_ms: float = Field(..., description="90th percentile latency in ms")
    p95_ms: float = Field(..., description="95th percentile latency in ms")
    p99_ms: float = Field(..., description="99th percentile latency in ms")
    min_ms: float = Field(..., description="Minimum recorded latency in ms")
    max_ms: float = Field(..., description="Maximum recorded latency in ms")
    std_dev_ms: float = Field(..., description="Standard deviation of latency in ms")


class ThroughputMetrics(BaseModel):
    samples_per_second: float = Field(..., description="Inference throughput in samples (items) per second")
    queries_per_second: float = Field(..., description="Batches / queries processed per second")


class MemoryMetrics(BaseModel):
    peak_ram_mb: float = Field(..., description="Peak process resident RAM allocated during run (MB)")
    delta_ram_mb: float = Field(..., description="Net RAM increase during benchmark run (MB)")
    peak_vram_mb: Optional[float] = Field(None, description="Peak GPU VRAM allocated (MB) if accelerator present")


class BenchmarkMetrics(BaseModel):
    latency: LatencyMetrics
    throughput: ThroughputMetrics
    memory: MemoryMetrics
    sample_size: int
    warm_up_iterations: int
    measured_iterations: int
    notes: Optional[str] = None


def compute_metrics(
    latencies_ms: List[float],
    batch_size: int,
    warm_up_iters: int,
    measured_iters: int,
    peak_ram_mb: float,
    delta_ram_mb: float,
    peak_vram_mb: Optional[float] = None,
) -> BenchmarkMetrics:
    """Computes genuine statistical percentiles, throughput, and memory measurements."""
    if not latencies_ms:
        raise ValueError("Cannot compute metrics on empty latency measurements.")

    arr = np.array(latencies_ms, dtype=np.float64)
    n = len(arr)

    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    p90_val = float(np.percentile(arr, 90))
    p95_val = float(np.percentile(arr, 95))
    p99_val = float(np.percentile(arr, 99))

    # Throughput calculation: total samples processed / total measured time (sec)
    total_time_sec = float(np.sum(arr)) / 1000.0
    total_samples = n * batch_size

    if total_time_sec > 0:
        samples_per_sec = total_samples / total_time_sec
        queries_per_sec = n / total_time_sec
    else:
        samples_per_sec = 0.0
        queries_per_sec = 0.0

    notes = None
    if n < 50:
        notes = "Sample size < 50: p95 and p99 percentiles may exhibit high variance."

    return BenchmarkMetrics(
        latency=LatencyMetrics(
            mean_ms=round(mean_val, 3),
            median_p50_ms=round(median_val, 3),
            p90_ms=round(p90_val, 3),
            p95_ms=round(p95_val, 3),
            p99_ms=round(p99_val, 3),
            min_ms=round(min_val, 3),
            max_ms=round(max_val, 3),
            std_dev_ms=round(std_val, 3),
        ),
        throughput=ThroughputMetrics(
            samples_per_second=round(samples_per_sec, 2),
            queries_per_second=round(queries_per_sec, 2),
        ),
        memory=MemoryMetrics(
            peak_ram_mb=round(peak_ram_mb, 2),
            delta_ram_mb=round(delta_ram_mb, 2),
            peak_vram_mb=round(peak_vram_mb, 2) if peak_vram_mb is not None else None,
        ),
        sample_size=n,
        warm_up_iterations=warm_up_iters,
        measured_iterations=measured_iters,
        notes=notes,
    )
