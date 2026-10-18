"""Unit tests for metric calculations, percentiles, and throughput."""

import pytest
from ai_inference_bench.core.metrics import compute_metrics


def test_compute_metrics_known_distribution():
    # 100 values from 1.0 to 100.0 ms
    latencies = [float(i) for i in range(1, 101)]

    metrics = compute_metrics(
        latencies_ms=latencies,
        batch_size=2,
        warm_up_iters=5,
        measured_iters=100,
        peak_ram_mb=128.5,
        delta_ram_mb=12.0,
    )

    # Median of 1..100 is 50.5
    assert metrics.latency.median_p50_ms == 50.5
    # Mean of 1..100 is 50.5
    assert metrics.latency.mean_ms == 50.5
    # Min is 1.0, Max is 100.0
    assert metrics.latency.min_ms == 1.0
    assert metrics.latency.max_ms == 100.0
    # p90 is 90.1, p95 is 95.05, p99 is 99.01
    assert pytest.approx(metrics.latency.p90_ms, abs=0.2) == 90.1
    assert pytest.approx(metrics.latency.p95_ms, abs=0.2) == 95.05
    assert pytest.approx(metrics.latency.p99_ms, abs=0.2) == 99.01

    # Throughput: Total samples = 100 * 2 = 200. Total time = 5050 ms = 5.05 s
    # Samples / sec = 200 / 5.05 ~= 39.60
    assert pytest.approx(metrics.throughput.samples_per_second, abs=0.1) == 39.60
    assert metrics.sample_size == 100
    assert metrics.memory.peak_ram_mb == 128.5


def test_empty_metrics_raises_value_error():
    with pytest.raises(ValueError):
        compute_metrics(
            latencies_ms=[],
            batch_size=1,
            warm_up_iters=0,
            measured_iters=0,
            peak_ram_mb=0.0,
            delta_ram_mb=0.0,
        )
