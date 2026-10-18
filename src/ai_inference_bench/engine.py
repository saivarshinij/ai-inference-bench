"""Core Benchmarking Engine with High-Precision Timing and Memory Tracking."""

import gc
import time
import tracemalloc
import uuid
from pathlib import Path
from typing import Optional
import psutil
import torch

from ai_inference_bench.backends import get_backend
from ai_inference_bench.core.config import BenchmarkConfig
from ai_inference_bench.core.metrics import compute_metrics
from ai_inference_bench.core.result import BenchmarkResult
from ai_inference_bench.core.system_info import collect_system_info
from ai_inference_bench.models.registry import get_model_spec


class BenchmarkRunner:
    """Executes high-precision, reproducible inference benchmarks across models and backends."""

    def __init__(self, config: BenchmarkConfig, results_dir: Optional[Path] = None):
        self.config = config
        self.results_dir = Path(results_dir or "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> BenchmarkResult:
        """Executes full warm-up, measured inference loop, memory profiling, and persistence."""
        model_spec = get_model_spec(self.config.model_name)
        model = model_spec.create_model()

        device_str = self.config.device if self.config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        sample_input = model_spec.generate_input(batch_size=self.config.batch_size, device=device_str)

        # Initialize backend
        backend = get_backend(self.config)
        prepared_model = backend.prepare(model, sample_input)

        # 1. Warm-up Phase (Excludes compilation/lazy initialization from measured stats)
        for _ in range(self.config.warm_up_iterations):
            _ = backend.run_inference(sample_input)
        backend.synchronize()

        # 2. Memory Baseline
        process = psutil.Process()
        start_ram_mb = process.memory_info().rss / (1024 * 1024)
        tracemalloc.start()

        if torch.cuda.is_available() and device_str == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # 3. Measured Iterations
        latencies_ms = []
        gc.disable()  # Prevent non-deterministic GC pauses from skewing p99 percentiles

        try:
            for _ in range(self.config.iterations):
                t_start = time.perf_counter_ns()
                _ = backend.run_inference(sample_input)
                backend.synchronize()
                t_end = time.perf_counter_ns()

                lat_ms = (t_end - t_start) / 1_000_000.0
                latencies_ms.append(lat_ms)
        finally:
            gc.enable()

        # 4. Post-run Memory Profiling
        _, peak_traced_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_ram_mb = process.memory_info().rss / (1024 * 1024)
        peak_ram_mb = max(end_ram_mb, start_ram_mb + (peak_traced_bytes / (1024 * 1024)))
        delta_ram_mb = max(0.0, end_ram_mb - start_ram_mb)

        peak_vram_mb = None
        if torch.cuda.is_available() and device_str == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # 5. Compute Statistical Metrics
        metrics = compute_metrics(
            latencies_ms=latencies_ms,
            batch_size=self.config.batch_size,
            warm_up_iters=self.config.warm_up_iterations,
            measured_iters=self.config.iterations,
            peak_ram_mb=peak_ram_mb,
            delta_ram_mb=delta_ram_mb,
            peak_vram_mb=peak_vram_mb,
        )

        system_info = collect_system_info()
        run_id = f"run_{uuid.uuid4().hex[:12]}"

        result = BenchmarkResult(
            run_id=run_id,
            config=self.config,
            system_info=system_info,
            metrics=metrics,
            raw_latencies_ms=latencies_ms,
        )

        # 6. Save persistent JSON and CSV outputs
        json_path = self.results_dir / f"{run_id}.json"
        csv_path = self.results_dir / f"{run_id}.csv"
        result.save_json(json_path)
        result.save_csv(csv_path)

        backend.cleanup()
        return result
