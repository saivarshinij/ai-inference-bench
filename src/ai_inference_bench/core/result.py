"""Benchmark Result Serialization and Persistence (JSON & CSV)."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field

from ai_inference_bench.core.config import BenchmarkConfig
from ai_inference_bench.core.metrics import BenchmarkMetrics
from ai_inference_bench.core.system_info import SystemInfo


class BenchmarkResult(BaseModel):
    """Complete, self-contained record of a single benchmark run."""

    run_id: str = Field(..., description="Unique UUID identifying this benchmark execution")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config: BenchmarkConfig
    system_info: SystemInfo
    metrics: BenchmarkMetrics
    raw_latencies_ms: List[float] = Field(default_factory=list, description="Raw per-iteration latencies in ms")

    def save_json(self, file_path: Path) -> Path:
        """Saves full structured JSON record."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return path

    def save_csv(self, file_path: Path) -> Path:
        """Saves raw latency iterations to CSV for external statistical tools."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "latency_ms"])
            for idx, lat in enumerate(self.raw_latencies_ms):
                writer.writerow([idx + 1, lat])
        return path

    @classmethod
    def load_json(cls, file_path: Path) -> "BenchmarkResult":
        """Loads benchmark result from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
