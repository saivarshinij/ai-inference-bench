"""FastAPI Web Dashboard for Visualizing Benchmark Results."""

import json
from pathlib import Path
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_inference_bench.core.result import BenchmarkResult
from ai_inference_bench.reporting.comparator import compare_runs

app = FastAPI(title="AI Inference Benchmark Lab Explorer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_DIR = Path("results")
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    """Scans and lists all saved benchmark runs in the results directory."""
    if not RESULTS_DIR.exists():
        return []

    runs = []
    for file in sorted(RESULTS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                runs.append({
                    "run_id": data["run_id"],
                    "timestamp": data.get("timestamp", ""),
                    "model_name": data["config"]["model_name"],
                    "backend": data["config"]["backend"],
                    "precision": data["config"]["precision"],
                    "batch_size": data["config"]["batch_size"],
                    "mean_latency_ms": data["metrics"]["latency"]["mean_ms"],
                    "p95_latency_ms": data["metrics"]["latency"]["p95_ms"],
                    "throughput_fps": data["metrics"]["throughput"]["samples_per_second"],
                    "peak_ram_mb": data["metrics"]["memory"]["peak_ram_mb"],
                    "device": data["config"]["device"],
                })
        except Exception:
            continue
    return runs


@app.get("/api/runs/{run_id}")
def get_run_detail(run_id: str):
    """Retrieves full details and raw latencies for a single benchmark run."""
    path = RESULTS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return BenchmarkResult.load_json(path)


@app.get("/api/compare")
def compare_runs_endpoint(base_id: str, target_id: str):
    """Compares two benchmark runs."""
    base_path = RESULTS_DIR / f"{base_id}.json"
    target_path = RESULTS_DIR / f"{target_id}.json"
    if not base_path.exists() or not target_path.exists():
        raise HTTPException(status_code=404, detail="One or both run IDs not found.")

    b = BenchmarkResult.load_json(base_path)
    t = BenchmarkResult.load_json(target_path)
    return compare_runs(b, t)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def serve_dashboard():
        index = STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"message": "Benchmark API is running. Results at /api/runs"}
