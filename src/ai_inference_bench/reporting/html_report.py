"""Interactive HTML Benchmark Report Generator with Chart.js Visualizations."""

import json
from pathlib import Path
from typing import Optional
from ai_inference_bench.core.result import BenchmarkResult
from ai_inference_bench.reporting.comparator import ComparisonMetrics


def generate_html_report(
    result: BenchmarkResult,
    comparison: Optional[ComparisonMetrics] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Generates a standalone, beautiful HTML report with interactive latency charts."""
    cfg = result.config
    sys_info = result.system_info
    metrics = result.metrics
    lat = metrics.latency

    # Percentiles data for chart
    percentiles_labels = ["Min", "p50 (Median)", "Mean", "p90", "p95", "p99", "Max"]
    percentiles_values = [
        lat.min_ms,
        lat.median_p50_ms,
        lat.mean_ms,
        lat.p90_ms,
        lat.p95_ms,
        lat.p99_ms,
        lat.max_ms,
    ]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Benchmark Report - {cfg.model_name} ({cfg.backend})</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #0a0d14;
      --card-bg: #111726;
      --border: rgba(255, 255, 255, 0.08);
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #6366f1;
      --success: #10b981;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 30px;
      line-height: 1.5;
    }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    .header {{ margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }}
    .badge {{ background: rgba(99, 102, 241, 0.2); color: #818cf8; padding: 4px 10px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 24px; }}
    .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 18px; }}
    .card-title {{ font-size: 0.75rem; text-transform: uppercase; color: var(--muted); }}
    .card-val {{ font-size: 1.5rem; font-weight: 700; margin-top: 4px; }}
    .chart-container {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.88rem; }}
    th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <span class="badge">Inference Benchmark</span>
      <h1 style="margin-top: 8px;">Model: {cfg.model_name}</h1>
      <p style="color: var(--muted); font-size: 0.9rem;">
        Run ID: <code>{result.run_id}</code> | Backend: <strong>{cfg.backend} ({cfg.precision})</strong> | Device: <strong>{cfg.device}</strong>
      </p>
    </div>

    <div class="stats-grid">
      <div class="card">
        <div class="card-title">Mean Latency</div>
        <div class="card-val" style="color: var(--accent);">{lat.mean_ms:.2f} ms</div>
      </div>
      <div class="card">
        <div class="card-title">p95 Latency</div>
        <div class="card-val">{lat.p95_ms:.2f} ms</div>
      </div>
      <div class="card">
        <div class="card-title">Throughput</div>
        <div class="card-val" style="color: var(--success);">{metrics.throughput.samples_per_second:.1f} fps</div>
      </div>
      <div class="card">
        <div class="card-title">Peak RAM</div>
        <div class="card-val">{metrics.memory.peak_ram_mb:.1f} MB</div>
      </div>
    </div>

    <div class="chart-container">
      <h3>Latency Distribution & Percentiles</h3>
      <canvas id="latencyChart" height="120"></canvas>
    </div>

    <div class="card">
      <h3>System Hardware & Runtime</h3>
      <table>
        <tr><th>Property</th><th>Specification</th></tr>
        <tr><td>CPU Model</td><td>{sys_info.cpu_model}</td></tr>
        <tr><td>Cores</td><td>{sys_info.physical_cores} Physical / {sys_info.logical_cores} Logical</td></tr>
        <tr><td>System RAM</td><td>{sys_info.total_ram_gb} GB</td></tr>
        <tr><td>PyTorch Version</td><td>{sys_info.pytorch_version}</td></tr>
        <tr><td>ONNX Runtime</td><td>{sys_info.onnxruntime_version or 'N/A'}</td></tr>
        <tr><td>Hardware Accelerator</td><td>{sys_info.accelerator_type.upper()}</td></tr>
      </table>
    </div>
  </div>

  <script>
    const ctx = document.getElementById('latencyChart').getContext('2d');
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: {json.dumps(percentiles_labels)},
        datasets: [{{
          label: 'Latency (ms)',
          data: {json.dumps(percentiles_values)},
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          fill: true,
          tension: 0.3,
          pointRadius: 5,
        }}]
      }},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
          y: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
          x: {{ grid: {{ color: 'rgba(255, 255, 255, 0.05)' }}, ticks: {{ color: '#94a3b8' }} }}
        }}
      }}
    }});
  </script>
</body>
</html>
"""
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content
