# Inference Benchmark Report: `mlp_classifier`

**Run ID:** `run_55cbb73f8f44`  
**Timestamp (UTC):** `2026-10-18T17:09:17.957242+00:00`  
**Backend:** `onnxruntime` | **Precision:** `fp32` | **Device:** `cpu`  
**Batch Size:** `1` | **Warm-up Iterations:** `10` | **Measured Iterations:** `50`  

## Hardware & Environment

- **CPU Model:** AMD Ryzen 3 7320U with Radeon Graphics
- **Cores:** 4 Physical / 8 Logical
- **System RAM:** 7.21 GB (0.79 GB available)
- **OS:** Windows 10 (AMD64)
- **Runtimes:** Python `3.10.10` | PyTorch `2.14.1+cpu` | ONNX Runtime `1.23.2`
- **Accelerator Detected:** `CPU` (CUDA Available: `False`)

## Measured Performance Metrics

### 1. Latency Percentiles (Milliseconds)

| Metric        |   Value (ms) |
|---------------|--------------|
| Mean          |        0.066 |
| Median (p50)  |        0.046 |
| p90           |        0.084 |
| p95           |        0.099 |
| p99           |        0.353 |
| Min           |        0.04  |
| Max           |        0.527 |
| Std Deviation |        0.07  |

### 2. Throughput & Memory Consumption

- **Throughput (Samples / sec):** **`15212.36`** items/sec
- **Throughput (Queries / sec):** `15212.36` batches/sec
- **Peak Process RAM:** `370.51` MB
- **Net RAM Delta:** `0.00` MB