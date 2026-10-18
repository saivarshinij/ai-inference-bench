# Inference Benchmark Report: `mlp_classifier`

**Run ID:** `run_c65ac7b673e0`  
**Timestamp (UTC):** `2026-10-18T17:09:04.344892+00:00`  
**Backend:** `torch_quantized_int8` | **Precision:** `int8` | **Device:** `cpu`  
**Batch Size:** `1` | **Warm-up Iterations:** `10` | **Measured Iterations:** `50`  

## Hardware & Environment

- **CPU Model:** AMD Ryzen 3 7320U with Radeon Graphics
- **Cores:** 4 Physical / 8 Logical
- **System RAM:** 7.21 GB (0.88 GB available)
- **OS:** Windows 10 (AMD64)
- **Runtimes:** Python `3.10.10` | PyTorch `2.14.1+cpu` | ONNX Runtime `1.23.2`
- **Accelerator Detected:** `CPU` (CUDA Available: `False`)

## Measured Performance Metrics

### 1. Latency Percentiles (Milliseconds)

| Metric        |   Value (ms) |
|---------------|--------------|
| Mean          |        0.706 |
| Median (p50)  |        0.681 |
| p90           |        0.836 |
| p95           |        0.91  |
| p99           |        1.417 |
| Min           |        0.507 |
| Max           |        1.533 |
| Std Deviation |        0.181 |

### 2. Throughput & Memory Consumption

- **Throughput (Samples / sec):** **`1415.83`** items/sec
- **Throughput (Queries / sec):** `1415.83` batches/sec
- **Peak Process RAM:** `202.11` MB
- **Net RAM Delta:** `0.05` MB