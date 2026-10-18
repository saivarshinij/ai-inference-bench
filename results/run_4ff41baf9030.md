# Inference Benchmark Report: `mlp_classifier`

**Run ID:** `run_4ff41baf9030`  
**Timestamp (UTC):** `2026-10-18T17:08:58.008008+00:00`  
**Backend:** `torchscript` | **Precision:** `fp32` | **Device:** `cpu`  
**Batch Size:** `1` | **Warm-up Iterations:** `10` | **Measured Iterations:** `50`  

## Hardware & Environment

- **CPU Model:** AMD Ryzen 3 7320U with Radeon Graphics
- **Cores:** 4 Physical / 8 Logical
- **System RAM:** 7.21 GB (0.73 GB available)
- **OS:** Windows 10 (AMD64)
- **Runtimes:** Python `3.10.10` | PyTorch `2.14.1+cpu` | ONNX Runtime `1.23.2`
- **Accelerator Detected:** `CPU` (CUDA Available: `False`)

## Measured Performance Metrics

### 1. Latency Percentiles (Milliseconds)

| Metric        |   Value (ms) |
|---------------|--------------|
| Mean          |        0.115 |
| Median (p50)  |        0.105 |
| p90           |        0.144 |
| p95           |        0.163 |
| p99           |        0.462 |
| Min           |        0.061 |
| Max           |        0.659 |
| Std Deviation |        0.085 |

### 2. Throughput & Memory Consumption

- **Throughput (Samples / sec):** **`8724.63`** items/sec
- **Throughput (Queries / sec):** `8724.63` batches/sec
- **Peak Process RAM:** `241.95` MB
- **Net RAM Delta:** `0.01` MB