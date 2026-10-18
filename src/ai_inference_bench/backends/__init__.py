"""Backends package."""

from ai_inference_bench.backends.base import BaseInferenceBackend
from ai_inference_bench.backends.onnx_backend import ONNXRuntimeBackend
from ai_inference_bench.backends.pytorch_eager import PyTorchEagerBackend
from ai_inference_bench.backends.quantized import TorchQuantizedBackend
from ai_inference_bench.backends.torchscript import TorchScriptBackend
from ai_inference_bench.core.config import BenchmarkConfig


def get_backend(config: BenchmarkConfig) -> BaseInferenceBackend:
    """Instantiates backend engine according to configuration."""
    backend_map = {
        "pytorch_eager": PyTorchEagerBackend,
        "torchscript": TorchScriptBackend,
        "torch_quantized_int8": TorchQuantizedBackend,
        "onnxruntime": ONNXRuntimeBackend,
    }

    if config.backend not in backend_map:
        raise ValueError(
            f"Unsupported backend '{config.backend}'. Available: {', '.join(backend_map.keys())}"
        )

    return backend_map[config.backend](config)


__all__ = [
    "BaseInferenceBackend",
    "PyTorchEagerBackend",
    "TorchScriptBackend",
    "TorchQuantizedBackend",
    "ONNXRuntimeBackend",
    "get_backend",
]
