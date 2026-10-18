"""Benchmark Configuration Schema and Validation."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


BackendType = Literal["pytorch_eager", "torchscript", "torch_quantized_int8", "onnxruntime"]
PrecisionType = Literal["fp32", "fp16", "int8"]
DeviceType = Literal["cpu", "cuda", "mps", "rocm", "auto"]


class BenchmarkConfig(BaseModel):
    """Reproducible configuration parameters for an inference benchmark run."""

    model_name: str = Field("mobilenet_v3_small", description="Model identifier from registry")
    batch_size: int = Field(1, ge=1, le=256, description="Inference batch size")
    warm_up_iterations: int = Field(10, ge=0, le=1000, description="Warm-up cycles excluded from metrics")
    iterations: int = Field(50, ge=1, le=10000, description="Measured iterations for statistical profiling")
    backend: BackendType = Field("pytorch_eager", description="Inference execution backend")
    precision: PrecisionType = Field("fp32", description="Numerical precision")
    device: DeviceType = Field("cpu", description="Target hardware compute device")
    seq_length: Optional[int] = Field(64, ge=8, le=512, description="Sequence length for NLP models")
    image_size: Optional[int] = Field(224, ge=32, le=512, description="Image height/width for vision models")

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, v: str, info) -> str:
        data = info.data
        backend = data.get("backend")
        device = data.get("device", "cpu")

        if backend == "torch_quantized_int8" and v != "int8":
            raise ValueError("Backend 'torch_quantized_int8' requires precision='int8'.")

        if v == "fp16" and device == "cpu":
            # Note: PyTorch CPU native half-precision can be slow or unsupported on some ops
            pass

        return v
