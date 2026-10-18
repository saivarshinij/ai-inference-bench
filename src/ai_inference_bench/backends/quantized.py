"""Dynamic INT8 Quantization Backend for High-Efficiency CPU Inference."""

from typing import Any
import torch
from ai_inference_bench.backends.base import BaseInferenceBackend
from ai_inference_bench.core.config import BenchmarkConfig


class TorchQuantizedBackend(BaseInferenceBackend):
    """Dynamic INT8 quantization backend for CPU inference optimization."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.quantized_model = None

    def prepare(self, model: torch.nn.Module, sample_input: torch.Tensor) -> Any:
        # Dynamic quantization applies to CPU models
        model = model.to(device="cpu")
        model.eval()

        # Dynamically quantize Linear layers to qint8
        self.quantized_model = torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        return self.quantized_model

    def run_inference(self, input_data: Any) -> Any:
        with torch.inference_mode():
            if isinstance(input_data, torch.Tensor):
                inp = input_data.to(device="cpu")
                out = self.quantized_model(inp)
            elif isinstance(input_data, (tuple, list)):
                inp = [x.to(device="cpu") for x in input_data]
                out = self.quantized_model(*inp)
            else:
                out = self.quantized_model(input_data)

            return out
