"""PyTorch Eager Inference Backend."""

from typing import Any
import torch
from ai_inference_bench.backends.base import BaseInferenceBackend
from ai_inference_bench.core.config import BenchmarkConfig


class PyTorchEagerBackend(BaseInferenceBackend):
    """Standard PyTorch eager mode inference engine with FP32/FP16 precision controls."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.model: torch.nn.Module = None
        self.device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if (config.precision == "fp16" and self.device.type != "cpu") else torch.float32

    def prepare(self, model: torch.nn.Module, sample_input: torch.Tensor) -> Any:
        self.model = model.to(device=self.device)
        # Convert model weights if FP16 on accelerator
        if self.config.precision == "fp16" and self.device.type != "cpu":
            self.model = self.model.half()
        self.model.eval()
        return self.model

    def run_inference(self, input_data: Any) -> Any:
        with torch.inference_mode():
            if isinstance(input_data, torch.Tensor):
                inp = input_data.to(device=self.device)
                if self.config.precision == "fp16" and self.device.type != "cpu" and inp.dtype == torch.float32:
                    inp = inp.half()
                out = self.model(inp)
            elif isinstance(input_data, (tuple, list)):
                inp = [x.to(device=self.device) for x in input_data]
                out = self.model(*inp)
            else:
                out = self.model(input_data)

            self.synchronize()
            return out
