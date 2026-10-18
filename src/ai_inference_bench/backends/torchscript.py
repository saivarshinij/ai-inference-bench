"""TorchScript JIT Compiled Inference Backend."""

from typing import Any
import torch
from ai_inference_bench.backends.base import BaseInferenceBackend
from ai_inference_bench.core.config import BenchmarkConfig


class TorchScriptBackend(BaseInferenceBackend):
    """JIT Traced and frozen TorchScript backend for optimized inference execution."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.scripted_model = None
        self.device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    def prepare(self, model: torch.nn.Module, sample_input: torch.Tensor) -> Any:
        model = model.to(device=self.device)
        model.eval()
        inp = sample_input.to(device=self.device)

        with torch.inference_mode():
            try:
                traced = torch.jit.trace(model, inp)
                self.scripted_model = torch.jit.freeze(traced)
            except Exception:
                # Fallback to scripting if tracing encounters dynamic control flow
                scripted = torch.jit.script(model)
                self.scripted_model = torch.jit.freeze(scripted)

        return self.scripted_model

    def run_inference(self, input_data: Any) -> Any:
        with torch.inference_mode():
            if isinstance(input_data, torch.Tensor):
                inp = input_data.to(device=self.device)
                out = self.scripted_model(inp)
            elif isinstance(input_data, (tuple, list)):
                inp = [x.to(device=self.device) for x in input_data]
                out = self.scripted_model(*inp)
            else:
                out = self.scripted_model(input_data)

            self.synchronize()
            return out
