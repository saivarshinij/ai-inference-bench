"""Base Inference Backend Abstract Interface."""

from abc import ABC, abstractmethod
from typing import Any, Tuple
import torch
from ai_inference_bench.core.config import BenchmarkConfig


class BaseInferenceBackend(ABC):
    """Abstract interface for benchmark inference engines."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    @abstractmethod
    def prepare(self, model: torch.nn.Module, sample_input: torch.Tensor) -> Any:
        """Prepares, compiles, or exports model for benchmark execution."""
        pass

    @abstractmethod
    def run_inference(self, input_data: Any) -> Any:
        """Executes a single forward inference pass."""
        pass

    def synchronize(self) -> None:
        """Ensures all asynchronous compute queues (e.g. CUDA streams) are flushed."""
        if torch.cuda.is_available() and self.config.device == "cuda":
            torch.cuda.synchronize()

    def cleanup(self) -> None:
        """Performs optional memory and resource cleanup."""
        pass
