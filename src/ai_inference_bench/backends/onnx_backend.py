"""ONNX Runtime Cross-Platform Inference Backend."""

import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
from ai_inference_bench.backends.base import BaseInferenceBackend
from ai_inference_bench.core.config import BenchmarkConfig


class ONNXRuntimeBackend(BaseInferenceBackend):
    """ONNX Runtime execution provider backend."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.session = None
        self.input_name = None
        self.onnx_path = None

    def prepare(self, model: torch.nn.Module, sample_input: torch.Tensor) -> Any:
        import onnxruntime as ort

        model = model.to(device="cpu")
        model.eval()

        export_dir = Path("onnx_models")
        export_dir.mkdir(parents=True, exist_ok=True)
        self.onnx_path = export_dir / f"{self.config.model_name}_b{self.config.batch_size}.onnx"

        # Export PyTorch model to ONNX
        with torch.no_grad():
            inp_cpu = sample_input.to(device="cpu")
            torch.onnx.export(
                model,
                inp_cpu,
                str(self.onnx_path),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )

        # Set execution providers
        providers = ["CPUExecutionProvider"]
        if torch.cuda.is_available() and self.config.device == "cuda":
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(self.onnx_path), sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        return self.session

    def run_inference(self, input_data: Any) -> Any:
        if isinstance(input_data, torch.Tensor):
            numpy_inp = input_data.detach().cpu().numpy()
        elif isinstance(input_data, np.ndarray):
            numpy_inp = input_data
        else:
            raise ValueError(f"Unsupported input data type for ONNX: {type(input_data)}")

        outputs = self.session.run(None, {self.input_name: numpy_inp})
        return outputs[0]

    def cleanup(self) -> None:
        if self.onnx_path and self.onnx_path.exists():
            try:
                self.onnx_path.unlink()
            except Exception:
                pass
