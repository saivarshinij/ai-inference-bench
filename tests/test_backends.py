"""Unit tests for inference runtime execution backends."""

import pytest
import torch
from ai_inference_bench.backends import get_backend
from ai_inference_bench.core.config import BenchmarkConfig
from ai_inference_bench.models.registry import get_model_spec


def test_pytorch_eager_backend():
    config = BenchmarkConfig(model_name="mlp_classifier", batch_size=2, backend="pytorch_eager")
    spec = get_model_spec("mlp_classifier")
    model = spec.create_model()
    inp = spec.generate_input(batch_size=2)

    backend = get_backend(config)
    backend.prepare(model, inp)
    out = backend.run_inference(inp)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2


def test_torchscript_backend():
    config = BenchmarkConfig(model_name="mlp_classifier", batch_size=2, backend="torchscript")
    spec = get_model_spec("mlp_classifier")
    model = spec.create_model()
    inp = spec.generate_input(batch_size=2)

    backend = get_backend(config)
    backend.prepare(model, inp)
    out = backend.run_inference(inp)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2


def test_quantized_int8_backend():
    config = BenchmarkConfig(model_name="mlp_classifier", batch_size=2, backend="torch_quantized_int8", precision="int8")
    spec = get_model_spec("mlp_classifier")
    model = spec.create_model()
    inp = spec.generate_input(batch_size=2)

    backend = get_backend(config)
    backend.prepare(model, inp)
    out = backend.run_inference(inp)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == 2


def test_onnxruntime_backend():
    config = BenchmarkConfig(model_name="mlp_classifier", batch_size=2, backend="onnxruntime")
    spec = get_model_spec("mlp_classifier")
    model = spec.create_model()
    inp = spec.generate_input(batch_size=2)

    backend = get_backend(config)
    try:
        backend.prepare(model, inp)
        out = backend.run_inference(inp)
        assert out is not None
        assert out.shape[0] == 2
    finally:
        backend.cleanup()
