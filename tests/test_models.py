"""Unit tests for model registry and synthetic tensor data generation."""

import pytest
import torch
from ai_inference_bench.models.registry import (
    MODEL_REGISTRY,
    get_model_spec,
    list_registered_models,
)


@pytest.mark.parametrize("model_name", ["mobilenet_v3_small", "resnet18", "tiny_transformer", "mlp_classifier"])
def test_model_instantiation_and_forward_pass(model_name: str):
    spec = get_model_spec(model_name)
    model = spec.create_model()
    assert isinstance(model, torch.nn.Module)
    assert not model.training  # Verify eval mode

    sample_input = spec.generate_input(batch_size=2, device="cpu")
    with torch.no_grad():
        output = model(sample_input)

    assert output is not None
    assert output.shape[0] == 2  # Matches batch size


def test_list_registered_models():
    models = list_registered_models()
    assert len(models) == 4
    names = [m["name"] for m in models]
    assert "mobilenet_v3_small" in names
    assert "tiny_transformer" in names


def test_unknown_model_raises_error():
    with pytest.raises(ValueError):
        get_model_spec("non_existent_super_model")
