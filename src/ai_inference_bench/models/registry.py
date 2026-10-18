"""Model Architectures and Registry for Benchmarking."""

import math
from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn


# -------------------------------------------------------------
# 1. Vision Models: MobileNetV3-Small & ResNet-18 Implementations
# -------------------------------------------------------------
class ConvBNActivation(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.Hardswish(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_c, exp_c, out_c, kernel_size, stride):
        super().__init__()
        self.use_res_connect = stride == 1 and in_c == out_c
        layers = []
        if exp_c != in_c:
            layers.append(ConvBNActivation(in_c, exp_c, kernel_size=1))
        layers.append(ConvBNActivation(exp_c, exp_c, kernel_size=kernel_size, stride=stride, groups=exp_c))
        layers.append(nn.Conv2d(exp_c, out_c, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_c))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class MobileNetV3Small(nn.Module):
    """Compact MobileNetV3-Small architecture for edge and CPU vision inference."""

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNActivation(3, 16, stride=2),
            InvertedResidual(16, 16, 16, 3, 2),
            InvertedResidual(16, 72, 24, 3, 2),
            InvertedResidual(24, 88, 24, 3, 1),
            InvertedResidual(24, 96, 40, 5, 2),
            InvertedResidual(40, 240, 40, 5, 1),
            ConvBNActivation(40, 96, kernel_size=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(96, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class ResNet18(nn.Module):
    """ResNet-18 vision benchmark architecture."""

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# -------------------------------------------------------------
# 2. NLP Model: TinyTransformer
# -------------------------------------------------------------
class TinyTransformer(nn.Module):
    """Compact Transformer Encoder for text sequence benchmarking and dynamic quantization."""

    def __init__(self, vocab_size: int = 2000, d_model: int = 128, nhead: int = 4, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len) of token IDs
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        out = self.transformer_encoder(emb)
        # Global mean pooling over sequence dimension
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


# -------------------------------------------------------------
# 3. Tabular / Classical Deep Learning: MLPClassifier
# -------------------------------------------------------------
class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for tabular / feature-based inference benchmarking."""

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------------------------------------------
# Model Registry Metadata & Lookup
# -------------------------------------------------------------
class ModelSpec:
    def __init__(
        self,
        name: str,
        domain: str,
        description: str,
        factory: Callable[[], nn.Module],
        input_type: str,  # "image", "sequence", "tabular"
        default_shape: Tuple[int, ...],
    ):
        self.name = name
        self.domain = domain
        self.description = description
        self.factory = factory
        self.input_type = input_type
        self.default_shape = default_shape

    def create_model(self) -> nn.Module:
        model = self.factory()
        model.eval()
        return model

    def generate_input(self, batch_size: int, device: str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
        torch.manual_seed(42)
        if self.input_type == "image":
            # (batch_size, 3, 224, 224)
            shape = (batch_size, 3, self.default_shape[1], self.default_shape[2])
            return torch.randn(shape, device=device, dtype=dtype)
        elif self.input_type == "sequence":
            # (batch_size, seq_length) integer token IDs
            seq_len = self.default_shape[0]
            return torch.randint(0, 1000, (batch_size, seq_len), device=device, dtype=torch.long)
        elif self.input_type == "tabular":
            # (batch_size, features)
            feat_dim = self.default_shape[0]
            return torch.randn((batch_size, feat_dim), device=device, dtype=dtype)
        raise ValueError(f"Unknown input type: {self.input_type}")


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "mobilenet_v3_small": ModelSpec(
        name="mobilenet_v3_small",
        domain="Vision",
        description="Lightweight inverted-residual CNN for mobile and edge vision inference.",
        factory=MobileNetV3Small,
        input_type="image",
        default_shape=(3, 224, 224),
    ),
    "resnet18": ModelSpec(
        name="resnet18",
        domain="Vision",
        description="Standard 18-layer residual network for computer vision benchmarks.",
        factory=ResNet18,
        input_type="image",
        default_shape=(3, 224, 224),
    ),
    "tiny_transformer": ModelSpec(
        name="tiny_transformer",
        domain="NLP",
        description="2-layer self-attention transformer encoder for NLP inference.",
        factory=TinyTransformer,
        input_type="sequence",
        default_shape=(64,),
    ),
    "mlp_classifier": ModelSpec(
        name="mlp_classifier",
        domain="Tabular",
        description="3-layer dense feedforward network for high-throughput tabular inference.",
        factory=MLPClassifier,
        input_type="tabular",
        default_shape=(64,),
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not found in registry. Available models: {', '.join(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


def list_registered_models() -> List[Dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "domain": spec.domain,
            "description": spec.description,
            "input_type": spec.input_type,
            "default_shape": spec.default_shape,
        }
        for spec in MODEL_REGISTRY.values()
    ]
