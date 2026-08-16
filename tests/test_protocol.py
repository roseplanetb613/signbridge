import torch
import torch.nn as nn

from signbridge.models.protocol import SkeletonClassifier


def test_protocol_exposes_interface():
    assert hasattr(SkeletonClassifier, "forward")
    assert hasattr(SkeletonClassifier, "predict")
    assert "num_classes" in SkeletonClassifier.__annotations__
    assert "num_nodes" in SkeletonClassifier.__annotations__


def test_implementation_conforms_via_structure():
    class MyModel(nn.Module):
        num_classes = 3
        num_nodes = 21

        def forward(self, x):
            return torch.zeros(x.shape[0], 3)

        def predict(self, x):
            return torch.zeros(x.shape[0], dtype=torch.int64)

    m = MyModel()
    assert isinstance(m, nn.Module)
    assert isinstance(m, SkeletonClassifier)  # runtime_checkable 结构检查
