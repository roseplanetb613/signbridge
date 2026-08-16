import torch.nn as nn

from signbridge.models.protocol import SkeletonClassifier


def test_protocol_is_nn_module_compatible():
    assert issubclass(SkeletonClassifier, nn.Module)


def test_protocol_exposes_interface():
    assert hasattr(SkeletonClassifier, "forward")
    assert hasattr(SkeletonClassifier, "predict")
    assert hasattr(SkeletonClassifier, "num_classes")
    assert hasattr(SkeletonClassifier, "num_nodes")
