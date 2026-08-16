import numpy as np
import pytest
import torch
import torch.nn.functional as F

from signbridge.core.graphs import build_hand_graph
from signbridge.models.stgcn import STGCN


def _adj():
    return build_hand_graph(num_hands=1)


def test_forward_shape_single_hand():
    model = STGCN(num_classes=5, adjacency=_adj())
    out = model(torch.randn(2, 3, 64, 21))
    assert out.shape == (2, 5)


def test_forward_shape_two_hands():
    model = STGCN(num_classes=5, adjacency=build_hand_graph(num_hands=2))
    out = model(torch.randn(2, 3, 64, 42))
    assert out.shape == (2, 5)


def test_backward_gradients_exist():
    model = STGCN(num_classes=5, adjacency=_adj())
    y = torch.tensor([1, 3])
    loss = F.cross_entropy(model(torch.randn(2, 3, 64, 21)), y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"参数 {name} 无梯度"


def test_adaptive_false_forward():
    model = STGCN(num_classes=5, adjacency=_adj(), adaptive=False)
    out = model(torch.randn(2, 3, 64, 21))
    assert out.shape == (2, 5)
    assert not any("B" in n for n, _ in model.named_parameters())


def test_custom_config_forward():
    model = STGCN(num_classes=3, adjacency=_adj(),
                  channels=(32, 64, 128), strides=(1, 2, 2), dropout=0.1)
    out = model(torch.randn(2, 3, 64, 21))
    assert out.shape == (2, 3)


def test_predict_returns_argmax_indices():
    model = STGCN(num_classes=5, adjacency=_adj())
    model.eval()
    pred = model.predict(torch.randn(4, 3, 64, 21))
    assert pred.shape == (4,)
    assert pred.dtype == torch.int64
    assert pred.min() >= 0 and pred.max() < 5


def test_short_t_raises():
    model = STGCN(num_classes=5, adjacency=_adj(), kernel_size=9)
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 5, 21))   # T=5 < kernel_size=9


def test_normalized_adjacency_buffer():
    model = STGCN(num_classes=5, adjacency=_adj())
    a = model.blocks[0].gcn.adjacency
    assert a.shape == (21, 21)
    assert torch.allclose(a, a.T, atol=1e-6)      # 对称归一化保持对称
    # 腕部(0)邻居为 1,5,17 → 度 4（含自环）→ 对角 = 1/4
    assert torch.allclose(a[0, 0], torch.tensor(1.0 / 4.0), atol=1e-5)


def test_wrong_node_count_raises():
    model = STGCN(num_classes=5, adjacency=_adj())
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 64, 20))          # V=20 != 21


def test_channels_strides_mismatch_raises():
    with pytest.raises(ValueError):
        STGCN(num_classes=5, adjacency=_adj(),
              channels=(64, 64), strides=(1, 1, 1))
