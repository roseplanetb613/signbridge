import numpy as np
import pytest
import torch
import torch.nn.functional as F

from signbridge.core.graphs import build_hand_graph
from signbridge.models.stgcn_ctc import STGCNCTC


def _adj():
    return build_hand_graph(num_hands=1)


def test_forward_shape_single_hand():
    model = STGCNCTC(num_classes=10, adjacency=_adj())
    out = model(torch.randn(2, 3, 128, 21))
    assert out.dim() == 3
    assert out.shape[0] == 2 and out.shape[2] == 11   # K+1
    assert out.shape[1] == 32                          # T' = 128/2/2


def test_forward_shape_two_hands():
    model = STGCNCTC(num_classes=10, adjacency=build_hand_graph(num_hands=2))
    out = model(torch.randn(2, 3, 128, 42))
    assert out.shape == (2, 32, 11)


def test_log_probs_shape_and_normalized():
    model = STGCNCTC(num_classes=10, adjacency=_adj())
    lp = model.log_probs(torch.randn(2, 3, 128, 21))
    assert lp.shape == (32, 2, 11)                     # (T', N, K+1)
    assert torch.allclose(lp.exp().sum(dim=2), torch.ones(32, 2), atol=1e-4)


def test_decode_merges_repeats_and_removes_blank():
    model = STGCNCTC(num_classes=3, adjacency=_adj())
    model.eval()
    # argmax 序列 = [0,1,1,0,2,0,0,2]：
    # 相邻重复合并（1,1→1），blank(0) 剔除；但两个 2 被 blank 隔开 → 不合并
    logits = torch.full((1, 8, 4), -10.0)
    seq = [0, 1, 1, 0, 2, 0, 0, 2]
    for t, c in enumerate(seq):
        logits[0, t, c] = 10.0
    decoded = model.decode(logits)
    assert decoded == [[1, 2, 2]]


def test_decode_empty_output():
    model = STGCNCTC(num_classes=3, adjacency=_adj())
    logits = torch.full((1, 8, 4), -10.0)
    for t in range(8):
        logits[0, t, 0] = 10.0                        # 全 blank
    assert model.decode(logits) == [[]]


def test_ctc_backward_gradients_exist():
    model = STGCNCTC(num_classes=5, adjacency=_adj())
    x = torch.randn(2, 3, 128, 21)
    targets = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])  # 词 id（0 填充）
    target_lengths = torch.tensor([2, 1])
    loss = F.ctc_loss(model.log_probs(x), targets,
                      input_lengths=torch.full((2,), 32),
                      target_lengths=target_lengths)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"参数 {name} 无梯度"


def test_custom_config_forward():
    model = STGCNCTC(num_classes=3, adjacency=_adj(),
                     channels=(32, 64, 128), strides=(1, 2, 2))
    out = model(torch.randn(2, 3, 128, 21))
    assert out.shape == (2, 32, 4)


def test_invalid_input_raises():
    model = STGCNCTC(num_classes=3, adjacency=_adj())
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 128, 20))              # V=20 != 21
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 5, 21))                # T < kernel_size


def test_embed_shape_and_t_invariance():
    model = STGCNCTC(num_classes=5, adjacency=_adj())
    model.eval()
    with torch.no_grad():
        e1 = model.embed(torch.randn(2, 3, 128, 21))
        e2 = model.embed(torch.randn(2, 3, 64, 21))    # 不同 T（下采样后 T' 不同）
    assert e1.shape == (2, 256)                        # C_last = 256
    assert e2.shape == (2, 256)                        # 时间均值 → 与 T 无关
