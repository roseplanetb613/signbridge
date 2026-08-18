"""骨架版 TFNet（STGCNCTCTF）测试：时频双分支、多 CTC、SeqKD、迁移。"""

import torch
import torch.nn.functional as F
import pytest

from signbridge.core.graphs import build_hand_graph
from signbridge.models.stgcn_ctc import STGCNCTC
from signbridge.models.stgcn_ctc_tf import STGCNCTCTF, SeqKD


def _model(k=5):
    return STGCNCTCTF(num_classes=k,
                      adjacency=build_hand_graph(num_hands=2))


def test_forward_returns_three_logits():
    model = _model()
    out = model(torch.randn(2, 3, 128, 42))
    assert len(out) == 3
    for logits in out:
        assert logits.shape == (2, 32, 6)       # (N, T'=32, K+1)


def test_frequency_branch_gradients():
    model = _model()
    x = torch.randn(2, 3, 128, 42)
    targets = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
    ylen = torch.tensor([2, 1])
    loss = sum(F.ctc_loss(model.log_probs(l), targets,
                          input_lengths=torch.full((2,), 32),
                          target_lengths=ylen)
               for l in model(x))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"参数 {name} 无梯度"
    # 频域分支参数确实收到梯度
    assert model.freq_conv[0].weight.grad is not None
    assert model.head_f.weight.grad is not None
    assert model.head_fusion.weight.grad is not None


def test_seqkd_loss():
    kld = SeqKD()
    a = torch.randn(2, 32, 6)
    b = torch.randn(2, 32, 6)
    loss = kld(a, b)                      # prediction=a, ref=b
    assert loss.ndim == 0 and torch.isfinite(loss)
    # 相同分布 → 接近 0
    same = torch.full((2, 32, 6), 0.5)
    assert kld(same, same) < 1e-3


def test_load_temporal_state_from_stgcnctc():
    """从 STGCNCTC checkpoint 迁移 blocks + head（head_t 权重 squeeze）。"""
    src = STGCNCTC(num_classes=5, adjacency=build_hand_graph(num_hands=2))
    model = _model()
    n = model.load_temporal_state(src.state_dict())
    assert n > 0
    # head_t 权重 = 源 head 权重 squeeze（形状 (K+1, 256, 1)）
    assert torch.allclose(
        model.head_t.weight,
        src.head.weight.squeeze(-1), atol=1e-6)
    assert torch.allclose(model.head_t.bias, src.head.bias, atol=1e-6)
    # blocks 权重一致
    assert torch.allclose(
        model.blocks[0].gcn.conv.weight,
        src.blocks[0].gcn.conv.weight, atol=1e-6)


def test_decode_and_beam():
    model = _model()
    model.eval()
    logits = torch.randn(2, 32, 6)
    assert len(model.decode(logits)) == 2
    assert len(model.beam_decode(logits, beam_width=5)) == 2


def test_invalid_input_raises():
    model = _model()
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 128, 20))       # V 不匹配
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 5, 42))         # T < kernel_size
