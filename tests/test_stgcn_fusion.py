"""FusionSTGCNCTC 三流融合模型测试（resnet_pretrained=False，不下载权重）。"""

import torch

torch.set_num_threads(1)   # 规避 CPU 多线程下的崩溃路径

import numpy as np
import pytest
import torch.nn.functional as F

from signbridge.core.graphs import build_adjacency, build_hand_graph
from signbridge.models.stgcn_fusion import FusionSTGCNCTC

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


def _inputs(n=2, t=128, roi_size=112, s=42):
    hand = torch.randn(n, 3, t, s)
    pose = torch.randn(n, 3, t, 33)
    roi = torch.randint(0, 256, (n, t, 3, roi_size, roi_size)).float()
    return hand, pose, roi


def _model(k=5, pretrained=False):
    return FusionSTGCNCTC(
        num_classes=k,
        hand_adjacency=build_hand_graph(num_hands=2),
        pose_adjacency=build_adjacency(POSE_CONNECTIONS, 33),
        resnet_pretrained=pretrained,
    )


def test_forward_shape():
    model = _model()
    out = model(*_inputs())
    assert out.shape == (2, 32, 6)          # (N, T'=32, K+1)


def test_log_probs_normalized():
    model = _model()
    lp = model.log_probs(*_inputs())
    assert lp.shape == (32, 2, 6)
    assert torch.allclose(lp.exp().sum(dim=2), torch.ones(32, 2), atol=1e-4)


def test_ctc_backward_all_streams():
    model = _model()
    hand, pose, roi = _inputs()
    targets = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
    loss = F.ctc_loss(model.log_probs(hand, pose, roi), targets,
                      input_lengths=torch.full((2,), 32),
                      target_lengths=torch.tensor([2, 1]))
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"参数 {name} 无梯度"


def test_decode_and_beam():
    model = _model()
    logits = torch.randn(2, 32, 6)
    assert model.decode(logits) == [[]] * 2 or len(model.decode(logits)) == 2
    assert len(model.beam_decode(logits)) == 2


def test_beam_decode_length_bonus():
    """length_bonus 透传：不报错、输出 token 合法（行为测试见 test_decoding）。"""
    model = _model()
    logits = torch.randn(2, 32, 6)
    out = model.beam_decode(logits, beam_width=5, length_bonus=1.0)
    assert len(out) == 2
    assert all(0 <= c <= 5 for seq in out for c in seq)


def test_roi_size_mismatch_raises():
    model = _model()
    hand, pose, roi = _inputs(roi_size=128)   # 与 roi_input_size=112 不一致
    with pytest.raises(ValueError):
        model(hand, pose, roi)


def test_t_mismatch_raises():
    model = _model()
    hand, pose, roi = _inputs()
    with pytest.raises(ValueError):
        model(hand, pose, roi[:, :64])        # roi T 不一致


def test_custom_pose_channels():
    model = FusionSTGCNCTC(
        num_classes=3,
        hand_adjacency=build_hand_graph(num_hands=2),
        pose_adjacency=build_adjacency(POSE_CONNECTIONS, 33),
        pose_channels=(32, 64, 64), pose_strides=(1, 2, 2),
        resnet_pretrained=False,
    )
    out = model(*_inputs())
    assert out.shape == (2, 32, 4)
