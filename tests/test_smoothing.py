import numpy as np
import pytest

from signbridge.core.smoothing import LandmarkSmoother, OneEuroSmoother


def _pts(v):
    return np.full((21, 3), v, dtype=np.float32)


def test_protocol_exposes_update_and_reset():
    assert hasattr(LandmarkSmoother, "update")
    assert hasattr(LandmarkSmoother, "reset")


def test_first_frame_passthrough():
    s = OneEuroSmoother()
    pts = np.random.default_rng(1).random((21, 3)).astype(np.float32)
    out = s.update(pts.copy())
    assert np.array_equal(out, pts)


def test_constant_sequence_converges():
    s = OneEuroSmoother()
    out = None
    for _ in range(200):
        out = s.update(_pts(0.5))
    assert out is not None
    assert np.allclose(out, _pts(0.5), atol=1e-3)


def test_step_response_smoothed_then_converges():
    s = OneEuroSmoother()
    for _ in range(50):
        s.update(_pts(0.0))
    first = s.update(_pts(1.0))
    assert np.all(first < 1.0)  # 阶跃被平滑，未立即跳满
    for _ in range(400):
        last = s.update(_pts(1.0))
    assert np.allclose(last, _pts(1.0), atol=1e-2)


def test_none_input_keeps_state():
    s = OneEuroSmoother()
    pts = np.random.default_rng(0).random((21, 3)).astype(np.float32)
    a = s.update(pts.copy())
    assert s.update(None) is None
    b = s.update(pts.copy())
    assert np.allclose(a, b, atol=1e-6)  # None 不改变内部状态（仅浮点舍入差异）


def test_reset_clears_memory():
    s = OneEuroSmoother()
    for _ in range(100):
        s.update(_pts(1.0))
    s.reset()
    jump = _pts(0.0)
    out = s.update(jump.copy())
    assert np.array_equal(out, jump)  # reset 后无记忆，首帧直通
