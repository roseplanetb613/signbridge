import numpy as np
import pytest

from signbridge.core.landmarks import HandFrame
from signbridge.hands.sequence import HandSequence, HandSequenceBuffer


def test_window_slides(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=60, coordinate="image", smoother=None)
    pts = hand_pts()
    for _ in range(70):
        buf.update(make_hand_frame([("Left", pts)]))
    seq = buf.update(make_hand_frame([("Left", pts)]))[0]
    assert seq.data.shape == (60, 21, 3)
    assert seq.valid_mask.all()
    assert seq.frame_indices[0] == 11   # 前 11 帧滑出
    assert seq.frame_indices[-1] == 70


def test_lost_frame_occupies_slot_with_nan(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=None,
                             max_lost_frames=5)
    pts = hand_pts()
    for _ in range(5):
        buf.update(make_hand_frame([("Left", pts)]))
    buf.update(HandFrame())                          # 第 6 帧：无手
    seq = buf.update(make_hand_frame([("Left", pts)]))[0]  # 第 7 帧：手回来
    assert seq.data.shape == (7, 21, 3)
    assert seq.valid_mask[5] == False
    assert np.isnan(seq.data[5]).all()
    assert seq.valid_mask[6] == True
    assert seq.frame_indices[-1] == 6


def test_wrist_normalized_to_origin(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=None)
    pts = hand_pts()
    for _ in range(5):
        buf.update(make_hand_frame([("Left", pts)]))
    seq = buf.update(make_hand_frame([("Left", pts)]))[0]
    assert np.allclose(seq.data[:, 0, :], 0.0, atol=1e-6)


def test_two_hands_independent_sequences(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=None)
    left = hand_pts(center=(0.2, 0.5), seed=0)
    right = hand_pts(center=(0.8, 0.5), seed=1)
    for _ in range(5):
        buf.update(make_hand_frame([("Left", left), ("Right", right)]))
    seqs = buf.update(make_hand_frame([("Left", left), ("Right", right)]))
    assert len(seqs) == 2
    assert [s.hand_id for s in seqs] == sorted(s.hand_id for s in seqs)
    lseq = next(s for s in seqs if s.handedness == "Left")
    rseq = next(s for s in seqs if s.handedness == "Right")
    assert not np.array_equal(lseq.data, rseq.data)


def test_smoother_called_per_valid_frame(make_hand_frame, hand_pts):
    class _Recorder:
        """记录调用的假平滑器。Buffer 会 deepcopy 实例，__deepcopy__ 返回自身以共享计数。"""

        def __init__(self):
            self.calls = []

        def update(self, pts):
            self.calls.append(None if pts is None else pts.copy())
            return pts

        def reset(self):
            self.calls.clear()

        def __deepcopy__(self, memo):
            return self

    smoother = _Recorder()
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=smoother)
    pts = hand_pts()
    for _ in range(5):
        buf.update(make_hand_frame([("Left", pts)]))
    buf.update(HandFrame())  # 无手帧 → smoother 收到 None
    assert len(smoother.calls) == 6
    assert smoother.calls[-1] is None


def test_reset_clears_state(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=None)
    pts = hand_pts()
    for _ in range(5):
        buf.update(make_hand_frame([("Left", pts)]))
    buf.reset()
    seq = buf.update(make_hand_frame([("Left", pts)]))[0]
    assert seq.frame_indices[0] == 0
    assert len(seq.data) == 1


def test_empty_frames_yield_no_sequences(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=None)
    assert buf.update(HandFrame()) == ()
    assert buf.update(HandFrame()) == ()
