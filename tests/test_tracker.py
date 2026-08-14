import numpy as np
import pytest

from signbridge.core.landmarks import HandFrame
from signbridge.core.matching import Matching
from signbridge.hands.sequence import HandSequenceBuffer


def test_single_hand_stable_id(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None)
    pts = hand_pts()
    ids = set()
    for _ in range(10):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            ids.add(s.hand_id)
    assert len(ids) == 1


def test_cross_swap_keeps_identity(make_hand_frame, hand_pts):
    """双手互换位置：ID 数稳定、序列完整、无 NaN 丢失帧。"""
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None)
    a_left = hand_pts(center=(0.2, 0.5), seed=0)   # 手 X（形 A）初始在左
    b_right = hand_pts(center=(0.8, 0.5), seed=1)  # 手 Y（形 B）初始在右
    a_right = hand_pts(center=(0.8, 0.5), seed=0)  # X 移到右
    b_left = hand_pts(center=(0.2, 0.5), seed=1)   # Y 移到左
    frames = []
    for _ in range(5):
        frames.append(make_hand_frame([("Left", a_left), ("Right", b_right)]))
    for _ in range(5):  # 位置互换（标记随位置走，模拟镜像/误判）
        frames.append(make_hand_frame([("Left", b_left), ("Right", a_right)]))
    ids = set()
    for hf in frames:
        for s in buf.update(hf):
            ids.add(s.hand_id)
    assert len(ids) == 2  # 交换不产生新 ID、不丢 ID
    seqs = buf.update(frames[-1])  # 第 11 次 update
    assert len(seqs) == 2
    for s in seqs:
        assert s.data.shape == (11, 21, 3)
        assert s.valid_mask.all()       # 全程无丢失帧
        assert not np.isnan(s.data).any()


def test_lost_within_k_frames_keeps_id(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None,
                             max_lost_frames=10)
    pts = hand_pts()
    ids = set()
    for _ in range(5):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            ids.add(s.hand_id)
    for _ in range(5):  # 消失 5 帧（<= 10）
        buf.update(HandFrame())
    for _ in range(5):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            ids.add(s.hand_id)
    assert len(ids) == 1


def test_lost_beyond_k_frames_recycles_id(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=60, coordinate="image", smoother=None,
                             max_lost_frames=10)
    pts = hand_pts()
    first = set()
    for _ in range(5):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            first.add(s.hand_id)
    for _ in range(15):  # 消失 15 帧（> 10）
        buf.update(HandFrame())
    second = set()
    for _ in range(5):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            second.add(s.hand_id)
    assert first.isdisjoint(second)  # 回收后不复用


def test_new_hand_gets_new_id(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None)
    pts = hand_pts()
    for _ in range(3):
        buf.update(make_hand_frame([("Left", pts)]))
    seqs = buf.update(make_hand_frame(
        [("Left", pts), ("Right", hand_pts(seed=1))]))
    assert len(seqs) == 2
    assert len({s.hand_id for s in seqs}) == 2


def test_left_right_hand_id_properties(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None)
    pts = hand_pts()
    buf.update(make_hand_frame([("Left", pts)]))
    assert buf.left_hand_id >= 0 and buf.right_hand_id == -1
    buf.update(make_hand_frame([("Left", pts), ("Right", hand_pts(seed=1))]))
    assert buf.left_hand_id >= 0 and buf.right_hand_id >= 0
    assert buf.left_hand_id != buf.right_hand_id


class _FixedMatcher:
    """固定返回指定匹配结果的假匹配器（可插拔验证）。

    首帧（上一帧轨迹为空）时返回全不匹配——这是任何匹配器的合理行为。
    """

    def __init__(self, matching: Matching):
        self._m = matching
        self.calls = 0

    def match(self, current_centroids, previous_centroids):
        self.calls += 1
        if len(previous_centroids) == 0:
            return Matching(
                matched=(),
                unmatched_current=tuple(range(len(current_centroids))),
                unmatched_previous=(),
            )
        return self._m


def test_custom_matcher_full_match_reuses_id(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(
        window_size=10, coordinate="image", smoother=None,
        matcher=_FixedMatcher(Matching(matched=((0, 0),),
                                       unmatched_current=(), unmatched_previous=())),
    )
    pts = hand_pts()
    ids = set()
    for _ in range(4):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            ids.add(s.hand_id)
    assert len(ids) == 1  # 匹配对驱动 ID 续用


def test_custom_matcher_full_unmatched_creates_new_ids(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(
        window_size=10, coordinate="image", smoother=None,
        matcher=_FixedMatcher(Matching(matched=(),
                                       unmatched_current=(0,), unmatched_previous=(0,))),
    )
    pts = hand_pts()
    ids = set()
    for _ in range(4):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            ids.add(s.hand_id)
    assert len(ids) == 4  # 每次都不匹配 → 每帧新 ID
