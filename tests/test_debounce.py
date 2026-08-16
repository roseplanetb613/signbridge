"""handedness 时序防抖（滞后切换）单元测试。"""

import numpy as np
import pytest

from signbridge.hands.sequence import HandSequenceBuffer


def _flip_frames(make_hand_frame, hand_pts, pattern):
    """按 pattern 生成帧序列：'L'=判定左手，'R'=判定右手。"""
    pts = hand_pts()
    frames = []
    for tag in pattern:
        h = "Left" if tag == "L" else "Right"
        frames.append(make_hand_frame([(h, pts)]))
    return frames


def test_debounce_suppresses_short_flips(make_hand_frame, hand_pts):
    """判定短暂翻转（< M 帧）→ 标签保持原样。"""
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None,
                             handedness_debounce=5)
    # 左手稳定 10 帧 → 判定变 R 3 帧（抖动）→ 恢复 L 5 帧
    pattern = "L" * 10 + "R" * 3 + "L" * 5
    labels = []
    for hf in _flip_frames(make_hand_frame, hand_pts, pattern):
        seqs = buf.update(hf)
        labels.append(seqs[0].handedness if seqs else None)
    assert all(l == "Left" for l in labels)  # 抖动被完全抑制


def test_debounce_allows_real_switch(make_hand_frame, hand_pts):
    """判定持续相反 ≥ M 帧 → 切换标签。"""
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None,
                             handedness_debounce=5)
    pattern = "L" * 5 + "R" * 8
    labels = []
    for hf in _flip_frames(make_hand_frame, hand_pts, pattern):
        seqs = buf.update(hf)
        labels.append(seqs[0].handedness if seqs else None)
    # 前 5 帧 Left；第 5 个 R 帧（累计 flip=5）起切换
    assert labels[:5] == ["Left"] * 5
    assert labels[10:] == ["Right"] * 3


def test_debounce_zero_follows_immediately(make_hand_frame, hand_pts):
    """handedness_debounce=0 → 关闭防抖，逐帧跟随。"""
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None,
                             handedness_debounce=0)
    pattern = "L" * 3 + "R" * 3 + "L" * 3
    labels = []
    for hf in _flip_frames(make_hand_frame, hand_pts, pattern):
        seqs = buf.update(hf)
        labels.append(seqs[0].handedness if seqs else None)
    assert labels == ["Left"] * 3 + ["Right"] * 3 + ["Left"] * 3


def test_debounce_per_track_independent(make_hand_frame, hand_pts):
    """双手各自独立防抖（右手抖动不影响左手标签）。"""
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None,
                             handedness_debounce=5)
    left = hand_pts(center=(0.2, 0.5), seed=0)
    right = hand_pts(center=(0.8, 0.5), seed=1)
    labels_by_id = {}
    for i in range(12):
        if 4 <= i < 7:   # 右手（@0.8）判定短暂抖动为 Left
            hf = make_hand_frame([("Left", left), ("Left", right)])
        else:
            hf = make_hand_frame([("Left", left), ("Right", right)])
        for s in buf.update(hf):
            labels_by_id.setdefault(s.hand_id, []).append(s.handedness)
    assert len(labels_by_id) == 2               # 两个稳定轨迹
    for hid, labels in labels_by_id.items():
        assert len(set(labels)) == 1            # 各自标签全程稳定（无跳变）
    ids = list(labels_by_id)
    assert labels_by_id[ids[0]][0] != labels_by_id[ids[1]][0]  # 左右分开
