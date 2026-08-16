import numpy as np
import pytest

from signbridge.core.segmentation import extract_segments


def test_no_valid_frames():
    assert extract_segments(np.zeros(10, dtype=bool)) == []


def test_single_segment():
    valid = np.array([0, 1, 1, 1, 1, 0, 0], dtype=bool)
    assert extract_segments(valid, min_len=3) == [(1, 4)]


def test_short_segment_dropped():
    valid = np.array([1, 1, 0, 0, 1, 1, 1, 1], dtype=bool)
    assert extract_segments(valid, min_len=3, merge_gap=0) == [(4, 4)]


def test_gap_merge():
    valid = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1, 1], dtype=bool)
    # 缝隙 1 帧（idx3）与 1 帧（idx6）均 ≤ merge_gap=2 → 合并为一段
    segs = extract_segments(valid, min_len=5, merge_gap=2)
    assert segs == [(0, 10)]


def test_gap_too_large_not_merged():
    valid = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    segs = extract_segments(valid, min_len=3, merge_gap=1)
    assert segs == [(0, 3), (6, 3)]


def test_full_valid():
    valid = np.ones(20, dtype=bool)
    assert extract_segments(valid, min_len=5) == [(0, 20)]
