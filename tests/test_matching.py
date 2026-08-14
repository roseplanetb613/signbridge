import numpy as np
import pytest

from signbridge.core.matching import HungarianMatcher, Matcher, Matching


def _pts(*xy):
    return np.array(xy, dtype=np.float32)


def test_protocol_exposes_match():
    assert hasattr(Matcher, "match")


def test_basic_assignment():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(_pts((0.0, 0.0), (1.0, 1.0)), _pts((0.1, 0.1), (0.9, 0.9)))
    assert set(res.matched) == {(0, 0), (1, 1)}
    assert res.unmatched_current == ()
    assert res.unmatched_previous == ()


def test_cross_swap_matches_by_nearest():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(_pts((0.9, 0.9), (0.1, 0.1)), _pts((0.1, 0.1), (0.9, 0.9)))
    assert set(res.matched) == {(0, 1), (1, 0)}  # 每只手仍绑定自己的轨迹


def test_distance_beyond_threshold_unmatched():
    m = HungarianMatcher(distance_threshold=0.3)
    res = m.match(_pts((0.0, 0.0)), _pts((0.9, 0.9)))
    assert res.matched == ()
    assert res.unmatched_current == (0,)
    assert res.unmatched_previous == (0,)


def test_asymmetric_counts():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(
        _pts((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)),  # 当前 3 手
        _pts((0.0, 0.0), (1.0, 1.0)),              # 轨迹 2 条
    )
    assert len(res.matched) == 2
    assert len(res.unmatched_current) == 1
    assert res.unmatched_previous == ()


def test_empty_side():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(np.zeros((0, 2), dtype=np.float32), _pts((0.0, 0.0)))
    assert res.matched == ()
    assert res.unmatched_current == ()
    assert res.unmatched_previous == (0,)


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        HungarianMatcher(distance_threshold=-0.1)
