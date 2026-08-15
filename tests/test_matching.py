import numpy as np
import pytest

from signbridge.core.features import DistanceFeatureVerifier, HandShapeFeature
from signbridge.core.matching import (
    FeatureHungarianMatcher,
    HandDescriptor,
    HungarianMatcher,
    Matcher,
    Matching,
)


def _desc(*xy, feature=None):
    """构造 HandDescriptor 列表：_desc((x0,y0),(x1,y1), feature=...)"""
    out = []
    for i, p in enumerate(xy):
        feat = None if feature is None else feature[i]
        out.append(HandDescriptor(centroid=np.array(p, dtype=np.float32), feature=feat))
    return out


def _features(seed_a=0, seed_b=1):
    f = HandShapeFeature()
    a = f.extract(_pts21(seed_a))
    b = f.extract(_pts21(seed_b))
    return [a, b]


def _pts21(seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.1, 0.1, size=(21, 3)).astype(np.float32)
    pts[0] = (0.5, 0.5, 0.0)
    return pts


def test_protocol_exposes_match():
    assert hasattr(Matcher, "match")


def test_basic_assignment():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(_desc((0.0, 0.0), (1.0, 1.0)), _desc((0.1, 0.1), (0.9, 0.9)))
    assert set(res.matched) == {(0, 0), (1, 1)}
    assert res.unmatched_current == ()
    assert res.unmatched_previous == ()


def test_cross_swap_matches_by_nearest():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(_desc((0.9, 0.9), (0.1, 0.1)), _desc((0.1, 0.1), (0.9, 0.9)))
    assert set(res.matched) == {(0, 1), (1, 0)}


def test_distance_beyond_threshold_unmatched():
    m = HungarianMatcher(distance_threshold=0.3)
    res = m.match(_desc((0.0, 0.0)), _desc((0.9, 0.9)))
    assert res.matched == ()
    assert res.unmatched_current == (0,)
    assert res.unmatched_previous == (0,)


def test_asymmetric_counts():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(
        _desc((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)),
        _desc((0.0, 0.0), (1.0, 1.0)),
    )
    assert len(res.matched) == 2
    assert len(res.unmatched_current) == 1
    assert res.unmatched_previous == ()


def test_empty_side():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match([], _desc((0.0, 0.0)))
    assert res.matched == ()
    assert res.unmatched_current == ()
    assert res.unmatched_previous == (0,)


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        HungarianMatcher(distance_threshold=-0.1)


# ---- FeatureHungarianMatcher ----


def test_feature_matcher_position_near_behaves_like_position_only():
    m = FeatureHungarianMatcher(distance_threshold=0.5)
    res = m.match(
        _desc((0.0, 0.0), (1.0, 1.0), feature=_features()),
        _desc((0.1, 0.1), (0.9, 0.9), feature=_features()),
    )
    assert set(res.matched) == {(0, 0), (1, 1)}


def test_feature_recovery_far_but_same_shape():
    m = FeatureHungarianMatcher(confidence_threshold=0.85)
    fa = HandShapeFeature().extract(_pts21(0))
    cur = _desc((0.8, 0.5), feature=[fa])      # 画面另一侧
    prev = _desc((0.2, 0.5), feature=[fa])     # 同手形
    res = m.match(cur, prev)
    assert set(res.matched) == {(0, 0)}        # 特征恢复匹配
    assert res.unmatched_current == ()
    assert res.unmatched_previous == ()


def test_feature_recovery_rejects_different_shape():
    m = FeatureHungarianMatcher(confidence_threshold=0.85)
    fa = HandShapeFeature().extract(_pts21(0))
    fb = HandShapeFeature().extract(_pts21(1))
    cur = _desc((0.8, 0.5), feature=[fb])      # 异手形
    prev = _desc((0.2, 0.5), feature=[fa])
    res = m.match(cur, prev)
    assert res.matched == ()
    assert res.unmatched_current == (0,)
    assert res.unmatched_previous == (0,)


def test_feature_recovery_requires_confidence_threshold():
    m = FeatureHungarianMatcher(confidence_threshold=0.999)
    fa = HandShapeFeature().extract(_pts21(0))
    cur = _desc((0.8, 0.5), feature=[fa])
    prev = _desc((0.2, 0.5), feature=[fa + 0.1])  # 扰动后置信度低于阈值
    res = m.match(cur, prev)
    assert res.matched == ()


def test_feature_recovery_skipped_when_feature_none():
    m = FeatureHungarianMatcher(confidence_threshold=0.85)
    cur = _desc((0.8, 0.5))   # feature=None
    prev = _desc((0.2, 0.5))
    res = m.match(cur, prev)
    assert res.matched == ()


def test_feature_recovery_greedy_no_double_match():
    m = FeatureHungarianMatcher(confidence_threshold=0.85)
    fa = HandShapeFeature().extract(_pts21(0))
    cur = _desc((0.8, 0.5), (0.85, 0.5), feature=[fa, fa])   # 两只同形新手
    prev = _desc((0.2, 0.5), feature=[fa])                    # 一条轨迹
    res = m.match(cur, prev)
    assert len(res.matched) == 1                              # 每边最多一次
    assert len(res.unmatched_current) == 1
    assert res.unmatched_previous == ()
