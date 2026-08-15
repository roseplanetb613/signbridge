import numpy as np
import pytest

from signbridge.core.features import (
    DistanceFeatureVerifier,
    FeatureExtractor,
    FeatureVerifier,
    HandShapeFeature,
)


def _pts(seed=0, center=(0.5, 0.5)):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.1, 0.1, size=(21, 3)).astype(np.float32)
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    pts[0] = (center[0], center[1], 0.0)
    return pts


def test_protocols_expose_methods():
    assert hasattr(FeatureExtractor, "extract")
    assert hasattr(FeatureVerifier, "verify")


def test_output_is_210_dim():
    f = HandShapeFeature()
    vec = f.extract(_pts())
    assert vec.shape == (210,)
    assert vec.dtype == np.float32


def test_translation_invariant():
    f = HandShapeFeature()
    a = f.extract(_pts(center=(0.2, 0.5)))
    b = f.extract(_pts(center=(0.8, 0.5)))
    assert np.allclose(a, b, atol=1e-5)


def test_rotation_invariant():
    f = HandShapeFeature()
    pts = _pts()
    theta = np.pi / 4
    rot_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]], dtype=np.float32)
    rotated = pts @ rot_z.T
    a = f.extract(pts)
    b = f.extract(rotated)
    assert np.allclose(a, b, atol=1e-4)


def test_scale_invariant():
    f = HandShapeFeature()
    pts = _pts()
    a = f.extract(pts)
    b = f.extract(pts * 0.5)  # 整体缩放（腕点保持原点附近）
    assert np.allclose(a, b, atol=1e-4)


def test_different_shapes_are_far_apart():
    f = HandShapeFeature()
    a = f.extract(_pts(seed=0))
    b = f.extract(_pts(seed=1))
    d_ab = float(np.linalg.norm(a - b))
    d_aa = float(np.linalg.norm(a - f.extract(_pts(seed=0))))
    assert d_ab > 5 * d_aa


def test_verifier_same_feature_is_1():
    v = DistanceFeatureVerifier()
    fvec = HandShapeFeature().extract(_pts())
    assert v.verify(fvec, fvec) == pytest.approx(1.0)


def test_verifier_monotonic():
    v = DistanceFeatureVerifier(sigma=0.3)
    fvec = HandShapeFeature().extract(_pts())
    close = v.verify(fvec, fvec + 1e-3)
    far = v.verify(fvec, fvec + 0.5)
    assert close > far
    assert 0.0 <= close <= 1.0 and 0.0 <= far <= 1.0


def test_verifier_invalid_sigma_raises():
    with pytest.raises(ValueError):
        DistanceFeatureVerifier(sigma=0.0)
