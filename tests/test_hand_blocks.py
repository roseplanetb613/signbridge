import numpy as np
import pytest

from signbridge.core.landmarks import Hand, Landmark
from signbridge.hands.sequence import classify_two_hands, to_normalized


def _hand(handedness, x):
    lms = tuple(
        Landmark(x=x + 0.01 * i, y=0.5, z=0.0) for i in range(21)
    )
    world = tuple(
        Landmark(x=0.0 + 0.01 * i, y=0.1, z=0.2) for i in range(21)
    )
    return Hand(landmarks=lms, world_landmarks=world,
                handedness=handedness, score=0.9)


def test_classify_by_handedness():
    left = _hand("Left", 0.2)
    right = _hand("Right", 0.8)
    b0, b1 = classify_two_hands(left, right)
    assert b0 is left and b1 is right
    b0, b1 = classify_two_hands(right, left)
    assert b0 is left and b1 is right


def test_classify_conflict_by_position():
    left_a = _hand("Left", 0.8)   # 双同侧，位置右
    left_b = _hand("Left", 0.2)   # 位置左
    b0, b1 = classify_two_hands(left_a, left_b)
    assert b0 is left_b and b1 is left_a


def test_to_normalized_wrist_origin():
    hand = _hand("Right", 0.5)
    pts = to_normalized(hand)
    assert pts.shape == (21, 3)
    assert np.allclose(pts[0], 0.0, atol=1e-6)   # 腕点 = 原点
    assert pts[1][0] == pytest.approx(0.01)       # 相对腕点偏移
