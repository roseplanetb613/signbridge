import numpy as np

from signbridge.core.landmarks import Hand, HandFrame, Landmark
from signbridge.hands.draw import draw_landmarks_depth


def _hand_frame(handedness: str, z_values) -> HandFrame:
    lms = tuple(
        Landmark(x=0.1 + 0.03 * i, y=0.2 + 0.02 * (i % 5), z=z)
        for i, z in enumerate(z_values)
    )
    return HandFrame(
        hands=(Hand(landmarks=lms, handedness=handedness, score=0.9),)
    )


def _frame(h=240, w=320):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_left_hand_is_blue():
    out = draw_landmarks_depth(_frame(), _hand_frame("Left", [0.0] * 21))
    blue_vs_red = out[:, :, 0].astype(int) - out[:, :, 2].astype(int)
    assert blue_vs_red.max() > 50


def test_right_hand_is_green():
    out = draw_landmarks_depth(_frame(), _hand_frame("Right", [0.0] * 21))
    green_vs_red = out[:, :, 1].astype(int) - out[:, :, 2].astype(int)
    assert green_vs_red.max() > 50


def test_nearer_point_brighter_than_farther():
    # 前 15 点 z=0.5（远），后 6 点 z=-0.5（近）；手内 min-max 归一化后近点亮、远点暗
    zs = [0.5] * 15 + [-0.5] * 6
    hf = _hand_frame("Left", zs)
    out = draw_landmarks_depth(_frame(), hf)
    h, w = out.shape[:2]
    far_pt = (int(0.1 * w), int(0.2 * h))                      # 点 0（远）
    near_pt = (int((0.1 + 0.03 * 15) * w), int((0.2 + 0.02 * (15 % 5)) * h))  # 点 15（近）
    assert int(out[near_pt[1], near_pt[0], 0]) > int(out[far_pt[1], far_pt[0], 0])


def test_input_frame_not_mutated():
    frame = np.full((240, 320, 3), 100, dtype=np.uint8)
    before = frame.copy()
    draw_landmarks_depth(frame, _hand_frame("Left", [0.0] * 21))
    assert np.array_equal(frame, before)


def test_empty_frame_returns_unchanged_copy():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = draw_landmarks_depth(frame, HandFrame())
    assert out is not frame
    assert out.shape == frame.shape and out.dtype == frame.dtype
    assert np.array_equal(out, frame)
