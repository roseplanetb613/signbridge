import numpy as np

from signbridge.core.landmarks import Hand, HandFrame, Landmark
from signbridge.hands.draw import draw_landmarks


def _open_hand_frame() -> HandFrame:
    lms = tuple(
        Landmark(x=0.1 + 0.03 * i, y=0.2 + 0.02 * (i % 5), z=0.0)
        for i in range(21)
    )
    return HandFrame(hands=(Hand(landmarks=lms, handedness="Right", score=0.9),))


def test_empty_frame_returns_unchanged_copy():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = draw_landmarks(frame, HandFrame())
    assert out is not frame
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    assert np.array_equal(out, frame)


def test_drawn_frame_keeps_shape_and_does_not_mutate_input():
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    before = frame.copy()
    out = draw_landmarks(frame, _open_hand_frame())
    assert out.shape == frame.shape and out.dtype == frame.dtype
    assert np.array_equal(frame, before)
    assert not np.array_equal(out, frame)


def test_handedness_colors_differ():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    lms = _open_hand_frame().hands[0].landmarks
    left = HandFrame(hands=(Hand(landmarks=lms, handedness="Left", score=0.9),))
    right = HandFrame(hands=(Hand(landmarks=lms, handedness="Right", score=0.9),))
    assert not np.array_equal(draw_landmarks(frame, left), draw_landmarks(frame, right))


def test_explicit_color_used():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = draw_landmarks(frame, _open_hand_frame(), color=(0, 0, 255))
    assert np.any(out[:, :, 0] > 0)
