import pytest

from signbridge.core.landmarks import (
    HAND_CONNECTIONS,
    HAND_LANDMARK_NAMES,
    Hand,
    HandFrame,
    Landmark,
)


def test_landmark_dataclass_is_frozen():
    lm = Landmark(x=0.1, y=0.2, z=0.3)
    with pytest.raises(AttributeError):
        lm.x = 0.5


def test_hand_frame_defaults_to_empty():
    frame = HandFrame()
    assert frame.hands == ()
    assert frame.timestamp_ms == 0
    assert frame.frame_index == 0


def test_hand_frame_is_frozen():
    frame = HandFrame()
    with pytest.raises(AttributeError):
        frame.hands = ()


def test_landmark_names_have_21_entries():
    assert len(HAND_LANDMARK_NAMES) == 21


def test_landmark_names_exact():
    expected = (
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    )
    assert HAND_LANDMARK_NAMES == expected


def test_connections_have_21_edges():
    assert len(HAND_CONNECTIONS) == 21


def test_connections_exact():
    expected = {
        (0, 1), (1, 2), (2, 3), (3, 4),          # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),          # 食指
        (5, 9), (9, 10), (10, 11), (11, 12),     # 中指
        (9, 13), (13, 14), (14, 15), (15, 16),   # 无名指
        (13, 17), (17, 18), (18, 19), (19, 20),  # 小指
        (0, 17),                                 # 手掌（腕→小指根）
    }
    assert set(HAND_CONNECTIONS) == expected


def test_connections_indices_in_range_no_self_loops():
    for a, b in HAND_CONNECTIONS:
        assert 0 <= a <= 20 and 0 <= b <= 20
        assert a != b
