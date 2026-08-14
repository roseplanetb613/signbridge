"""真实模型 + 真实图片的集成测试。

首次运行会通过 ensure_model() 下载 7.8MB 模型到 ~/.cache/signbridge/（幂等，之后秒过）。
"""

import cv2
import numpy as np
import pytest

from signbridge.hands.detector import HandDetector


@pytest.fixture(scope="module")
def detector():
    with HandDetector(max_num_hands=2) as d:
        yield d


def test_detects_hand_in_open_palm(detector, hand_open_path):
    frame = cv2.imread(str(hand_open_path))
    result = detector.detect(frame)
    assert len(result.hands) >= 1
    hand = result.hands[0]
    assert len(hand.landmarks) == 21
    for lm in hand.landmarks:
        assert 0.0 <= lm.x <= 1.0
        assert 0.0 <= lm.y <= 1.0
    assert hand.handedness in ("Left", "Right")
    assert 0.0 <= hand.score <= 1.0


def test_detects_thumbs_up(detector, thumbs_up_path):
    frame = cv2.imread(str(thumbs_up_path))
    result = detector.detect(frame)
    assert len(result.hands) >= 1


def test_blank_image_has_no_hands(detector):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(blank)
    assert result.hands == ()
