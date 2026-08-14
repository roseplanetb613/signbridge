from pathlib import Path

import numpy as np
import pytest

from signbridge.core.landmarks import Hand, HandFrame, Landmark

ASSETS = Path(__file__).parent / "assets"
HAND_OPEN = ASSETS / "hand_open.jpg"
THUMBS_UP = ASSETS / "thumbs_up.jpg"


@pytest.fixture
def hand_open_path() -> Path:
    return HAND_OPEN


@pytest.fixture
def thumbs_up_path() -> Path:
    return THUMBS_UP


def _make_hand_frame(hands, ts=0):
    """构造测试用 HandFrame。

    hands: [(handedness, pts(21,3) numpy 或 None), ...]
    pts 为图像归一化坐标（x,y∈[0,1]）。
    """
    out = []
    for handedness, pts in hands:
        lms = tuple(
            Landmark(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts
        )
        out.append(Hand(landmarks=lms, handedness=handedness, score=0.9))
    return HandFrame(hands=tuple(out), timestamp_ms=ts, frame_index=0)


@pytest.fixture
def make_hand_frame():
    return _make_hand_frame


@pytest.fixture
def hand_pts():
    def _factory(center=(0.5, 0.5), seed=0):
        rng = np.random.default_rng(seed)
        pts = rng.uniform(-0.1, 0.1, size=(21, 3)).astype(np.float32)
        pts[:, 0] += center[0]
        pts[:, 1] += center[1]
        pts[0] = (center[0], center[1], 0.0)  # WRIST 在中心
        return pts

    return _factory
