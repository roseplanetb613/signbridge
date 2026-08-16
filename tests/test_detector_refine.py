"""ROI 放大精化（refine_roi）单元测试：两层检测 mock，验证坐标换算与回退。"""

from types import SimpleNamespace

import numpy as np
import pytest

from signbridge.core.landmarks import HandFrame
from signbridge.hands.detector import HandDetector


def _fake_category(name: str, score: float):
    from mediapipe.tasks.python.components.containers.category import Category

    return Category(index=0, score=score, display_name="", category_name=name)


def _norm_landmarks(xs, ys):
    from mediapipe.tasks.python.components.containers.landmark import (
        NormalizedLandmark,
    )

    return [NormalizedLandmark(x=x, y=y, z=0.0) for x, y in zip(xs, ys)]


def _small_hand_lms(w=320, h=240):
    """全图小手的 landmarks：集中在画面中央约 40px 区域。"""
    return _norm_landmarks(
        [0.44 + 0.03 * (i % 5) for i in range(21)],
        [0.42 + 0.03 * (i % 7) for i in range(21)],
    )


def _refined_lms():
    """ROI 重检测的精细结果（在 ROI 内占较大范围）。"""
    return _norm_landmarks(
        [0.2 + 0.6 * (i % 5) / 4 for i in range(21)],
        [0.2 + 0.6 * (i % 7) / 6 for i in range(21)],
    )


def _two_stage_landmarker(second_fail: bool = False):
    """两层检测 fake：按输入图像尺寸区分全图（320x240）与 ROI。"""
    first_lms = _small_hand_lms()

    def detect(img):
        if img.width == 320 and img.height == 240:      # 第一遍：全图
            return SimpleNamespace(
                hand_landmarks=[first_lms],
                hand_world_landmarks=[first_lms],
                handedness=[[_fake_category("Right", 0.6)]],
            )
        # 第二遍：ROI（尺寸任意）
        if second_fail:
            return SimpleNamespace(
                hand_landmarks=[], hand_world_landmarks=[], handedness=[]
            )
        return SimpleNamespace(
            hand_landmarks=[_refined_lms()],
            hand_world_landmarks=[_refined_lms()],
            handedness=[[_fake_category("Right", 0.98)]],
        )

    return detect


@pytest.fixture
def refine_detector(monkeypatch, tmp_path, second_fail=False):
    model = tmp_path / "m.task"
    model.write_bytes(b"x")

    def make(*args, **kwargs):
        return SimpleNamespace(
            detect=_two_stage_landmarker(second_fail), close=lambda: None
        )

    monkeypatch.setattr("signbridge.hands.detector._create_landmarker", make)
    return HandDetector(model_path=model, refine_roi=True)


def _frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_refine_uses_second_stage_coords(refine_detector):
    hf = refine_detector.detect(_frame())
    assert len(hf.hands) == 1
    hand = hf.hands[0]
    # 精细结果（ROI 内 0.2~0.8 范围）换算回原图应落在 ROI 附近区域
    # 全图小手 bbox ≈ x∈[0.44,0.56], y∈[0.42,0.60]；margin 后 ROI 略大
    xs = [lm.x for lm in hand.landmarks]
    ys = [lm.y for lm in hand.landmarks]
    assert max(xs) > 0.44 and min(xs) < 0.56     # 仍在手区域
    assert 0.30 < min(xs) < 0.50                 # 未被放大到全图角落
    assert hand.score > 0.9                       # 采用第二遍的高置信度


def test_refine_fallback_to_first_stage(monkeypatch, tmp_path):
    model = tmp_path / "m.task"
    model.write_bytes(b"x")
    monkeypatch.setattr(
        "signbridge.hands.detector._create_landmarker",
        lambda *a, **k: SimpleNamespace(
            detect=_two_stage_landmarker(second_fail=True), close=lambda: None
        ),
    )
    detector = HandDetector(model_path=model, refine_roi=True)
    hf = detector.detect(_frame())
    assert len(hf.hands) == 1
    # 第二遍失败 → 回退第一遍（score 0.6，坐标 = 全图检测结果）
    assert hf.hands[0].score == pytest.approx(0.6)


def test_refine_off_behaves_legacy(monkeypatch, tmp_path):
    model = tmp_path / "m.task"
    model.write_bytes(b"x")
    monkeypatch.setattr(
        "signbridge.hands.detector._create_landmarker",
        lambda *a, **k: SimpleNamespace(
            detect=_two_stage_landmarker(), close=lambda: None
        ),
    )
    detector = HandDetector(model_path=model)   # refine_roi 默认 False
    hf = detector.detect(_frame())
    assert hf.hands[0].score == pytest.approx(0.6)   # 只有第一遍


def test_refine_no_hands_stays_empty(monkeypatch, tmp_path):
    model = tmp_path / "m.task"
    model.write_bytes(b"x")

    def detect(img):
        return SimpleNamespace(
            hand_landmarks=[], hand_world_landmarks=[], handedness=[]
        )

    monkeypatch.setattr(
        "signbridge.hands.detector._create_landmarker",
        lambda *a, **k: SimpleNamespace(detect=detect, close=lambda: None),
    )
    detector = HandDetector(model_path=model, refine_roi=True)
    assert detector.detect(_frame()).hands == ()
