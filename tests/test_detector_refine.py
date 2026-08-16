"""ROI 两级候选检测（refine_roi）单元测试：低阈值候选 + 放大确认 + 噪声丢弃。"""

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


def _small_hand_lms():
    """全图小手的 landmarks：集中在画面中央约 40px 区域。"""
    return _norm_landmarks(
        [0.44 + 0.03 * (i % 5) for i in range(21)],
        [0.42 + 0.03 * (i % 7) for i in range(21)],
    )


def _refined_lms():
    """ROI 确认的精细结果（在 ROI 内占较大范围）。"""
    return _norm_landmarks(
        [0.2 + 0.6 * (i % 5) / 4 for i in range(21)],
        [0.2 + 0.6 * (i % 7) / 6 for i in range(21)],
    )


def _two_stage_factory(candidate_ok=True, confirm_ok=True):
    """fake _create_landmarker：按 min_detection_confidence 区分候选/确认。"""
    first_lms = _small_hand_lms()

    def factory(model_path, num_hands, min_detection, min_tracking):
        def detect(img):
            if min_detection < 0.2:      # 候选（低阈值）→ 全图找手
                if not candidate_ok:
                    return SimpleNamespace(
                        hand_landmarks=[], hand_world_landmarks=[], handedness=[]
                    )
                return SimpleNamespace(
                    hand_landmarks=[first_lms],
                    hand_world_landmarks=[first_lms],
                    handedness=[[_fake_category("Right", 0.25)]],
                )
            # 确认（正常阈值）→ ROI 输入
            if not confirm_ok:
                return SimpleNamespace(
                    hand_landmarks=[], hand_world_landmarks=[], handedness=[]
                )
            return SimpleNamespace(
                hand_landmarks=[_refined_lms()],
                hand_world_landmarks=[_refined_lms()],
                handedness=[[_fake_category("Right", 0.98)]],
            )

        return SimpleNamespace(detect=detect, close=lambda: None)

    return factory


def _make_detector(monkeypatch, tmp_path, factory, refine=True):
    model = tmp_path / "m.task"
    model.write_bytes(b"x")
    monkeypatch.setattr("signbridge.hands.detector._create_landmarker", factory)
    return HandDetector(model_path=model, refine_roi=refine,
                        min_detection_confidence=0.5)


def _frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_refine_uses_confirmed_stage_coords(monkeypatch, tmp_path):
    det = _make_detector(monkeypatch, tmp_path, _two_stage_factory())
    hf = det.detect(_frame())
    assert len(hf.hands) == 1
    hand = hf.hands[0]
    # 精细结果（ROI 内 0.2~0.8）换算回原图应落在手区域附近
    xs = [lm.x for lm in hand.landmarks]
    assert max(xs) > 0.44 and min(xs) < 0.56
    assert 0.30 < min(xs) < 0.50
    assert hand.score > 0.9       # 采用确认阶段的高置信度


def test_refine_falls_back_to_candidate_when_confirm_fails(monkeypatch, tmp_path):
    det = _make_detector(monkeypatch, tmp_path,
                         _two_stage_factory(confirm_ok=False))
    hf = det.detect(_frame())
    assert len(hf.hands) == 1                     # 确认失败 → 回退候选
    assert hf.hands[0].score == pytest.approx(0.25)   # 候选的置信度


def test_refine_no_candidate_stays_empty(monkeypatch, tmp_path):
    det = _make_detector(monkeypatch, tmp_path,
                         _two_stage_factory(candidate_ok=False))
    assert det.detect(_frame()).hands == ()


def test_refine_off_uses_single_stage(monkeypatch, tmp_path):
    # refine 关闭：只用正常阈值 landmarker（无候选阶段）
    called = []

    def factory(model_path, num_hands, min_detection, min_tracking):
        called.append(min_detection)

        def detect(img):
            return SimpleNamespace(
                hand_landmarks=[_small_hand_lms()],
                hand_world_landmarks=[_small_hand_lms()],
                handedness=[[_fake_category("Right", 0.6)]],
            )

        return SimpleNamespace(detect=detect, close=lambda: None)

    det = _make_detector(monkeypatch, tmp_path, factory, refine=False)
    hf = det.detect(_frame())
    assert len(called) == 1                     # 只创建一个 landmarker
    assert hf.hands[0].score == pytest.approx(0.6)


def test_refine_creates_two_landmarkers(monkeypatch, tmp_path):
    created = []

    def factory(model_path, num_hands, min_detection, min_tracking):
        created.append(min_detection)

        def detect(img):
            return SimpleNamespace(
                hand_landmarks=[], hand_world_landmarks=[], handedness=[]
            )

        return SimpleNamespace(detect=detect, close=lambda: None)

    det = _make_detector(monkeypatch, tmp_path, factory, refine=True)
    det.detect(_frame())
    assert len(created) == 2                    # 确认(0.5) + 候选(0.15)
    assert created[0] == pytest.approx(0.5)
    assert created[1] == pytest.approx(0.15)
