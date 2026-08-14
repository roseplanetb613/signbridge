from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from signbridge.core.errors import InvalidArgumentError, ModelNotFoundError
from signbridge.core.landmarks import HandFrame
from signbridge.hands.detector import HandDetector


def _fake_category(name: str, score: float):
    from mediapipe.tasks.python.components.containers.category import Category

    return Category(index=0, score=score, display_name="", category_name=name)


def _fake_landmarks(n=21):
    from mediapipe.tasks.python.components.containers.landmark import (
        NormalizedLandmark,
    )

    return [NormalizedLandmark(x=i / 20, y=0.5, z=0.0) for i in range(n)]


def _fake_result(n_hands=1):
    return SimpleNamespace(
        hand_landmarks=[_fake_landmarks() for _ in range(n_hands)],
        hand_world_landmarks=[_fake_landmarks() for _ in range(n_hands)],
        handedness=[[_fake_category("Left", 0.95)] for _ in range(n_hands)],
    )


@pytest.fixture
def fake_landmarker(monkeypatch):
    captured = {"close_called": False}

    def fake_create(*args, **kwargs):
        return SimpleNamespace(
            detect=lambda img: _fake_result(),
            close=lambda: captured.__setitem__("close_called", True),
        )

    monkeypatch.setattr("signbridge.hands.detector._create_landmarker", fake_create)
    return captured


def _bgr_frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _model_file(tmp_path) -> Path:
    model = tmp_path / "m.task"
    model.write_bytes(b"x")
    return model


def test_detect_returns_converted_hand_frame(fake_landmarker, tmp_path):
    detector = HandDetector(model_path=_model_file(tmp_path))
    result = detector.detect(_bgr_frame())
    assert isinstance(result, HandFrame)
    assert len(result.hands) == 1
    hand = result.hands[0]
    assert len(hand.landmarks) == 21
    assert len(hand.world_landmarks) == 21
    assert hand.handedness == "Left"
    assert hand.score == pytest.approx(0.95)
    assert result.frame_index == 0
    assert result.timestamp_ms > 0
    detector.close()


def test_detect_empty_result_yields_empty_hands(tmp_path, monkeypatch):
    def fake_create(*args, **kwargs):
        return SimpleNamespace(
            detect=lambda img: SimpleNamespace(
                hand_landmarks=[], hand_world_landmarks=[], handedness=[]
            ),
            close=lambda: None,
        )

    monkeypatch.setattr("signbridge.hands.detector._create_landmarker", fake_create)
    detector = HandDetector(model_path=_model_file(tmp_path))
    assert detector.detect(_bgr_frame()).hands == ()
    detector.close()


def test_detect_increments_frame_index(fake_landmarker, tmp_path):
    detector = HandDetector(model_path=_model_file(tmp_path))
    detector.detect(_bgr_frame())
    assert detector.detect(_bgr_frame()).frame_index == 1
    detector.close()


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        HandDetector(model_path=tmp_path / "nope.task")


def test_invalid_max_hands_raises(tmp_path):
    with pytest.raises(InvalidArgumentError):
        HandDetector(model_path=_model_file(tmp_path), max_num_hands=3)


def test_context_manager_closes(fake_landmarker, tmp_path):
    with HandDetector(model_path=_model_file(tmp_path)) as detector:
        detector.detect(_bgr_frame())
    assert fake_landmarker["close_called"] is True


def test_detect_after_close_raises(fake_landmarker, tmp_path):
    detector = HandDetector(model_path=_model_file(tmp_path))
    detector.close()
    with pytest.raises(RuntimeError):
        detector.detect(_bgr_frame())
