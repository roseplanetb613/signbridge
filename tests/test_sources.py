import cv2
import numpy as np
import pytest

from signbridge.core.errors import SourceOpenError
from signbridge.hands.sources import CameraSource, ImageSource, VideoSource


def test_image_source_yields_single_frame(hand_open_path):
    src = ImageSource(hand_open_path)
    frames = list(src)
    assert len(frames) == 1
    frame, index, ts = frames[0]
    assert frame.shape[2] == 3 and frame.dtype == np.uint8
    assert index == 0
    assert ts == 0.0
    src.close()


def test_image_source_missing_file_raises(tmp_path):
    with pytest.raises(SourceOpenError):
        ImageSource(tmp_path / "nope.jpg")


def test_image_source_repeat(hand_open_path):
    src = ImageSource(hand_open_path, repeat=True)
    frames = [next(src) for _ in range(3)]
    assert len(frames) == 3
    assert frames[0][1] == 0 and frames[2][1] == 2
    src.close()
    with pytest.raises(StopIteration):
        next(src)


def test_video_source_yields_all_frames(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    src = VideoSource(path)
    assert src.meta["frame_count"] == 5
    assert src.meta["width"] == 64 and src.meta["height"] == 48
    frames = list(src)
    assert len(frames) == 5
    assert frames[0][1] == 0 and frames[4][1] == 4
    src.close()
    with pytest.raises(StopIteration):
        next(src)


def test_video_source_missing_file_raises(tmp_path):
    with pytest.raises(SourceOpenError):
        VideoSource(tmp_path / "nope.avi")


class _FakeCapture:
    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True

    def isOpened(self):
        return True

    def get(self, prop):
        return 0.0


def test_camera_source_yields_frames(monkeypatch):
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
    fake = _FakeCapture(frames)
    monkeypatch.setattr("signbridge.hands.sources._open_capture", lambda cam_id: fake)

    src = CameraSource(0)
    got = list(src)
    assert len(got) == 3
    src.close()
    assert fake.released


def test_camera_source_open_failure_raises(monkeypatch):
    monkeypatch.setattr("signbridge.hands.sources._open_capture", lambda cam_id: None)
    with pytest.raises(SourceOpenError):
        CameraSource(0)
