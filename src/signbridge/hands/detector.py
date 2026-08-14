"""手部关键点检测器（封装 MediaPipe Tasks API 的 HandLandmarker）。"""

import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from signbridge.core.errors import InvalidArgumentError, ModelNotFoundError
from signbridge.core.landmarks import Hand, HandFrame, Landmark
from signbridge.hands.model import ensure_model

_MAX_HANDS_ALLOWED = (1, 2)


def _create_landmarker(model_path, num_hands, min_detection, min_tracking):
    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection,
        min_hand_presence_confidence=min_detection,
        min_tracking_confidence=min_tracking,
    )
    return vision.HandLandmarker.create_from_options(options)


def _to_landmark(lm) -> Landmark:
    return Landmark(x=lm.x, y=lm.y, z=lm.z)


def _to_hand(landmarks, world_landmarks, handedness) -> Hand:
    name = handedness[0].category_name if handedness else "Unknown"
    score = handedness[0].score if handedness else 0.0
    return Hand(
        landmarks=tuple(_to_landmark(lm) for lm in landmarks),
        world_landmarks=tuple(_to_landmark(lm) for lm in world_landmarks),
        handedness=name,
        score=score,
    )


class HandDetector:
    """BGR 帧 → HandFrame 的手部关键点检测器。

    无帧间历史状态；仅维护自增帧计数与单调时钟时间戳（时序缓冲是后续步骤职责）。
    支持 with 语句；close() 释放底层资源。
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | Path | None = None,
    ) -> None:
        if max_num_hands not in _MAX_HANDS_ALLOWED:
            raise InvalidArgumentError(
                f"max_num_hands 必须是 1 或 2，收到 {max_num_hands!r}"
            )
        if model_path is None:
            model_path = ensure_model()
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelNotFoundError(
                f"模型文件不存在: {model_path}。可运行 "
                "`python -m signbridge.hands.cli --download-model` 下载。"
            )
        self._landmarker = _create_landmarker(
            model_path,
            max_num_hands,
            min_detection_confidence,
            min_tracking_confidence,
        )
        self._closed = False
        self._frame_index = 0

    def detect(self, frame) -> HandFrame:
        """检测一帧 BGR 图像（H×W×3 uint8），返回 HandFrame（无手时 hands 为空）。"""
        if self._closed:
            raise RuntimeError("HandDetector 已 close()，不可再检测")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect(image)
        hands = tuple(
            _to_hand(h, w, c)
            for h, w, c in zip(
                result.hand_landmarks,
                result.hand_world_landmarks,
                result.handedness,
            )
        )
        hand_frame = HandFrame(
            hands=hands,
            timestamp_ms=int(time.monotonic() * 1000),
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return hand_frame

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
