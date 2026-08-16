"""手部关键点检测器（封装 MediaPipe Tasks API 的 HandLandmarker）。"""

import time
from pathlib import Path

import cv2
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

    refine_roi=True 时启用 ROI 放大精化（改善小手/远端手部识别）：
    全图检测定位 → 裁剪手部包围盒（含 margin）→ 放大重检测 → 精细关键点换算回原图；
    第二遍失败时回退第一遍结果。
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | Path | None = None,
        refine_roi: bool = False,
        roi_target_size: int = 256,
        roi_margin: float = 0.35,
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
        self.refine_roi = refine_roi
        self._roi_target_size = roi_target_size
        self._roi_margin = roi_margin
        self._closed = False
        self._frame_index = 0

    def detect(self, frame) -> HandFrame:
        """检测一帧 BGR 图像（H×W×3 uint8），返回 HandFrame（无手时 hands 为空）。"""
        if self._closed:
            raise RuntimeError("HandDetector 已 close()，不可再检测")
        # OpenCV 帧是 BGR，MediaPipe SRGB 期望 RGB —— 通道顺序错误会导致检测不到手
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if self.refine_roi and result.hand_landmarks:
            fh, fw = frame.shape[:2]
            hands = tuple(
                self._refine_hand(frame, fw, fh, h, w, c)
                for h, w, c in zip(
                    result.hand_landmarks,
                    result.hand_world_landmarks,
                    result.handedness,
                )
            )
        else:
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

    def _refine_hand(self, frame, fw, fh, landmarks, world_landmarks, handedness):
        """ROI 放大重检测一只手；失败回退第一遍结果。"""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        bw = (max(xs) - min(xs)) * fw
        bh = (max(ys) - min(ys)) * fh
        if bw < 8 or bh < 8:
            return _to_hand(landmarks, world_landmarks, handedness)
        mx = bw * self._roi_margin
        my = bh * self._roi_margin
        x0 = max(int(min(xs) * fw - mx), 0)
        x1 = min(int(max(xs) * fw + mx), fw)
        y0 = max(int(min(ys) * fh - my), 0)
        y1 = min(int(max(ys) * fh + my), fh)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return _to_hand(landmarks, world_landmarks, handedness)
        roi = frame[y0:y1, x0:x1]
        scale = self._roi_target_size / max(x1 - x0, y1 - y0)
        rw = max(int((x1 - x0) * scale), 8)
        rh = max(int((y1 - y0) * scale), 8)
        resized = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._landmarker.detect(img)
        if not res.hand_landmarks:
            return _to_hand(landmarks, world_landmarks, handedness)
        lms2 = res.hand_landmarks[0]
        wl = (
            res.hand_world_landmarks[0]
            if res.hand_world_landmarks else world_landmarks
        )
        hd = res.handedness[0] if res.handedness else handedness
        # 换算回原图归一化空间（用 ROI 原始尺寸，非放大尺寸）
        norm = tuple(
            Landmark(
                x=(x0 + lm.x * (x1 - x0)) / fw,
                y=(y0 + lm.y * (y1 - y0)) / fh,
                z=lm.z * (x1 - x0) / fw,
            )
            for lm in lms2
        )
        return Hand(
            landmarks=norm,
            world_landmarks=tuple(_to_landmark(lm) for lm in wl),
            handedness=hd[0].category_name if hd else "Unknown",
            score=hd[0].score if hd else 0.0,
        )

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
