"""帧输入源：摄像头 / 视频文件 / 图片文件。

统一产出 (frame, frame_index, timestamp_ms) 三元组，消费方与具体源解耦。
timestamp_ms 优先取 OpenCV 帧时间戳（CAP_PROP_POS_MSEC），不可用时回退系统单调时钟。
"""

import time
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from signbridge.core.errors import SourceOpenError

FrameTuple = tuple[np.ndarray, int, float]


class FrameSource(Protocol):
    """帧源协议：可迭代产出 (BGR 帧, 帧索引, 时间戳毫秒)。"""

    def __iter__(self) -> Iterator[FrameTuple]: ...

    def __next__(self) -> FrameTuple: ...

    def close(self) -> None: ...


def _open_capture(camera_id: int):
    """可被测试替换的摄像头打开函数。"""
    return cv2.VideoCapture(camera_id)


class CameraSource:
    """摄像头帧源。帧读取失败（断开）时迭代结束。"""

    def __init__(self, camera_id: int = 0) -> None:
        self._cap = _open_capture(camera_id)
        if self._cap is None or not self._cap.isOpened():
            raise SourceOpenError(f"无法打开摄像头 #{camera_id}")
        self._index = 0

    def __iter__(self) -> "CameraSource":
        return self

    def __next__(self) -> FrameTuple:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise StopIteration
        ts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        if ts <= 0:
            ts = time.monotonic() * 1000
        result = (frame, self._index, ts)
        self._index += 1
        return result

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoSource:
    """视频文件帧源。meta 提供 width/height/fps/frame_count。"""

    def __init__(self, path: str | Path) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise SourceOpenError(f"无法打开视频文件: {path}")
        self._index = 0
        self.meta = {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def __iter__(self) -> "VideoSource":
        return self

    def __next__(self) -> FrameTuple:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise StopIteration
        ts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        if ts <= 0:
            ts = time.monotonic() * 1000
        result = (frame, self._index, ts)
        self._index += 1
        return result

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageSource:
    """单张图片帧源；repeat=True 时循环输出同一帧。"""

    def __init__(self, path: str | Path, repeat: bool = False) -> None:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SourceOpenError(f"无法读取图片: {path}")
        self._frame = frame
        self._repeat = repeat
        self._index = 0
        self._done = False

    def __iter__(self) -> "ImageSource":
        return self

    def __next__(self) -> FrameTuple:
        if self._done:
            raise StopIteration
        if not self._repeat:
            self._done = True
        result = (self._frame, self._index, 0.0)
        self._index += 1
        return result

    def close(self) -> None:
        self._done = True
