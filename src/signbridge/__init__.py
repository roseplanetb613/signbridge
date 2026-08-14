"""SignBridge: 手语翻译项目 —— MediaPipe 关键点组件库。"""

__version__ = "0.1.0"

from signbridge.core.errors import SignBridgeError
from signbridge.core.landmarks import (
    HAND_CONNECTIONS,
    HAND_LANDMARK_NAMES,
    Hand,
    HandFrame,
    Landmark,
)
from signbridge.core.matching import HungarianMatcher
from signbridge.core.smoothing import OneEuroSmoother
from signbridge.hands.detector import HandDetector
from signbridge.hands.draw import draw_landmarks
from signbridge.hands.sequence import HandSequence, HandSequenceBuffer
from signbridge.hands.sources import CameraSource, ImageSource, VideoSource

__all__ = [
    "__version__",
    "SignBridgeError",
    "HAND_CONNECTIONS",
    "HAND_LANDMARK_NAMES",
    "Hand",
    "HandFrame",
    "Landmark",
    "HungarianMatcher",
    "OneEuroSmoother",
    "HandDetector",
    "draw_landmarks",
    "HandSequence",
    "HandSequenceBuffer",
    "CameraSource",
    "ImageSource",
    "VideoSource",
]
