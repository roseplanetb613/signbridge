"""SignBridge: 手语翻译项目 —— MediaPipe 关键点组件库。"""

__version__ = "0.5.0"

from signbridge.core.errors import SignBridgeError
from signbridge.core.features import (
    DistanceFeatureVerifier,
    FeatureExtractor,
    FeatureVerifier,
    HandShapeFeature,
)
from signbridge.core.graphs import build_hand_graph
from signbridge.core.landmarks import (
    HAND_CONNECTIONS,
    HAND_LANDMARK_NAMES,
    Hand,
    HandFrame,
    Landmark,
)
from signbridge.core.matching import FeatureHungarianMatcher, HungarianMatcher
from signbridge.core.smoothing import OneEuroSmoother
from signbridge.hands.detector import HandDetector
from signbridge.hands.draw import draw_landmarks
from signbridge.hands.sequence import HandSequence, HandSequenceBuffer
from signbridge.hands.sources import CameraSource, ImageSource, VideoSource
from signbridge.models.protocol import SkeletonClassifier
from signbridge.models.stgcn import STGCN
from signbridge.models.stgcn_ctc import STGCNCTC, STGCNCTCEmb
from signbridge.models.stgcn_ctc_tf import STGCNCTCTF, SeqKD
from signbridge.models.stgcn_fusion import FusionSTGCNCTC

__all__ = [
    "__version__",
    "SignBridgeError",
    "HAND_CONNECTIONS",
    "HAND_LANDMARK_NAMES",
    "Hand",
    "HandFrame",
    "Landmark",
    "FeatureExtractor",
    "FeatureVerifier",
    "HandShapeFeature",
    "DistanceFeatureVerifier",
    "HungarianMatcher",
    "FeatureHungarianMatcher",
    "OneEuroSmoother",
    "build_hand_graph",
    "SkeletonClassifier",
    "STGCN",
    "STGCNCTC",
    "STGCNCTCEmb",
    "STGCNCTCTF",
    "SeqKD",
    "FusionSTGCNCTC",
    "HandDetector",
    "draw_landmarks",
    "HandSequence",
    "HandSequenceBuffer",
    "CameraSource",
    "ImageSource",
    "VideoSource",
]
