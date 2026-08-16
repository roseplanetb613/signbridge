"""signbridge.core: 各组件共享的基础设施（数据结构、图谱常量、异常体系）。"""

from signbridge.core.features import (
    DistanceFeatureVerifier,
    FeatureExtractor,
    FeatureVerifier,
    HandShapeFeature,
)
from signbridge.core.graphs import (
    build_adjacency,
    build_block_diagonal_graph,
    build_hand_graph,
    normalize_adjacency,
)
from signbridge.core.landmarks import (
    HAND_CONNECTIONS,
    HAND_LANDMARK_NAMES,
    Hand,
    HandFrame,
    Landmark,
)
from signbridge.core.matching import (
    FeatureHungarianMatcher,
    HandDescriptor,
    HungarianMatcher,
    Matcher,
    Matching,
)
from signbridge.core.smoothing import LandmarkSmoother, OneEuroSmoother

__all__ = [
    "HAND_CONNECTIONS",
    "HAND_LANDMARK_NAMES",
    "Hand",
    "HandFrame",
    "Landmark",
    "FeatureExtractor",
    "FeatureVerifier",
    "HandShapeFeature",
    "DistanceFeatureVerifier",
    "build_adjacency",
    "build_block_diagonal_graph",
    "build_hand_graph",
    "normalize_adjacency",
    "HandDescriptor",
    "HungarianMatcher",
    "FeatureHungarianMatcher",
    "Matcher",
    "Matching",
    "LandmarkSmoother",
    "OneEuroSmoother",
]
