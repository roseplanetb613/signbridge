"""手部关键点数据结构与 21 点图谱常量。

图谱与 MediaPipe 官方定义一致：
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
HAND_CONNECTIONS 即未来 ST-GCN 的图边。
"""

from dataclasses import dataclass

HAND_LANDMARK_NAMES: tuple[str, ...] = (
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),          # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),     # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),   # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),  # 小指
    (0, 17),                                 # 手掌（腕→小指根）
)


@dataclass(frozen=True)
class Landmark:
    """单个关键点坐标。图像坐标模式下 x/y ∈ [0,1]、z 为相对深度；world 模式下为米制。"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class Hand:
    """一只手的 21 个关键点。"""

    landmarks: tuple[Landmark, ...] = ()
    world_landmarks: tuple[Landmark, ...] = ()
    handedness: str = "Unknown"
    score: float = 0.0


@dataclass(frozen=True)
class HandFrame:
    """一帧的检测结果：0~N 只手。"""

    hands: tuple[Hand, ...] = ()
    timestamp_ms: int = 0
    frame_index: int = 0
