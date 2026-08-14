"""手部关键点可视化：在帧副本上绘制关键点、骨骼连线与标签。"""

import cv2
import numpy as np

from signbridge.core.landmarks import HAND_CONNECTIONS, HandFrame

_COLORS = {
    "Left": (0, 255, 0),     # BGR 绿色
    "Right": (0, 165, 255),  # BGR 橙色
    "Unknown": (200, 200, 200),
}

DEPTH_COLORS = {
    "Left": (255, 0, 0),     # BGR 蓝色
    "Right": (0, 255, 0),    # BGR 绿色
    "Unknown": (200, 200, 200),
}

_MIN_SHADE = 0.35  # 最暗处的明度系数


def _shade(color, t: float):
    """按明度系数 t ∈ [0,1] 缩放颜色（t=1 为原色，t 越小越暗）。"""
    return tuple(int(c * t) for c in color)


def _depth_alphas(z_values) -> list[float]:
    """z（相对深度，腕部为原点，负值朝镜头）→ 每点明度系数 t ∈ [0.35, 1.0]。

    手内 min-max 归一化：最近点 t=1.0（亮），最远点 t=0.35（暗）；全部同深时全亮。
    """
    zmin, zmax = min(z_values), max(z_values)
    span = zmax - zmin
    if span <= 1e-9:
        return [1.0] * len(z_values)
    return [1.0 - (1.0 - _MIN_SHADE) * (z - zmin) / span for z in z_values]


def _color_for(handedness: str, color):
    if color is not None:
        return tuple(int(c) for c in color)
    return _COLORS.get(handedness, _COLORS["Unknown"])


def draw_landmarks(
    frame: np.ndarray, hand_frame: HandFrame, color=None
) -> np.ndarray:
    """在 BGR 帧的副本上绘制手部关键点与骨骼连线，返回新帧（不改原帧）。"""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    for hand in hand_frame.hands:
        c = _color_for(hand.handedness, color)
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], c, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(canvas, pt, 3, c, -1, cv2.LINE_AA)
        label = f"{hand.handedness} {hand.score:.2f}"
        if pts:
            cv2.putText(
                canvas,
                label,
                (pts[0][0] - 10, max(pts[0][1] - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                c,
                1,
                cv2.LINE_AA,
            )
    return canvas


def draw_landmarks_depth(
    frame: np.ndarray, hand_frame: HandFrame
) -> np.ndarray:
    """在帧副本上绘制带深度明暗着色的关键点与骨骼连线，返回新帧（不改原帧）。

    左手蓝色、右手绿色；每个点按自身 z 深度着色（近亮远暗），
    连线按两端点平均深度着色；标签用基础色。
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    for hand in hand_frame.hands:
        base = DEPTH_COLORS.get(hand.handedness, DEPTH_COLORS["Unknown"])
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]
        alphas = _depth_alphas([lm.z for lm in hand.landmarks])
        for a, b in HAND_CONNECTIONS:
            t = (alphas[a] + alphas[b]) / 2
            cv2.line(canvas, pts[a], pts[b], _shade(base, t), 2, cv2.LINE_AA)
        for pt, t in zip(pts, alphas):
            cv2.circle(canvas, pt, 3, _shade(base, t), -1, cv2.LINE_AA)
        label = f"{hand.handedness} {hand.score:.2f}"
        if pts:
            cv2.putText(
                canvas,
                label,
                (pts[0][0] - 10, max(pts[0][1] - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                base,
                1,
                cv2.LINE_AA,
            )
    return canvas
