"""手部关键点可视化：在帧副本上绘制关键点、骨骼连线与标签。"""

import cv2
import numpy as np

from signbridge.core.landmarks import HAND_CONNECTIONS, HandFrame

_COLORS = {
    "Left": (0, 255, 0),     # BGR 绿色
    "Right": (0, 165, 255),  # BGR 橙色
    "Unknown": (200, 200, 200),
}


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
