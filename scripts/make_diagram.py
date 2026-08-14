"""生成 README 用的手部关键点编号示意图（基于真实检测结果）。

用法: python scripts/make_diagram.py
"""

from pathlib import Path

import cv2

from signbridge.core.landmarks import HAND_CONNECTIONS
from signbridge.hands.detector import HandDetector
from signbridge.hands.sources import ImageSource

ROOT = Path(__file__).resolve().parent.parent
ASSET = ROOT / "tests" / "assets" / "hand_open.jpg"
OUT = ROOT / "docs" / "images" / "hand_landmark_diagram.png"


def main() -> None:
    src = ImageSource(ASSET)
    frame, _, _ = next(iter(src))
    src.close()
    with HandDetector(max_num_hands=2) as detector:
        hand_frame = detector.detect(frame)
        if not hand_frame.hands:
            raise SystemExit("测试图片未检测到手，无法生成示意图")
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        hand = hand_frame.hands[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            cv2.circle(canvas, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(i),
                (pt[0] + 6, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT), canvas)
        print(f"示意图已生成: {OUT}")


if __name__ == "__main__":
    main()
