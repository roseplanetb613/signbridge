"""摄像头姿态+手部联合演示：PoseLandmarker 33 点骨架 + 手部深度叠加。

用法: python scripts/demo/pose_demo.py [--camera-id 0]
按 q/Esc 退出。
"""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from signbridge import CameraSource, HandDetector
from signbridge.core.errors import SignBridgeError
from signbridge.hands.draw import draw_landmarks_depth

# MediaPipe Pose 33 点连接（官方定义）
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)
POSE_COLOR = (0, 255, 255)     # BGR 黄色
POSE_NOSE = 0


def draw_pose(canvas, landmarks):
    """在画布上绘制 33 点姿态骨架。landmarks: [(x, y, z) 归一化]。"""
    h, w = canvas.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], POSE_COLOR, 2, cv2.LINE_AA)
    for i, pt in enumerate(pts):
        cv2.circle(canvas, pt, 3, POSE_COLOR, -1, cv2.LINE_AA)
    if pts:
        cv2.putText(canvas, "pose 33pts", (pts[POSE_NOSE][0] - 20,
                                           max(pts[POSE_NOSE][1] - 15, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, POSE_COLOR, 2, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="姿态+手部演示")
    parser.add_argument("--camera-id", type=int, default=0)
    args = parser.parse_args()

    pose_model = Path.home() / ".cache" / "signbridge" / "pose_landmarker_full.task"
    if not pose_model.exists():
        print(f"姿态模型缺失: {pose_model}")
        return 1
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(pose_model)),
        num_poses=1)
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

    try:
        src = CameraSource(args.camera_id)
    except SignBridgeError as exc:
        print(f"摄像头错误: {exc}")
        return 1
    print("摄像头已打开（姿态+手部，q/Esc 退出）", flush=True)
    try:
        with HandDetector(max_num_hands=2) as detector:
            for frame, _, _ in src:
                hf = detector.detect(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pr = pose_landmarker.detect(img)
                canvas = draw_landmarks_depth(frame, hf)
                if pr.pose_landmarks:
                    lm = pr.pose_landmarks[0]
                    pts = np.array([[p.x, p.y, p.z] for p in lm],
                                   dtype=np.float32)
                    draw_pose(canvas, pts)
                cv2.imshow("SignBridge 姿态+手部（q/Esc 退出）", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        src.close()
        pose_landmarker.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
