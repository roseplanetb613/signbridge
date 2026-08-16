"""摄像头录制工具：保存原始视频 + 叠加视频 + 每帧检测数据（NPZ）。

用法: python scripts/record_camera.py [--duration 60] [--out data/recordings/xxx]
结束后 NPZ 包含：frame_indices/timestamps/n_hands/handedness(编码)/scores/
landmarks(N,2,21,3)/world(N,2,21,3)——供 handedness 稳定性分析。
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from signbridge import CameraSource, HandDetector
from signbridge.core.errors import SignBridgeError
from signbridge.hands.draw import DEPTH_COLORS, draw_landmarks_depth

HANDEDNESS_CODE = {"Left": 0, "Right": 1}


def main() -> int:
    parser = argparse.ArgumentParser(description="摄像头录制（含检测数据）")
    parser.add_argument("--duration", type=float, default=60.0, help="录制秒数")
    parser.add_argument("--out", type=str,
                        default="data/recordings/record")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--refine", action="store_true", help="开启两级候选检测")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw.mp4"
    overlay_path = out_dir / "overlay.mp4"
    npz_path = out_dir / "detections.npz"

    try:
        source = CameraSource(args.camera_id)
    except SignBridgeError as exc:
        print(f"摄像头错误: {exc}")
        return 1
    detector = HandDetector(max_num_hands=2, refine_roi=args.refine)

    # 先读一帧确定尺寸
    frame, _, _ = next(iter(source))
    h, w = frame.shape[:2]
    raw_writer = cv2.VideoWriter(str(raw_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    overlay_writer = cv2.VideoWriter(str(overlay_path),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))

    frame_indices, timestamps, n_hands = [], [], []
    handedness, scores = [], []
    landmarks, worlds = [], []

    start = time.monotonic()
    fidx = 0
    print(f"录制开始 {args.duration:.0f}s → {out_dir}（请在此期间摆各种手势，"
          f"包括容易触发左右手误判的姿势）", flush=True)
    while time.monotonic() - start < args.duration:
        try:
            frame, _, _ = next(iter(source))
        except StopIteration:
            break
        hf = detector.detect(frame)
        raw_writer.write(frame)
        overlay_writer.write(draw_landmarks_depth(frame, hf))

        frame_indices.append(fidx)
        timestamps.append(hf.timestamp_ms)
        n_hands.append(len(hf.hands))
        hs = np.full(2, -1, dtype=np.int8)
        sc = np.zeros(2, dtype=np.float32)
        lm = np.full((2, 21, 3), np.nan, dtype=np.float32)
        wl = np.full((2, 21, 3), np.nan, dtype=np.float32)
        for i, hand in enumerate(hf.hands[:2]):
            hs[i] = HANDEDNESS_CODE.get(hand.handedness, -1)
            sc[i] = hand.score
            lm[i] = np.array([[p.x, p.y, p.z] for p in hand.landmarks])
            wl[i] = np.array([[p.x, p.y, p.z] for p in hand.world_landmarks])
        handedness.append(hs)
        scores.append(sc)
        landmarks.append(lm)
        worlds.append(wl)
        fidx += 1

    source.close()
    detector.close()
    raw_writer.release()
    overlay_writer.release()

    np.savez_compressed(
        npz_path,
        frame_indices=np.array(frame_indices),
        timestamps=np.array(timestamps),
        n_hands=np.array(n_hands),
        handedness=np.array(handedness),
        scores=np.array(scores),
        landmarks=np.array(landmarks),
        world=np.array(worlds),
    )
    print(f"录制完成: {fidx} 帧")
    print(f"  raw      → {raw_path}")
    print(f"  overlay  → {overlay_path}")
    print(f"  detections → {npz_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
