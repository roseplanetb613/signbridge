"""ROI 精化对比验证：同一视频 refine on/off 的检测率 / 置信度 / 帧间稳定性。

用法: python scripts/verify_roi_refine.py <video_path> [--conf 0.3]
"""

import argparse
import glob
import sys

import cv2
import numpy as np

from signbridge import HandDetector


def _frame_displacement(prev_pts, cur_pts):
    """最近邻匹配后相邻帧关键点平均位移（归一化空间）。"""
    if prev_pts is None or cur_pts is None:
        return None
    d = np.linalg.norm(cur_pts[:, None, :] - prev_pts[None, :, :], axis=-1)
    # 贪心最近邻（数量少，够用）
    used = set()
    total = 0.0
    count = 0
    for i in range(len(cur_pts)):
        j = int(np.argmin(d[i]))
        if j in used:
            continue
        used.add(j)
        total += d[i, j]
        count += 1
    return total / max(count, 1)


def verify(path: str, conf: float, refine: bool) -> dict:
    cap = cv2.VideoCapture(path)
    det_frames = 0
    total = 0
    scores = []
    displacements = []
    prev_pts = None
    with HandDetector(max_num_hands=2, min_detection_confidence=conf,
                      refine_roi=refine) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            hf = detector.detect(frame)
            if hf.hands:
                det_frames += 1
                scores.append(hf.hands[0].score)
                cur = np.array([[lm.x, lm.y, lm.z] for lm in hf.hands[0].landmarks])
                disp = _frame_displacement(prev_pts, cur)
                if disp is not None:
                    displacements.append(disp)
                prev_pts = cur
            else:
                prev_pts = None
    cap.release()
    return {
        "detection": det_frames / max(total, 1),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "mean_disp": float(np.mean(displacements)) if displacements else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ROI 精化对比验证")
    parser.add_argument("video", type=str)
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()
    path = glob.glob(args.video)[0]
    name = path.split("\\")[-1]

    off = verify(path, args.conf, refine=False)
    on = verify(path, args.conf, refine=True)
    print(f"{name} (conf={args.conf}):")
    print(f"  检测率    : off {off['detection']:.0%} → on {on['detection']:.0%} "
          f"({'+' if on['detection'] > off['detection'] else ''}"
          f"{on['detection'] - off['detection']:.0%})")
    print(f"  平均置信度: off {off['mean_score']:.3f} → on {on['mean_score']:.3f}")
    print(f"  帧间位移  : off {off['mean_disp']:.5f} → on {on['mean_disp']:.5f} "
          f"(越小越稳定)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
