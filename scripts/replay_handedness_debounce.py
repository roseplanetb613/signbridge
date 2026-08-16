"""录制数据回放验证 handedness 防抖效果。

从 detections.npz 重建 HandFrame 流喂给 HandSequenceBuffer，
对比 handedness_debounce=0（关闭）与 =5（开启）时各轨迹的标签翻转次数。

用法: python scripts/replay_handedness_debounce.py [--npz ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from signbridge.core.landmarks import Hand, HandFrame, Landmark
from signbridge.hands.sequence import HandSequenceBuffer

CODE_TO_NAME = {0: "Left", 1: "Right"}


def replay(npz_path: Path, debounce: int):
    data = np.load(npz_path)
    n = len(data["frame_indices"])
    buf = HandSequenceBuffer(window_size=2000, coordinate="image",
                             smoother=None, handedness_debounce=debounce)
    history = {}   # hand_id -> [(t, handedness)]
    for t in range(n):
        hs = data["handedness"][t]
        lms = data["landmarks"][t]
        scores = data["scores"][t]
        hands = []
        for i in range(2):
            if hs[i] >= 0 and not np.isnan(lms[i]).all():
                pts = tuple(
                    Landmark(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                    for p in lms[i]
                )
                hands.append(Hand(landmarks=pts,
                                  handedness=CODE_TO_NAME[int(hs[i])],
                                  score=float(scores[i])))
        hf = HandFrame(hands=tuple(hands),
                       timestamp_ms=int(data["timestamps"][t]),
                       frame_index=int(data["frame_indices"][t]))
        for s in buf.update(hf):
            history.setdefault(s.hand_id, []).append((t, s.handedness))
    # 统计每条轨迹的标签翻转
    stats = {}
    for hid, seq in history.items():
        changes = sum(1 for k in range(1, len(seq))
                      if seq[k][1] != seq[k - 1][1])
        stats[hid] = {
            "frames": len(seq),
            "flips": changes,
            "label": seq[-1][1],
            "first_frame": seq[0][0],
        }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="防抖回放验证")
    parser.add_argument("--npz", type=str,
                        default="data/recordings/record/detections.npz")
    args = parser.parse_args()
    path = Path(args.npz)
    if not path.exists():
        print(f"未找到 {path}")
        return 1

    off = replay(path, debounce=0)
    on = replay(path, debounce=5)

    print(f"{'轨迹':>4} | {'关闭防抖':^22} | {'开启防抖(=5)':^22}")
    print(f"{'':>4} | {'帧数':>6} {'翻转':>6} {'标签':>6} | "
          f"{'帧数':>6} {'翻转':>6} {'标签':>6}")
    all_ids = sorted(set(off) | set(on))
    total_off = total_on = 0
    for hid in all_ids:
        a = off.get(hid)
        b = on.get(hid)
        ao = (a["frames"], a["flips"], a["label"]) if a else ("-", "-", "-")
        bo = (b["frames"], b["flips"], b["label"]) if b else ("-", "-", "-")
        total_off += a["flips"] if a else 0
        total_on += b["flips"] if b else 0
        print(f"{hid:>4} | {ao[0]:>6} {ao[1]:>6} {str(ao[2]):>6} | "
              f"{bo[0]:>6} {bo[1]:>6} {str(bo[2]):>6}")
    print(f"\n标签翻转总数：关闭 {total_off} → 开启 {total_on} "
          f"（减少 {total_off - total_on}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
