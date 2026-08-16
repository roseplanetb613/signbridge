"""handedness 稳定性分析：从录制 NPZ 中找出左右手标签跳变事件。

方法：帧间按质心最近邻关联手 → 每条轨迹检测 handedness 翻转点 →
跳变前后手形特征距离（HandShapeFeature）衡量「手形是否连续」——
特征距离小 = 同一只手被误判左右，距离大 = 可能真的是换手/检测切换。

用法: python scripts/analyze_handedness.py [--npz data/recordings/record/detections.npz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from signbridge.core.features import HandShapeFeature

MATCH_DIST = 0.25      # 帧间手关联距离阈值（归一化空间）
FEATURE = HandShapeFeature()


def _centroid(pts):
    return pts[:, :2].mean(axis=0)


def build_tracks(hs, lm):
    """帧间质心最近邻关联 → tracks: list of dict(frames, handedness, pts)"""
    tracks = []            # 每个元素 dict
    active = []            # [(track_idx, centroid)]
    track_of = []          # 每帧: [(track_idx, hand_idx)]
    for t in range(len(hs)):
        hands = [(i, lm[t, i]) for i in range(2)
                 if hs[t, i] >= 0 and not np.isnan(lm[t, i]).all()]
        assigned = []
        for hand_idx, pts in hands:
            c = _centroid(pts)
            best = None
            best_d = MATCH_DIST
            for k, (tid, ac) in enumerate(active):
                d = float(np.linalg.norm(c - ac))
                if d < best_d:
                    best_d = d
                    best = k
            if best is not None:
                tid = active[best][0]
                active[best] = (tid, c)
                tracks[tid]["frames"].append(t)
                tracks[tid]["handedness"].append(hs[t, hand_idx])
                tracks[tid]["pts"].append(pts)
                assigned.append((tid, hand_idx))
            else:
                tid = len(tracks)
                tracks.append({"frames": [t], "handedness": [hs[t, hand_idx]],
                               "pts": [pts]})
                active.append((tid, c))
                assigned.append((tid, hand_idx))
        # 未匹配的活动轨迹保留（允许短丢帧后重连由后续帧匹配）
        keep = [a for a in active if any(tid == x[0] for x in assigned)]
        active = keep
        track_of.append(assigned)
    return tracks


def analyze(npz_path: Path):
    data = np.load(npz_path)
    hs = data["handedness"]
    lm = data["landmarks"]
    frame_indices = data["frame_indices"]

    tracks = build_tracks(hs, lm)
    events = []
    for tid, tr in enumerate(tracks):
        seq = tr["handedness"]
        pts = tr["pts"]
        frames = tr["frames"]
        if len(seq) < 3:
            continue
        # 找翻转点
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                # 翻转段：连续相同新值的长度
                j = i
                while j < len(seq) and seq[j] == seq[i]:
                    j += 1
                duration = j - i
                # 手形特征：翻转前 3 帧均值 vs 翻转后 3 帧均值
                def _feat_avg(idx_range):
                    vecs = [FEATURE.extract(pts[k]) for k in idx_range
                            if not np.isnan(pts[k]).all()]
                    return np.mean(vecs, axis=0) if vecs else None

                before = _feat_avg(range(max(0, i - 3), i))
                after = _feat_avg(range(i, min(len(seq), j + 2)))
                feat_dist = (
                    float(np.linalg.norm(before - after))
                    if before is not None and after is not None else None
                )
                events.append({
                    "track": tid,
                    "frame": int(frame_indices[frames[i]]),
                    "flip": f"{seq[i-1]}→{seq[i]}",
                    "duration": duration,
                    "feat_dist": feat_dist,
                })
                i = j - 1

    print(f"总帧数: {len(hs)}，轨迹数: {len(tracks)}，"
          f"handedness 跳变事件: {len(events)}")
    if not events:
        print("未检测到跳变（本次录制可能未触发）")
        return
    print(f"{'帧':>5} {'轨迹':>3} {'翻转':>6} {'持续帧':>5} {'手形特征距离':>10}")
    for e in events:
        fd = f"{e['feat_dist']:.4f}" if e["feat_dist"] is not None else "N/A"
        print(f"{e['frame']:>5} {e['track']:>3} {e['flip']:>6} "
              f"{e['duration']:>5} {fd:>10}")
    fd_vals = [e["feat_dist"] for e in events if e["feat_dist"] is not None]
    if fd_vals:
        print(f"\n手形特征距离（跳变前后）：均值 {np.mean(fd_vals):.4f}，"
              f"中位 {np.median(fd_vals):.4f}，最大 {max(fd_vals):.4f}")
        print("解读：距离 <0.3 表示手形几乎未变 → 确属 handedness 误判；"
              "距离大 → 可能是检测切换/换手")


def main() -> int:
    parser = argparse.ArgumentParser(description="handedness 跳变分析")
    parser.add_argument("--npz", type=str,
                        default="data/recordings/record/detections.npz")
    args = parser.parse_args()
    analyze(Path(args.npz))
    return 0


if __name__ == "__main__":
    sys.exit(main())
