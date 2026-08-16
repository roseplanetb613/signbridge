"""随机抽样 N 个 CE-CSL 视频跑完整链路验证并汇总（优化版）。

优化点：
1. 手势段切分：连续双手有效帧 ≥ min_len 构成一个样本段（不再要求全片有手）
2. 质量标记：检测率 / 手 bbox 均值记录；低检测率样本标记 LOW_DETECTION
3. 检测阈值可调（--conf，小手样本推荐 0.3）

用法: python scripts/verify_dataset_batch.py [--count 10] [--seed 42] [--conf 0.3]
"""

import argparse
import glob
import random
import sys

import numpy as np
import torch

from signbridge import (
    HandDetector,
    HandSequenceBuffer,
    OneEuroSmoother,
    STGCN,
    VideoSource,
    build_hand_graph,
)
from signbridge.core.segmentation import extract_segments
from signbridge.hands.sequence import classify_two_hands, to_normalized

MODEL = STGCN(num_classes=100, adjacency=build_hand_graph(num_hands=2))
MODEL.eval()

MIN_SEGMENT = 9      # ST-GCN kernel_size
LOW_DETECTION = 0.3  # 检测率阈值


def verify_one(video_path: str, window: int, conf: float) -> dict:
    src = VideoSource(video_path)
    buf = HandSequenceBuffer(window_size=window, coordinate="world",
                             smoother=OneEuroSmoother())
    rows = []            # 双手同帧的 42x3 行
    hand_frames = 0      # 检测到至少一只手的帧
    total = 0
    hand_sizes = []      # 手 bbox 归一化面积
    with HandDetector(max_num_hands=2, min_detection_confidence=conf) as detector:
        for frame_index, (frame, _, _) in enumerate(src):
            hf = detector.detect(frame)
            buf.update(hf)
            total += 1
            hands = list(hf.hands)
            if hands:
                hand_frames += 1
                for hand in hands:
                    xs = [lm.x for lm in hand.landmarks]
                    ys = [lm.y for lm in hand.landmarks]
                    hand_sizes.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
            # 有手帧都构造 42 节点行：双手按方案 B 分块；
            # 单手帧 → 该手入块 0、块 1 零填充（兼容单手手语视频）
            if len(hands) == 2:
                b0, b1 = classify_two_hands(hands[0], hands[1])
                row = np.full((42, 3), np.nan, dtype=np.float32)
                row[:21] = to_normalized(b0)
                row[21:] = to_normalized(b1)
                rows.append(row)
            elif len(hands) == 1:
                row = np.zeros((42, 3), dtype=np.float32)
                row[:21] = to_normalized(hands[0])
                rows.append(row)
    src.close()

    detection_rate = hand_frames / max(total, 1)
    avg_size = float(np.mean(hand_sizes)) if hand_sizes else 0.0
    segs = extract_segments(np.ones(len(rows), dtype=bool), MIN_SEGMENT)
    seg_tensors = 0
    for start, length in segs:
        tensor = np.stack(rows[start:start + length])
        x = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            MODEL(x)
        seg_tensors += 1
    if segs:
        quality = "OK" if detection_rate >= LOW_DETECTION else "LOW_DETECTION"
    else:
        quality = "NO_SEGMENT"
    return {
        "file": video_path.split("\\")[-1],
        "total": total,
        "detection_rate": detection_rate,
        "avg_size": avg_size,
        "segments": [(s, l) for s, l in segs],
        "seg_tensors": seg_tensors,
        "quality": quality,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CE-CSL 随机抽样链路验证（段切分版）")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--window", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    videos = sorted(glob.glob(rf"E:\SignBridge\data\CE-CSL\video\{args.split}\*\*.mp4"))
    if not videos:
        print("未找到视频")
        return 1
    rng = random.Random(args.seed)
    sample = rng.sample(videos, min(args.count, len(videos)))

    results = []
    for v in sample:
        r = verify_one(v, args.window, args.conf)
        results.append(r)
        longest = max((l for _, l in r["segments"]), default=0)
        print(f"{r['file']}: 帧{r['total']:>3} 检测率{r['detection_rate']:.0%} "
              f"bbox{10000 * r['avg_size']:>6.0f}e-4 段{r['seg_tensors']:>2}个 "
              f"(最长{longest:>3}帧) [{r['quality']}]", flush=True)

    n = len(results)
    ok = sum(1 for r in results if r["quality"] == "OK")
    low = sum(1 for r in results if r["quality"] == "LOW_DETECTION")
    noseg = sum(1 for r in results if r["quality"] == "NO_SEGMENT")
    segs_total = sum(r["seg_tensors"] for r in results)
    print(f"\n汇总: {n} 视频 → OK {ok} / LOW_DETECTION {low} / NO_SEGMENT {noseg}")
    print(f"  提取手势段样本总数: {segs_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
