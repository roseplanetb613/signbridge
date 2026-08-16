"""随机抽样 N 个 CE-CSL 视频跑完整链路验证并汇总。

用法: python scripts/verify_dataset_batch.py [--count 10] [--seed 42] [--split train]
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
from verify_dataset_pipeline import classify_two_hands, to_normalized

MODEL = STGCN(num_classes=100, adjacency=build_hand_graph(num_hands=2))
MODEL.eval()


def verify_one(video_path: str, window: int) -> dict:
    src = VideoSource(video_path)
    buf = HandSequenceBuffer(window_size=window, coordinate="world",
                             smoother=OneEuroSmoother())
    two = one = none = conflict = 0
    rows = []
    with HandDetector(max_num_hands=2) as detector:
        for frame_index, (frame, _, _) in enumerate(src):
            hf = detector.detect(frame)
            buf.update(hf)
            hands = list(hf.hands)
            if len(hands) == 2:
                two += 1
                if hands[0].handedness == hands[1].handedness:
                    conflict += 1
                b0, b1 = classify_two_hands(hands[0], hands[1])
                row = np.full((42, 3), np.nan, dtype=np.float32)
                row[:21] = to_normalized(b0)
                row[21:] = to_normalized(b1)
                rows.append(row)
            elif len(hands) == 1:
                one += 1
            else:
                none += 1
    src.close()
    total = two + one + none
    tensor_ok = False
    if rows:
        tensor = np.stack(rows)
        valid = ~np.isnan(tensor).any(axis=(1, 2))
        tensor = tensor[valid]
        if len(tensor) >= 9:  # ST-GCN kernel_size=9
            x = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                MODEL(x)
            tensor_ok = True
    return {
        "file": video_path.split("\\")[-1],
        "total": total,
        "two": two, "one": one, "none": none,
        "conflict": conflict,
        "tensor_ok": tensor_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CE-CSL 随机抽样链路验证")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--window", type=int, default=300)
    args = parser.parse_args()

    videos = sorted(glob.glob(
        rf"E:\SignBridge\data\CE-CSL\video\{args.split}\*\*.mp4"))
    if not videos:
        print("未找到视频")
        return 1
    rng = random.Random(args.seed)
    sample = rng.sample(videos, min(args.count, len(videos)))

    results = []
    for v in sample:
        r = verify_one(v, args.window)
        results.append(r)
        pct_two = 100 * r["two"] / max(r["total"], 1)
        print(f"{r['file']}: {r['total']:>3}帧 双手{pct_two:>3.0f}% "
              f"单手{r['one']:>2} 无手{r['none']:>2} 冲突{r['conflict']:>2} "
              f"张量{'OK' if r['tensor_ok'] else 'FAIL'}", flush=True)

    n = len(results)
    ok = sum(1 for r in results if r["tensor_ok"])
    two_total = sum(r["two"] for r in results)
    total = sum(r["total"] for r in results)
    conflict_total = sum(r["conflict"] for r in results)
    print(f"\n汇总: {n} 个视频，张量构造成功 {ok}/{n}")
    print(f"  双手帧占比 {100 * two_total / max(total, 1):.0f}%")
    print(f"  handedness 冲突帧占比 {100 * conflict_total / max(two_total, 1):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
