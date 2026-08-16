"""真实数据集小样本链路验证：视频 → 检测 → 追踪缓冲 → 双手张量 → ST-GCN 前向。

用法: python scripts/verify_dataset_pipeline.py <video_path> [--window N]
"""

import argparse
import sys
from pathlib import Path

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


def align_two_hands(left, right, window: int):
    """按 frame_indices 对齐两只手的序列 → (T, 42, 3) 张量。

    左手块 0-20、右手块 21-41（与 build_hand_graph(num_hands=2) 分块一致）。
    某帧某手缺失 → 该行 NaN（由调用方决定有效段）。
    """
    li = left.frame_indices
    ri = right.frame_indices
    lo = {int(f): i for i, f in enumerate(li)}
    ro = {int(f): i for i, f in enumerate(ri)}
    common = sorted(set(lo) & set(ro))
    if not common:
        return None
    t = np.full((len(common), 42, 3), np.nan, dtype=np.float32)
    for k, fidx in enumerate(common):
        t[k, :21, :] = left.data[lo[fidx]]
        t[k, 21:, :] = right.data[ro[fidx]]
    return t


def main() -> int:
    parser = argparse.ArgumentParser(description="CE-CSL 链路验证")
    parser.add_argument("video", type=str)
    parser.add_argument("--window", type=int, default=300)
    args = parser.parse_args()

    src = VideoSource(args.video)
    buf = HandSequenceBuffer(
        window_size=args.window,
        coordinate="world",
        smoother=OneEuroSmoother(),
    )
    total_frames = 0
    final = ()
    with HandDetector(max_num_hands=2) as detector:
        for frame, _, _ in src:
            final = buf.update(detector.detect(frame))
            total_frames += 1
    src.close()

    by_hand = {s.handedness: s for s in final}

    print(f"video frames processed: {total_frames}")
    for s in final:
        valid = int(s.valid_mask.sum())
        print(f"  id{s.hand_id} {s.handedness}: window={len(s.data)} valid={valid} "
              f"({100 * valid / len(s.data):.0f}%)")

    left = by_hand.get("Left")
    right = by_hand.get("Right")
    if left is None or right is None:
        print("需要双手都在场才能拼 42 节点张量（当前缺一只手）")
        return 1

    tensor = align_two_hands(left, right, args.window)
    if tensor is None:
        print("双手时间轴无重叠帧")
        return 1

    # 有效段截取（全有效的连续段）
    valid_rows = ~np.isnan(tensor).any(axis=(1, 2))
    tensor = tensor[valid_rows]
    print(f"aligned tensor: {tensor.shape} (T, 42, 3)，NaN 行已剔除 "
          f"({valid_rows.sum()}/{len(valid_rows)} 行有效)")

    # → (C, T, V)
    x = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).float()
    print(f"ST-GCN input: {tuple(x.shape)} (N=1, C=3, T, V=42)")

    model = STGCN(num_classes=100, adjacency=build_hand_graph(num_hands=2))
    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = model.predict(x)
    print(f"ST-GCN forward: logits {tuple(logits.shape)} → 预测类别 {int(pred[0])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
