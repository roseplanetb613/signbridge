"""真实数据集小样本链路验证：视频 → 检测 → 追踪缓冲 → 双手张量 → ST-GCN 前向。

双手分块规则（方案 B）：优先 handedness（Left→块0 / Right→块1），
handedness 冲突（双同侧）时按画面 x 位置（左侧→块0，右侧→块1）。

用法: python scripts/verify_dataset_pipeline.py <video_path>
"""

import argparse
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


def classify_two_hands(hand_a, hand_b):
    """方案 B 分块：返回 (块0 的手, 块1 的手)。"""
    ha, hb = hand_a, hand_b
    if ha.handedness != hb.handedness:
        if ha.handedness == "Left":
            return ha, hb
        return hb, ha
    # handedness 冲突：按画面 x 位置（左侧 → 块 0）
    xa = ha.landmarks[0].x
    xb = hb.landmarks[0].x
    if xa <= xb:
        return ha, hb
    return hb, ha


def to_normalized(hand):
    """腕点归一化 (21,3)（world 米制坐标）。"""
    lms = hand.world_landmarks
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    return pts - pts[0]


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

    rows = []            # (frame_idx, 42x3)
    two_hand_frames = 0
    one_hand_frames = 0
    no_hand_frames = 0
    conflict_frames = 0  # handedness 冲突（双同侧）帧数
    total_frames = 0

    with HandDetector(max_num_hands=2) as detector:
        for frame_index, (frame, _, _) in enumerate(src):
            hf = detector.detect(frame)
            buf.update(hf)
            total_frames += 1
            hands = list(hf.hands)
            if len(hands) == 2:
                two_hand_frames += 1
                if hands[0].handedness == hands[1].handedness:
                    conflict_frames += 1
                b0, b1 = classify_two_hands(hands[0], hands[1])
                row = np.full((42, 3), np.nan, dtype=np.float32)
                row[:21] = to_normalized(b0)
                row[21:] = to_normalized(b1)
                rows.append((frame_index, row))
            elif len(hands) == 1:
                one_hand_frames += 1
            else:
                no_hand_frames += 1
    src.close()

    print(f"video frames: {total_frames}")
    print(f"  two-hand: {two_hand_frames} ({100 * two_hand_frames / total_frames:.0f}%)  "
          f"one-hand: {one_hand_frames}  no-hand: {no_hand_frames}")
    if two_hand_frames:
        print(f"  handedness 冲突帧（双同侧，按位置分块）: "
              f"{conflict_frames} ({100 * conflict_frames / two_hand_frames:.0f}%)")

    if not rows:
        print("无双手同帧，无法构造 42 节点张量")
        return 1

    tensor = np.stack([r for _, r in rows])           # (T, 42, 3)
    valid = ~np.isnan(tensor).any(axis=(1, 2))
    tensor = tensor[valid]
    print(f"aligned tensor: {tensor.shape} (T, 42, 3)，有效行 {valid.sum()}/{len(valid)}")

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
