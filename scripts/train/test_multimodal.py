"""三模态小批量训练/评估链路测试（30 段）。

验证：
1. hand 骨架 → STGCNCTC（现有图）+ best.pt 评估（WER）
2. pose 33 点 → STGCNCTC（POSE_CONNECTIONS 图）前向 + CTC 一步
3. ROI JPEG → 解码 → 统计（RGB 融合数据就绪验证）

用法: python scripts/train/test_multimodal.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from signbridge import STGCNCTC, build_hand_graph
from signbridge.core.graphs import build_adjacency

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)


def to_tensor(seg, target_t=128):
    """段 (T,V,3) → (1,3,T,V) 张量（截断/重复填充）。"""
    t = len(seg)
    if t >= target_t:
        arr = seg[:target_t]
    else:
        reps = int(np.ceil(target_t / t))
        arr = np.tile(seg, (reps, 1, 1))[:target_t]
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()


def main() -> int:
    base = Path("data/dataset")
    hand = np.load(base / "train.npz", allow_pickle=True)
    pose = np.load(base / "train_pose.npz", allow_pickle=True)
    roi = np.load(base / "train_roi.npz", allow_pickle=True)
    n = len(hand["data"])
    print(f"样本 {n}（hand/pose/roi 对齐: "
          f"{n == len(pose['pose_img']) == len(roi['roi'])}）")

    # ---- 1. hand 骨架评估（best.pt）----
    ckpt = torch.load("checkpoints/best.pt", map_location="cpu")
    vocab = list(ckpt["vocab"])
    model = STGCNCTC(num_classes=len(vocab),
                     adjacency=build_hand_graph(num_hands=2))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    xs = [to_tensor(np.asarray(hand["data"][i], dtype=np.float32))
          for i in range(n)]
    x = torch.cat(xs)
    with torch.no_grad():
        logits = model(x)
        decoded = model.beam_decode(logits, beam_width=5)
    ref_all = []
    hyp_all = []
    for i in range(n):
        gloss = str(hand["glosses"][i])
        ref = [w for w in gloss.split("/") if w.strip() and w.strip() not in "。，？！、；："]
        hyp = [vocab[c - 1] for c in decoded[i] if 0 < c <= len(vocab)]
        ref_all.append(ref)
        hyp_all.append(hyp)
    err = sum(_levenshtein(r, h) for r, h in zip(ref_all, hyp_all))
    total_ref = sum(len(r) for r in ref_all)
    print(f"[hand] 30 段训练集评估: WER {err / max(total_ref, 1):.3f} "
          f"（注意：这些段在全量 train 内，数字不代表泛化）")
    for i in range(3):
        print(f"  例{i + 1} 真值: {''.join(ref_all[i])}")
        print(f"       预测: {''.join(hyp_all[i]) or '(空)'}")

    # ---- 2. pose 33 点 → STGCNCTC（pose 图）链路验证 ----
    pose_adj = build_adjacency(POSE_CONNECTIONS, 33)
    pose_model = STGCNCTC(num_classes=5, adjacency=pose_adj)
    pi = np.asarray(pose["pose_img"][0], dtype=np.float32)
    pi = np.nan_to_num(pi, nan=0.0)          # 缺失姿态帧补 0
    xp = to_tensor(pi)
    lp = pose_model.log_probs(xp)
    targets = torch.tensor([[1, 2]])
    loss = F.ctc_loss(lp, targets,
                      input_lengths=torch.tensor([32]),
                      target_lengths=torch.tensor([2]))
    loss.backward()
    grad_ok = all(p.grad is not None for p in pose_model.parameters()
                  if p.requires_grad)
    print(f"[pose] 33 点图前向+CTC 反向: loss {loss.item():.3f}，"
          f"梯度正常 {grad_ok}，输入 {tuple(xp.shape)}")

    # ---- 3. ROI JPEG 解码统计 ----
    decoded_frames = 0
    total = 0
    sizes = []
    for r in roi["roi"]:
        for b in np.asarray(r, dtype=object):
            total += 1
            if b is not None:
                img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                decoded_frames += 1
                if img is not None:
                    sizes.append(img.shape)
    print(f"[roi] 解码 {decoded_frames}/{total} 帧，尺寸统一 "
          f"{len(set(sizes)) == 1}（{sizes[0] if sizes else 'N/A'}）")
    return 0


def _levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


if __name__ == "__main__":
    sys.exit(main())
