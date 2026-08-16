"""段数据接入 ST-GCN 实测：加载 NPZ → 张量化 → 训练/验证循环。

目的：验证「真实骨架段 ↔ ST-GCN」接口通畅（张量格式、批处理、梯度流、
loss 下降），非追求精度。段长对齐：截断或重复填充到固定 T。

用法: python scripts/test_model_on_data.py [--npz data/extracted/segments.npz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from signbridge import STGCN, build_hand_graph


def to_tensor_batch(samples, target_t: int):
    """段列表 → (N, 3, T, 42) 张量。段长 > T 截断，< T 尾帧重复填充。"""
    batch = []
    for data in samples:
        t = len(data)
        if t >= target_t:
            arr = data[:target_t]
        else:
            reps = int(np.ceil(target_t / t))
            arr = np.tile(data, (reps, 1, 1))[:target_t]
        batch.append(arr)
    x = np.stack(batch)                    # (N, T, 42, 3)
    x = np.transpose(x, (0, 3, 1, 2))      # (N, C=3, T, V=42)
    return torch.from_numpy(x).float()


def main() -> int:
    parser = argparse.ArgumentParser(description="段数据接 ST-GCN 实测")
    parser.add_argument("--npz", type=str, default="data/extracted/segments.npz")
    parser.add_argument("--target-t", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.npz)
    if not path.exists():
        print(f"未找到 {path}，请先运行 extract_segments.py")
        return 1
    data = np.load(path, allow_pickle=True)
    samples = data["data"]
    labels = data["labels"]
    vocab = list(data["vocab"])
    n_classes = len(vocab)
    print(f"样本 {len(samples)}，类别 {n_classes}，词表: {vocab}")

    x = to_tensor_batch(samples, args.target_t)
    y = torch.from_numpy(labels)
    print(f"张量: {tuple(x.shape)} (N, C=3, T={args.target_t}, V=42)")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(x))
    n_val = max(int(len(x) * args.val_ratio), 1)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    print(f"train {len(train_idx)} / val {len(val_idx)}")

    torch.manual_seed(args.seed)
    model = STGCN(num_classes=n_classes,
                  adjacency=build_hand_graph(num_hands=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = 0.0
    print(f"{'epoch':>5} {'train_loss':>10} {'train_acc':>9} "
          f"{'val_loss':>9} {'val_acc':>7}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            train_acc = (model.predict(x_train) == y_train).float().mean()
            v_logits = model(x_val)
            v_loss = F.cross_entropy(v_logits, y_val)
            v_acc = (model.predict(x_val) == y_val).float().mean()
            model.train()
        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>5} {loss.item():>10.4f} {train_acc.item():>9.3f} "
                  f"{v_loss.item():>9.4f} {v_acc.item():>7.3f}")
        if v_acc.item() > best_val:
            best_val = v_acc.item()
    print(f"\n最佳 val_acc: {best_val:.3f}")
    print("接口验证：前向/反向/批处理/段长对齐全部正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
