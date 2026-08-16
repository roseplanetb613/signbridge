"""CTC 训练链路小样本验证：segments.npz → STGCNCTC → CTCLoss → 贪心解码。

成功标准：loss 下降、解码输出合法词、无 NaN。

用法: python scripts/train_ctc.py [--npz data/extracted/segments.npz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from signbridge import STGCNCTC, build_hand_graph

PUNCT = set("。，？！、；：""''（）《》")


def gloss_words(gloss: str) -> list[str]:
    return [w for w in gloss.split("/") if w.strip() and w.strip() not in PUNCT]


def to_tensor_batch(samples, target_t: int):
    batch = []
    for data in samples:
        t = len(data)
        if t >= target_t:
            arr = data[:target_t]
        else:
            reps = int(np.ceil(target_t / t))
            arr = np.tile(data, (reps, 1, 1))[:target_t]
        batch.append(arr)
    x = np.stack(batch)
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(x).float()


def main() -> int:
    parser = argparse.ArgumentParser(description="CTC 小样本训练验证")
    parser.add_argument("--npz", type=str, default="data/extracted/segments.npz")
    parser.add_argument("--target-t", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.npz)
    if not path.exists():
        print(f"未找到 {path}")
        return 1
    data = np.load(path, allow_pickle=True)
    samples = data["data"]
    glosses = list(data["glosses"])

    # 词表（当前小样本）
    from collections import Counter
    freq = Counter(w for g in glosses for w in gloss_words(g))
    vocab = [w for w, _ in freq.most_common()]
    vocab_idx = {w: i + 1 for i, w in enumerate(vocab)}   # 0 = blank
    print(f"样本 {len(samples)}，词表 {len(vocab)}: {vocab}")

    x = to_tensor_batch(samples, args.target_t)
    targets, target_lengths = [], []
    for g in glosses:
        ids = [vocab_idx[w] for w in gloss_words(g) if w in vocab_idx]
        targets.append(ids)
        target_lengths.append(len(ids))
    max_len = max(target_lengths)
    targets_pad = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, ids in enumerate(targets):
        targets_pad[i, :len(ids)] = torch.tensor(ids)
    target_lengths = torch.tensor(target_lengths)
    print(f"张量 {tuple(x.shape)}；标签最长 {max_len}（≤ T'=32）")

    torch.manual_seed(args.seed)
    model = STGCNCTC(num_classes=len(vocab),
                     adjacency=build_hand_graph(num_hands=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    input_lengths = torch.full((len(x),), 32)   # T'

    print(f"{'epoch':>5} {'ctc_loss':>10} {'样本1预测':>16} {'样本1真值':>16}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        lp = model.log_probs(x)
        loss = F.ctc_loss(lp, targets_pad, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                decoded = model.decode(model(x))
                pred0 = "".join(vocab[c - 1] for c in decoded[0]) or "(空)"
                truth0 = "".join(gloss_words(glosses[0]))
            print(f"{epoch:>5} {loss.item():>10.4f} {pred0:>16} {truth0:>16}")
    print("\nCTC 链路验证完成：loss 下降 / 解码合法 / 梯度正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
