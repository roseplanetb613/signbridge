"""正式 CTC 训练：全量数据集 → STGCNCTC → CTC 训练 + WER 评估。

数据：data/dataset/{train,dev,test}.npz（extract_dataset.py 产出）
评估：每 epoch dev 集 CTC loss + 贪心解码词错误率（WER）+ 句准确率
保存：checkpoints/best.pt（按 dev WER 最优）

用法: python scripts/train_full.py [--data-dir data/dataset] [--epochs 30]
                                    [--batch-size 32] [--device auto]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from signbridge import STGCNCTC, build_hand_graph

PUNCT = set("。，？！、；：""''（）《》")
MIN_DETECTION = 0.3     # 质量过滤：检测率下限
MAX_T = 256             # 段长上限（截断）


def gloss_words(gloss: str) -> list[str]:
    return [w for w in gloss.split("/") if w.strip() and w.strip() not in PUNCT]


def levenshtein(a: list, b: list) -> int:
    """词序列编辑距离（WER 用）。"""
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


def to_tensor_batch(samples, target_t: int):
    batch = []
    for data in samples:
        data = np.asarray(data, dtype=np.float32)   # object 数组反序列化防御
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


def load_split(path: Path, vocab_idx: dict | None, min_det: float,
               target_t: int):
    d = np.load(path, allow_pickle=True)
    keep = [i for i in range(len(d["data"]))
            if d["detection_rates"][i] >= min_det]
    samples = [d["data"][i] for i in keep]
    glosses = [str(d["glosses"][i]) for i in keep]
    x = to_tensor_batch(samples, target_t)

    targets, target_lengths = [], []
    for g in glosses:
        ids = [vocab_idx[w] for w in gloss_words(g) if w in vocab_idx]
        targets.append(ids)
        target_lengths.append(len(ids))
    max_len = max(target_lengths) if target_lengths else 0
    targets_pad = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, ids in enumerate(targets):
        targets_pad[i, :len(ids)] = torch.tensor(ids)
    return x, targets_pad, torch.tensor(target_lengths), glosses


def decode_and_wer(model, x, targets, target_lengths, vocab, device):
    """贪心解码 + WER/句准确率。返回 (wer, acc, loss)。"""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        lp = torch.log_softmax(logits, dim=2).permute(1, 0, 2)
        loss = F.ctc_loss(lp, targets.to(device),
                          input_lengths=torch.full((len(x),), 32,
                                                   device=device),
                          target_lengths=target_lengths.to(device))
        decoded = model.decode(logits)
    total_err = total_ref = 0
    correct = 0
    for i, hyp in enumerate(decoded):
        ref = [vocab[c - 1]
               for c in targets[i][:int(target_lengths[i])].tolist()]
        hyp_words = [vocab[c - 1] for c in hyp if 0 < c <= len(vocab)]
        total_err += levenshtein(ref, hyp_words)
        total_ref += len(ref)
        if ref == hyp_words:
            correct += 1
    wer = total_err / max(total_ref, 1)
    return wer, correct / max(len(decoded), 1), loss.item()


def main() -> int:
    parser = argparse.ArgumentParser(description="正式 CTC 训练")
    parser.add_argument("--data-dir", type=str, default="data/dataset")
    parser.add_argument("--target-t", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.npz"
    dev_path = data_dir / "dev.npz"
    vocab_path = data_dir / "vocab.npz"
    for p in (train_path, dev_path, vocab_path):
        if not p.exists():
            print(f"缺少 {p}（请先运行 extract_dataset.py）")
            return 1

    vocab = list(np.load(vocab_path, allow_pickle=True)["words"])
    vocab_idx = {w: i + 1 for i, w in enumerate(vocab)}
    print(f"词表 {len(vocab)} 词（0=blank）")

    x_train, y_train, ylen_train, _ = load_split(
        train_path, vocab_idx, MIN_DETECTION, args.target_t)
    x_dev, y_dev, ylen_dev, _ = load_split(
        dev_path, vocab_idx, MIN_DETECTION, args.target_t)
    print(f"train {len(x_train)} 段 / dev {len(x_dev)} 段")

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    model = STGCNCTC(num_classes=len(vocab),
                     adjacency=build_hand_graph(num_hands=2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    dataset = TensorDataset(x_train, y_train, ylen_train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_wer = 1.0
    print(f"{'epoch':>5} {'train_loss':>10} {'dev_loss':>9} "
          f"{'dev_WER':>8} {'dev_acc':>7}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batch = 0
        for xb, yb, ylb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            ylb = ylb.to(device)
            optimizer.zero_grad()
            lp = model.log_probs(xb)
            loss = F.ctc_loss(
                lp, yb,
                input_lengths=torch.full((len(xb),), 32, device=device),
                target_lengths=ylb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        train_loss = total_loss / max(n_batch, 1)

        wer, acc, dev_loss = decode_and_wer(
            model, x_dev, y_dev, ylen_dev, vocab, device)
        scheduler.step(dev_loss)
        if wer < best_wer:
            best_wer = wer
            torch.save({"state_dict": model.state_dict(),
                        "vocab": vocab,
                        "config": vars(args),
                        "best_wer": best_wer},
                       out_dir / "best.pt")
            suffix = " *"
        else:
            suffix = ""
        print(f"{epoch:>5} {train_loss:>10.4f} {dev_loss:>9.4f} "
              f"{wer:>8.3f} {acc:>7.3f}{suffix}", flush=True)
    print(f"\n最佳 dev WER: {best_wer:.3f} → checkpoints/best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
