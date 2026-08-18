"""正式 CTC 训练：全量数据集 → STGCNCTC → CTC 训练 + WER 评估。

数据：data/dataset/{train,dev,test}.npz（extract_dataset.py 产出）
评估：每 epoch dev 集 CTC loss + 贪心解码词错误率（WER）+ 句准确率
保存：checkpoints/best.pt（按 dev WER 最优）

用法: python scripts/train_full.py [--data-dir data/dataset] [--epochs 30]
                                    [--batch-size 32] [--device auto]
"""

import os
# Windows 排坑：PyTorch OpenMP 与 numpy OpenMP 冲突 → npz object 数组
# 反序列化慢 5000 倍；必须在 import torch/numpy 之前强制单线程。
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from signbridge import STGCNCTC, build_hand_graph

PUNCT = set("。，？！、；：""''（）《》")
MIN_DETECTION = 0.3     # 质量过滤：检测率下限
MAX_T = 256             # 段长上限（截断）


def time_augment(data: np.ndarray, target_t: int) -> np.ndarray:
    """时间增强：长段随机窗口裁剪 + 短段随机（重复填充 / 时间插值缩放）。"""
    t = len(data)
    if t > target_t:
        start = np.random.randint(0, t - target_t + 1)
        return data[start:start + target_t]
    if t < target_t:
        if np.random.rand() < 0.5:
            # 时间插值缩放（线性，向量化）——比重复填充更平滑
            x_old = np.linspace(0.0, 1.0, t)
            x_new = np.linspace(0.0, 1.0, target_t)
            pos = x_new * (t - 1)
            i0 = np.floor(pos).astype(int)
            i1 = np.minimum(i0 + 1, t - 1)
            frac = (pos - i0)[:, None, None].astype(np.float32)
            return data[i0] * (1 - frac) + data[i1] * frac
        reps = int(np.ceil(target_t / t))
        return np.tile(data, (reps, 1, 1))[:target_t]
    return data


def space_augment(data: np.ndarray, noise: float = 0.01,
                  rot: float = 0.1, scale_range=(0.9, 1.1)) -> np.ndarray:
    """空间增强：随机缩放 + 绕腕点小角度旋转 + 高斯关节噪声。"""
    s = np.random.uniform(*scale_range)
    data = data * s
    theta = np.random.uniform(-rot, rot)
    c, sn = np.cos(theta), np.sin(theta)
    rot_z = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]], dtype=np.float32)
    data = data @ rot_z.T
    data = data + np.random.normal(0, noise, data.shape).astype(np.float32)
    return data


def align_length(data: np.ndarray, target_t: int) -> np.ndarray:
    """无增强的对齐：长段截断、短段重复填充。"""
    t = len(data)
    if t >= target_t:
        return data[:target_t]
    reps = int(np.ceil(target_t / t))
    return np.tile(data, (reps, 1, 1))[:target_t]


class SkeletonDataset(Dataset):
    """骨架段数据集（可选在线增强）。

    samples: [(T,42,3)]；targets: (N, L) padded 词 id；target_lengths: (N,)。
    """

    def __init__(self, samples, targets, target_lengths, target_t: int,
                 augment: bool = False):
        self.samples = samples
        self.targets = targets
        self.target_lengths = target_lengths
        self.target_t = target_t
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        data = np.asarray(self.samples[i], dtype=np.float32)
        if self.augment:
            data = time_augment(data, self.target_t)
            data = space_augment(data)
        else:
            data = align_length(data, self.target_t)
        x = np.transpose(data, (2, 0, 1))       # (3, T, 42)
        return (torch.from_numpy(x).float(),
                self.targets[i], self.target_lengths[i])


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
    # 注意：NpzFile 每次 d["data"] 访问都会重新解压整个数组（无缓存）！
    # 必须先取引用再循环，否则列表推导 = 数千次全量解压（卡死）。
    d = np.load(path, allow_pickle=True)
    data_arr = d["data"]
    rates = d["detection_rates"]
    gloss_arr = d["glosses"]
    keep = [i for i in range(len(data_arr)) if rates[i] >= min_det]
    samples = [data_arr[i] for i in keep]
    glosses = [str(gloss_arr[i]) for i in keep]

    targets, target_lengths = [], []
    for g in glosses:
        ids = [vocab_idx[w] for w in gloss_words(g) if w in vocab_idx]
        targets.append(ids)
        target_lengths.append(len(ids))
    max_len = max(target_lengths) if target_lengths else 0
    targets_pad = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, ids in enumerate(targets):
        targets_pad[i, :len(ids)] = torch.tensor(ids)
    return (samples, targets_pad, torch.tensor(target_lengths), glosses)


def decode_and_wer(model, x, targets, target_lengths, vocab, device,
                   beam_width: int = 1):
    """贪心/束搜索解码 + WER/句准确率。返回 (wer, acc, loss)。"""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        lp = torch.log_softmax(logits, dim=2).permute(1, 0, 2)
        loss = F.ctc_loss(lp, targets.to(device),
                          input_lengths=torch.full((len(x),), 32,
                                                   device=device),
                          target_lengths=target_lengths.to(device))
        if beam_width > 1:
            decoded = model.beam_decode(logits, beam_width=beam_width)
        else:
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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--beam-width", type=int, default=5,
                        help="评估解码束宽（1=贪心）")
    parser.add_argument("--augment", action="store_true",
                        help="开启训练在线增强（时间随机窗口/插值 + 空间扰动）")
    parser.add_argument("--min-count", type=int, default=3,
                        help="词表低频过滤：出现次数 < min_count 的词剔除（0=不过滤）")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估 checkpoint（--checkpoint），不训练")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--eval-splits", nargs="+", default=["dev", "test"],
                        help="--eval-only 时评估的 split")
    parser.add_argument("--show-examples", type=int, default=5,
                        help="--eval-only 时打印解码示例数")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 恢复训练（权重/优化器/epoch）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.npz"
    dev_path = data_dir / "dev.npz"
    vocab_path = data_dir / "vocab.npz"
    for p in (train_path, dev_path, vocab_path):
        if not p.exists():
            print(f"缺少 {p}（请先运行 extract_dataset.py）")
            return 1

    vocab_raw = list(np.load(vocab_path, allow_pickle=True)["words"])
    if args.eval_only:
        # 评估模式：直接使用 checkpoint 中的词表
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        vocab = list(ckpt["vocab"])
        print(f"加载 checkpoint: {args.checkpoint}（词表 {len(vocab)}，"
              f"训练时 best WER {ckpt.get('best_wer', '?')}）")
        vocab_idx = {w: i + 1 for i, w in enumerate(vocab)}
        device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if args.device == "auto" else args.device
        model = STGCNCTC(num_classes=len(vocab),
                         adjacency=build_hand_graph(num_hands=2)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        for split in args.eval_splits:
            sp = data_dir / f"{split}.npz"
            if not sp.exists():
                print(f"缺少 {sp}")
                continue
            samples, y, ylen, glosses = load_split(
                sp, vocab_idx, 0.0, args.target_t)
            x = to_tensor_batch(
                [align_length(np.asarray(s, dtype=np.float32), args.target_t)
                 for s in samples], args.target_t)
            wer, acc, loss = decode_and_wer(
                model, x, y, ylen, vocab, device,
                beam_width=args.beam_width)
            print(f"[{split}] {len(samples)} 段：loss {loss:.3f} | "
                  f"WER {wer:.3f} | 句准确率 {acc:.1%}")
            if args.show_examples:
                model.eval()
                with torch.no_grad():
                    logits = model(x.to(device))
                    decoded = model.beam_decode(
                        logits, beam_width=args.beam_width) \
                        if args.beam_width > 1 else model.decode(logits)
                for i in range(min(args.show_examples, len(decoded))):
                    ref = "".join(gloss_words(str(glosses[i])))
                    hyp = "".join(vocab[c - 1] for c in decoded[i]
                                  if 0 < c <= len(vocab))
                    mark = "✓" if ref == hyp else "✗"
                    print(f"  例{i + 1} {mark} 真值: {ref}")
                    print(f"       预测: {hyp or '(空)'}")
        return 0

    if args.min_count > 1:
        # 从 train glosses 统计词频，过滤低频词（先读 glosses）
        d_train_raw = np.load(train_path, allow_pickle=True)
        freq = Counter()
        for g in d_train_raw["glosses"]:
            for w in gloss_words(str(g)):
                freq[w] += 1
        vocab = [w for w in vocab_raw if freq.get(w, 0) >= args.min_count]
        print(f"词表过滤: {len(vocab_raw)} → {len(vocab)} 词"
              f"（min_count={args.min_count}）")
    else:
        vocab = vocab_raw
    vocab_idx = {w: i + 1 for i, w in enumerate(vocab)}
    print(f"词表 {len(vocab)} 词（0=blank）")

    train_samples, y_train, ylen_train, _ = load_split(
        train_path, vocab_idx, MIN_DETECTION, args.target_t)
    dev_samples, y_dev, ylen_dev, _ = load_split(
        dev_path, vocab_idx, MIN_DETECTION, args.target_t)
    print(f"train {len(train_samples)} 段 / dev {len(dev_samples)} 段")

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    model = STGCNCTC(num_classes=len(vocab),
                     adjacency=build_hand_graph(num_hands=2)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    # --resume：恢复权重/优化器/调度器/epoch/best_wer
    start_epoch = 1
    best_wer = 1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        if list(ckpt.get("vocab", [])) != vocab:
            print(f"⚠️ checkpoint 词表与当前不一致，仅加载模型权重")
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_wer = float(ckpt.get("best_wer", 1.0))
        print(f"从 {args.resume} 恢复：epoch {start_epoch} 起，"
              f"best_wer 初始 {best_wer:.3f}")

    train_ds = SkeletonDataset(train_samples, y_train, ylen_train,
                               args.target_t, augment=args.augment)
    # num_workers=0：脚本式运行 + Windows spawn 的兼容性要求
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)
    # dev 用固定对齐（无增强）全量张量化供评估
    x_dev = to_tensor_batch(
        [align_length(np.asarray(s, dtype=np.float32), args.target_t)
         for s in dev_samples], args.target_t)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'epoch':>5} {'train_loss':>10} {'dev_loss':>9} "
          f"{'dev_WER':>8} {'dev_acc':>7}")
    for epoch in range(start_epoch, args.epochs + 1):
        # 学习率 warmup：前 warmup_epochs 个 epoch 线性升温到 base lr
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = lr
        model.train()
        total_loss = 0.0
        n_batch = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}",
                    unit="batch", ncols=110, leave=False)
        for xb, yb, ylb in pbar:
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
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        pbar.close()
        train_loss = total_loss / max(n_batch, 1)

        wer, acc, dev_loss = decode_and_wer(
            model, x_dev, y_dev, ylen_dev, vocab, device,
            beam_width=args.beam_width)
        scheduler.step(dev_loss)
        if wer < best_wer:
            best_wer = wer
            torch.save({"state_dict": model.state_dict(),
                        "vocab": vocab,
                        "config": vars(args),
                        "best_wer": best_wer,
                        "epoch": epoch,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict()},
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
