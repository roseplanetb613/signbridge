"""骨架版 TFNet 训练（方案 A）：STGCNCTCTF → 多 CTC + SeqKD。

结构：STGCN blocks（时域路径）+ |FFT| 频域分支 + 融合头。
Loss：CTC_t + CTC_f + CTC_fusion + 25 × SeqKD(频域→时域)。

可选 --init-from checkpoints/best.pt：加载 STGCNCTC 的 blocks + head
作为时域路径初始化（从 0.740 起步，增益直接可见）。

数据：data/dataset/{train,dev,test}.npz（hand 骨架）
保存：checkpoints/best_tf.pt（按 dev WER 最优）

用法: python scripts/train/train_full_tf.py --augment [--init-from checkpoints/best.pt]
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from signbridge import STGCNCTCTF, SeqKD, build_hand_graph

PUNCT = set("。，？！、；：""''（）《》")
MIN_DETECTION = 0.3
MAX_T = 256
KLD_WEIGHT = 25.0          # 论文同款


def time_augment(data: np.ndarray, target_t: int) -> np.ndarray:
    """时间增强：长段随机窗口裁剪 + 短段随机（重复填充 / 时间插值缩放）。"""
    t = len(data)
    if t > target_t:
        start = np.random.randint(0, t - target_t + 1)
        return data[start:start + target_t]
    if t < target_t:
        if np.random.rand() < 0.5:
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
    s = np.random.uniform(*scale_range)
    data = data * s
    theta = np.random.uniform(-rot, rot)
    c, sn = np.cos(theta), np.sin(theta)
    rot_z = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]], dtype=np.float32)
    data = data @ rot_z.T
    data = data + np.random.normal(0, noise, data.shape).astype(np.float32)
    return data


def align_length(data: np.ndarray, target_t: int) -> np.ndarray:
    t = len(data)
    if t >= target_t:
        return data[:target_t]
    reps = int(np.ceil(target_t / t))
    return np.tile(data, (reps, 1, 1))[:target_t]


class SkeletonDataset(Dataset):
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
        data = np.asarray(data, dtype=np.float32)
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
               target_t: int, max_samples: int = 0, verbose: bool = True):
    # NpzFile 无缓存：先取引用再循环（否则每次 d["data"] 全量解压卡死）
    t0 = time.monotonic()
    d = np.load(path, allow_pickle=True)
    data_arr = d["data"]
    rates = d["detection_rates"]
    gloss_arr = d["glosses"]
    if verbose:
        print(f"[数据] {path.stem}.npz 解压（{time.monotonic() - t0:.0f}s）",
              flush=True)
    t0 = time.monotonic()
    keep = [i for i in range(len(data_arr)) if rates[i] >= min_det]
    if max_samples > 0:
        keep = keep[:max_samples]
    if verbose:
        print(f"[数据]   质量过滤 {len(keep)} 段"
              f"（{time.monotonic() - t0:.0f}s）", flush=True)
    t0 = time.monotonic()
    samples = [data_arr[i] for i in keep]
    glosses = [str(gloss_arr[i]) for i in keep]
    if verbose:
        print(f"[数据]   逐段访问 {len(samples)} 段"
              f"（{time.monotonic() - t0:.0f}s）", flush=True)
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
                   beam_width: int = 10, batch_size: int = 32):
    """融合头（logits_fusion）评估：WER/句准确率/loss。分批防 OOM。"""
    model.eval()
    total_err = total_ref = 0
    correct = 0
    total_loss = 0.0
    n_batch = 0
    decoded_all = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            xb = x[s:s + batch_size].to(device)
            yb = targets[s:s + batch_size].to(device)
            ylb = target_lengths[s:s + batch_size].to(device)
            _, _, logits_fusion = model(xb)
            lp = torch.log_softmax(logits_fusion, dim=2).permute(1, 0, 2)
            loss = F.ctc_loss(lp, yb,
                              input_lengths=torch.full((len(xb),), 32,
                                                       device=device),
                              target_lengths=ylb)
            total_loss += loss.item()
            n_batch += 1
            decoded_all.extend(model.beam_decode(logits_fusion,
                                                 beam_width=beam_width))
    for i, hyp in enumerate(decoded_all):
        ref = [vocab[c - 1]
               for c in targets[i][:int(target_lengths[i])].tolist()]
        hyp_words = [vocab[c - 1] for c in hyp if 0 < c <= len(vocab)]
        total_err += levenshtein(ref, hyp_words)
        total_ref += len(ref)
        if ref == hyp_words:
            correct += 1
    wer = total_err / max(total_ref, 1)
    return wer, correct / max(len(decoded_all), 1), total_loss / max(n_batch, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="骨架版 TFNet 训练（方案 A）")
    parser.add_argument("--data-dir", type=str, default="data/dataset")
    parser.add_argument("--target-t", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--kld-weight", type=float, default=KLD_WEIGHT)
    parser.add_argument("--beam-width", type=int, default=10,
                        help="评估解码束宽（论文同款 10）")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--init-from", type=str, default=None,
                        help="STGCNCTC checkpoint（best.pt）：加载 blocks+head"
                             " 作为时域路径初始化")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best_tf.pt")
    parser.add_argument("--eval-splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.npz"
    dev_path = data_dir / "dev.npz"
    vocab_path = data_dir / "vocab.npz"
    for p in (train_path, dev_path, vocab_path):
        if not p.exists():
            print(f"缺少 {p}")
            return 1

    vocab_raw = list(np.load(vocab_path, allow_pickle=True)["words"])
    if args.eval_only:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        vocab = list(ckpt["vocab"])
        print(f"加载 checkpoint: {args.checkpoint}（词表 {len(vocab)}，"
              f"best WER {ckpt.get('best_wer', '?')}）")
        vocab_idx = {w: i + 1 for i, w in enumerate(vocab)}
        device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if args.device == "auto" else args.device
        model = STGCNCTCTF(num_classes=len(vocab),
                           adjacency=build_hand_graph(num_hands=2)).to(device)
        model.load_state_dict(ckpt["state_dict"])
        for split in args.eval_splits:
            sp = data_dir / f"{split}.npz"
            if not sp.exists():
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
        return 0

    if args.min_count > 1:
        print("[数据] 统计 train 词频...", flush=True)
        t0 = time.monotonic()
        d_train_raw = np.load(train_path, allow_pickle=True)
        freq = Counter()
        for g in d_train_raw["glosses"]:
            for w in gloss_words(str(g)):
                freq[w] += 1
        print(f"[数据]   词频统计完成（{time.monotonic() - t0:.0f}s）",
              flush=True)
        vocab = [w for w in vocab_raw if freq.get(w, 0) >= args.min_count]
        print(f"词表过滤: {len(vocab_raw)} → {len(vocab)} 词")
    else:
        vocab = vocab_raw
    vocab_idx = {w: i + 1 for i, w in enumerate(vocab)}
    print(f"词表 {len(vocab)} 词（0=blank）| kld 权重 {args.kld_weight}")

    train_samples, y_train, ylen_train, _ = load_split(
        train_path, vocab_idx, MIN_DETECTION, args.target_t,
        args.max_samples)
    dev_samples, y_dev, ylen_dev, _ = load_split(
        dev_path, vocab_idx, MIN_DETECTION, args.target_t,
        args.max_samples)
    print(f"train {len(train_samples)} 段 / dev {len(dev_samples)} 段")

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    model = STGCNCTCTF(num_classes=len(vocab),
                       adjacency=build_hand_graph(num_hands=2)).to(device)

    # 时域路径初始化（从骨架 STGCNCTC 迁移 blocks + head_t）
    if args.init_from:
        src = torch.load(args.init_from, map_location="cpu")
        if list(src.get("vocab", [])) != vocab:
            print("⚠️ init-from 词表与当前不一致，仅加载匹配权重")
        n = model.load_temporal_state(src["state_dict"])
        print(f"时域路径初始化 ← {args.init_from}（{n} 组权重，"
              f"源 best_wer {src.get('best_wer', '?')}）")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    kld_fn = SeqKD()

    start_epoch = 1
    best_wer = 1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
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
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)
    x_dev = to_tensor_batch(
        [align_length(np.asarray(s, dtype=np.float32), args.target_t)
         for s in dev_samples], args.target_t)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_tf.pt"
    print(f"{'epoch':>5} {'train_loss':>10} {'dev_loss':>9} "
          f"{'dev_WER':>8} {'dev_acc':>7}")
    for epoch in range(start_epoch, args.epochs + 1):
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
            logits_t, logits_f, logits_fusion = model(xb)
            lp_t = model.log_probs(logits_t)
            lp_f = model.log_probs(logits_f)
            lp_fu = model.log_probs(logits_fusion)
            loss = (F.ctc_loss(lp_t, yb,
                               input_lengths=torch.full((len(xb),), 32,
                                                        device=device),
                               target_lengths=ylb)
                    + F.ctc_loss(lp_f, yb,
                                 input_lengths=torch.full((len(xb),), 32,
                                                          device=device),
                                 target_lengths=ylb)
                    + F.ctc_loss(lp_fu, yb,
                                 input_lengths=torch.full((len(xb),), 32,
                                                          device=device),
                                 target_lengths=ylb)
                    + args.kld_weight * kld_fn(logits_f, logits_t))
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
                       ckpt_path)
            suffix = " *"
        else:
            suffix = ""
        print(f"{epoch:>5} {train_loss:>10.4f} {dev_loss:>9.4f} "
              f"{wer:>8.3f} {acc:>7.3f}{suffix}", flush=True)
    print(f"\n最佳 dev WER: {best_wer:.3f} → {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
