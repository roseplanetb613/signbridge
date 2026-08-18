"""纯 RGB（ROI-only）CTC 训练：ResNet18 → 时间下采样 → CTC 头。

目的：验证 RGB 单流的翻译能力，对照骨架 baseline（dev WER 0.740）
与融合模型（三流）。可选 --init-from 从 fusion_best.pt 迁移
resnet / roi_tconv 权重（已学 13+ epochs 手部特征），仅新分类头
随机初始化 → 收敛快。

数据：data/dataset/{split}_roi.npz（JPEG 字节，即时解码）
      + {split}.npz（detection_rates 质量过滤 + glosses 标签）

支持：--resume 断点续训（roi_latest.pt > roi_best.pt 自动恢复）、
--eval-only、batch 级断点（每 100 batch 保存 latest）、
ROI 在线增强（随机裁剪/翻转/噪声）、分层 lr（ResNet 主干 ×0.1）。

用法: python scripts/train/train_roi_only.py [--augment] [--init-from checkpoints/fusion_best.pt]
"""

import os
# Windows 排坑：PyTorch OpenMP 与 numpy OpenMP 冲突 → npz object 数组
# 反序列化慢 5000 倍；必须在 import torch/numpy 之前强制单线程。
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset

PUNCT = set("。，？！、；：""''（）《》")
MIN_DETECTION = 0.3
TARGET_T = 128          # 输入 T（与融合训练一致）
ROI_SIZE = 128          # 提取尺寸
CROP_SIZE = 112         # 训练输入（随机裁剪/中心裁剪）


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


def align_series(data: np.ndarray, target_t: int) -> np.ndarray:
    """T 对齐：截断或重复填充（第 0 维为时间）。"""
    t = len(data)
    if t >= target_t:
        return data[:target_t]
    reps = (int(np.ceil(target_t / t)),) + (1,) * (data.ndim - 1)
    return np.tile(data, reps)[:target_t]


class _TemporalDownsample(nn.Module):
    """时间维下采样（stride 2 卷积 + BN + ReLU），与融合模型一致。"""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1),
                              stride=(2, 1), padding=(kernel_size // 2, 0))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class ROICTC(nn.Module):
    """纯 RGB 流 CTC 模型：ResNet18 逐帧 → 时间下采样 ×2 → 分类头。

    输入 roi (N,T,3,112,112) → logits (N, T'=32, K+1)。
    """

    def __init__(self, num_classes, resnet_pretrained=True):
        super().__init__()
        self.num_classes = int(num_classes)
        weights = (torchvision.models.ResNet18_Weights.DEFAULT
                   if resnet_pretrained else None)
        resnet = torchvision.models.resnet18(weights=weights)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet_dim = 512
        self.roi_tconv1 = _TemporalDownsample(self.resnet_dim)
        self.roi_tconv2 = _TemporalDownsample(self.resnet_dim)
        self.head = nn.Conv2d(self.resnet_dim, self.num_classes + 1,
                              kernel_size=1)

    def forward(self, roi):
        """roi (N,T,3,S,S) → logits (N, T', K+1)。"""
        n, t = roi.shape[0], roi.shape[1]
        r = roi.float() / 255.0
        r = r.reshape(n * t, 3, CROP_SIZE, CROP_SIZE)
        r = self.resnet(r).flatten(1)               # (N·T, 512)
        r = r.reshape(n, t, self.resnet_dim).permute(0, 2, 1)  # (N,512,T)
        r = r.unsqueeze(-1)
        r = self.roi_tconv1(r)
        r = self.roi_tconv2(r).squeeze(-1)          # (N, 512, T')
        out = self.head(r.unsqueeze(-1)).squeeze(-1)  # (N, K+1, T')
        return out.permute(0, 2, 1)                 # (N, T', K+1)

    def log_probs(self, roi):
        """→ (T', N, K+1) log-softmax（CTCLoss 标准输入）。"""
        logits = self.forward(roi)
        return torch.log_softmax(logits, dim=2).permute(1, 0, 2)

    def decode(self, logits):
        pred = logits.argmax(dim=2)
        out = []
        for row in pred:
            seq = []
            prev = -1
            for c in row.tolist():
                if c != prev and c != 0:
                    seq.append(c)
                prev = c
            out.append(seq)
        return out

    def beam_decode(self, logits, beam_width=10, top_tokens=20,
                    length_bonus: float = 0.0):
        from signbridge.models.decoding import ctc_beam_search
        lp = torch.log_softmax(logits, dim=2).cpu().numpy()
        return [ctc_beam_search(x, blank=0, beam_width=beam_width,
                                top_tokens=top_tokens,
                                length_bonus=length_bonus)
                for x in lp]


class ROIDataset(Dataset):
    """ROI-only 数据集（JPEG 即时解码；增强逻辑与融合训练一致）。"""

    def __init__(self, rois, targets, target_lengths, augment: bool = False):
        self.rois = rois
        self.targets = targets
        self.target_lengths = target_lengths
        self.augment = augment

    def __len__(self):
        return len(self.rois)

    def __getitem__(self, i):
        frames = []
        for b in self.rois[i]:
            if b is None:
                frames.append(np.zeros((ROI_SIZE, ROI_SIZE, 3),
                                       dtype=np.uint8))
            else:
                frames.append(cv2.imdecode(np.frombuffer(b, np.uint8),
                                           cv2.IMREAD_COLOR))
        roi = np.stack(frames) if frames else np.zeros(
            (TARGET_T, ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8)
        roi = align_series(roi, TARGET_T)
        if self.augment:
            x0 = np.random.randint(0, ROI_SIZE - CROP_SIZE + 1)
            y0 = np.random.randint(0, ROI_SIZE - CROP_SIZE + 1)
            roi = roi[:, y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE, :]
            if np.random.rand() < 0.5:
                roi = roi[:, :, ::-1, :]
            roi = roi.astype(np.float32) + np.random.normal(
                0, 4, roi.shape).astype(np.float32)
        else:
            off = (ROI_SIZE - CROP_SIZE) // 2
            roi = roi[:, off:off + CROP_SIZE, off:off + CROP_SIZE, :]
        rt = torch.from_numpy(
            np.ascontiguousarray(roi)).permute(0, 3, 1, 2)    # (T,3,112,112)
        return rt, self.targets[i], self.target_lengths[i]


def load_roi_split(base: Path, split: str, vocab_idx: dict, min_det: float,
                   max_samples: int = 0, verbose: bool = True):
    """加载 ROI 数据 + 标签（detection_rates 来自 hand npz）。"""
    t0 = time.monotonic()
    hand = np.load(base / f"{split}.npz", allow_pickle=True)
    rates = hand["detection_rates"]
    gloss_arr = hand["glosses"]
    if verbose:
        print(f"[数据] {split}.npz 加载（{time.monotonic() - t0:.0f}s）",
              flush=True)
    t0 = time.monotonic()
    roi_arr = np.load(base / f"{split}_roi.npz",
                      allow_pickle=True)["roi"]
    if verbose:
        print(f"[数据] {split}_roi.npz 加载（{time.monotonic() - t0:.0f}s）",
              flush=True)
    keep = [i for i in range(len(rates)) if rates[i] >= min_det]
    if max_samples > 0:
        keep = keep[:max_samples]
    rois = [roi_arr[i] for i in keep]
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
    return (rois, targets_pad, torch.tensor(target_lengths), glosses)


def decode_and_wer(model, rois, targets, target_lengths, vocab, device,
                   beam_width, length_bonus=0.0):
    """ROI 评估：CTC loss + WER + 句准确率。"""
    model.eval()
    ds = ROIDataset(rois, targets, target_lengths, augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    total_err = total_ref = correct = 0
    total_loss = 0.0
    n_batch = 0
    decoded_all = []
    with torch.no_grad():
        for rt, yb, ylb in loader:
            rt = rt.to(device)
            yb, ylb = yb.to(device), ylb.to(device)
            logits = model(rt)
            lp = torch.log_softmax(logits, dim=2).permute(1, 0, 2)
            loss = F.ctc_loss(lp, yb,
                              input_lengths=torch.full((len(rt),), 32,
                                                       device=device),
                              target_lengths=ylb)
            total_loss += loss.item()
            n_batch += 1
            decoded = (model.beam_decode(logits, beam_width=beam_width,
                                         length_bonus=length_bonus)
                       if beam_width > 1 else model.decode(logits))
            decoded_all.extend(decoded)
    for i, hyp in enumerate(decoded_all):
        ref = [vocab[c - 1]
               for c in targets[i][:int(target_lengths[i])].tolist()]
        hyp_words = [vocab[c - 1] for c in hyp if 0 < c <= len(vocab)]
        total_err += levenshtein(ref, hyp_words)
        total_ref += len(ref)
        if ref == hyp_words:
            correct += 1
    return (total_err / max(total_ref, 1),
            correct / max(len(decoded_all), 1),
            total_loss / max(n_batch, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="纯 RGB（ROI-only）CTC 训练")
    parser.add_argument("--data-dir", type=str, default="data/dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="16GB 内存机器建议 ≤4（ROI 解码占内存）")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resnet-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--length-bonus", type=float, default=0.0,
                        help="评估解码长度偏置（0=不启用）")
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--init-from", type=str, default=None,
                        help="迁移源 checkpoint（fusion_best.pt）："
                             "继承 resnet/roi_tconv 权重，仅新分类头随机")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/roi_best.pt")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="只取前 N 段（0=全部），调试用")
    args = parser.parse_args()

    base = Path(args.data_dir)
    for f in ("train.npz", "train_roi.npz", "dev.npz", "dev_roi.npz",
              "vocab.npz"):
        if not (base / f).exists():
            print(f"缺少 {base / f}")
            return 1

    if args.eval_only:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        vocab = list(ckpt["vocab"])
        model = ROICTC(num_classes=len(vocab), resnet_pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if args.device == "auto" else args.device
        model = model.to(device)
        vocab_idx = {w: i + 1 for i, w in enumerate(vocab)}
        for split in ("dev", "test"):
            sp = base / f"{split}_roi.npz"
            if not sp.exists():
                continue
            rois, y, ylen, _ = load_roi_split(base, split, vocab_idx, 0.0,
                                              args.max_samples)
            wer, acc, loss = decode_and_wer(
                model, rois, y, ylen, vocab, device, args.beam_width,
                args.length_bonus)
            print(f"[{split}] {len(rois)} 段：loss {loss:.3f} | "
                  f"WER {wer:.3f} | 句准确率 {acc:.1%}")
        return 0

    # 词表（min_count 过滤，与 train_fusion 一致）
    vocab_raw = list(np.load(base / "vocab.npz",
                             allow_pickle=True)["words"])
    if args.min_count > 1:
        print("[数据] 加载 train.npz 统计词频...", flush=True)
        t0 = time.monotonic()
        d_raw = np.load(base / "train.npz", allow_pickle=True)
        freq = Counter()
        for g in d_raw["glosses"]:
            for w in gloss_words(str(g)):
                freq[w] += 1
        print(f"[数据]   词频统计完成（{time.monotonic() - t0:.0f}s）",
              flush=True)
        vocab = [w for w in vocab_raw if freq.get(w, 0) >= args.min_count]
        print(f"词表过滤: {len(vocab_raw)} → {len(vocab)}")
    else:
        vocab = vocab_raw
    vocab_idx = {w: i + 1 for i, w in enumerate(vocab)}

    tr = load_roi_split(base, "train", vocab_idx, MIN_DETECTION,
                        args.max_samples)
    de = load_roi_split(base, "dev", vocab_idx, MIN_DETECTION,
                        args.max_samples)
    rois_tr, y_tr, ylen_tr, _ = tr
    rois_de, y_de, ylen_de, _ = de
    print(f"train {len(rois_tr)} 段 / dev {len(rois_de)} 段 / "
          f"词表 {len(vocab)}")

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    model = ROICTC(num_classes=len(vocab),
                   resnet_pretrained=True).to(device)

    # 迁移：从融合 checkpoint 继承 resnet/roi_tconv（新 head 随机）
    if args.init_from:
        src = torch.load(args.init_from, map_location="cpu")
        sd = src["state_dict"]
        own = model.state_dict()
        picked = {k: v for k, v in sd.items()
                  if k.startswith(("resnet.", "roi_tconv")) and k in own}
        own.update(picked)
        model.load_state_dict(own)
        print(f"迁移 {len(picked)} 组权重 ← {args.init_from}"
              f"（best_wer {src.get('best_wer', '?')}）")

    # 分层 lr：ResNet 主干 lr × factor
    resnet_params = set(id(p) for p in model.resnet.parameters())
    base_params = [p for p in model.parameters()
                   if id(p) not in resnet_params]
    resnet_params_list = [p for p in model.parameters()
                          if id(p) in resnet_params]
    optimizer = torch.optim.AdamW([
        {"params": base_params},
        {"params": resnet_params_list, "lr": args.lr * args.resnet_lr_factor},
    ], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    start_epoch = 1
    best_wer = 1.0
    resume_path = args.resume
    if resume_path is None:
        out_dir0 = Path(args.out_dir)
        for cand in ("roi_latest.pt", "roi_best.pt"):
            if (out_dir0 / cand).exists():
                resume_path = str(out_dir0 / cand)
                print(f"检测到 {cand}，自动断点恢复", flush=True)
                break
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_wer = float(ckpt.get("best_wer", 1.0))
        print(f"恢复 {resume_path}：epoch {start_epoch} 起，"
              f"best_wer {best_wer:.3f}")

    train_ds = ROIDataset(rois_tr, y_tr, ylen_tr, augment=args.augment)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "roi_best.pt"
    latest_path = out_dir / "roi_latest.pt"

    def _save_ckpt(path, epoch, best):
        torch.save({"state_dict": model.state_dict(),
                    "vocab": vocab,
                    "config": vars(args),
                    "best_wer": best,
                    "epoch": epoch,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict()},
                   path)

    print(f"{'epoch':>5} {'train_loss':>10} {'dev_loss':>9} "
          f"{'dev_WER':>8} {'dev_acc':>7}")
    for epoch in range(start_epoch, args.epochs + 1):
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            lr = args.lr * epoch / args.warmup_epochs
            for g in optimizer.param_groups:
                g["lr"] = lr if "params" not in g or g is optimizer.param_groups[0] \
                    else max(lr * args.resnet_lr_factor, 1e-6)
        model.train()
        total_loss = 0.0
        n_batch = 0
        from tqdm import tqdm
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}",
                    unit="batch", ncols=110, leave=False)
        for rt, yb, ylb in pbar:
            rt = rt.to(device)
            yb, ylb = yb.to(device), ylb.to(device)
            optimizer.zero_grad()
            lp = model.log_probs(rt)
            loss = F.ctc_loss(lp, yb,
                              input_lengths=torch.full((len(rt),), 32,
                                                       device=device),
                              target_lengths=ylb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
            if n_batch % 100 == 0:
                _save_ckpt(latest_path, epoch, best_wer)
        pbar.close()
        _save_ckpt(latest_path, epoch, best_wer)   # epoch 结束也保存
        train_loss = total_loss / max(n_batch, 1)

        wer, acc, dev_loss = decode_and_wer(
            model, rois_de, y_de, ylen_de, vocab, device, args.beam_width,
            args.length_bonus)
        scheduler.step(dev_loss)
        if wer < best_wer:
            best_wer = wer
            _save_ckpt(ckpt_path, epoch, best_wer)
            suffix = " *"
        else:
            suffix = ""
        print(f"{epoch:>5} {train_loss:>10.4f} {dev_loss:>9.4f} "
              f"{wer:>8.3f} {acc:>7.3f}{suffix}", flush=True)
    print(f"\n最佳 dev WER: {best_wer:.3f} → {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
