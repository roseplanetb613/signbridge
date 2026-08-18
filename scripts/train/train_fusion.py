"""三流融合 CTC 训练：hand + pose + ROI → FusionSTGCNCTC。

数据：data/dataset/{split}.npz + {split}_pose.npz + {split}_roi.npz（同顺序对齐）
支持：--resume 断点续训（optimizer/scheduler/epoch 全恢复）、--eval-only、
分层学习率（ResNet 主干 lr×0.1）、ROI 在线增强（随机裁剪/翻转/颜色抖动）。

用法: python scripts/train/train_fusion.py [--epochs 30] [--batch-size 8]
                                    [--resume checkpoints/fusion_best.pt]
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from signbridge import FusionSTGCNCTC, build_hand_graph
from signbridge.core.graphs import build_adjacency
from signbridge.core.segmentation import extract_segments  # noqa: F401（保持依赖）

try:
    from lm_score import NGramLM
except ImportError:                                     # 独立运行（eval 场景）
    from scripts.train.lm_score import NGramLM          # noqa: E402

PUNCT = set("。，？！、；：""''（）《》")
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)
MIN_DETECTION = 0.3
TARGET_T = 128
ROI_SIZE = 128          # 提取尺寸
CROP_SIZE = 112         # 训练输入（随机裁剪）


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
    """T 对齐：截断或重复填充（支持任意维度，第 0 维为时间）。"""
    t = len(data)
    if t >= target_t:
        return data[:target_t]
    reps = int(np.ceil(target_t / t))
    reps_tuple = (reps,) + (1,) * (data.ndim - 1)
    return np.tile(data, reps_tuple)[:target_t]


class FusionDataset(Dataset):
    """三流数据集：hand/pose 骨架 + ROI 图像（即时 JPEG 解码）。"""

    def __init__(self, hands, poses, rois, targets, target_lengths,
                 augment: bool = False):
        self.hands = hands
        self.poses = poses
        self.rois = rois
        self.targets = targets
        self.target_lengths = target_lengths
        self.augment = augment

    def __len__(self):
        return len(self.hands)

    def __getitem__(self, i):
        h = np.asarray(self.hands[i], dtype=np.float32)
        p = np.asarray(self.poses[i], dtype=np.float32)
        p = np.nan_to_num(p, nan=0.0)
        h = align_series(h, TARGET_T)
        p = align_series(p, TARGET_T)

        # ROI：JPEG 解码 → 128 → 随机/中心裁剪 112 → (T,112,112,3)
        frames = []
        for b in self.rois[i]:
            if b is None:
                frames.append(np.zeros((ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8))
            else:
                img = cv2.imdecode(np.frombuffer(b, np.uint8),
                                   cv2.IMREAD_COLOR)
                frames.append(img)
        roi = np.stack(frames) if frames else \
            np.zeros((TARGET_T, ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8)
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

        ht = torch.from_numpy(h.transpose(2, 0, 1)).float()   # (3,T,42)
        pt = torch.from_numpy(p.transpose(2, 0, 1)).float()   # (3,T,33)
        rt = torch.from_numpy(
            np.ascontiguousarray(roi)).permute(0, 3, 1, 2)    # (T,3,112,112)
        return ht, pt, rt, self.targets[i], self.target_lengths[i]


def load_split(base: Path, split: str, vocab_idx: dict, min_det: float,
               verbose: bool = True):
    # np.load 是惰性的：真正耗时在首次访问数组（解压+反序列化）
    t0 = time.monotonic()
    hand = np.load(base / f"{split}.npz", allow_pickle=True)
    hand_data = hand["data"]
    rates = hand["detection_rates"]
    gloss_arr = hand["glosses"]
    print(f"[数据] {split}.npz 解压完成（{time.monotonic() - t0:.0f}s）",
          flush=True)
    t0 = time.monotonic()
    pose_img = np.load(base / f"{split}_pose.npz",
                       allow_pickle=True)["pose_img"]
    print(f"[数据] {split}_pose.npz 解压完成（{time.monotonic() - t0:.0f}s）",
          flush=True)
    t0 = time.monotonic()
    roi_arr = np.load(base / f"{split}_roi.npz", allow_pickle=True)["roi"]
    print(f"[数据] {split}_roi.npz 解压完成（{time.monotonic() - t0:.0f}s，"
          f"最大文件）", flush=True)
    keep = [i for i in range(len(hand_data))
            if rates[i] >= min_det]
    hands = [hand_data[i] for i in keep]
    poses = [pose_img[i] for i in keep]
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
    return (hands, poses, rois, targets_pad,
            torch.tensor(target_lengths), glosses)


def decode_and_wer(model, hands, poses, rois, targets, target_lengths,
                   vocab, device, beam_width, lm=None, lm_alpha=0.8):
    """三流评估：CTC loss + WER + 句准确率（可选 LM 重打分）。"""
    model.eval()
    ds = FusionDataset(hands, poses, rois, targets, target_lengths,
                       augment=False)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    total_err = total_ref = correct = 0
    total_loss = 0.0
    n_batch = 0
    decoded_all = []
    with torch.no_grad():
        for ht, pt, rt, yb, ylb in loader:
            ht, pt, rt = ht.to(device), pt.to(device), rt.to(device)
            yb, ylb = yb.to(device), ylb.to(device)
            logits = model(ht, pt, rt)
            lp = torch.log_softmax(logits, dim=2).permute(1, 0, 2)
            loss = F.ctc_loss(lp, yb,
                              input_lengths=torch.full((len(ht),), 32,
                                                       device=device),
                              target_lengths=ylb)
            total_loss += loss.item()
            n_batch += 1
            if lm is not None and beam_width > 1:
                # top-k 束候选 + LM 重打分
                from signbridge.models.decoding import ctc_beam_search_topk

                lp_np = lp.cpu().numpy()          # (T', N, K+1)
                decoded = []
                for row in lp_np.T:               # 每样本 (T', K+1)
                    cands = ctc_beam_search_topk(row, blank=0,
                                                 beam_width=beam_width,
                                                 topk=5)
                    decoded.append(lm.rescore(cands, alpha=lm_alpha))
            else:
                decoded = (model.beam_decode(logits, beam_width=beam_width)
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
    return (total_err / max(total_ref, 1), correct / max(len(decoded_all), 1),
            total_loss / max(n_batch, 1))


def main() -> int:
    parser = argparse.ArgumentParser(description="三流融合 CTC 训练")
    parser.add_argument("--data-dir", type=str, default="data/dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="16GB 内存机器建议 ≤4（ROI 数据占内存大）")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resnet-lr-factor", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/fusion_best.pt")
    parser.add_argument("--lm", type=str, default=None,
                        help="词级 n-gram LM 路径（train_lm.py 产出），启用重打分")
    parser.add_argument("--lm-alpha", type=float, default=0.8,
                        help="LM 重打分权重")
    args = parser.parse_args()

    base = Path(args.data_dir)
    for f in ("train.npz", "train_pose.npz", "train_roi.npz",
              "dev.npz", "dev_pose.npz", "dev_roi.npz", "vocab.npz"):
        if not (base / f).exists():
            print(f"缺少 {base / f}（请先完成全量提取）")
            return 1

    if args.eval_only:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        vocab = list(ckpt["vocab"])
        model = FusionSTGCNCTC(
            num_classes=len(vocab),
            hand_adjacency=build_hand_graph(num_hands=2),
            pose_adjacency=build_adjacency(POSE_CONNECTIONS, 33),
            resnet_pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if args.device == "auto" else args.device
        model = model.to(device)
        vocab_idx = {w: i + 1 for i, w in enumerate(vocab)}
        lm = NGramLM(args.lm) if args.lm else None
        for split in ("dev", "test"):
            sp = base / f"{split}.npz"
            if not sp.exists():
                continue
            data = load_split(base, split, vocab_idx, 0.0)
            hands, poses, rois, y, ylen, glosses = data
            wer, acc, loss = decode_and_wer(
                model, hands, poses, rois, y, ylen, vocab, device,
                args.beam_width, lm=lm, lm_alpha=args.lm_alpha)
            print(f"[{split}] {len(hands)} 段：loss {loss:.3f} | "
                  f"WER {wer:.3f} | 句准确率 {acc:.1%}"
                  f"{'（LM 重打分）' if lm else ''}")
        return 0

    # 词表（与 train_full 相同的 min_count 过滤）
    vocab_raw = list(np.load(base / "vocab.npz", allow_pickle=True)["words"])
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

    tr = load_split(base, "train", vocab_idx, MIN_DETECTION)
    de = load_split(base, "dev", vocab_idx, MIN_DETECTION)
    hands_tr, poses_tr, rois_tr, y_tr, ylen_tr, _ = tr
    hands_de, poses_de, rois_de, y_de, ylen_de, _ = de
    print(f"train {len(hands_tr)} 段 / dev {len(hands_de)} 段 / "
          f"词表 {len(vocab)}")

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    model = FusionSTGCNCTC(
        num_classes=len(vocab),
        hand_adjacency=build_hand_graph(num_hands=2),
        pose_adjacency=build_adjacency(POSE_CONNECTIONS, 33),
        resnet_pretrained=True).to(device)

    # 分层学习率：ResNet 主干 lr × factor
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
        # 自动恢复顺序：batch 级 latest > epoch 级 best（兼容旧版进程的产物）
        out_dir0 = Path(args.out_dir)
        for cand in ("fusion_latest.pt", "fusion_best.pt"):
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
        print(f"恢复 {args.resume}：epoch {start_epoch} 起，"
              f"best_wer {best_wer:.3f}")

    train_ds = FusionDataset(hands_tr, poses_tr, rois_tr, y_tr, ylen_tr,
                             augment=args.augment)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "fusion_best.pt"
    latest_path = out_dir / "fusion_latest.pt"

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
        for ht, pt, rt, yb, ylb in pbar:
            ht, pt, rt = ht.to(device), pt.to(device), rt.to(device)
            yb, ylb = yb.to(device), ylb.to(device)
            optimizer.zero_grad()
            lp = model.log_probs(ht, pt, rt)
            loss = F.ctc_loss(lp, yb,
                              input_lengths=torch.full((len(ht),), 32,
                                                       device=device),
                              target_lengths=ylb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}")
            # batch 级断点：每 100 batch 自动保存 latest（崩溃损失 ≤100 batch）
            if n_batch % 100 == 0:
                _save_ckpt(latest_path, epoch, best_wer)
        pbar.close()
        _save_ckpt(latest_path, epoch, best_wer)   # epoch 结束也保存
        train_loss = total_loss / max(n_batch, 1)

        wer, acc, dev_loss = decode_and_wer(
            model, hands_de, poses_de, rois_de, y_de, ylen_de, vocab, device,
            args.beam_width,
            lm=NGramLM(args.lm) if args.lm else None,
            lm_alpha=args.lm_alpha)
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
