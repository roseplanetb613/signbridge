"""三流输入遮挡消融：一次前向提取三流特征，head 层组合遮挡测试。

回答：融合模型里每条流（hand 手骨架 / pose 姿态 / roi RGB）对最终
解码的贡献——把某流输入置零（特征缺失），WER 相对全流涨得越多，
说明该流贡献越大。

背景：单流 head 切片方案不可行（head 权重与单流特征不兼容，输出
发散，见 v1 实验），故采用输入遮挡（input ablation）。

用法:
  python scripts/analyze/ablate_fusion_streams.py --max-samples 40   # 小样本
  python scripts/analyze/ablate_fusion_streams.py                    # 全量 dev

输出:
  - 控制台对比表（7 种输入组合 × length_bonus）
  - reports/ablate_fusion/results.json
"""

import os
# Windows 排坑：PyTorch OpenMP 与 numpy OpenMP 冲突 → npz object 数组
# 反序列化慢 5000 倍；必须在 import torch/numpy 之前强制单线程。
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wer_buckets as wb

from signbridge import FusionSTGCNCTC, build_hand_graph
from signbridge.core.graphs import build_adjacency

TARGET_T = 128
FUSION_BATCH = 8
TOP_TOKENS = 20
ROI_SIZE = 128
ROI_CROP = 112

# 输入组合：特征级遮挡（h/p/r 为各流均值池化后的 (N, C, T') 特征）
CONFIGS = {
    "全流 hand+pose+roi": (True, True, True),
    "无 hand（pose+roi）": (False, True, True),
    "无 pose（hand+roi）": (True, False, True),
    "无 roi（hand+pose）": (True, True, False),
    "仅 hand": (True, False, False),
    "仅 pose": (False, True, False),
    "仅 roi": (False, False, True),
}


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


def compute_metrics(decoded_ids, refs, vocab) -> dict:
    wer_num = wer_den = 0
    dels = inss = subs = 0
    seg_acc = 0
    hyp_lens = []
    for ids, ref in zip(decoded_ids, refs):
        hyp = [vocab[c - 1] for c in ids if 0 < c <= len(vocab)]
        ops = wb.align_words(ref, hyp)
        wer_num += sum(1 for op, _ in ops if op != "match")
        wer_den += len(ref)
        dels += sum(1 for op, _ in ops if op == "del")
        inss += sum(1 for op, _ in ops if op == "ins")
        subs += sum(1 for op, _ in ops if op == "sub")
        seg_acc += 1 if not any(op != "match" for op, _ in ops) else 0
        hyp_lens.append(len(hyp))
    return {
        "wer": round(wer_num / max(wer_den, 1), 4),
        "seg_acc": round(seg_acc / max(len(refs), 1), 4),
        "del": dels, "ins": inss, "sub": subs,
        "hyp_avg": round(float(np.mean(hyp_lens)), 2),
    }


def _roi_batch(roi_samples, target_t):
    """ROI 单流 batch（与 wer_buckets._fusion_batch 的 ROI 路径一致）。"""
    import cv2  # 延迟导入
    out = []
    for frames in roi_samples:
        roi_frames = []
        for b in frames:
            if b is None:
                roi_frames.append(np.zeros((ROI_SIZE, ROI_SIZE, 3),
                                           dtype=np.uint8))
            else:
                roi_frames.append(cv2.imdecode(np.frombuffer(b, np.uint8),
                                               cv2.IMREAD_COLOR))
        roi = np.stack(roi_frames) if roi_frames else np.zeros(
            (target_t, ROI_SIZE, ROI_SIZE, 3), dtype=np.uint8)
        roi = wb.align_length(roi, target_t)
        off = (ROI_SIZE - ROI_CROP) // 2
        roi = roi[:, off:off + ROI_CROP, off:off + ROI_CROP, :]
        out.append(np.ascontiguousarray(roi).transpose(0, 3, 1, 2))
    return torch.from_numpy(np.stack(out)).float()


def stream_features(full, ht, pt, rt):
    """三流骨干特征（与 FusionSTGCNCTC.forward 前半段一致）。"""
    with torch.no_grad():
        h = ht
        for block in full.hand_blocks:
            h = block(h)
        h = h.mean(dim=3)                           # (N, 256, T')

        p = pt
        for block in full.pose_blocks:
            p = block(p)
        p = p.mean(dim=3)                           # (N, 256, T')

        n, t = rt.shape[0], rt.shape[1]
        r = rt.float() / 255.0
        r = r.reshape(n * t, 3, ROI_CROP, ROI_CROP)
        r = full.resnet(r).flatten(1)               # (N·T, 512)
        r = r.reshape(n, t, 512).permute(0, 2, 1).unsqueeze(-1)
        r = full.roi_tconv1(r)
        r = full.roi_tconv2(r).squeeze(-1)          # (N, 512, T')
    return h, p, r


def head_logits(full, h, p, r):
    f = torch.cat([h, p, r], dim=1).unsqueeze(-1)
    out = full.head(f).squeeze(-1)                  # (N, K+1, T')
    return out.permute(0, 2, 1)                     # (N, T', K+1)


def beam_decode(logits, beam_width, length_bonus):
    from signbridge.models.decoding import ctc_beam_search
    lp = torch.log_softmax(logits, dim=2).cpu().numpy()
    return [ctc_beam_search(x, blank=0, beam_width=beam_width,
                            top_tokens=TOP_TOKENS,
                            length_bonus=length_bonus)
            for x in lp]


def greedy_decode(logits):
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


def main() -> int:
    parser = argparse.ArgumentParser(description="融合模型三流输入遮挡消融")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/fusion_best.pt")
    parser.add_argument("--data-dir", type=str, default="data/dataset")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--bonuses", type=str, default="0.0 1.0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--min-det", type=float, default=wb.MIN_DETECTION)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="reports/ablate_fusion")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    bonuses = [float(x) for x in args.bonuses.split()]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = list(ckpt["vocab"])
    full = FusionSTGCNCTC(
        num_classes=len(vocab),
        hand_adjacency=build_hand_graph(num_hands=2),
        pose_adjacency=build_adjacency(wb.POSE_CONNECTIONS, 33),
        resnet_pretrained=False).to(args.device)
    full.load_state_dict(ckpt["state_dict"])
    full.eval()
    print(f"checkpoint: {args.checkpoint}（词表 {len(vocab)}，"
          f"best_wer {ckpt.get('best_wer', '?')}）| device {args.device}")

    base = Path(args.data_dir)
    d = np.load(base / f"{args.split}.npz", allow_pickle=True)
    data_arr = d["data"]     # NpzFile 无缓存：先取引用，避免重复解压
    pose_img = np.load(base / f"{args.split}_pose.npz",
                       allow_pickle=True)["pose_img"]
    roi_arr = np.load(base / f"{args.split}_roi.npz",
                      allow_pickle=True)["roi"]
    keep = [i for i in range(len(data_arr))
            if float(d["detection_rates"][i]) >= args.min_det]
    if args.max_samples > 0:
        keep = keep[:args.max_samples]
    refs = [wb.gloss_words(str(d["glosses"][i])) for i in keep]
    print(f"{args.split} {len(refs)} 段 | beam {args.beam_width} | "
          f"bonuses {bonuses}")

    # 一次前向：分批提取三流特征
    t0 = time.monotonic()
    H, P, R = [], [], []
    with torch.no_grad():
        for s in range(0, len(keep), FUSION_BATCH):
            idx = keep[s:s + FUSION_BATCH]
            ht = wb.to_tensor_batch(
                [wb.align_length(np.asarray(data_arr[i], dtype=np.float32),
                                 TARGET_T) for i in idx], TARGET_T)
            pt = wb.to_tensor_batch(
                [wb.align_length(np.asarray(pose_img[i], dtype=np.float32),
                                 TARGET_T) for i in idx], TARGET_T)
            rt = _roi_batch([roi_arr[i] for i in idx], TARGET_T)
            h, p, r = stream_features(full, ht.to(args.device),
                                      pt.to(args.device), rt.to(args.device))
            H.append(h.cpu().numpy())
            P.append(p.cpu().numpy())
            R.append(r.cpu().numpy())
    H = torch.from_numpy(np.concatenate(H, axis=0))
    P = torch.from_numpy(np.concatenate(P, axis=0))
    R = torch.from_numpy(np.concatenate(R, axis=0))
    print(f"特征提取完成 {len(refs)} 段（{time.monotonic() - t0:.0f}s）",
          flush=True)

    rows = []
    for cfg_name, (use_h, use_p, use_r) in CONFIGS.items():
        hh = H if use_h else torch.zeros_like(H)
        pp = P if use_p else torch.zeros_like(P)
        rr = R if use_r else torch.zeros_like(R)
        with torch.no_grad():
            logits = head_logits(full, hh, pp, rr)
        for bonus in bonuses:
            t0 = time.monotonic()
            with torch.no_grad():
                decoded = (beam_decode(logits, args.beam_width, bonus)
                           if args.beam_width > 1 else greedy_decode(logits))
            m = compute_metrics(decoded, refs, vocab)
            m.update({"config": cfg_name, "bonus": bonus})
            rows.append(m)
            print(f"[{cfg_name}] bonus={bonus}: WER {m['wer']:.4f} "
                  f"seg_acc {m['seg_acc']:.3f} del {m['del']} ins {m['ins']} "
                  f"sub {m['sub']} hyp_avg {m['hyp_avg']}"
                  f"（{time.monotonic() - t0:.0f}s）", flush=True)

    print("\n## 输入遮挡消融（dev，beam=%d）" % args.beam_width)
    print("| 输入组合 | bonus | WER | seg_acc | del | ins | sub | hyp_avg |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in sorted(rows, key=lambda x: (x["bonus"], x["wer"])):
        print(f"| {r['config']} | {r['bonus']:.1f} | {r['wer']:.4f} "
              f"| {r['seg_acc']:.3f} | {r['del']} | {r['ins']} "
              f"| {r['sub']} | {r['hyp_avg']} |")
    print("\n参照: 骨架 STGCNCTC 全量训练 best.pt = 0.740；"
          "融合三流（训练中 best）= %.4f" % ckpt.get("best_wer", 0.0))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"checkpoint": args.checkpoint, "split": args.split,
           "beam_width": args.beam_width, "n_segments": len(refs),
           "rows": rows}
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\n结果 → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
