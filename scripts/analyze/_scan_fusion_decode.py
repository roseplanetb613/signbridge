"""融合模型解码侧优化扫描：beam_width × length_bonus × LM 网格。

设计：一次前向缓存 logits（CPU 前向 ~30 分钟/全量 dev），之后所有
解码组合在缓存上秒级扫描（骨架版 _scan_length_bonus.py 每配置重跑
全量前向，融合模型下不可行）。

用法:
  python scripts/analyze/_scan_fusion_decode.py --max-samples 60   # 小样本验证
  python scripts/analyze/_scan_fusion_decode.py                    # 全量 dev

输出:
  - 控制台 markdown 表格（全组合 + WER 排序 Top）
  - reports/scan_fusion/results.json（全部组合指标）
  - reports/scan_fusion/{split}_logits.npy + _meta.json（缓存，复用）
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from lm_score import NGramLM

from signbridge import FusionSTGCNCTC, build_hand_graph
from signbridge.core.graphs import build_adjacency

TARGET_T = 128      # 与 train_fusion.py 一致
TOP_TOKENS = 20
LM_TOPK = 5


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
    """WER / 句准确率 / S/D/I / hyp 均长。"""
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


def main() -> int:
    parser = argparse.ArgumentParser(description="融合模型解码侧优化扫描")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/fusion_best.pt")
    parser.add_argument("--data-dir", type=str, default="data/dataset")
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="只取前 N 段（0=全部），小样本验证用")
    parser.add_argument("--cache-dir", type=str, default="reports/scan_fusion")
    parser.add_argument("--lm", type=str, default="checkpoints/gloss_lm.json",
                        help="LM 路径；空字符串=不加载")
    parser.add_argument("--alpha", type=float, default=0.8, help="LM 权重")
    parser.add_argument("--beam-widths", type=str, default="1 5 10")
    parser.add_argument("--bonuses", type=str, default="0.0 0.5 1.0 1.5 2.0")
    parser.add_argument("--threads", type=int, default=4,
                        help="CPU 线程数（训练期间请调小，避免抢资源）")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    beams = [int(x) for x in args.beam_widths.split()]
    bonuses = [float(x) for x in args.bonuses.split()]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = list(ckpt["vocab"])
    device = args.device
    model = FusionSTGCNCTC(
        num_classes=len(vocab),
        hand_adjacency=build_hand_graph(num_hands=2),
        pose_adjacency=build_adjacency(wb.POSE_CONNECTIONS, 33),
        resnet_pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"checkpoint: {args.checkpoint}（词表 {len(vocab)}，"
          f"best_wer {ckpt.get('best_wer', '?')}）| device {device}")

    lm = NGramLM(args.lm) if args.lm and Path(args.lm).exists() else None
    if lm:
        print(f"LM: {args.lm}（{len(lm.vocab)} 词，alpha {args.alpha}）")

    base = Path(args.data_dir)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_n{args.max_samples or 'all'}"
    logits_path = cache / f"{tag}_logits.npy"
    meta_path = cache / f"{tag}_meta.json"

    if logits_path.exists() and meta_path.exists():
        print(f"复用 logits 缓存: {logits_path}")
        logits = np.load(logits_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        refs = meta["refs"]
    else:
        t0 = time.monotonic()
        d = np.load(base / f"{args.split}.npz", allow_pickle=True)
        data_arr = d["data"]     # NpzFile 无缓存：先取引用，避免重复解压
        pose_img = np.load(base / f"{args.split}_pose.npz",
                           allow_pickle=True)["pose_img"]
        roi_arr = np.load(base / f"{args.split}_roi.npz",
                          allow_pickle=True)["roi"]
        keep = [i for i in range(len(data_arr))
                if float(d["detection_rates"][i]) >= wb.MIN_DETECTION]
        if args.max_samples > 0:
            keep = keep[:args.max_samples]
        refs = [wb.gloss_words(str(d["glosses"][i])) for i in keep]
        print(f"数据 {len(refs)} 段（解压 {time.monotonic() - t0:.0f}s），"
              f"分批前向...", flush=True)
        t0 = time.monotonic()
        logits_list = []
        with torch.no_grad():
            for s in range(0, len(keep), wb.FUSION_BATCH):
                idx = keep[s:s + wb.FUSION_BATCH]
                ht, pt, rt = wb._fusion_batch(
                    [data_arr[i] for i in idx],
                    [pose_img[i] for i in idx],
                    [roi_arr[i] for i in idx], TARGET_T)
                lg = model(ht.to(device), pt.to(device),
                           rt.to(device)).cpu().numpy()
                logits_list.append(lg)
        logits = np.concatenate(logits_list, axis=0)
        print(f"前向完成 {len(refs)} 段 → {tuple(logits.shape)}"
              f"（{time.monotonic() - t0:.0f}s），保存缓存", flush=True)
        np.save(logits_path, logits)
        meta_path.write_text(
            json.dumps({"split": args.split, "n": len(refs), "refs": refs},
                       ensure_ascii=False), encoding="utf-8")

    # ---- 解码扫描 ----
    rows = []
    lg_all = torch.from_numpy(logits)
    for beam in beams:
        for bonus in bonuses:
            if beam == 1 and bonus != bonuses[0]:
                continue        # 贪心与 bonus 无关，只测一次
            if beam == 1:
                with torch.no_grad():
                    decoded = model.decode(lg_all)
                m = compute_metrics(decoded, refs, vocab)
                rows.append({"beam": 1, "bonus": 0.0, "lm": False,
                             "alpha": None, **m})
                print(f"beam=1(贪心): WER {m['wer']:.4f} seg_acc {m['seg_acc']:.3f} "
                      f"del {m['del']} ins {m['ins']} hyp_avg {m['hyp_avg']}",
                      flush=True)
                continue
            for use_lm in (False, True):
                if use_lm and lm is None:
                    continue
                t0 = time.monotonic()
                if use_lm:
                    from signbridge.models.decoding import ctc_beam_search_topk
                    decoded = []
                    for x in logits:
                        lp = torch.log_softmax(torch.from_numpy(x),
                                               dim=1).numpy()
                        cands = ctc_beam_search_topk(
                            lp, blank=0, beam_width=beam,
                            top_tokens=TOP_TOKENS, topk=LM_TOPK,
                            length_bonus=bonus)
                        decoded.append(lm.rescore(cands, alpha=args.alpha))
                else:
                    with torch.no_grad():
                        decoded = model.beam_decode(
                            lg_all, beam_width=beam,
                            top_tokens=TOP_TOKENS, length_bonus=bonus)
                m = compute_metrics(decoded, refs, vocab)
                rows.append({"beam": beam, "bonus": bonus, "lm": use_lm,
                             "alpha": args.alpha if use_lm else None, **m})
                print(f"beam={beam} bonus={bonus} lm={int(use_lm)}: "
                      f"WER {m['wer']:.4f} seg_acc {m['seg_acc']:.3f} "
                      f"del {m['del']} ins {m['ins']} hyp_avg {m['hyp_avg']} "
                      f"（{time.monotonic() - t0:.0f}s）", flush=True)

    # ---- 报告 ----
    rows.sort(key=lambda r: r["wer"])
    print("\n## 全组合（按 WER 升序）")
    print("| beam | bonus | LM | WER | seg_acc | del | ins | sub | hyp_avg |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['beam']} | {r['bonus']:.1f} | {'on' if r['lm'] else '-'} "
              f"| {r['wer']:.4f} | {r['seg_acc']:.3f} | {r['del']} | "
              f"{r['ins']} | {r['sub']} | {r['hyp_avg']} |")
    best = rows[0]
    print(f"\n推荐: beam={best['beam']} bonus={best['bonus']:.1f} "
          f"LM={'on' if best['lm'] else 'off'} → WER {best['wer']:.4f}")

    out = {"checkpoint": args.checkpoint, "split": args.split,
           "n_segments": len(refs), "alpha": args.alpha,
           "rows": rows}
    out_path = cache / "results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"\n结果 → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
