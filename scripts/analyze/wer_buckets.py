"""WER 分桶分析：按句子长度 / 手语者 / 词频 细粒度定位错误。

用法:
  python scripts/analyze/wer_buckets.py [--checkpoint checkpoints/best.pt]
                                        [--splits dev test]
                                        [--beam-width 5] [--min-det 0.3]
                                        [--out-dir reports/wer_buckets]
                                        [--show-examples 3]

输出:
  - 控制台 markdown 表格（句长桶 / signer 桶 / 词频桶，含 S/D/I 细分）
  - reports/wer_buckets/results.json（全量逐样本 + 聚合）
  - reports/wer_buckets/wer_buckets.png（三张柱状图）

对齐口径：WER 分子 = 替换 + 删除 + 插入；词级错误率 = (替换+删除) / ref 词数。
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from signbridge import STGCNCTC, build_hand_graph

PUNCT = set("。，？！、；：""''（）《》")
MIN_DETECTION = 0.3
MAX_T = 256


def gloss_words(gloss: str) -> list[str]:
    return [w for w in gloss.split("/") if w.strip() and w.strip() not in PUNCT]


def align_words(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None]]:
    """词级 DP 对齐（回溯），返回 [(op, word)]，op ∈ match/sub/del/ins。

    match/sub/del 对应 ref 词；ins 的 word 为 hyp 词。
    """
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    ops: list[tuple[str, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (
                0 if ref[i - 1] == hyp[j - 1] else 1):
            op = "match" if ref[i - 1] == hyp[j - 1] else "sub"
            ops.append((op, ref[i - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1]))
            i -= 1
        else:
            ops.append(("ins", hyp[j - 1]))
            j -= 1
    return list(reversed(ops))


def word_freq_bucket(freq: Counter, word: str) -> str:
    c = freq.get(word, 0)
    if c >= 50:
        return "高频 ≥50"
    if c >= 20:
        return "中频 20-49"
    if c >= 5:
        return "低频 5-19"
    return "极低频 1-4"


def align_length(data: np.ndarray, target_t: int) -> np.ndarray:
    t = len(data)
    if t >= target_t:
        return data[:target_t]
    reps = int(np.ceil(target_t / t))
    return np.tile(data, (reps, 1, 1))[:target_t]


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


def len_bucket(n_words: int) -> str:
    if n_words <= 3:
        return "短句 ≤3"
    if n_words <= 5:
        return "中句 4-5"
    if n_words <= 7:
        return "长句 6-7"
    return "超长句 ≥8"


def agg_bucket(bucket_key: str, rows: list[dict]) -> dict:
    """聚合一个桶：WER / S/D/I 计数 / 句准确率。"""
    wer = sum(r["errors"] for r in rows) / max(
        sum(r["n_ref"] for r in rows), 1)
    counts = Counter(op for r in rows for op, _ in r["ops"])
    return {
        "bucket": bucket_key,
        "n_segments": len(rows),
        "n_ref_words": sum(r["n_ref"] for r in rows),
        "n_hyp_words": sum(len(r["hyp"]) for r in rows),
        "wer": round(wer, 4),
        "sub": counts["sub"], "del": counts["del"], "ins": counts["ins"],
        "match": counts["match"],
        "seg_acc": sum(1 for r in rows if r["errors"] == 0)
        / max(len(rows), 1),
    }


def analyze_split(model, split_path: Path, vocab, vocab_idx, device,
                  target_t: int, beam_width: int, min_det: float,
                  train_freq: Counter, length_bonus: float = 0.0) -> dict:
    d = np.load(split_path, allow_pickle=True)
    keep = [i for i in range(len(d["data"]))
            if float(d["detection_rates"][i]) >= min_det]
    samples = [d["data"][i] for i in keep]
    glosses = [str(d["glosses"][i]) for i in keep]
    translators = [str(d["translators"][i]) for i in keep]
    videos = [str(d["videos"][i]) for i in keep]

    x = to_tensor_batch(
        [align_length(np.asarray(s, dtype=np.float32), target_t)
         for s in samples], target_t)
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        decoded = (model.beam_decode(logits, beam_width=beam_width,
                                     length_bonus=length_bonus)
                   if beam_width > 1 else model.decode(logits))

    per_sample = []
    for i, hyp_ids in enumerate(decoded):
        ref = gloss_words(glosses[i])
        hyp = [vocab[c - 1] for c in hyp_ids if 0 < c <= len(vocab)]
        ops = align_words(ref, hyp)
        err = sum(1 for op, _ in ops if op != "match")
        per_sample.append({
            "video": videos[i],
            "translator": translators[i],
            "ref": ref,
            "hyp": hyp,
            "ops": ops,
            "errors": err,
            "n_ref": len(ref),
        })

    def agg(bucket_key: str, rows: list[dict]) -> dict:
        return agg_bucket(bucket_key, rows)

    # 1) 句长桶
    len_groups = defaultdict(list)
    for r in per_sample:
        len_groups[len_bucket(r["n_ref"])].append(r)
    len_rows = [agg(k, v) for k, v in
                sorted(len_groups.items(), key=lambda kv: len(kv[0]))]

    # 2) signer 桶（H/I 样本少，单独成桶并提示）
    signer_groups = defaultdict(list)
    for r in per_sample:
        signer_groups[r["translator"]].append(r)
    signer_rows = [agg(k, v) for k, v in sorted(signer_groups.items())]

    # 3) 词频桶：按 ref 词频率统计词级 match/sub/del
    freq_groups = defaultdict(lambda: {"n": 0, "match": 0, "err": 0})
    for r in per_sample:
        for op, w in r["ops"]:
            if op == "ins":
                continue
            b = word_freq_bucket(train_freq, w)
            freq_groups[b]["n"] += 1
            freq_groups[b]["err"] += 0 if op == "match" else 1
            freq_groups[b]["match"] += 1 if op == "match" else 0
    freq_rows = [{
        "bucket": k,
        "n_ref_words": v["n"],
        "word_err_rate": round(v["err"] / max(v["n"], 1), 4),
        "match": v["match"],
        "err": v["err"],
    } for k, v in sorted(freq_groups.items(),
                         key=lambda kv: -kv[1]["n"])]

    return {
        "split": split_path.stem,
        "n_segments": len(per_sample),
        "overall_wer": round(
            sum(r["errors"] for r in per_sample)
            / max(sum(r["n_ref"] for r in per_sample), 1), 4),
        "seg_acc": round(
            sum(1 for r in per_sample if r["errors"] == 0)
            / max(len(per_sample), 1), 4),
        "len_buckets": len_rows,
        "signer_buckets": signer_rows,
        "freq_buckets": freq_rows,
        "per_sample": per_sample,
    }


def print_table(title: str, rows: list[dict], cols: list[str]) -> None:
    print(f"\n## {title}")
    header = " | ".join(cols)
    print(header)
    print(" | ".join("---" for _ in cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))


def main() -> int:
    parser = argparse.ArgumentParser(description="WER 分桶分析")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--data-dir", type=str, default="data/dataset")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    parser.add_argument("--target-t", type=int, default=128)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--length-bonus", type=float, default=0.0,
                        help="解码长度偏置：每输出一词乘 (1+bonus)，"
                             "缓解欠预测（0=不启用）")
    parser.add_argument("--min-det", type=float, default=MIN_DETECTION)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="reports/wer_buckets")
    parser.add_argument("--show-examples", type=int, default=0,
                        help="每个句长桶打印最差样例数（0=不打印）")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = list(ckpt["vocab"])
    vocab_idx = {w: i + 1 for i, w in enumerate(vocab)}
    print(f"checkpoint: {args.checkpoint}（词表 {len(vocab)}，"
          f"训练 best WER {ckpt.get('best_wer', '?')}）")

    # 训练集词频（词频桶用）
    train_path = data_dir / "train.npz"
    train_freq = Counter()
    if train_path.exists():
        dt = np.load(train_path, allow_pickle=True)
        for g in dt["glosses"]:
            for w in gloss_words(str(g)):
                train_freq[w] += 1
        print(f"train 词频统计: {len(train_freq)} 词")

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    model = STGCNCTC(num_classes=len(vocab),
                     adjacency=build_hand_graph(num_hands=2)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for split in args.splits:
        sp = data_dir / f"{split}.npz"
        if not sp.exists():
            print(f"缺少 {sp}")
            continue
        res = analyze_split(model, sp, vocab, vocab_idx, device,
                            args.target_t, args.beam_width, args.min_det,
                            train_freq, args.length_bonus)
        results[split] = res
        print(f"\n=== [{split}] {res['n_segments']} 段 | "
              f"整体 WER {res['overall_wer']:.3f} | "
              f"句准确率 {res['seg_acc']:.1%} ===")
        print_table("按句子长度", res["len_buckets"],
                    ["bucket", "n_segments", "n_ref_words", "wer",
                     "sub", "del", "ins", "seg_acc"])
        print_table("按手语者", res["signer_buckets"],
                    ["bucket", "n_segments", "n_ref_words", "wer",
                     "sub", "del", "ins", "seg_acc"])
        print_table("按词频（词级）", res["freq_buckets"],
                    ["bucket", "n_ref_words", "word_err_rate", "match", "err"])
        if args.show_examples > 0:
            worst = sorted(res["per_sample"], key=lambda r: -r["errors"])[
                :args.show_examples]
            print(f"\n最差 {args.show_examples} 例:")
            for r in worst:
                print(f"  {r['video']} ({r['translator']}, "
                      f"{r['n_ref']} 词, {r['errors']} 错)")
                print(f"    真值: {'/'.join(r['ref'])}")
                print(f"    预测: {'/'.join(r['hyp']) or '(空)'}")

    # 全量 JSON（含逐样本，供后续分析）
    slim = {}
    for split, res in results.items():
        slim[split] = {k: v for k, v in res.items() if k != "per_sample"}
        slim[split]["per_sample"] = [
            {k: v for k, v in r.items() if k != "ops"} | {"ops":
             [f"{op}:{w}" for op, w in r["ops"]]}
            for r in res["per_sample"]]
    (out_dir / "results.json").write_text(
        json.dumps(slim, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果 → {out_dir / 'results.json'}")

    # 图表
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                           "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
        if n == 1:
            axes = [axes]
        for ax, (split, res) in zip(axes, results.items()):
            labels = [r["bucket"] for r in res["len_buckets"]]
            wers = [r["wer"] for r in res["len_buckets"]]
            ax.bar(range(len(labels)), wers, color="#4c72b0")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("WER")
            ax.set_title(f"{split} 按句子长度\n整体 WER {res['overall_wer']:.3f}")
            for i, w in enumerate(wers):
                ax.text(i, w + 0.01, f"{w:.2f}", ha="center", fontsize=8)
        fig.tight_layout()
        fig_path = out_dir / "wer_buckets.png"
        fig.savefig(fig_path, dpi=150)
        print(f"图表 → {fig_path}")
    except ImportError:
        print("matplotlib 不可用，跳过图表")
    return 0


if __name__ == "__main__":
    sys.exit(main())
