"""词级 n-gram 语言模型训练（trigram + stupid backoff，纯 Python 零依赖）。

用途：CTC 束搜索解码重打分（低概率词序列用 LM 校正）。

用法: python scripts/train/train_lm.py [--data-dir data/dataset]
                                       [--out checkpoints/gloss_lm.json]
                                       [--n 3]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PUNCT = set("。，？！、；：""''（）《》")
BACKOFF = 0.4   # stupid backoff 因子


def gloss_words(gloss: str) -> list[str]:
    return [w for w in gloss.split("/") if w.strip() and w.strip() not in PUNCT]


def train_ngram(sentences: list[list[str]], n: int):
    """n-gram 计数 + vocab。返回 (vocab, counts_by_order)。"""
    vocab = sorted({w for s in sentences for w in s})
    counts = []
    for order in range(1, n + 1):
        c = Counter()
        for s in sentences:
            padded = ["<s>"] * (order - 1) + s + ["</s>"]
            for i in range(len(padded) - order + 1):
                c[tuple(padded[i:i + order])] += 1
        counts.append(c)
    return vocab, counts


def log_prob(counts, n, backoff):
    """对词序列计算 stupid backoff log 概率。"""
    import math

    total_trigrams = sum(counts[2].values())
    total_bigrams = sum(counts[1].values())
    total_unigrams = sum(counts[0].values())

    def p(seq):
        """seq: 词列表（无 <s>/</s>）。"""
        lp = 0.0
        prev2, prev1 = "<s>", "<s>"
        for w in seq:
            tri = counts[2][(prev2, prev1, w)]
            if tri > 0:
                lp += math.log(tri / counts[1][(prev1,)])
            else:
                bi = counts[1][(prev1, w)]
                if bi > 0:
                    lp += math.log(BACKOFF * bi / counts[0][(prev1,)])
                else:
                    uni = counts[0][(w,)]
                    lp += math.log(BACKOFF * BACKOFF * uni / total_unigrams)
            prev2, prev1 = prev1, w
        # 句尾
        tri_e = counts[2][(prev2, prev1, "</s>")]
        if tri_e > 0:
            lp += math.log(tri_e / counts[1][(prev1,)])
        else:
            bi_e = counts[1][(prev1, "</s>")]
            lp += math.log(BACKOFF * bi_e / counts[0][(prev1,)]) if bi_e > 0 \
                else math.log(BACKOFF * BACKOFF / total_unigrams)
        return lp

    return p


def main() -> int:
    parser = argparse.ArgumentParser(description="gloss n-gram LM 训练")
    parser.add_argument("--data-dir", type=str, default="data/dataset")
    parser.add_argument("--out", type=str, default="checkpoints/gloss_lm.json")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    train_path = Path(args.data_dir) / "train.npz"
    if not train_path.exists():
        print(f"缺少 {train_path}")
        return 1
    glosses = np.load(train_path, allow_pickle=True)["glosses"]
    sentences = [gloss_words(str(g)) for g in glosses]
    sentences = [s for s in sentences if s]
    print(f"句子 {len(sentences)}，词数 {sum(len(s) for s in sentences)}")

    vocab, counts = train_ngram(sentences, args.n)
    print(f"词表 {len(vocab)}，trigram 数 {len(counts[2])}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "vocab": vocab,
            "n": args.n,
            "backoff": BACKOFF,
            "counts": [[list(k) + [v] for k, v in c.items()]
                       for c in counts],
        }, f, ensure_ascii=False)
    print(f"LM 保存 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
