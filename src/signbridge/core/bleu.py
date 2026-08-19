"""BLEU 评估指标：n-gram 精确率 + 长度惩罚（BP）。

标准 BLEU（Papineni 2002，corpus 级累计计数）：

    BLEU_N = BP · exp( (1/N) · Σ_n log p_n )
    p_n = Σ min(Count_cand(g), Count_ref(g)) / Σ Count_cand(g)   （n-gram 精确率，截断计数）
    BP  = 1                    （c > r）
          exp(1 - r/c)         （c <= r）

平滑：简单位加一（+1），避免 p_n = 0 时 log 无定义。
"""

import math
from collections import Counter


def ngrams(tokens, n: int):
    """词序列 → n-gram 元组列表。"""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def corpus_bleu(references, hypotheses, max_n: int = 4, smooth: bool = True):
    """corpus 级 BLEU。

    references / hypotheses: list[list[str]]（词序列，长度一致）。
    返回 (bleu, p_n, bp)。
    """
    if not references or len(references) != len(hypotheses):
        raise ValueError("references 与 hypotheses 长度必须一致且非空")

    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)

    p_n = []
    for n in range(1, max_n + 1):
        clipped = 0
        total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_counts = Counter(ngrams(ref, n))
            hyp_counts = Counter(ngrams(hyp, n))
            total += sum(hyp_counts.values())
            clipped += sum(min(c, ref_counts.get(g, 0))
                           for g, c in hyp_counts.items())
        if smooth:
            p_n.append((clipped + 1.0) / (total + 1.0))
        else:
            p_n.append(clipped / total if total > 0 else 0.0)

    if any(p <= 0.0 for p in p_n):
        return 0.0, p_n, 0.0            # 某阶无匹配 → BLEU=0（无平滑时）
    log_sum = sum(math.log(p) for p in p_n) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(
        1.0 - ref_len / max(hyp_len, 1))
    return bp * math.exp(log_sum), p_n, bp


def sentence_bleu(reference, hypothesis, max_n: int = 4):
    """单句 BLEU（corpus_bleu 的便捷封装）。"""
    return corpus_bleu([reference], [hypothesis], max_n=max_n)[0]
