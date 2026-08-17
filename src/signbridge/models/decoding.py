"""CTC 解码器：前缀束搜索（经典 Hannun 2016 实现）。

与贪心解码相比，束搜索合并同一前缀的多条对齐路径概率，
可修正"相邻重复/blank 竞争"场景的贪心次优解。
ctc_beam_search_topk 返回 top-k 候选，供语言模型重打分。
"""

import numpy as np


def ctc_beam_search_topk(log_probs, blank: int = 0, beam_width: int = 10,
                         top_tokens: int = 20, topk: int = 5,
                         length_bonus: float = 0.0):
    """CTC 前缀束搜索，返回 top-k 候选 [(prob, tokens), ...]。

    length_bonus: 每输出一个非 blank 词额外乘 (1+length_bonus)，
    鼓励更长的序列（缓解 CTC 欠预测/漏词问题）。0=不启用。
    log_probs: (T, K+1) float64 log 概率（行和为 1）。
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    T, K = lp.shape
    if T == 0:
        return [(1.0, [])]

    len_boost = 1.0 + max(length_bonus, 0.0)
    beams = {(): [1.0, 0.0]}   # prefix -> [p_total, p_blank]
    for t in range(T):
        probs = np.exp(lp[t])
        top = np.argsort(probs)[::-1][:min(top_tokens, K)]
        new_beams: dict = {}
        for prefix, (p_total, p_blank) in beams.items():
            for c in top:
                p = probs[c]
                if p <= 0.0:
                    continue
                if c == blank:
                    entry = new_beams.setdefault(prefix, [0.0, 0.0])
                    add = p_total * p
                    entry[0] += add
                    entry[1] += add
                else:
                    if prefix and prefix[-1] == c:
                        if p_blank > 0.0:
                            add = p_blank * p * len_boost
                            key = prefix + (c,)
                            entry = new_beams.setdefault(key, [0.0, 0.0])
                            entry[0] += add
                    else:
                        add = p_total * p * len_boost
                        key = prefix + (c,)
                        entry = new_beams.setdefault(key, [0.0, 0.0])
                        entry[0] += add
        beams = dict(
            sorted(new_beams.items(), key=lambda kv: kv[1][0], reverse=True)[
                :beam_width]
        )
    ranked = sorted(beams.items(), key=lambda kv: kv[1][0], reverse=True)
    return [(float(prob), list(tokens))
            for tokens, (prob, _) in ranked[:topk]]


def ctc_beam_search(log_probs, blank: int = 0, beam_width: int = 10,
                    top_tokens: int = 20, length_bonus: float = 0.0
                    ) -> list[int]:
    """CTC 前缀束搜索，返回最优路径 token 序列（兼容接口）。

    length_bonus 语义同 ctc_beam_search_topk。
    """
    ranked = ctc_beam_search_topk(log_probs, blank=blank,
                                  beam_width=beam_width,
                                  top_tokens=top_tokens, topk=1,
                                  length_bonus=length_bonus)
    return ranked[0][1] if ranked else []
