"""CTC 解码器：前缀束搜索（经典 Hannun 2016 实现）。

与贪心解码相比，束搜索合并同一前缀的多条对齐路径概率，
可修正"相邻重复/blank 竞争"场景的贪心次优解。
"""

import numpy as np


def ctc_beam_search(log_probs, blank: int = 0, beam_width: int = 10,
                    top_tokens: int = 20) -> list[int]:
    """CTC 前缀束搜索。

    log_probs: (T, K+1) float64 log 概率（行和为 1）。
    返回最优路径的 token 序列（已按 CTC 规则去重合并、去 blank）。
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    T, K = lp.shape
    if T == 0:
        return []

    # prefix -> [p_total, p_blank]（线性概率空间）
    beams = {(): [1.0, 0.0]}
    for t in range(T):
        probs = np.exp(lp[t])
        # 每步只考虑概率最高的 top_tokens 个 token（加速，大词表必需）
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
                        # 重复 token：仅 blank 分隔路径可追加
                        if p_blank > 0.0:
                            add = p_blank * p
                            key = prefix + (c,)
                            entry = new_beams.setdefault(key, [0.0, 0.0])
                            entry[0] += add
                    else:
                        add = p_total * p
                        key = prefix + (c,)
                        entry = new_beams.setdefault(key, [0.0, 0.0])
                        entry[0] += add
        beams = dict(
            sorted(new_beams.items(), key=lambda kv: kv[1][0], reverse=True)[
                :beam_width]
        )
    best = max(beams.items(), key=lambda kv: kv[1][0])[0]
    return list(best)
