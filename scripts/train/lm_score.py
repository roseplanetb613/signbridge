"""词级 n-gram 语言模型加载与 CTC 解码重打分。

用法（train_fusion/train_full 评估集成）：
    lm = NGramLM("checkpoints/gloss_lm.json")
    best = lm.rescore(candidates, vocab, alpha=0.8)
    # candidates: [(ctc_prob, [token_id...]), ...]（来自 ctc_beam_search_topk）
"""

import json
import math
from pathlib import Path

BACKOFF = 0.4


class NGramLM:
    """trigram + stupid backoff 词级语言模型。"""

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = list(data["vocab"])
        self.n = int(data["n"])
        self.backoff = float(data.get("backoff", BACKOFF))
        counts = data["counts"]
        self.c1 = {tuple(k[:-1]): k[-1] for k in counts[0]}
        self.c2 = {tuple(k[:-1]): k[-1] for k in counts[1]}
        self.c3 = {tuple(k[:-1]): k[-1] for k in counts[2]}
        self.total1 = sum(self.c1.values())
        # 句尾标记计数
        self.eos2 = sum(v for (a, b), v in self.c2.items() if b == "</s>")

    def log_prob(self, tokens):
        """词 id 序列（含 blank 语义已由 CTC 处理）→ 归一化 log 概率。"""
        if not tokens:
            return 0.0
        lp = 0.0
        prev2, prev1 = "<s>", "<s>"
        for t in tokens:
            w = self.vocab[t - 1] if 0 < t <= len(self.vocab) else "<unk>"
            tri = self.c3.get((prev2, prev1, w), 0)
            if tri > 0:
                lp += math.log(tri / self.c2.get((prev1,), 1))
            else:
                bi = self.c2.get((prev1, w), 0)
                if bi > 0:
                    lp += math.log(self.backoff * bi / self.c1.get((prev1,), 1))
                else:
                    uni = self.c1.get((w,), 0)
                    lp += math.log(self.backoff * self.backoff *
                                   max(uni, 1) / self.total1)
            prev2, prev1 = prev1, w
        # 句尾
        tri_e = self.c3.get((prev2, prev1, "</s>"), 0)
        if tri_e > 0:
            lp += math.log(tri_e / self.c2.get((prev1,), 1))
        else:
            bi_e = self.c2.get((prev1, "</s>"), 0)
            lp += math.log(self.backoff * bi_e / self.c1.get((prev1,), 1)) \
                if bi_e > 0 else math.log(self.backoff * self.backoff /
                                          self.total1)
        # 长度归一化（避免长句惩罚）
        return lp / max(len(tokens), 1)

    def rescore(self, candidates, alpha=0.8):
        """对束搜索候选 [(ctc_log_prob, tokens), ...] 重打分，返回最优 tokens。

        score = ctc_log_prob + alpha * lm_log_prob（长度归一）。
        """
        if not candidates:
            return []
        best_tokens, best_score = None, -1e18
        for ctc_prob, tokens in candidates:
            lm_score = self.log_prob(tokens)
            combined = math.log(max(ctc_prob, 1e-300)) + alpha * lm_score
            if combined > best_score:
                best_score = combined
                best_tokens = tokens
        return best_tokens
