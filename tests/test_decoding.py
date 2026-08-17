"""CTC 束搜索解码测试。"""

import numpy as np
import pytest

from signbridge.models.decoding import ctc_beam_search


def _log_probs(probs_rows):
    """概率行列表 → (T, K) log 概率（自动归一化）。"""
    arr = np.array(probs_rows, dtype=np.float64)
    arr = arr / arr.sum(axis=1, keepdims=True)
    return np.log(arr)


def test_beam_width_1_equals_greedy():
    # 高置信度序列：每步 argmax 明确
    probs = [[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]]
    lp = _log_probs(probs)          # K=2（0=blank, 1=token）
    beam = ctc_beam_search(lp, blank=0, beam_width=1)
    assert beam == [1, 1]           # 与贪心一致（blank 分隔的两个 1）


def test_beam_beats_greedy():
    """标准构造：多个对齐路径合并使 [A] 最优，贪心却输出 [A,A]。"""
    probs = [
        [0.4, 0.6],   # t0: blank 0.4 / A 0.6
        [0.6, 0.4],   # t1: blank 0.6 / A 0.4
        [0.4, 0.6],   # t2: blank 0.4 / A 0.6
    ]
    lp = _log_probs(probs)
    greedy = [int(np.argmax(row)) for row in lp]
    # 贪心：t0→A, t1→blank, t2→A → 解码 [A, A]
    gseq = []
    prev = -1
    for c in greedy:
        if c != prev and c != 0:
            gseq.append(c)
        prev = c
    assert gseq == [1, 1]
    # 束搜索（宽度 3）应找到合并路径 → [A]
    beam = ctc_beam_search(lp, blank=0, beam_width=3)
    assert beam == [1]


def test_empty_output():
    lp = _log_probs([[0.9, 0.1], [0.8, 0.2]])
    assert ctc_beam_search(lp, blank=0, beam_width=5) == []


def test_single_token():
    lp = _log_probs([[0.2, 0.8], [0.1, 0.9]])
    assert ctc_beam_search(lp, blank=0, beam_width=5) == [1]


def test_repeated_token_needs_blank():
    # [A, blank, A] → [A, A]（blank 分隔的重复）
    lp = _log_probs([[0.2, 0.8], [0.9, 0.1], [0.2, 0.8]])
    assert ctc_beam_search(lp, blank=0, beam_width=5) == [1, 1]


def test_larger_vocab():
    probs = [[0.1, 0.6, 0.3], [0.2, 0.1, 0.7], [0.8, 0.1, 0.1]]
    lp = _log_probs(probs)
    beam = ctc_beam_search(lp, blank=0, beam_width=5)
    assert beam == [1, 2]           # token1, blank, token2 → [1, 2]


def test_length_bonus_prefers_longer():
    """低置信度下默认输出短序列，加 length_bonus 后倾向更长。"""
    probs = [[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]]
    lp = _log_probs(probs)          # 每步 blank 占优；T=3 最多 2 词（重复需 blank 分隔）
    assert ctc_beam_search(lp, blank=0, beam_width=5) == [1]
    # bonus 足够大 → 输出满长 [1, blank, 1]
    assert ctc_beam_search(lp, blank=0, beam_width=5,
                           length_bonus=5.0) == [1, 1]


def test_length_bonus_zero_is_noop():
    probs = [[0.6, 0.4], [0.6, 0.4]]
    lp = _log_probs(probs)
    assert ctc_beam_search(lp, blank=0, beam_width=5,
                           length_bonus=0.0) == [1]


def test_length_bonus_monotonic():
    """bonus 越大输出越长（单调不减）。"""
    probs = [[0.55, 0.45], [0.55, 0.45], [0.55, 0.45], [0.55, 0.45]]
    lp = _log_probs(probs)
    lens = [len(ctc_beam_search(lp, blank=0, beam_width=8,
                                length_bonus=b))
            for b in (0.0, 0.3, 0.8, 2.0)]
    assert lens == sorted(lens)
