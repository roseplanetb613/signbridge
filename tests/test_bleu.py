"""BLEU 指标测试（用标准示例验证公式实现）。"""

import math

import pytest

from signbridge import corpus_bleu, sentence_bleu

REF = "I LOVE YOU VERY MUCH".split()
HYP = "I LOVE YOU VERY GOOD".split()


class TestCorpusBleu:
    def test_standard_example_pn(self):
        """用户示例：p1=0.8, p2=0.75, p3=0.667, p4=0.5（c=r 无 BP）。"""
        bleu, p_n, bp = corpus_bleu([REF], [HYP], smooth=False)
        assert p_n[0] == pytest.approx(0.8, abs=1e-6)
        assert p_n[1] == pytest.approx(0.75, abs=1e-6)
        assert p_n[2] == pytest.approx(2 / 3, abs=1e-6)
        assert p_n[3] == pytest.approx(0.5, abs=1e-6)
        assert bp == pytest.approx(1.0)

    def test_standard_example_bleu4(self):
        """BLEU-4 = exp((log0.8+log0.75+log0.667+log0.5)/4) ≈ 0.669。"""
        bleu, _, _ = corpus_bleu([REF], [HYP], smooth=False)
        expect = math.exp((math.log(0.8) + math.log(0.75)
                           + math.log(2 / 3) + math.log(0.5)) / 4)
        assert bleu == pytest.approx(expect, abs=1e-6)

    def test_exact_match_bleu1(self):
        assert corpus_bleu([REF], [REF])[0] == pytest.approx(1.0, abs=1e-6)

    def test_brevity_penalty(self):
        """预测过短 → BP < 1。ref 5 词，hyp 3 词：BP=exp(1-5/3)=exp(-2/3)。

        平滑开启时 p4（无 4-gram）平滑为 1 → BLEU = BP。
        """
        hyp = "I LOVE YOU".split()
        bleu, p_n, bp = corpus_bleu([REF], [hyp], smooth=True)
        assert bp == pytest.approx(math.exp(-2 / 3), abs=1e-6)
        assert bleu == pytest.approx(bp, abs=1e-6)
        assert bleu < 1.0

    def test_short_hyp_no_smooth_returns_zero(self):
        """无平滑时 hyp 短于 max_n → 高阶 p_n 无定义 → BLEU=0（标准行为）。"""
        hyp = "I LOVE".split()
        assert corpus_bleu([REF], [hyp], smooth=False)[0] == 0.0

    def test_no_match_returns_zero(self):
        bleu, _, _ = corpus_bleu(["A B".split()], ["C D".split()],
                                 smooth=False)
        assert bleu == 0.0

    def test_smoothing_avoids_zero(self):
        """无匹配时平滑后有限值。"""
        bleu, p_n, _ = corpus_bleu(["A B".split()], ["C D".split()],
                                   smooth=True)
        assert bleu >= 0.0
        assert all(p > 0 for p in p_n)

    def test_sentence_bleu_wrapper(self):
        assert sentence_bleu(REF, HYP) == corpus_bleu([REF], [HYP])[0]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            corpus_bleu([REF], [REF, HYP])
