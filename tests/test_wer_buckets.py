"""WER 分桶分析核心逻辑测试（scripts/analyze/wer_buckets.py）。"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analyze"))
import wer_buckets as wb


class TestAlignWords:
    def test_exact_match(self):
        ops = wb.align_words(["a", "b"], ["a", "b"])
        assert ops == [("match", "a"), ("match", "b")]

    def test_substitution(self):
        ops = wb.align_words(["a", "b"], ["a", "c"])
        assert ops == [("match", "a"), ("sub", "b")]

    def test_deletion(self):
        ops = wb.align_words(["a", "b"], ["a"])
        assert ops == [("match", "a"), ("del", "b")]

    def test_insertion(self):
        ops = wb.align_words(["a"], ["a", "x"])
        assert ops == [("match", "a"), ("ins", "x")]

    def test_complex_mixed(self):
        # 真值 [a,b,c,d] 预测 [a,x,d] → a match, b del, c sub→x, d match
        ops = wb.align_words(["a", "b", "c", "d"], ["a", "x", "d"])
        assert ops == [("match", "a"), ("del", "b"), ("sub", "c"), ("match", "d")]

    def test_empty_ref(self):
        assert wb.align_words([], ["x"]) == [("ins", "x")]

    def test_empty_hyp(self):
        assert wb.align_words(["a"], []) == [("del", "a")]

    def test_minimal_edit_choice(self):
        # [a,b] vs [b]：应选删除 a（1 次编辑）而非替换 a→b
        ops = wb.align_words(["a", "b"], ["b"])
        assert ops == [("del", "a"), ("match", "b")]


class TestBuckets:
    def test_len_bucket_boundaries(self):
        assert wb.len_bucket(3) == "短句 ≤3"
        assert wb.len_bucket(4) == "中句 4-5"
        assert wb.len_bucket(5) == "中句 4-5"
        assert wb.len_bucket(6) == "长句 6-7"
        assert wb.len_bucket(7) == "长句 6-7"
        assert wb.len_bucket(8) == "超长句 ≥8"

    def test_freq_bucket_boundaries(self):
        freq = Counter({"高": 50, "中": 20, "低": 5, "极低": 4, "无": 0})
        assert wb.word_freq_bucket(freq, "高") == "高频 ≥50"
        assert wb.word_freq_bucket(freq, "中") == "中频 20-49"
        assert wb.word_freq_bucket(freq, "低") == "低频 5-19"
        assert wb.word_freq_bucket(freq, "极低") == "极低频 1-4"
        assert wb.word_freq_bucket(freq, "无") == "极低频 1-4"
        assert wb.word_freq_bucket(freq, "oov") == "极低频 1-4"

    def test_gloss_words_filters_punct(self):
        assert wb.gloss_words("你好/世界/。") == ["你好", "世界"]

    def test_wer_aggregation(self):
        rows = [
            {"errors": 1, "n_ref": 3, "hyp": ["a"], "ops": [
                ("match", "a"), ("del", "b"), ("sub", "c")]},
            {"errors": 0, "n_ref": 2, "hyp": ["a", "b"], "ops": [
                ("match", "a"), ("match", "b")]},
        ]
        agg = wb.agg_bucket("x", rows)
        assert agg["wer"] == pytest.approx(1 / 5)
        assert agg["sub"] == 1 and agg["del"] == 1 and agg["ins"] == 0
        assert agg["seg_acc"] == pytest.approx(0.5)
