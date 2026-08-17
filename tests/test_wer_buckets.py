"""WER 分桶分析核心逻辑测试（scripts/analyze/wer_buckets.py）。"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)   # 规避 CPU 多线程下的崩溃路径

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analyze"))
import wer_buckets as wb

from signbridge import FusionSTGCNCTC, build_hand_graph
from signbridge.core.graphs import build_adjacency


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


def _jpeg_frame(size=128, seed=0):
    import cv2
    img = np.full((size, size, 3), seed % 255, dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _fusion_model(k=3):
    return FusionSTGCNCTC(
        num_classes=k,
        hand_adjacency=build_hand_graph(num_hands=2),
        pose_adjacency=build_adjacency(wb.POSE_CONNECTIONS, 33),
        resnet_pretrained=False)


class TestFusionBatch:
    def test_shapes_and_nan_handling(self):
        rng = np.random.default_rng(0)
        hand = rng.standard_normal((10, 42, 3)).astype(np.float32)   # (T,V,C)
        pose = rng.standard_normal((10, 33, 3)).astype(np.float32)
        pose[3, 0, 0] = np.nan          # 验证 nan→0
        rois = [_jpeg_frame(seed=i) for i in range(10)] + [None]  # 末帧缺失
        ht, pt, rt = wb._fusion_batch([hand], [pose], [rois], target_t=16)
        assert ht.shape == (1, 3, 16, 42)
        assert pt.shape == (1, 3, 16, 33)
        assert rt.shape == (1, 16, 3, 112, 112)
        assert torch.isfinite(pt).all()     # nan 已清零
        assert torch.isfinite(rt).all()

    def test_align_repeat_short_series(self):
        rng = np.random.default_rng(1)
        hand = rng.standard_normal((3, 42, 3)).astype(np.float32)
        pose = rng.standard_normal((3, 33, 3)).astype(np.float32)
        rois = [_jpeg_frame(seed=0)] * 3
        ht, pt, rt = wb._fusion_batch([hand], [pose], [rois], target_t=16)
        assert ht.shape[2] == 16 and pt.shape[2] == 16 and rt.shape[1] == 16
        # 重复填充：前 3 帧与后 3 帧（第二次重复）一致
        assert torch.equal(ht[0, :, 3:6], ht[0, :, 6:9])


class TestAnalyzeSplitFusion:
    def _write_splits(self, tmp_path, rng):
        vocab = ["你好", "世界", "谢谢"]
        # 3 段：hand/pose 12 帧（短于 target_t，验证对齐），ROI 12 帧 JPEG
        hands, poses, rois, glosses, trans, videos = [], [], [], [], [], []
        specs = [("你好/世界", "A", "v1"), ("谢谢", "B", "v2"),
                 ("你好", "A", "v3")]
        for gloss, tr, vid in specs:
            hands.append(rng.standard_normal((12, 42, 3)).astype(np.float32))
            poses.append(rng.standard_normal((12, 33, 3)).astype(np.float32))
            rois.append([_jpeg_frame(seed=i) for i in range(12)])
            glosses.append(gloss)
            trans.append(tr)
            videos.append(vid)
        np.savez(tmp_path / "dev.npz",
                 data=np.array(hands, dtype=object),
                 detection_rates=np.array([0.9, 0.8, 0.7]),
                 glosses=np.array(glosses, dtype=object),
                 translators=np.array(trans, dtype=object),
                 videos=np.array(videos, dtype=object))
        np.savez(tmp_path / "dev_pose.npz",
                 pose_img=np.array(poses, dtype=object))
        np.savez(tmp_path / "dev_roi.npz",
                 roi=np.array(rois, dtype=object))
        return vocab

    def test_end_to_end(self, tmp_path):
        rng = np.random.default_rng(42)
        vocab = self._write_splits(tmp_path, rng)
        model = _fusion_model(k=len(vocab))
        train_freq = Counter({"你好": 30, "世界": 10, "谢谢": 5})
        res = wb.analyze_split_fusion(
            model, tmp_path, "dev", vocab, "cpu", target_t=16,
            beam_width=5, min_det=0.0, train_freq=train_freq)
        assert res["split"] == "dev"
        assert res["n_segments"] == 3
        assert len(res["per_sample"]) == 3
        assert all({"video", "translator", "ref", "hyp", "ops",
                    "errors", "n_ref"} <= set(r) for r in res["per_sample"])
        assert {r["translator"] for r in res["per_sample"]} == {"A", "B"}
        assert {r["video"] for r in res["per_sample"]} == {"v1", "v2", "v3"}
        assert res["len_buckets"] and res["signer_buckets"] and res["freq_buckets"]
        assert res["len_buckets"][0]["n_segments"] == 3

    def test_end_to_end_length_bonus(self, tmp_path):
        rng = np.random.default_rng(7)
        vocab = self._write_splits(tmp_path, rng)
        model = _fusion_model(k=len(vocab))
        res = wb.analyze_split_fusion(
            model, tmp_path, "dev", vocab, "cpu", target_t=16,
            beam_width=5, min_det=0.0, train_freq=Counter(),
            length_bonus=1.0)          # 透传不报错
        assert res["n_segments"] == 3

    def test_min_det_filters(self, tmp_path):
        rng = np.random.default_rng(3)
        vocab = self._write_splits(tmp_path, rng)
        model = _fusion_model(k=len(vocab))
        res = wb.analyze_split_fusion(
            model, tmp_path, "dev", vocab, "cpu", target_t=16,
            beam_width=5, min_det=0.85, train_freq=Counter())
        assert res["n_segments"] == 1       # 仅 detection_rate 0.9 保留
