"""从 CE-CSL 视频提取手势段张量 → NPZ 数据集（供模型接入测试）。

段：连续有手帧（方案 B 双手分块 + 单手零填充），gap≤2 合并，≥9 帧。
标签：句子 gloss 首个非标点词 → top-K 词表索引（简化分类任务，接口验证用）。

用法: python scripts/extract_segments.py --count 30 --seed 123 [--out data/extracted/segments.npz]
"""

import argparse
import csv
import glob
import random
import sys
from pathlib import Path

import numpy as np

from signbridge import HandDetector, VideoSource
from signbridge.core.segmentation import extract_segments
from signbridge.hands.sequence import classify_two_hands, to_normalized

PUNCT = set("。，？！、；：""''（）《》")


def load_gloss(split: str) -> dict:
    rows = list(csv.DictReader(
        open(rf"E:\SignBridge\data\CE-CSL\label\{split}.csv", encoding="utf-8")))
    return {r["Number"]: r["Gloss"] for r in rows}


def first_word(gloss: str):
    for w in gloss.split("/"):
        w = w.strip()
        if w and w not in PUNCT:
            return w
    return None


def extract_video(path: str, detector, min_seg=9, merge_gap=2):
    """返回该视频的手势段列表 [(data(T,42,3), frame_span)]。"""
    rows = []
    for frame_index, (frame, _, _) in enumerate(VideoSource(path)):
        hf = detector.detect(frame)
        hands = list(hf.hands)
        if len(hands) == 2:
            b0, b1 = classify_two_hands(hands[0], hands[1])
            row = np.full((42, 3), np.nan, dtype=np.float32)
            row[:21] = to_normalized(b0)
            row[21:] = to_normalized(b1)
            rows.append(row)
        elif len(hands) == 1:
            row = np.zeros((42, 3), dtype=np.float32)
            row[:21] = to_normalized(hands[0])
            rows.append(row)
    segs = extract_segments(np.ones(len(rows), dtype=bool), min_seg, merge_gap)
    return [(np.stack(rows[s:s + l]), (s, s + l)) for s, l in segs]


def main() -> int:
    parser = argparse.ArgumentParser(description="CE-CSL 手势段提取")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", type=str, default="data/extracted/segments.npz")
    args = parser.parse_args()

    gloss_map = load_gloss(args.split)
    videos = sorted(glob.glob(
        rf"E:\SignBridge\data\CE-CSL\video\{args.split}\*\*.mp4"))
    rng = random.Random(args.seed)
    sample = rng.sample(videos, min(args.count, len(videos)))

    samples = []   # dict(video, data, gloss, fw)
    detector = HandDetector(max_num_hands=2, min_detection_confidence=0.3)
    for v in sample:
        number = Path(v).stem
        gloss = gloss_map.get(number, "")
        fw = first_word(gloss)
        segs = extract_video(v, detector)
        for data, span in segs:
            samples.append({"video": number, "data": data, "gloss": gloss,
                            "fw": fw, "span": span})
        print(f"{number}: {len(segs)} 段 ({fw})", flush=True)
    detector.close()

    # 词表：按词频取 top-K（只统计有首词的样本）
    from collections import Counter
    freq = Counter(s["fw"] for s in samples if s["fw"])
    vocab = [w for w, _ in freq.most_common(args.top_k)]
    vocab_idx = {w: i for i, w in enumerate(vocab)}

    kept = [s for s in samples if s["fw"] in vocab_idx]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        data=np.array([s["data"] for s in kept], dtype=object),
        labels=np.array([vocab_idx[s["fw"]] for s in kept], dtype=np.int64),
        glosses=np.array([s["gloss"] for s in kept], dtype=object),
        videos=np.array([s["video"] for s in kept], dtype=object),
        vocab=np.array(vocab, dtype=object),
    )
    lens = [len(s["data"]) for s in kept]
    print(f"\n提取视频 {len(sample)}，段样本 {len(samples)}，"
          f"有效样本（首词在词表）{len(kept)}")
    print(f"词表 top-{args.top_k}: {vocab}")
    print(f"段长: min {min(lens) if lens else 0} 中位 "
          f"{sorted(lens)[len(lens)//2] if lens else 0} max "
          f"{max(lens) if lens else 0}")
    print(f"保存 → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
