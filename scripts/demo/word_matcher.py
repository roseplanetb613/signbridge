"""词识别演示：SpreadTheSign 词模板库 + 特征检索匹配。

模板：每词最长手势段 → 双手分块 → HandShapeFeature（210 维×2 手=420 维）
每帧特征 → 段内平均 → 模板特征。
查询：视频/摄像头 → 同样特征提取 → L2 最近邻 → top-k 词。

用法:
  python scripts/demo/word_matcher.py --query <视频路径> [--topk 5]
  python scripts/demo/word_matcher.py --self-test          # 自查询验证
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from signbridge import HandDetector, VideoSource
from signbridge.core.features import HandShapeFeature
from signbridge.hands.sequence import classify_two_hands

FEATURE = HandShapeFeature()


def segment_feature(data: np.ndarray) -> np.ndarray:
    """段张量 (T,42,3) → 420 维平均特征（块0+块1 各 210 维）。"""
    feats = []
    for row in data:
        if np.isnan(row).all():
            continue
        left = FEATURE.extract(row[:21])
        right = FEATURE.extract(row[21:])
        feats.append(np.concatenate([left, right]))
    if not feats:
        return np.zeros(420, dtype=np.float32)
    return np.mean(np.stack(feats), axis=0)


def build_templates(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    templates = {}
    for i in range(len(d["data"])):
        word = str(d["words"][i])
        if word not in templates:          # 同词多视频取第一个（或平均？先第一个）
            templates[word] = segment_feature(
                np.asarray(d["data"][i], dtype=np.float32))
    words = list(templates)
    mat = np.stack([templates[w] for w in words])
    return words, mat


def query_video(video_path: str, detector) -> np.ndarray:
    """视频 → 420 维查询特征（全程帧平均）。"""
    rows = []
    for frame_index, (frame, _, _) in enumerate(VideoSource(video_path)):
        hf = detector.detect(frame)
        hands = list(hf.hands)
        if len(hands) == 2:
            b0, b1 = classify_two_hands(hands[0], hands[1])
            from signbridge.hands.sequence import to_normalized
            row = np.full((42, 3), np.nan, dtype=np.float32)
            row[:21] = to_normalized(b0)
            row[21:] = to_normalized(b1)
            rows.append(row)
        elif len(hands) == 1:
            from signbridge.hands.sequence import to_normalized
            row = np.zeros((42, 3), dtype=np.float32)
            row[:21] = to_normalized(hands[0])
            rows.append(row)
    if not rows:
        return None
    return segment_feature(np.stack(rows))


def match(query_feat: np.ndarray, words, mat, topk: int):
    dists = np.linalg.norm(mat - query_feat, axis=1)
    order = np.argsort(dists)[:topk]
    return [(words[i], float(dists[i])) for i in order]


def main() -> int:
    parser = argparse.ArgumentParser(description="词识别演示")
    parser.add_argument("--npz", type=str, default="data/dataset/spreadthesign.npz")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    npz = Path(args.npz)
    if not npz.exists():
        print(f"未找到 {npz}（先运行 convert_spreadthesign.py）")
        return 1
    words, mat = build_templates(npz)
    print(f"模板库: {len(words)} 词")

    if args.self_test:
        d = np.load(npz, allow_pickle=True)
        hits = 0
        total = 0
        for i in range(len(d["data"])):
            feat = segment_feature(np.asarray(d["data"][i], dtype=np.float32))
            top = match(feat, words, mat, 1)
            total += 1
            if top[0][0] == str(d["words"][i]):
                hits += 1
        print(f"自查询 top-1 命中率: {hits}/{total} = {hits / max(total, 1):.1%}")
        return 0

    if not args.query:
        print("需要 --query <视频路径> 或 --self-test")
        return 1
    with HandDetector(max_num_hands=2,
                      min_detection_confidence=0.3) as detector:
        feat = query_video(args.query, detector)
    if feat is None:
        print("查询视频未检测到手")
        return 1
    for rank, (word, dist) in enumerate(match(feat, words, mat, args.topk), 1):
        print(f"  {rank}. {word}  (距离 {dist:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
