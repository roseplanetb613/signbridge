"""SpreadTheSign 中文手语词汇集转换：视频 → 标准 NPZ 词模板集。

数据集仅 1 个词带现成 JSON，其余需 MediaPipe 提取（与 CE-CSL 同管线）。
每词视频取最长有效手势段（方案 B 双手分块 + 单手零填充）→ (T,42,3)。

产出 data/dataset/spreadthesign.npz：
  data/words/videos/sources/detection_rates/avg_bboxes + word_list

用法: python scripts/extract/convert_spreadthesign.py [--workers 4]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from signbridge import HandDetector, VideoSource
from signbridge.core.segmentation import extract_segments
from signbridge.hands.sequence import classify_two_hands, to_normalized

ROOT = Path(r"E:\SignBridge\data\SpreadTheSign中文手语词汇集")
OUT = Path("data/dataset/spreadthesign.npz")
MIN_SEGMENT = 9
MERGE_GAP = 2
DETECTION_CONF = 0.3


def process_word(video_path: str):
    """单个词视频 → 最长手势段张量 + 质量指标。"""
    rows = []
    hand_frames = 0
    total = 0
    bboxes = []
    with HandDetector(max_num_hands=2,
                      min_detection_confidence=DETECTION_CONF) as detector:
        for frame_index, (frame, _, _) in enumerate(VideoSource(video_path)):
            total += 1
            hf = detector.detect(frame)
            hands = list(hf.hands)
            if hands:
                hand_frames += 1
                for hand in hands:
                    xs = [lm.x for lm in hand.landmarks]
                    ys = [lm.y for lm in hand.landmarks]
                    bboxes.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
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
    det_rate = hand_frames / max(total, 1)
    avg_bbox = float(np.mean(bboxes)) if bboxes else 0.0
    segs = extract_segments(np.ones(len(rows), dtype=bool),
                            MIN_SEGMENT, MERGE_GAP)
    longest = max(segs, key=lambda s: s[1]) if segs else None
    if longest is None:
        return None
    s, l = longest
    return {
        "data": np.stack(rows[s:s + l]),
        "detection_rate": det_rate,
        "avg_bbox": avg_bbox,
    }


def _work(args_tuple):
    video_path, source = args_tuple
    return video_path, source, process_word(video_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="SpreadTheSign 转换")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = []
    for sub in ("CSL_basic_dataset", "CSL_common_dataset"):
        for mp4 in sorted((ROOT / sub).glob("*.mp4")):
            jobs.append((str(mp4), sub))

    samples, words, videos, sources, rates, bboxes = [], [], [], [], [], []
    skipped = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                video_path, source, result = fut.result()
            except Exception as exc:            # noqa: BLE001
                skipped.append(Path(jobs[i - 1][0]).name)
                continue
            if result is None:
                skipped.append(Path(video_path).name)
                continue
            samples.append(result["data"])
            words.append(Path(video_path).stem)
            videos.append(Path(video_path).name)
            sources.append(source)
            rates.append(result["detection_rate"])
            bboxes.append(result["avg_bbox"])
            if i % 200 == 0 or i == len(jobs):
                print(f"进度 {i}/{len(jobs)}（成功 {len(samples)}）", flush=True)

    if not samples:
        print("未提取到任何样本")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        data=np.array(samples, dtype=object),
        words=np.array(words, dtype=object),
        videos=np.array(videos, dtype=object),
        sources=np.array(sources, dtype=object),
        detection_rates=np.array(rates),
        avg_bboxes=np.array(bboxes),
        word_list=np.array(sorted(set(words)), dtype=object),
    )
    lens = sorted(len(s) for s in samples)
    n = len(samples)
    print(f"\n转换 {len(samples)} 词样本（跳过 {len(skipped)}）→ {OUT}")
    print(f"词表 {len(set(words))} 词")
    print(f"段长: min {lens[0]} 中位 {lens[n // 2]} max {lens[-1]}")
    print(f"检测率中位 {np.median(rates):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
