"""CE-CSL 全量骨架段提取（多进程 + 断点续跑）。

每视频：检测(conf=0.3, 隔帧采样) → 方案 B 双手分块（单手零填充）→ 段切分
(gap≤2, ≥9帧) → 段张量 (T,42,3) + 质量指标。每视频独立 part 文件，
断点续跑自动跳过已完成视频。最后合并为 split NPZ + 词表。

用法: python scripts/extract_dataset.py [--splits train dev test]
                                     [--workers 4] [--out data/dataset]
"""

import argparse
import csv
import glob
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from signbridge import HandDetector, VideoSource
from signbridge.core.segmentation import extract_segments
from signbridge.hands.sequence import classify_two_hands, to_normalized

PUNCT = set("。，？！、；：""''（）《》")
MIN_SEGMENT = 9
MERGE_GAP = 2
DETECTION_CONF = 0.3
FRAME_STRIDE = 2          # 隔帧采样：30fps → 15fps


def load_meta(split: str) -> dict:
    rows = list(csv.DictReader(
        open(rf"E:\SignBridge\data\CE-CSL\label\{split}.csv", encoding="utf-8")))
    return {r["Number"]: r for r in rows}


def process_video(video_path: str, split: str, meta: dict):
    """单个视频 → 段列表 [(data(T,42,3), span, det_rate, avg_bbox)]。"""
    number = Path(video_path).stem
    row = meta.get(number, {})
    gloss = row.get("Gloss", "")
    translator = row.get("Translator", "")

    rows = []
    hand_frames = 0
    total = 0
    bboxes = []
    with HandDetector(max_num_hands=2,
                      min_detection_confidence=DETECTION_CONF) as detector:
        for frame_index, (frame, _, _) in enumerate(VideoSource(video_path)):
            if frame_index % FRAME_STRIDE != 0:
                continue
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
                row42 = np.full((42, 3), np.nan, dtype=np.float32)
                row42[:21] = to_normalized(b0)
                row42[21:] = to_normalized(b1)
                rows.append(row42)
            elif len(hands) == 1:
                row42 = np.zeros((42, 3), dtype=np.float32)
                row42[:21] = to_normalized(hands[0])
                rows.append(row42)
    det_rate = hand_frames / max(total, 1)
    avg_bbox = float(np.mean(bboxes)) if bboxes else 0.0

    segs = extract_segments(np.ones(len(rows), dtype=bool),
                            MIN_SEGMENT, MERGE_GAP)
    segments = []
    for s, l in segs:
        segments.append({
            "data": np.stack(rows[s:s + l]),
            "span": (s * FRAME_STRIDE, (s + l) * FRAME_STRIDE),
        })
    return {
        "video": number,
        "split": split,
        "gloss": gloss,
        "translator": translator,
        "detection_rate": det_rate,
        "avg_bbox": avg_bbox,
        "segments": segments,
    }


def _work(args_tuple):
    """模块级 worker（Windows spawn 可 pickle）。"""
    video_path, split, meta = args_tuple
    return process_video(video_path, split, meta)


def main() -> int:
    parser = argparse.ArgumentParser(description="CE-CSL 全量骨架段提取")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="data/dataset")
    parser.add_argument("--limit", type=int, default=0,
                        help="每 split 处理上限（0=全部，测试用）")
    args = parser.parse_args()

    out_dir = Path(args.out)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    all_meta = {s: load_meta(s) for s in args.splits}
    for split in args.splits:
        videos = sorted(glob.glob(
            rf"E:\SignBridge\data\CE-CSL\video\{split}\*\*.mp4"))
        if args.limit:
            videos = videos[:args.limit]
        done = {p.stem for p in parts_dir.glob(f"{split}-*.npz")}
        pending = [v for v in videos if Path(v).stem not in done]
        print(f"[{split}] 共 {len(videos)}，已完成 {len(done)}，"
              f"待处理 {len(pending)}", flush=True)
        if not pending:
            continue

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_work, (v, split, all_meta[split])): v
                       for v in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    result = fut.result()
                except Exception as exc:          # noqa: BLE001
                    print(f"失败 {Path(futures[fut]).stem}: {exc}", flush=True)
                    continue
                np.savez_compressed(
                    parts_dir / f"{split}-{result['video']}.npz",
                    data=np.array(result["segments"], dtype=object),
                    gloss=result["gloss"],
                    translator=result["translator"],
                    detection_rate=result["detection_rate"],
                    avg_bbox=result["avg_bbox"],
                    span=np.array([s["span"] for s in result["segments"]]),
                )
                if i % 50 == 0 or i == len(pending):
                    print(f"[{split}] 进度 {i}/{len(pending)}", flush=True)

    # 合并 parts → split NPZ
    vocab = Counter()
    for split in args.splits:
        parts = sorted(parts_dir.glob(f"{split}-*.npz"))
        if not parts:
            continue
        all_data, glosses, videos, translators, rates, bboxes, spans = (
            [], [], [], [], [], [], [])
        for p in parts:
            d = np.load(p, allow_pickle=True)
            for seg in d["data"]:
                all_data.append(seg["data"])
                spans.append(tuple(seg["span"]))
            glosses.extend([d["gloss"]] * len(d["data"]))
            videos.extend([p.stem[len(split) + 1:]] * len(d["data"]))
            translators.extend([d["translator"]] * len(d["data"]))
            rates.extend([float(d["detection_rate"])] * len(d["data"]))
            bboxes.extend([float(d["avg_bbox"])] * len(d["data"]))
        np.savez_compressed(
            out_dir / f"{split}.npz",
            data=np.array(all_data, dtype=object),
            glosses=np.array(glosses, dtype=object),
            videos=np.array(videos, dtype=object),
            translators=np.array(translators, dtype=object),
            detection_rates=np.array(rates),
            avg_bboxes=np.array(bboxes),
            spans=np.array(spans, dtype=object),
        )
        lens = [len(d) for d in all_data]
        print(f"[{split}] 合并 {len(all_data)} 段 → {out_dir / (split + '.npz')}，"
              f"段长 min {min(lens) if lens else 0} 中位 "
              f"{sorted(lens)[len(lens)//2] if lens else 0} max "
              f"{max(lens) if lens else 0}", flush=True)
        if split == "train":
            for g in glosses:
                for w in str(g).split("/"):
                    w = w.strip()
                    if w and w not in PUNCT:
                        vocab[w] += 1
            np.savez_compressed(
                out_dir / "vocab.npz",
                words=np.array([w for w, _ in vocab.most_common()],
                               dtype=object),
            )
            print(f"[train] 词表 {len(vocab)} → vocab.npz", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
