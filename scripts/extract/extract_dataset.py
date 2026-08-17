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
import time
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
ROI_SIZE = 128            # 手部 ROI 目标尺寸
ROI_MARGIN = 0.35         # ROI margin（bbox 比例）
JPEG_QUALITY = 85


def load_meta(split: str) -> dict:
    rows = list(csv.DictReader(
        open(rf"E:\SignBridge\data\CE-CSL\label\{split}.csv", encoding="utf-8")))
    return {r["Number"]: r for r in rows}


def _roi_jpeg(frame, x0, y0, x1, y1):
    """裁剪手部 ROI → 128×128 JPEG bytes。"""
    import cv2

    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    roi = cv2.resize(roi, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def process_video(video_path: str, split: str, meta: dict):
    """单个视频 → 段列表（手部张量 + 姿态 + ROI，同一 span 对齐）。"""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    number = Path(video_path).stem
    row = meta.get(number, {})
    gloss = row.get("Gloss", "")
    translator = row.get("Translator", "")

    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(
            model_asset_path=str(Path.home() / ".cache" / "signbridge"
                                 / "pose_landmarker_full.task")),
        num_poses=1)
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

    rows = []            # 手部 42 节点行（段切分依据）
    pose_img = []        # 每帧姿态 image 坐标 (33,3)（NaN 填充）
    pose_world = []      # 每帧姿态 world 坐标 (33,3)
    roi_bytes = []       # 每帧块 0 手 ROI JPEG bytes（无手 None）
    hand_frames = 0
    total = 0
    bboxes = []
    try:
        with HandDetector(max_num_hands=2,
                          min_detection_confidence=DETECTION_CONF) as detector:
            for frame_index, (frame, _, _) in enumerate(VideoSource(video_path)):
                if frame_index % FRAME_STRIDE != 0:
                    continue
                total += 1
                # 姿态检测用全分辨率原帧（质量优先）
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hf = detector.detect(frame)
                pr = pose_landmarker.detect(mp_img)

                pi = np.full((33, 3), np.nan, dtype=np.float32)
                pw = np.full((33, 3), np.nan, dtype=np.float32)
                if pr.pose_landmarks:
                    lm = pr.pose_landmarks[0]
                    pi = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
                    wl = pr.pose_world_landmarks[0]
                    pw = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32)
                pose_img.append(pi)
                pose_world.append(pw)

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

                # 块 0 手 ROI（bbox + margin 裁剪）
                if hands:
                    if len(hands) == 2:
                        b0, _ = classify_two_hands(hands[0], hands[1])
                    else:
                        b0 = hands[0]
                    xs = [lm.x for lm in b0.landmarks]
                    ys = [lm.y for lm in b0.landmarks]
                    fh, fw = frame.shape[:2]
                    bw = (max(xs) - min(xs)) * fw
                    bh = (max(ys) - min(ys)) * fh
                    x0 = max(int(min(xs) * fw - bw * ROI_MARGIN), 0)
                    x1 = min(int(max(xs) * fw + bw * ROI_MARGIN), fw)
                    y0 = max(int(min(ys) * fh - bh * ROI_MARGIN), 0)
                    y1 = min(int(max(ys) * fh + bh * ROI_MARGIN), fh)
                    roi_bytes.append(_roi_jpeg(frame, x0, y0, x1, y1))
                else:
                    roi_bytes.append(None)
    finally:
        pose_landmarker.close()
    det_rate = hand_frames / max(total, 1)
    avg_bbox = float(np.mean(bboxes)) if bboxes else 0.0

    segs = extract_segments(np.ones(len(rows), dtype=bool),
                            MIN_SEGMENT, MERGE_GAP)
    segments = []
    for s, l in segs:
        segments.append({
            "data": np.stack(rows[s:s + l]),
            "span": (s * FRAME_STRIDE, (s + l) * FRAME_STRIDE),
            "pose_img": np.stack(pose_img[s:s + l]),
            "pose_world": np.stack(pose_world[s:s + l]),
            "roi": [b for b in roi_bytes[s:s + l]],
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


def _log_progress(split: str, done: int, total: int, start_time: float,
                  log_path: Path) -> None:
    """进度持久化：追加日志行（含时间戳与 ETA），供事后查看。"""
    import time as _time

    elapsed = _time.monotonic() - start_time
    per = elapsed / max(done, 1)
    eta_min = per * (total - done) / 60
    line = (f"[{_time.strftime('%H:%M:%S')}] {split}: {done}/{total} "
            f"({100 * done / max(total, 1):.0f}%) ETA {eta_min:.0f}min\n")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="CE-CSL 全量骨架段提取")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--workers", type=int, default=3,
                        help="进程数（16GB 内存机器建议 ≤3，防 OOM）")
    parser.add_argument("--out", type=str, default="data/dataset")
    parser.add_argument("--limit", type=int, default=0,
                        help="每 split 处理上限（0=全部，测试用）")
    args = parser.parse_args()

    out_dir = Path(args.out)
    parts_dir = out_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    all_meta = {s: load_meta(s) for s in args.splits}
    log_path = out_dir / "extract.log"
    for split in args.splits:
        videos = sorted(glob.glob(
            rf"E:\SignBridge\data\CE-CSL\video\{split}\*\*.mp4"))
        if args.limit:
            videos = videos[:args.limit]
        done = {p.stem[len(split) + 1:] for p in parts_dir.glob(f"{split}-*.npz")}
        pending = [v for v in videos if Path(v).stem not in done]
        print(f"[{split}] 共 {len(videos)}，已完成 {len(done)}，"
              f"待处理 {len(pending)}", flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {split} 开始："
                    f"已完成 {len(done)}，待处理 {len(pending)}\n")
        if not pending:
            continue

        start_time = time.monotonic()
        # 分块处理：每块重建进程池，释放 worker 内存（避免长时间运行 OOM）
        chunk_size = 400
        chunks = [pending[i:i + chunk_size]
                  for i in range(0, len(pending), chunk_size)]
        done_in_chunk = 0
        from tqdm import tqdm
        pbar = tqdm(total=len(pending), desc=f"[{split}]",
                    unit="video", ncols=100, dynamic_ncols=False)
        for chunk_idx, chunk in enumerate(chunks, 1):
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_work, (v, split, all_meta[split])): v
                           for v in chunk}
                for i, fut in enumerate(as_completed(futures), 1):
                    try:
                        result = fut.result()
                    except Exception as exc:          # noqa: BLE001
                        pbar.write(f"失败 {Path(futures[fut]).stem}: {exc}")
                        pbar.update(1)
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
                    done_in_chunk += 1
                    pbar.update(1)
                    if done_in_chunk % 50 == 0 or done_in_chunk == len(pending):
                        _log_progress(split, done_in_chunk, len(pending),
                                      start_time, log_path)
            pbar.write(f"[{split}] 块 {chunk_idx}/{len(chunks)} 完成"
                       f"（进程池已重建，内存已释放）")
        pbar.close()

    # 合并 parts → split NPZ（hand + pose + roi 三文件，同一顺序对齐）
    vocab = Counter()
    for split in args.splits:
        parts = sorted(parts_dir.glob(f"{split}-*.npz"))
        if not parts:
            continue
        all_data, glosses, videos, translators, rates, bboxes, spans = (
            [], [], [], [], [], [], [])
        pose_imgs, pose_worlds, rois = [], [], []
        for p in parts:
            d = np.load(p, allow_pickle=True)
            for seg in d["data"]:
                all_data.append(seg["data"])
                spans.append(tuple(seg["span"]))
                pose_imgs.append(seg["pose_img"])
                pose_worlds.append(seg["pose_world"])
                rois.append(np.array(seg["roi"], dtype=object))
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
        np.savez_compressed(
            out_dir / f"{split}_pose.npz",
            pose_img=np.array(pose_imgs, dtype=object),
            pose_world=np.array(pose_worlds, dtype=object),
            videos=np.array(videos, dtype=object),
            spans=np.array(spans, dtype=object),
        )
        np.savez_compressed(
            out_dir / f"{split}_roi.npz",
            roi=np.array(rois, dtype=object),
            videos=np.array(videos, dtype=object),
            spans=np.array(spans, dtype=object),
        )
        lens = [len(d) for d in all_data]
        roi_kb = sum(len(b) for r in rois for b in r if b is not None) / 1024
        print(f"[{split}] 合并 {len(all_data)} 段 → {out_dir / (split + '.npz')}"
              f"（+pose +roi {roi_kb / 1024:.1f} MB JPEG），"
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
