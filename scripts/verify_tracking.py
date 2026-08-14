"""验证视频手势跟踪：VideoSource + HandDetector + HandSequenceBuffer 全量跑一遍。

输出每帧摘要与 ID 生命周期统计（ID 稳定性、lost 事件、handedness 变化）。

用法: python scripts/verify_tracking.py <video_path> [--window N] [--max-frames N]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from signbridge import HandDetector, HandSequenceBuffer, OneEuroSmoother, VideoSource


def main() -> int:
    parser = argparse.ArgumentParser(description="视频手势跟踪验证")
    parser.add_argument("video", type=str)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    src = VideoSource(args.video)
    buf = HandSequenceBuffer(
        window_size=args.window,
        coordinate="world",
        smoother=OneEuroSmoother(),
    )

    id_first: dict[int, int] = {}
    id_last: dict[int, int] = {}
    id_handedness: dict[int, str] = {}
    id_frames: dict[int, int] = defaultdict(int)
    lost_events = 0
    prev_ids: set[int] = set()
    total_hands_frames = 0
    frames_with_hands = 0
    processed = 0

    with HandDetector(max_num_hands=2) as detector:
        for frame_index, (frame, _, _) in enumerate(src):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            hand_frame = detector.detect(frame)
            seqs = buf.update(hand_frame)
            processed += 1
            cur_ids = {s.hand_id for s in seqs}
            if cur_ids:
                frames_with_hands += 1
                total_hands_frames += len(cur_ids)
            for s in seqs:
                id_first.setdefault(s.hand_id, frame_index)
                id_last[s.hand_id] = frame_index
                id_frames[s.hand_id] += 1
                id_handedness[s.hand_id] = s.handedness
            # lost 事件：上一帧有、这一帧没出现在输出的 ID
            for hid in prev_ids - cur_ids:
                lost_events += 1
            prev_ids = cur_ids

    src.close()

    print(f"processed frames: {processed}")
    print(f"frames with hands: {frames_with_hands}/{processed} "
          f"({100 * frames_with_hands / max(processed, 1):.0f}%)")
    print(f"average hands per detected frame: "
          f"{total_hands_frames / max(frames_with_hands, 1):.2f}")
    print(f"ID lost events (track dropped from output): {lost_events}")
    print()
    print("ID lifecycle:")
    for hid in sorted(id_first):
        span = id_last[hid] - id_first[hid] + 1
        print(f"  id={hid} {id_handedness[hid]:>5s}  frames={id_frames[hid]:3d} "
              f"span={id_first[hid]}..{id_last[hid]} ({span}f)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
