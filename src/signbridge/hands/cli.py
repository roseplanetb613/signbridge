"""CLI 演示工具：实时显示手部关键点叠加效果。

用法：
    python -m signbridge.hands.cli --source camera --camera-id 0
    python -m signbridge.hands.cli --source video --path demo.mp4
    python -m signbridge.hands.cli --source image --path hand.jpg
    python -m signbridge.hands.cli --download-model
"""

import argparse
import sys

import cv2

from signbridge.core.landmarks import HandFrame
from signbridge.hands.detector import HandDetector
from signbridge.hands.draw import draw_landmarks
from signbridge.hands.model import ensure_model
from signbridge.hands.sources import CameraSource, FrameSource, ImageSource, VideoSource

WINDOW_NAME = "SignBridge Hands"
EXIT_KEYS = (ord("q"), 27)  # q / Esc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="signbridge-hands", description="SignBridge 手部关键点演示工具"
    )
    parser.add_argument(
        "--source", choices=["camera", "video", "image"], default="camera"
    )
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument(
        "--path", type=str, default=None, help="video/image 源的文件路径"
    )
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument(
        "--no-overlay", action="store_true", help="不显示窗口，仅打印每帧摘要"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="最多处理帧数（默认无限制）"
    )
    parser.add_argument(
        "--download-model", action="store_true", help="下载模型后退出"
    )
    return parser


def _make_source(args) -> FrameSource:
    if args.source == "camera":
        return CameraSource(args.camera_id)
    if args.path is None:
        raise SystemExit("--source video/image 需要 --path 指定文件")
    if args.source == "video":
        return VideoSource(args.path)
    return ImageSource(args.path)


def _summarize(frame_index: int, hand_frame: HandFrame) -> str:
    if not hand_frame.hands:
        return f"frame={frame_index} hands=0"
    hands = " ".join(f"{h.handedness}({h.score:.2f})" for h in hand_frame.hands)
    return f"frame={frame_index} hands={len(hand_frame.hands)} {hands}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.download_model:
        path = ensure_model()
        print(f"模型就绪: {path}")
        return 0
    source = _make_source(args)
    with HandDetector(max_num_hands=args.max_hands) as detector:
        for frame_index, (frame, _, _) in enumerate(source):
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
            hand_frame = detector.detect(frame)
            if args.no_overlay:
                print(_summarize(frame_index, hand_frame), flush=True)
                continue
            canvas = draw_landmarks(frame, hand_frame)
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in EXIT_KEYS:
                break
    source.close()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
