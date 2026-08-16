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
    total = len(d["data"])
    for i in range(total):
        word = str(d["words"][i])
        if word not in templates:          # 同词多视频取第一个（或平均？先第一个）
            templates[word] = segment_feature(
                np.asarray(d["data"][i], dtype=np.float32))
        if (i + 1) % 300 == 0:
            print(f"模板构建 {i + 1}/{total} ...", flush=True)
    words = list(templates)
    mat = np.stack([templates[w] for w in words])
    print(f"模板库: {len(words)} 词（构建完成）", flush=True)
    return words, mat


def build_templates_stgcn(npz_path: Path, model, device: str):
    """学习嵌入模板：整段一次前向 → 256 维嵌入（CTC 监督塑造的语义空间）。"""
    import torch

    d = np.load(npz_path, allow_pickle=True)
    templates = {}
    total = len(d["data"])
    for i in range(total):
        word = str(d["words"][i])
        if word in templates:
            continue
        data = np.asarray(d["data"][i], dtype=np.float32)   # (T,42,3)
        if len(data) < 9:                                   # kernel_size
            continue
        x = torch.from_numpy(data.transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            emb = model.embed(x.to(device)).squeeze(0).cpu().numpy()
        templates[word] = emb
        if (i + 1) % 300 == 0:
            print(f"嵌入模板 {i + 1}/{total} ...", flush=True)
    words = list(templates)
    mat = np.stack([templates[w] for w in words])
    print(f"嵌入模板库: {len(words)} 词（256 维）", flush=True)
    return words, mat


def query_video(video_path: str, detector) -> np.ndarray:
    """视频 → 420 维查询特征（全程帧平均）。"""
    rows = []
    for frame_index, (frame, _, _) in enumerate(VideoSource(video_path)):
        hf = detector.detect(frame)
        row = hands_to_row(hf.hands)
        if row is not None:
            rows.append(row)
    if not rows:
        return None
    return segment_feature(np.stack(rows))


def hands_to_row(hands) -> np.ndarray | None:
    """检测手列表 → 42 节点行（方案 B 分块 + 单手零填充）；无手返回 None。"""
    from signbridge.hands.sequence import to_normalized

    if len(hands) == 2:
        b0, b1 = classify_two_hands(hands[0], hands[1])
        row = np.full((42, 3), np.nan, dtype=np.float32)
        row[:21] = to_normalized(b0)
        row[21:] = to_normalized(b1)
        return row
    if len(hands) == 1:
        row = np.zeros((42, 3), dtype=np.float32)
        row[:21] = to_normalized(hands[0])
        return row
    return None


def run_camera(words, mat, camera_id: int, topk: int) -> None:
    """摄像头实时词识别：滑动窗口特征平均 → top-k 词叠加显示。"""
    import collections

    import cv2

    from signbridge import CameraSource, HandDetector
    from signbridge.core.errors import SignBridgeError
    from signbridge.hands.draw import draw_landmarks_depth

    print(f"正在打开摄像头 #{camera_id} ...", flush=True)
    try:
        src = CameraSource(camera_id)
    except SignBridgeError as exc:
        print(f"摄像头错误: {exc}", flush=True)
        return
    print("摄像头已打开，开始识别（q/Esc 退出）", flush=True)
    history = collections.deque(maxlen=15)
    with HandDetector(max_num_hands=2,
                      min_detection_confidence=0.3) as detector:
        for frame, _, _ in src:
            hf = detector.detect(frame)
            row = hands_to_row(hf.hands)
            if row is not None:
                f0 = FEATURE.extract(row[:21])
                f1 = (FEATURE.extract(row[21:])
                      if not np.isnan(row[21:]).all()
                      else np.zeros(210, dtype=np.float32))
                history.append(np.concatenate([f0, f1]))
            canvas = draw_landmarks_depth(frame, hf)
            if history:
                q = np.mean(np.stack(history), axis=0)
                for rank, (word, dist) in enumerate(
                        match(q, words, mat, topk), 1):
                    cv2.putText(canvas, f"{rank}. {word} ({dist:.2f})",
                                (10, 30 + rank * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("SignBridge 词识别（q/Esc 退出）", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    src.close()
    cv2.destroyAllWindows()


def match(query_feat: np.ndarray, words, mat, topk: int):
    dists = np.linalg.norm(mat - query_feat, axis=1)
    order = np.argsort(dists)[:topk]
    return [(words[i], float(dists[i])) for i in order]


def main() -> int:
    parser = argparse.ArgumentParser(description="词识别演示")
    parser.add_argument("--npz", type=str, default="data/dataset/spreadthesign.npz")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--camera", type=int, default=None,
                        help="摄像头实时词识别（指定 camera-id）")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--embedding", choices=["manual", "stgcn"],
                        default="manual",
                        help="manual=手工 420 维特征；stgcn=学习嵌入（best.pt）")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best.pt")
    args = parser.parse_args()

    npz = Path(args.npz)
    if not npz.exists():
        print(f"未找到 {npz}（先运行 convert_spreadthesign.py）")
        return 1

    if args.embedding == "stgcn":
        import torch

        from signbridge import STGCNCTC, build_hand_graph

        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model = STGCNCTC(num_classes=len(ckpt["vocab"]),
                         adjacency=build_hand_graph(num_hands=2))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        words, mat = build_templates_stgcn(npz, model, device)
    else:
        words, mat = build_templates(npz)
        print(f"模板库: {len(words)} 词")

    if args.camera is not None:
        run_camera(words, mat, args.camera, args.topk)
        return 0

    if args.self_test:
        import torch

        d = np.load(npz, allow_pickle=True)
        hits = 0
        total = 0
        for i in range(len(d["data"])):
            data = np.asarray(d["data"][i], dtype=np.float32)
            if args.embedding == "stgcn":
                if len(data) < 9:
                    continue
                x = torch.from_numpy(data.transpose(2, 0, 1)).unsqueeze(0).float()
                with torch.no_grad():
                    feat = model.embed(x.to(device)).squeeze(0).cpu().numpy()
            else:
                feat = segment_feature(data)
            top = match(feat, words, mat, 1)
            total += 1
            if top[0][0] == str(d["words"][i]):
                hits += 1
        print(f"自查询 top-1 命中率: {hits}/{total} = "
              f"{hits / max(total, 1):.1%}")
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
