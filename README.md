# SignBridge

手语翻译项目 —— MediaPipe 关键点组件库（基础层）。

当前版本实现**手部关键点提取**：输入摄像头 / 视频文件 / 图片，输出每帧 0~N 只手的
21 个关键点（归一化坐标 + 米制 3D world 坐标）与左右手判定，并支持叠加可视化。

技术栈：Python · MediaPipe Tasks API · OpenCV · NumPy · PyTorch（后续）· ST-GCN（后续）

## 安装

```bash
pip install -e ".[dev]"
```

要求 Python ≥ 3.10（开发环境 3.14）；首次运行会自动下载模型
（hand_landmarker.task，约 7.8MB，缓存到 `~/.cache/signbridge/`）。

## 快速上手

```python
import cv2
from signbridge.hands import HandDetector, ImageSource

src = ImageSource("hand.jpg")
frame, _, _ = next(iter(src))
src.close()

detector = HandDetector(max_num_hands=2)   # 首次自动下载模型
hand_frame = detector.detect(frame)        # HandFrame(hands=..., timestamp_ms=..., frame_index=...)

for hand in hand_frame.hands:              # 无手时 hands 为空元组
    print(hand.handedness, hand.score)     # "Right" 0.98
    print(len(hand.landmarks))             # 21
    print(hand.world_landmarks[0])         # 腕部米制 3D 坐标（ST-GCN 空间特征）

overlay = draw_landmarks(frame, hand_frame)  # 叠加可视化（不改原帧）
cv2.imwrite("overlay.jpg", overlay)
detector.close()
```

## CLI 演示工具

```bash
python -m signbridge.hands.cli --source camera --camera-id 0   # 摄像头实时叠加
python -m signbridge.hands.cli --source video --path demo.mp4  # 视频文件
python -m signbridge.hands.cli --source image --path hand.jpg  # 单张图片
python -m signbridge.hands.cli --source image --path hand.jpg --no-overlay  # 仅打印摘要
python -m signbridge.hands.cli --download-model                # 预下载模型
```

窗口内按 `q` / `Esc` 退出。

## API 速览

| 模块 | 内容 |
| --- | --- |
| `signbridge.core.landmarks` | `Landmark` / `Hand` / `HandFrame` 数据类；`HAND_CONNECTIONS`（21 条骨骼边）、`HAND_LANDMARK_NAMES`（21 点名） |
| `signbridge.core.errors` | `SignBridgeError` 及 `ModelNotFoundError` / `ModelDownloadError` / `SourceOpenError` / `InvalidArgumentError` |
| `signbridge.hands.detector` | `HandDetector(max_num_hands, min_detection_confidence, min_tracking_confidence, model_path=None)`；`detect(bgr_frame) -> HandFrame`；支持 with 语句 |
| `signbridge.hands.sources` | `CameraSource(camera_id)` / `VideoSource(path)`（含 `meta`）/ `ImageSource(path, repeat=False)`；统一产出 `(frame, frame_index, timestamp_ms)` |
| `signbridge.hands.draw` | `draw_landmarks(frame, hand_frame, color=None) -> 新帧` |
| `signbridge.hands.model` | `ensure_model()` / `cache_dir()` / `default_model_path()` |

## 手部关键点图谱（21 点）

![手部关键点编号示意图](docs/images/hand_landmark_diagram.png)

| 索引 | 名称 | 索引 | 名称 | 索引 | 名称 |
| --- | --- | --- | --- | --- | --- |
| 0 | WRIST（腕） | 7 | INDEX_FINGER_DIP | 14 | RING_FINGER_PIP |
| 1 | THUMB_CMC | 8 | INDEX_FINGER_TIP（指尖） | 15 | RING_FINGER_DIP |
| 2 | THUMB_MCP | 9 | MIDDLE_FINGER_MCP | 16 | RING_FINGER_TIP（指尖） |
| 3 | THUMB_IP | 10 | MIDDLE_FINGER_PIP | 17 | PINKY_MCP |
| 4 | THUMB_TIP（指尖） | 11 | MIDDLE_FINGER_DIP | 18 | PINKY_PIP |
| 5 | INDEX_FINGER_MCP | 12 | MIDDLE_FINGER_TIP（指尖） | 19 | PINKY_DIP |
| 6 | INDEX_FINGER_PIP | 13 | RING_FINGER_MCP | 20 | PINKY_TIP（指尖） |

`HAND_CONNECTIONS` 共 21 条边：每指 4 条指骨边 × 5 指 + 腕→小指根 1 条手掌边。
这套骨架与边列表即后续 ST-GCN 图卷积的**节点与图边**。

## 测试

```bash
python -m pytest -v
```

测试图片取自 Wikimedia Commons（CC0），来源见 `tests/assets/README.md`。

## 路线图

- [x] 第一步：手部关键点提取组件（本版本）
- [ ] 第二步：时序序列缓冲 → ST-GCN 图数据（滑动窗口 + 手腕原点归一化）
- [ ] 静态手势分类（MediaPipe 内置）
- [ ] 人体姿态组件 `signbridge.pose`
- [ ] ST-GCN 训练与推理（PyTorch）
- [ ] Qt 界面（PySide6）：摄像头预览 / 关键点叠加 / 翻译结果显示
