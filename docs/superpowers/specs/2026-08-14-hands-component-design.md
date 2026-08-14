# SignBridge 手部关键点提取组件设计（第一步）

日期：2026-08-14
状态：已获用户确认（分节确认）

## 1. 项目背景与目标

SignBridge 是一个手语翻译项目，技术栈为 Qt + PyTorch + MediaPipe + ST-GCN：

- MediaPipe 获取手势数据与人体姿态数据
- 建立时序图谱数据集
- ST-GCN 图卷积神经网络对手势和姿势数据进行分类与预测
- Qt 可视化界面呈现手语翻译结果

本设计文档覆盖**第一步**：创建一个纯 Python 的 MediaPipe 手部关键点提取组件库，作为整个项目的基础层。时序序列缓冲、ST-GCN 图数据对接、人体姿态、Qt 界面均为后续步骤，本步只预留接口，不实现。

## 2. 范围

### 本步实现（In Scope）

- 纯 Python 包 `signbridge`（src 布局，`pip install -e .` 安装）
- 子包 `signbridge.hands`：手部关键点提取组件
- 三种输入源：摄像头、视频文件、图片文件
- 关键点可视化叠加绘制
- `.task` 模型文件的下载与缓存管理
- CLI 演示工具（实时叠加显示 / 终端摘要）
- pytest 自动化测试 + 测试图片样本
- README 文档

### 本步不实现（Out of Scope）

- 时序序列缓冲 → ST-GCN 图数据（第二步，本步仅提供图谱常量与数据结构预留）
- 静态手势分类（MediaPipe 内置手势识别）
- 人体姿态提取（PoseLandmarker/Holistic）
- ST-GCN 模型训练与推理
- Qt 界面
- 数据采集录制保存

## 3. 技术选型

底层 API 选择 **MediaPipe Tasks API（`mediapipe.tasks.python.vision.HandLandmarker`）**，而非旧版 Solutions API（`mp.solutions.hands`）。

理由：

1. Tasks API 是官方推荐的现代 API，长期维护；Solutions API 已被标记 legacy，未来版本有移除风险
2. Tasks API 同时返回归一化坐标与 world_landmarks（米制 3D 坐标）——后者正是 ST-GCN 图数据需要的空间信息
3. GPU delegate（CUDA）支持好，实时摄像头更流畅
4. 可配置最大手数、置信度阈值，自带左右手判定（handedness）
5. 与后续人体姿态（PoseLandmarker/Holistic）同一套 API，可共用抽象层

代价：需要外部 `hand_landmarker.task` 模型文件（约 7MB），由库内置下载/缓存管理解决。

## 4. 架构与目录结构

```
E:\SignBridge\
├── pyproject.toml               # 包元数据（src 布局）
├── README.md                    # 安装/快速上手/API 速览/图谱说明
├── src\signbridge\
│   ├── __init__.py              # 版本号、公共 API 导出
│   ├── core\                    # 共享基础设施（未来 pose/stgcn 共用）
│   │   ├── __init__.py
│   │   ├── landmarks.py         # Landmark/Hand 数据类 + HAND_CONNECTIONS 图谱常量
│   │   └── errors.py            # SignBridgeError 异常体系
│   └── hands\                   # 本步：手部组件
│       ├── __init__.py
│       ├── detector.py          # HandDetector（封装 Tasks API）
│       ├── model.py             # .task 模型下载/缓存管理
│       ├── sources.py           # CameraSource / VideoSource / ImageSource
│       ├── draw.py              # 关键点/骨骼叠加绘制
│       └── cli.py               # python -m signbridge.hands.cli 演示工具
└── tests\
    ├── conftest.py              # 共享 fixture（测试图片路径等）
    ├── test_landmarks.py
    ├── test_detector.py
    ├── test_sources.py
    └── test_draw.py
```

## 5. 核心数据结构（`signbridge.core.landmarks`）

frozen dataclass，可直接转 torch.Tensor 或 JSON。

```python
@dataclass(frozen=True)
class Landmark:
    x: float   # 归一化坐标 [0,1]
    y: float
    z: float   # 相对深度（图像坐标模式）/ 米制（world 模式）

@dataclass(frozen=True)
class Hand:
    landmarks: tuple[Landmark, ...]        # 21 个，图像坐标
    world_landmarks: tuple[Landmark, ...]  # 21 个，米制 3D —— ST-GCN 空间特征来源
    handedness: str                        # "Left" / "Right"
    score: float                           # 置信度

@dataclass(frozen=True)
class HandFrame:
    hands: tuple[Hand, ...]                # 0~N 只手；无手时为空元组
    timestamp_ms: int
    frame_index: int
```

### 图谱常量（公开 API）

- `HAND_LANDMARK_NAMES`：21 个关键点名称（0=WRIST, 1-4=THUMB_*, 5-8=INDEX_*, 9-12=MIDDLE_*, 13-16=RING_*, 17-20=PINKY_*）
- `HAND_CONNECTIONS`：20 条骨骼边列表（腕→各指、指节间连接）——**未来直接作为 ST-GCN 的图边**

## 6. 检测器 API（`signbridge.hands.detector`）

```python
detector = HandDetector(
    max_num_hands=2,              # 1 或 2
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_path=None,              # 可选：自定义 .task 模型路径；None 时自动下载/缓存
)
hand_frame = detector.detect(bgr_frame)   # 无手 → hands=() 空元组，不抛异常
detector.close()                          # 释放资源
# 支持 with 语句上下文管理
```

- 构造时自动确保模型可用（懒下载）；模型文件缺失/下载失败抛 `ModelNotFoundError` / `ModelDownloadError`；`model_path` 指定的文件不存在同样抛 `ModelNotFoundError`
- `detect()` 接收 BGR numpy 帧（`np.ndarray`，H×W×3，uint8）
- `HandDetector` 是**无状态**的：不保存帧间历史（时序缓冲是后续步骤的职责）
- 重复调用安全；`close()` 后可复用或销毁

## 7. 输入源（`signbridge.hands.sources`）

统一实现 `FrameSource` 协议（`__iter__` / `__next__` / `close`），逐个产出 `(bgr_frame, frame_index, timestamp_ms)`。消费方代码与具体源解耦。`timestamp_ms` 取 OpenCV 帧时间戳（`CAP_PROP_POS_MSEC`），不可用时回退为系统单调时钟（`time.monotonic()`）。

- **`CameraSource(camera_id=0)`**：OpenCV 摄像头；循环 `read()`；`close()` 释放
- **`VideoSource(path)`**：OpenCV 视频文件逐帧读取；`meta` 属性暴露宽高/帧率/总帧数；迭代至结束
- **`ImageSource(path)`**：单帧，迭代一次即结束（可选 `repeat=True` 循环）

内部统一以 numpy BGR 帧为最小接口（后续 Qt 推送帧可再添 `NumpySource`）。

## 8. 可视化（`signbridge.hands.draw`）

- `draw_landmarks(frame, hand_frame, color=None) -> np.ndarray`：在 BGR 帧**副本**上绘制
  - 21 个关键点（圆点）
  - `HAND_CONNECTIONS` 骨骼连线（线段）
  - 手部标签（如 "Left 0.98"）
  - 左右手不同颜色
- 不修改原帧
- 简单够用，不做过多的视觉特效（可视化增强留给 Qt 阶段）

## 9. 模型管理（`signbridge.hands.model`）

- `ensure_model(version=None) -> Path`：懒下载 `hand_landmarker.task`（官方 Google Storage 地址），缓存到 `~/.cache/signbridge/hand_landmarker.task`；已存在则跳过（幂等）
- 首次运行自动触发（终端打印下载进度）
- `python -m signbridge.hands.cli --download-model` 手动预下载
- mediapipe 升级导致模型 API 不兼容时，抛出带指引的错误信息

## 10. CLI 演示工具（`signbridge.hands.cli`）

```
python -m signbridge.hands.cli --source camera --camera-id 0
python -m signbridge.hands.cli --source video --path demo.mp4
python -m signbridge.hands.cli --source image --path hand.jpg
```

- 打开窗口实时显示叠加效果，按 `q` / `Esc` 退出
- `--max-hands`、`--no-overlay`（只打印每帧检测摘要）、`--download-model` 等选项

## 11. 错误处理（`signbridge.core.errors`）

- 基类 `SignBridgeError(Exception)`
- `ModelNotFoundError`：模型缺失（带重试/下载指引）
- `ModelDownloadError`：下载失败（带重试指引）
- `SourceOpenError`：摄像头打不开 / 文件不存在或无法解码
- `InvalidArgumentError`：参数非法（如 max_num_hands 超范围）

原则：**检测不到手不是错误**（返回空 `HandFrame`）；打不开源、模型不可用才是错误。

## 12. 测试策略（pytest）

- `test_landmarks.py`：图谱常量完整性（21 点名、20 条边、索引范围合法）
- `test_detector.py`：测试图片（含手样本）→ 检测到手、21 个关键点、坐标 ∈ [0,1]、handedness 合法；无手图片返回空；模型缺失抛 `ModelNotFoundError`（mock）
- `test_sources.py`：图片/视频源迭代正确性、close 语义；摄像头用 mock 不碰真实设备
- `test_draw.py`：叠加后输出帧尺寸/通道正确、原帧未被修改

测试图片：下载公开许可（MIT/CC0）的含手图片存入 `tests/assets/`，README 注明来源与许可。

## 13. 依赖

- 运行依赖：`mediapipe>=0.10,<0.11`、`opencv-python`、`numpy`
- 开发依赖：`pytest`

## 14. 交付物清单

1. `pyproject.toml`（src 布局，可 `pip install -e .`）
2. `signbridge.core`：数据结构 + 图谱常量 + 异常体系
3. `signbridge.hands`：检测器、输入源、可视化、模型管理、CLI
4. `tests/`：pytest 测试 + 测试图片资产
5. `README.md`：安装/快速上手/API 速览/手部关键点图谱说明（含示意图）

## 15. 后续步骤（预留，不实现）

1. **时序序列缓冲**：滑动窗口缓存连续帧的 `HandFrame` → 输出 ST-GCN 图数据（节点=关键点，边=`HAND_CONNECTIONS`），并补充相对坐标归一化（以手腕为原点）
2. **静态手势分类**：MediaPipe 内置手势识别接入
3. **人体姿态组件** `signbridge.pose`：复用 `core` 数据结构与 `FrameSource` 协议
4. **ST-GCN 训练与推理**：PyTorch 实现，消费时序缓冲输出
5. **Qt 界面**：PySide6 集成摄像头预览、关键点叠加、翻译结果显示
