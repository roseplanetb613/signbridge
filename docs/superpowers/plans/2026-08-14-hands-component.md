# SignBridge 手部关键点提取组件实现计划（第一步）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `signbridge` Python 包的第一版：MediaPipe 手部关键点提取组件（检测器 + 三种输入源 + 可视化 + 模型管理 + CLI + 测试）。

**Architecture:** src 布局的单包多子包结构。`signbridge.core` 提供共享数据结构（Landmark/Hand/HandFrame）、21 点手部图谱常量（`HAND_CONNECTIONS`，未来 ST-GCN 的图边）与异常体系；`signbridge.hands` 提供 HandDetector（封装 mediapipe Tasks API HandLandmarker）、帧输入源（摄像头/视频/图片，统一 `FrameSource` 协议）、关键点可视化与 CLI 演示工具。模型文件懒下载缓存到 `~/.cache/signbridge/`。

**Tech Stack:** Python 3.14（已装）、mediapipe 0.10.35（已装，Tasks API）、opencv-python 4.13（已装）、numpy 2.5（已装）、pytest 9.1（已装）、setuptools（src 布局）、git。

**关键环境事实（已验证）：**
- 工作区 `E:\SignBridge`，git 已 init（main 分支），已有设计文档 commit `1eac26d`
- 模型下载 URL 可用（Python urllib 可访问外网，PowerShell/curl 不可——所有下载逻辑必须用 Python 实现）：
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`（7,819,105 字节）
- 测试图片（CC0）：`https://upload.wikimedia.org/wikipedia/commons/1/14/Woman%27s_Right_Hand.jpg`（右手张开，12280×9824，需缩放）与 `https://upload.wikimedia.org/wikipedia/commons/5/57/Fist-up.jpg`（拳头，4167×4167，需缩放）
- mediapipe 0.10.35 包结构：仅 `modules`/`tasks` 子包，无 `solutions`；Tasks API 用法：`mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)` + `vision.HandLandmarker.create_from_options(options)`
- `HAND_CONNECTIONS` 官方定义为 21 条边（规格文档已同步修正）

**执行约定：** 每步跑完测试再提交；提交用 `git add <具体文件> && git commit -m "..."`（仓库已配置本地 user.name/user.email，见 Task 1）。

---

### Task 1: 项目脚手架（pyproject + 包骨架 + 可安装）

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`（占位，Task 11 完善）
- Create: `.gitignore`
- Create: `src/signbridge/__init__.py`
- Create: `src/signbridge/core/__init__.py`
- Create: `src/signbridge/hands/__init__.py`

- [ ] **Step 1: 配置 git 本地身份**

```bash
git config user.name "SignBridge Dev"
git config user.email "dev@signbridge.local"
```

- [ ] **Step 2: 创建 `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "signbridge"
version = "0.1.0"
description = "SignBridge: MediaPipe 手部/姿态关键点组件库（手语翻译项目基础层）"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "mediapipe>=0.10,<0.11",
    "opencv-python>=4.8",
    "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: 创建 `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.venv/
dist/
build/
```

- [ ] **Step 4: 创建 `README.md` 占位**

```markdown
# SignBridge

MediaPipe 手部/姿态关键点组件库（手语翻译项目基础层）。

（完整文档见 Task 11）
```

- [ ] **Step 5: 创建三个 `__init__.py`（本步只有 docstring，导出随各实现任务逐步添加）**

`src/signbridge/__init__.py`:
```python
"""SignBridge: 手语翻译项目 —— MediaPipe 关键点组件库。"""

__version__ = "0.1.0"
```

`src/signbridge/core/__init__.py`:
```python
"""signbridge.core: 各组件共享的基础设施（数据结构、图谱常量、异常体系）。"""
```

`src/signbridge/hands/__init__.py`:
```python
"""signbridge.hands: 手部关键点提取组件。"""
```

- [ ] **Step 6: 安装并验证**

Run: `pip install -e ".[dev]"`（PowerShell 中引号必须保留）
Run: `python -c "import signbridge; print(signbridge.__version__)"`
Expected: `0.1.0`

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml README.md .gitignore src/
git commit -m "chore: 项目脚手架（src 布局、可 pip install -e）"
```

---

### Task 2: 异常体系 `signbridge.core.errors`（TDD）

**Files:**
- Create: `src/signbridge/core/errors.py`
- Test: `tests/test_errors.py`

- [ ] **Step 1: 写失败测试 `tests/test_errors.py`**

```python
from signbridge.core.errors import (
    InvalidArgumentError,
    ModelDownloadError,
    ModelNotFoundError,
    SignBridgeError,
    SourceOpenError,
)


def test_base_is_exception():
    assert issubclass(SignBridgeError, Exception)


def test_derived_errors_inherit_base():
    for cls in (ModelNotFoundError, ModelDownloadError, SourceOpenError, InvalidArgumentError):
        assert issubclass(cls, SignBridgeError)


def test_error_carries_message():
    err = SourceOpenError("webcam 0 not available")
    assert str(err) == "webcam 0 not available"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_errors.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.core.errors'`）

- [ ] **Step 3: 实现 `src/signbridge/core/errors.py`**

```python
"""SignBridge 异常体系。

原则：检测不到手不是错误（返回空 HandFrame）；打不开源、模型不可用才是错误。
"""


class SignBridgeError(Exception):
    """SignBridge 所有异常的基类。"""


class ModelNotFoundError(SignBridgeError):
    """模型文件缺失。"""


class ModelDownloadError(SignBridgeError):
    """模型下载失败。"""


class SourceOpenError(SignBridgeError):
    """输入源无法打开（摄像头打不开 / 文件不存在或无法解码）。"""


class InvalidArgumentError(SignBridgeError):
    """参数非法。"""
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_errors.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/core/errors.py tests/test_errors.py
git commit -m "feat: SignBridge 异常体系"
```

---

### Task 3: 数据结构与图谱常量 `signbridge.core.landmarks`（TDD）

**Files:**
- Create: `src/signbridge/core/landmarks.py`
- Test: `tests/test_landmarks.py`

- [ ] **Step 1: 写失败测试 `tests/test_landmarks.py`**

```python
import pytest

from signbridge.core.landmarks import (
    HAND_CONNECTIONS,
    HAND_LANDMARK_NAMES,
    Hand,
    HandFrame,
    Landmark,
)


def test_landmark_dataclass_is_frozen():
    lm = Landmark(x=0.1, y=0.2, z=0.3)
    with pytest.raises(AttributeError):
        lm.x = 0.5


def test_hand_frame_defaults_to_empty():
    frame = HandFrame()
    assert frame.hands == ()
    assert frame.timestamp_ms == 0
    assert frame.frame_index == 0


def test_hand_frame_is_frozen():
    frame = HandFrame()
    with pytest.raises(AttributeError):
        frame.hands = ()


def test_landmark_names_have_21_entries():
    assert len(HAND_LANDMARK_NAMES) == 21


def test_landmark_names_exact():
    expected = (
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
        "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
        "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    )
    assert HAND_LANDMARK_NAMES == expected


def test_connections_have_21_edges():
    assert len(HAND_CONNECTIONS) == 21


def test_connections_exact():
    expected = {
        (0, 1), (1, 2), (2, 3), (3, 4),          # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),          # 食指
        (5, 9), (9, 10), (10, 11), (11, 12),     # 中指
        (9, 13), (13, 14), (14, 15), (15, 16),   # 无名指
        (13, 17), (17, 18), (18, 19), (19, 20),  # 小指
        (0, 17),                                 # 手掌（腕→小指根）
    }
    assert set(HAND_CONNECTIONS) == expected


def test_connections_indices_in_range_no_self_loops():
    for a, b in HAND_CONNECTIONS:
        assert 0 <= a <= 20 and 0 <= b <= 20
        assert a != b
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_landmarks.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.core.landmarks'`）

- [ ] **Step 3: 实现 `src/signbridge/core/landmarks.py`**

```python
"""手部关键点数据结构与 21 点图谱常量。

图谱与 MediaPipe 官方定义一致：
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
HAND_CONNECTIONS 即未来 ST-GCN 的图边。
"""

from dataclasses import dataclass

HAND_LANDMARK_NAMES: tuple[str, ...] = (
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
)

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),          # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),     # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),   # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20),  # 小指
    (0, 17),                                 # 手掌（腕→小指根）
)


@dataclass(frozen=True)
class Landmark:
    """单个关键点坐标。图像坐标模式下 x/y ∈ [0,1]、z 为相对深度；world 模式下为米制。"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class Hand:
    """一只手的 21 个关键点。"""

    landmarks: tuple[Landmark, ...] = ()
    world_landmarks: tuple[Landmark, ...] = ()
    handedness: str = "Unknown"
    score: float = 0.0


@dataclass(frozen=True)
class HandFrame:
    """一帧的检测结果：0~N 只手。"""

    hands: tuple[Hand, ...] = ()
    timestamp_ms: int = 0
    frame_index: int = 0
```

- [ ] **Step 4: 更新 `src/signbridge/core/__init__.py` 导出并跑测试**

`src/signbridge/core/__init__.py`:
```python
"""signbridge.core: 各组件共享的基础设施（数据结构、图谱常量、异常体系）。"""

from signbridge.core.landmarks import (
    HAND_CONNECTIONS,
    HAND_LANDMARK_NAMES,
    Hand,
    HandFrame,
    Landmark,
)

__all__ = [
    "HAND_CONNECTIONS",
    "HAND_LANDMARK_NAMES",
    "Hand",
    "HandFrame",
    "Landmark",
]
```

Run: `pytest tests/test_landmarks.py -v`
Expected: 8 passed

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/core/ tests/test_landmarks.py
git commit -m "feat: 手部关键点数据结构与 21 点图谱常量"
```

---

### Task 4: 测试图片资产（CC0）与 conftest

**Files:**
- Create: `scripts/fetch_assets.py`
- Create: `tests/assets/README.md`
- Create: `tests/conftest.py`
- Result: `tests/assets/hand_open.jpg`、`tests/assets/fist.jpg`

- [ ] **Step 1: 创建 `scripts/fetch_assets.py`**

```python
"""下载并准备测试图片资产（CC0 许可，来源见 tests/assets/README.md）。

用法: python scripts/fetch_assets.py
"""

from pathlib import Path
import urllib.request

import cv2

ASSETS = Path(__file__).resolve().parent.parent / "tests" / "assets"

SOURCES = {
    "hand_open.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/1/14/Woman%27s_Right_Hand.jpg",
    ),
    "fist.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/5/57/Fist-up.jpg",
    ),
}

TARGET_WIDTH = 1024


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "SignBridge-dev/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
        f.write(resp.read())


def resize_to_width(img, width: int):
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, (url,) in SOURCES.items():
        raw = ASSETS / (name + ".raw")
        print(f"downloading {name} ...")
        download(url, raw)
        img = cv2.imread(str(raw))
        if img is None:
            raise SystemExit(f"failed to decode {raw}")
        img = resize_to_width(img, TARGET_WIDTH)
        cv2.imwrite(str(ASSETS / name), img)
        raw.unlink()
        print(f"  -> {ASSETS / name} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行下载脚本**

Run: `python scripts/fetch_assets.py`
Expected: 两个文件生成，hand_open.jpg 宽 1024；脚本退出码 0

- [ ] **Step 3: 创建 `tests/assets/README.md`（记录来源与许可）**

```markdown
# 测试图片资产

| 文件 | 来源 | 许可 |
| --- | --- | --- |
| hand_open.jpg | Wikimedia Commons [File:Woman's Right Hand.jpg](https://commons.wikimedia.org/wiki/File:Woman%27s_Right_Hand.jpg) | CC0 |
| fist.jpg | Wikimedia Commons [File:Fist-up.jpg](https://commons.wikimedia.org/wiki/File:Fist-up.jpg) | CC0 |

原图分辨率过大（分别为 12280×9824 与 4167×4167），经 `scripts/fetch_assets.py`
缩放（宽边 1024px，INTER_AREA）后入库。
```

- [ ] **Step 4: 创建 `tests/conftest.py`**

```python
from pathlib import Path

import pytest

ASSETS = Path(__file__).parent / "assets"
HAND_OPEN = ASSETS / "hand_open.jpg"
FIST = ASSETS / "fist.jpg"


@pytest.fixture
def hand_open_path() -> Path:
    return HAND_OPEN


@pytest.fixture
def fist_path() -> Path:
    return FIST
```

- [ ] **Step 5: 验证资产可解码并提交**

Run: `python -c "import cv2; [print(p, cv2.imread(p).shape) for p in ['tests/assets/hand_open.jpg','tests/assets/fist.jpg']]"`
Expected: 两个文件均输出 (h, 1024, 3)

```bash
git add scripts/ tests/assets/ tests/conftest.py
git commit -m "test: CC0 测试图片资产与 conftest"
```

---

### Task 5: 模型下载与缓存管理 `signbridge.hands.model`（TDD）

**Files:**
- Create: `src/signbridge/hands/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: 写失败测试 `tests/test_model.py`**

```python
import pytest

from signbridge.core.errors import ModelDownloadError
from signbridge.hands import model


def test_ensure_model_skips_when_cached(tmp_path, monkeypatch):
    dest = tmp_path / "hand_landmarker.task"
    dest.write_bytes(b"fake-model")
    calls = []

    def fake_download(url, tmp):
        calls.append(url)
        tmp.write_bytes(b"new")

    monkeypatch.setattr(model, "_download", fake_download)
    result = model.ensure_model(url="https://example.test/model.task", dest=dest)
    assert result == dest
    assert calls == []
    assert dest.read_bytes() == b"fake-model"


def test_ensure_model_downloads_when_missing(tmp_path, monkeypatch):
    dest = tmp_path / "hand_landmarker.task"

    def fake_download(url, tmp):
        assert url == "https://example.test/model.task"
        tmp.write_bytes(b"new")

    monkeypatch.setattr(model, "_download", fake_download)
    result = model.ensure_model(url="https://example.test/model.task", dest=dest)
    assert result == dest
    assert dest.read_bytes() == b"new"


def test_ensure_model_download_failure_raises_and_cleans(tmp_path, monkeypatch):
    dest = tmp_path / "hand_landmarker.task"

    def fake_download(url, tmp):
        raise OSError("network down")

    monkeypatch.setattr(model, "_download", fake_download)
    with pytest.raises(ModelDownloadError):
        model.ensure_model(url="https://example.test/model.task", dest=dest)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_default_model_path_is_under_cache_dir():
    path = model.default_model_path()
    assert path.name == "hand_landmarker.task"
    assert str(model.cache_dir()) in str(path)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_model.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.hands.model'`）

- [ ] **Step 3: 实现 `src/signbridge/hands/model.py`**

```python
"""手部关键点模型（hand_landmarker.task）的下载与缓存管理。"""

from pathlib import Path
import urllib.request

from signbridge.core.errors import ModelDownloadError

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def cache_dir() -> Path:
    """模型缓存目录（~/.cache/signbridge）。"""
    return Path.home() / ".cache" / "signbridge"


def default_model_path() -> Path:
    return cache_dir() / MODEL_FILENAME


def _download(url: str, dest: Path) -> None:
    """带进度的分块下载（必须用 urllib：本机 PowerShell/curl 网络受限）。"""
    req = urllib.request.Request(url, headers={"User-Agent": "SignBridge/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r下载模型 {done / total:.0%}", end="", flush=True)
    print()


def ensure_model(
    url: str = MODEL_URL,
    dest: Path | None = None,
    version: str | None = None,
) -> Path:
    """确保模型文件存在，缺失时自动下载；返回模型路径（幂等）。

    version 参数预留用于未来多模型版本切换；当前固定使用与
    mediapipe>=0.10,<0.11 配套的官方 float16 模型。
    """
    dest = dest or default_model_path()
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        _download(url, part)
        part.replace(dest)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"下载手部模型失败（{exc}）。请检查网络后重试，"
            f"或手动下载 {url} 并保存到 {dest}"
        ) from exc
    return dest
```

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_model.py -v`
Expected: 4 passed

- [ ] **Step 5: 真实下载一次模型（验证 urllib 通路与大小）**

Run: `python -c "from signbridge.hands.model import ensure_model; p = ensure_model(); print(p, p.stat().st_size)"`
Expected: 打印缓存路径与 7819105 字节；再次运行应秒回（缓存命中）

- [ ] **Step 6: 提交**

```bash
git add src/signbridge/hands/model.py tests/test_model.py
git commit -m "feat: 手部模型下载与缓存管理"
```

---

### Task 6: 检测器 `signbridge.hands.detector`（TDD 单元测试）

**Files:**
- Create: `src/signbridge/hands/detector.py`
- Test: `tests/test_detector.py`

- [ ] **Step 1: 写失败测试 `tests/test_detector.py`**

```python
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from signbridge.core.errors import InvalidArgumentError, ModelNotFoundError
from signbridge.core.landmarks import HandFrame
from signbridge.hands.detector import HandDetector


def _fake_category(name: str, score: float):
    from mediapipe.tasks.python.components.containers.category import Category

    return Category(index=0, score=score, display_name="", category_name=name)


def _fake_landmarks(n=21):
    from mediapipe.tasks.python.components.containers.landmark import (
        NormalizedLandmark,
    )

    return [NormalizedLandmark(x=i / 20, y=0.5, z=0.0) for i in range(n)]


def _fake_result(n_hands=1):
    return SimpleNamespace(
        hand_landmarks=[_fake_landmarks() for _ in range(n_hands)],
        hand_world_landmarks=[_fake_landmarks() for _ in range(n_hands)],
        handedness=[[_fake_category("Left", 0.95)] for _ in range(n_hands)],
    )


@pytest.fixture
def fake_landmarker(monkeypatch):
    captured = {"close_called": False}

    def fake_create(*args, **kwargs):
        return SimpleNamespace(
            detect=lambda img: _fake_result(),
            close=lambda: captured.__setitem__("close_called", True),
        )

    monkeypatch.setattr("signbridge.hands.detector._create_landmarker", fake_create)
    return captured


def _bgr_frame(w=320, h=240):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _model_file(tmp_path) -> Path:
    model = tmp_path / "m.task"
    model.write_bytes(b"x")
    return model


def test_detect_returns_converted_hand_frame(fake_landmarker, tmp_path):
    detector = HandDetector(model_path=_model_file(tmp_path))
    result = detector.detect(_bgr_frame())
    assert isinstance(result, HandFrame)
    assert len(result.hands) == 1
    hand = result.hands[0]
    assert len(hand.landmarks) == 21
    assert len(hand.world_landmarks) == 21
    assert hand.handedness == "Left"
    assert hand.score == pytest.approx(0.95)
    assert result.frame_index == 0
    assert result.timestamp_ms > 0
    detector.close()


def test_detect_empty_result_yields_empty_hands(tmp_path, monkeypatch):
    def fake_create(*args, **kwargs):
        return SimpleNamespace(
            detect=lambda img: SimpleNamespace(
                hand_landmarks=[], hand_world_landmarks=[], handedness=[]
            ),
            close=lambda: None,
        )

    monkeypatch.setattr("signbridge.hands.detector._create_landmarker", fake_create)
    detector = HandDetector(model_path=_model_file(tmp_path))
    assert detector.detect(_bgr_frame()).hands == ()
    detector.close()


def test_detect_increments_frame_index(fake_landmarker, tmp_path):
    detector = HandDetector(model_path=_model_file(tmp_path))
    detector.detect(_bgr_frame())
    assert detector.detect(_bgr_frame()).frame_index == 1
    detector.close()


def test_missing_model_raises(tmp_path):
    with pytest.raises(ModelNotFoundError):
        HandDetector(model_path=tmp_path / "nope.task")


def test_invalid_max_hands_raises(tmp_path):
    with pytest.raises(InvalidArgumentError):
        HandDetector(model_path=_model_file(tmp_path), max_num_hands=3)


def test_context_manager_closes(fake_landmarker, tmp_path):
    with HandDetector(model_path=_model_file(tmp_path)) as detector:
        detector.detect(_bgr_frame())
    assert fake_landmarker["close_called"] is True


def test_detect_after_close_raises(fake_landmarker, tmp_path):
    detector = HandDetector(model_path=_model_file(tmp_path))
    detector.close()
    with pytest.raises(RuntimeError):
        detector.detect(_bgr_frame())
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_detector.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.hands.detector'`）

- [ ] **Step 3: 实现 `src/signbridge/hands/detector.py`**

```python
"""手部关键点检测器（封装 MediaPipe Tasks API 的 HandLandmarker）。"""

import time
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from signbridge.core.errors import InvalidArgumentError, ModelNotFoundError
from signbridge.core.landmarks import Hand, HandFrame, Landmark
from signbridge.hands.model import ensure_model

_MAX_HANDS_ALLOWED = (1, 2)


def _create_landmarker(model_path, num_hands, min_detection, min_tracking):
    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection,
        min_hand_presence_confidence=min_detection,
        min_tracking_confidence=min_tracking,
    )
    return vision.HandLandmarker.create_from_options(options)


def _to_landmark(lm) -> Landmark:
    return Landmark(x=lm.x, y=lm.y, z=lm.z)


def _to_hand(landmarks, world_landmarks, handedness) -> Hand:
    name = handedness[0].category_name if handedness else "Unknown"
    score = handedness[0].score if handedness else 0.0
    return Hand(
        landmarks=tuple(_to_landmark(lm) for lm in landmarks),
        world_landmarks=tuple(_to_landmark(lm) for lm in world_landmarks),
        handedness=name,
        score=score,
    )


class HandDetector:
    """BGR 帧 → HandFrame 的手部关键点检测器。

    无帧间历史状态；仅维护自增帧计数与单调时钟时间戳（时序缓冲是后续步骤职责）。
    支持 with 语句；close() 释放底层资源。
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | Path | None = None,
    ) -> None:
        if max_num_hands not in _MAX_HANDS_ALLOWED:
            raise InvalidArgumentError(
                f"max_num_hands 必须是 1 或 2，收到 {max_num_hands!r}"
            )
        if model_path is None:
            model_path = ensure_model()
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelNotFoundError(
                f"模型文件不存在: {model_path}。可运行 "
                "`python -m signbridge.hands.cli --download-model` 下载。"
            )
        self._landmarker = _create_landmarker(
            model_path,
            max_num_hands,
            min_detection_confidence,
            min_tracking_confidence,
        )
        self._closed = False
        self._frame_index = 0

    def detect(self, frame) -> HandFrame:
        """检测一帧 BGR 图像（H×W×3 uint8），返回 HandFrame（无手时 hands 为空）。"""
        if self._closed:
            raise RuntimeError("HandDetector 已 close()，不可再检测")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self._landmarker.detect(image)
        hands = tuple(
            _to_hand(h, w, c)
            for h, w, c in zip(
                result.hand_landmarks,
                result.hand_world_landmarks,
                result.handedness,
            )
        )
        hand_frame = HandFrame(
            hands=hands,
            timestamp_ms=int(time.monotonic() * 1000),
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return hand_frame

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
```

- [ ] **Step 4: 更新 `src/signbridge/hands/__init__.py` 导出并跑测试**

`src/signbridge/hands/__init__.py`:
```python
"""signbridge.hands: 手部关键点提取组件。"""

from signbridge.hands.detector import HandDetector

__all__ = ["HandDetector"]
```

Run: `pytest tests/test_detector.py -v`
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/hands/ tests/test_detector.py
git commit -m "feat: HandDetector 手部关键点检测器（Tasks API 封装）"
```

---

### Task 7: 检测器集成测试（真实模型 + 真实图片）

**Files:**
- Create: `tests/test_detector_integration.py`

- [ ] **Step 1: 写集成测试 `tests/test_detector_integration.py`**

```python
"""真实模型 + 真实图片的集成测试。

首次运行会通过 ensure_model() 下载 7.8MB 模型到 ~/.cache/signbridge/（幂等，之后秒过）。
"""

import cv2
import numpy as np
import pytest

from signbridge.hands.detector import HandDetector


@pytest.fixture(scope="module")
def detector():
    with HandDetector(max_num_hands=2) as d:
        yield d


def test_detects_hand_in_open_palm(detector, hand_open_path):
    frame = cv2.imread(str(hand_open_path))
    result = detector.detect(frame)
    assert len(result.hands) >= 1
    hand = result.hands[0]
    assert len(hand.landmarks) == 21
    for lm in hand.landmarks:
        assert 0.0 <= lm.x <= 1.0
        assert 0.0 <= lm.y <= 1.0
    assert hand.handedness in ("Left", "Right")
    assert 0.0 <= hand.score <= 1.0


def test_detects_fist(detector, fist_path):
    frame = cv2.imread(str(fist_path))
    result = detector.detect(frame)
    assert len(result.hands) >= 1


def test_blank_image_has_no_hands(detector):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(blank)
    assert result.hands == ()
```

- [ ] **Step 2: 跑集成测试**

Run: `pytest tests/test_detector_integration.py -v`
Expected: 3 passed（首次运行含模型下载）

- [ ] **Step 3: 提交**

```bash
git add tests/test_detector_integration.py
git commit -m "test: 检测器集成测试（真实模型+真实图片）"
```

---

### Task 8: 输入源 `signbridge.hands.sources`（TDD）

**Files:**
- Create: `src/signbridge/hands/sources.py`
- Test: `tests/test_sources.py`

- [ ] **Step 1: 写失败测试 `tests/test_sources.py`**

```python
import cv2
import numpy as np
import pytest

from signbridge.core.errors import SourceOpenError
from signbridge.hands.sources import CameraSource, ImageSource, VideoSource


def test_image_source_yields_single_frame(hand_open_path):
    src = ImageSource(hand_open_path)
    frames = list(src)
    assert len(frames) == 1
    frame, index, ts = frames[0]
    assert frame.shape[2] == 3 and frame.dtype == np.uint8
    assert index == 0
    assert ts == 0.0
    src.close()


def test_image_source_missing_file_raises(tmp_path):
    with pytest.raises(SourceOpenError):
        ImageSource(tmp_path / "nope.jpg")


def test_image_source_repeat(hand_open_path):
    src = ImageSource(hand_open_path, repeat=True)
    frames = [next(src) for _ in range(3)]
    assert len(frames) == 3
    assert frames[0][1] == 0 and frames[2][1] == 2
    src.close()
    with pytest.raises(StopIteration):
        next(src)


def test_video_source_yields_all_frames(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    src = VideoSource(path)
    assert src.meta["frame_count"] == 5
    assert src.meta["width"] == 64 and src.meta["height"] == 48
    frames = list(src)
    assert len(frames) == 5
    assert frames[0][1] == 0 and frames[4][1] == 4
    src.close()
    with pytest.raises(StopIteration):
        next(src)


def test_video_source_missing_file_raises(tmp_path):
    with pytest.raises(SourceOpenError):
        VideoSource(tmp_path / "nope.avi")


class _FakeCapture:
    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False

    def read(self):
        if self._frames:
            return True, self._frames.pop(0)
        return False, None

    def release(self):
        self.released = True

    def isOpened(self):
        return True

    def get(self, prop):
        return 0.0


def test_camera_source_yields_frames(monkeypatch):
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3)]
    fake = _FakeCapture(frames)
    monkeypatch.setattr("signbridge.hands.sources._open_capture", lambda cam_id: fake)

    src = CameraSource(0)
    got = list(src)
    assert len(got) == 3
    src.close()
    assert fake.released


def test_camera_source_open_failure_raises(monkeypatch):
    monkeypatch.setattr("signbridge.hands.sources._open_capture", lambda cam_id: None)
    with pytest.raises(SourceOpenError):
        CameraSource(0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_sources.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.hands.sources'`）

- [ ] **Step 3: 实现 `src/signbridge/hands/sources.py`**

```python
"""帧输入源：摄像头 / 视频文件 / 图片文件。

统一产出 (frame, frame_index, timestamp_ms) 三元组，消费方与具体源解耦。
timestamp_ms 优先取 OpenCV 帧时间戳（CAP_PROP_POS_MSEC），不可用时回退系统单调时钟。
"""

import time
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from signbridge.core.errors import SourceOpenError

FrameTuple = tuple[np.ndarray, int, float]


class FrameSource(Protocol):
    """帧源协议：可迭代产出 (BGR 帧, 帧索引, 时间戳毫秒)。"""

    def __iter__(self) -> Iterator[FrameTuple]: ...

    def __next__(self) -> FrameTuple: ...

    def close(self) -> None: ...


def _open_capture(camera_id: int):
    """可被测试替换的摄像头打开函数。"""
    return cv2.VideoCapture(camera_id)


class CameraSource:
    """摄像头帧源。帧读取失败（断开）时迭代结束。"""

    def __init__(self, camera_id: int = 0) -> None:
        self._cap = _open_capture(camera_id)
        if self._cap is None or not self._cap.isOpened():
            raise SourceOpenError(f"无法打开摄像头 #{camera_id}")
        self._index = 0

    def __iter__(self) -> "CameraSource":
        return self

    def __next__(self) -> FrameTuple:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise StopIteration
        ts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        if ts <= 0:
            ts = time.monotonic() * 1000
        result = (frame, self._index, ts)
        self._index += 1
        return result

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class VideoSource:
    """视频文件帧源。meta 提供 width/height/fps/frame_count。"""

    def __init__(self, path: str | Path) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise SourceOpenError(f"无法打开视频文件: {path}")
        self._index = 0
        self.meta = {
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

    def __iter__(self) -> "VideoSource":
        return self

    def __next__(self) -> FrameTuple:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise StopIteration
        ts = self._cap.get(cv2.CAP_PROP_POS_MSEC)
        if ts <= 0:
            ts = time.monotonic() * 1000
        result = (frame, self._index, ts)
        self._index += 1
        return result

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageSource:
    """单张图片帧源；repeat=True 时循环输出同一帧。"""

    def __init__(self, path: str | Path, repeat: bool = False) -> None:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SourceOpenError(f"无法读取图片: {path}")
        self._frame = frame
        self._repeat = repeat
        self._index = 0
        self._done = False

    def __iter__(self) -> "ImageSource":
        return self

    def __next__(self) -> FrameTuple:
        if self._done:
            raise StopIteration
        if not self._repeat:
            self._done = True
        result = (self._frame, self._index, 0.0)
        self._index += 1
        return result

    def close(self) -> None:
        self._done = True
```

- [ ] **Step 4: 更新 `src/signbridge/hands/__init__.py` 导出并跑测试**

`src/signbridge/hands/__init__.py`:
```python
"""signbridge.hands: 手部关键点提取组件。"""

from signbridge.hands.detector import HandDetector
from signbridge.hands.sources import CameraSource, ImageSource, VideoSource

__all__ = ["HandDetector", "CameraSource", "ImageSource", "VideoSource"]
```

Run: `pytest tests/test_sources.py -v`
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/hands/sources.py src/signbridge/hands/__init__.py tests/test_sources.py
git commit -m "feat: 帧输入源（摄像头/视频/图片，统一 FrameSource 协议）"
```

---

### Task 9: 可视化 `signbridge.hands.draw`（TDD）

**Files:**
- Create: `src/signbridge/hands/draw.py`
- Test: `tests/test_draw.py`

- [ ] **Step 1: 写失败测试 `tests/test_draw.py`**

```python
import numpy as np

from signbridge.core.landmarks import Hand, HandFrame, Landmark
from signbridge.hands.draw import draw_landmarks


def _open_hand_frame() -> HandFrame:
    lms = tuple(
        Landmark(x=0.1 + 0.03 * i, y=0.2 + 0.02 * (i % 5), z=0.0)
        for i in range(21)
    )
    return HandFrame(hands=(Hand(landmarks=lms, handedness="Right", score=0.9),))


def test_empty_frame_returns_unchanged_copy():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = draw_landmarks(frame, HandFrame())
    assert out is not frame
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
    assert np.array_equal(out, frame)


def test_drawn_frame_keeps_shape_and_does_not_mutate_input():
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    before = frame.copy()
    out = draw_landmarks(frame, _open_hand_frame())
    assert out.shape == frame.shape and out.dtype == frame.dtype
    assert np.array_equal(frame, before)
    assert not np.array_equal(out, frame)


def test_handedness_colors_differ():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    lms = _open_hand_frame().hands[0].landmarks
    left = HandFrame(hands=(Hand(landmarks=lms, handedness="Left", score=0.9),))
    right = HandFrame(hands=(Hand(landmarks=lms, handedness="Right", score=0.9),))
    assert not np.array_equal(draw_landmarks(frame, left), draw_landmarks(frame, right))


def test_explicit_color_used():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = draw_landmarks(frame, _open_hand_frame(), color=(0, 0, 255))
    assert np.any(out[:, :, 0] > 0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_draw.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.hands.draw'`）

- [ ] **Step 3: 实现 `src/signbridge/hands/draw.py`**

```python
"""手部关键点可视化：在帧副本上绘制关键点、骨骼连线与标签。"""

import cv2
import numpy as np

from signbridge.core.landmarks import HAND_CONNECTIONS, HandFrame

_COLORS = {
    "Left": (0, 255, 0),     # BGR 绿色
    "Right": (0, 165, 255),  # BGR 橙色
    "Unknown": (200, 200, 200),
}


def _color_for(handedness: str, color):
    if color is not None:
        return tuple(int(c) for c in color)
    return _COLORS.get(handedness, _COLORS["Unknown"])


def draw_landmarks(
    frame: np.ndarray, hand_frame: HandFrame, color=None
) -> np.ndarray:
    """在 BGR 帧的副本上绘制手部关键点与骨骼连线，返回新帧（不改原帧）。"""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    for hand in hand_frame.hands:
        c = _color_for(hand.handedness, color)
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], c, 2, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(canvas, pt, 3, c, -1, cv2.LINE_AA)
        label = f"{hand.handedness} {hand.score:.2f}"
        if pts:
            cv2.putText(
                canvas,
                label,
                (pts[0][0] - 10, max(pts[0][1] - 10, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                c,
                1,
                cv2.LINE_AA,
            )
    return canvas
```

- [ ] **Step 4: 更新导出并跑测试**

`src/signbridge/hands/__init__.py`:
```python
"""signbridge.hands: 手部关键点提取组件。"""

from signbridge.hands.detector import HandDetector
from signbridge.hands.draw import draw_landmarks
from signbridge.hands.sources import CameraSource, ImageSource, VideoSource

__all__ = ["HandDetector", "draw_landmarks", "CameraSource", "ImageSource", "VideoSource"]
```

Run: `pytest tests/test_draw.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/hands/draw.py src/signbridge/hands/__init__.py tests/test_draw.py
git commit -m "feat: 手部关键点可视化叠加绘制"
```

---

### Task 10: CLI 演示工具 `signbridge.hands.cli`（TDD）

**Files:**
- Create: `src/signbridge/hands/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: 写失败测试 `tests/test_cli.py`**

```python
from signbridge.hands import cli


def test_parser_defaults():
    args = cli.build_parser().parse_args([])
    assert args.source == "camera"
    assert args.max_hands == 2
    assert args.no_overlay is False


def test_parser_image_source_accepts_path():
    args = cli.build_parser().parse_args(["--source", "image", "--path", "a.jpg"])
    assert args.path == "a.jpg"


def test_download_model_flag(monkeypatch, capsys):
    calls = []

    def fake_ensure(url=None, dest=None, version=None):
        calls.append(1)
        return "C:/cache/hand_landmarker.task"

    monkeypatch.setattr(cli, "ensure_model", fake_ensure)
    assert cli.main(["--download-model"]) == 0
    assert calls == [1]
    assert "C:/cache/hand_landmarker.task" in capsys.readouterr().out


def test_image_no_overlay_runs_and_prints_summary(hand_open_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from signbridge.core.landmarks import HandFrame

    def fake_detector(*args, **kwargs):
        return SimpleNamespace(
            detect=lambda frame: HandFrame(hands=(), timestamp_ms=0, frame_index=0),
            close=lambda: None,
            __enter__=lambda self: self,
            __exit__=lambda self, *a: False,
        )

    monkeypatch.setattr(cli, "HandDetector", fake_detector)
    code = cli.main(["--source", "image", "--path", str(hand_open_path), "--no-overlay"])
    assert code == 0
    assert "hands=0" in capsys.readouterr().out
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.hands.cli'`）

- [ ] **Step 3: 实现 `src/signbridge/hands/cli.py`**

```python
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
```

- [ ] **Step 4: 跑单元测试确认通过**

Run: `pytest tests/test_cli.py -v`
Expected: 4 passed

- [ ] **Step 5: 冒烟测试（真实模型 + 真实图片，无窗口）**

Run: `python -m signbridge.hands.cli --source image --path tests/assets/hand_open.jpg --no-overlay`
Expected: 打印一行 `frame=0 hands=N Left(...) ...`，退出码 0

- [ ] **Step 6: 提交**

```bash
git add src/signbridge/hands/cli.py tests/test_cli.py
git commit -m "feat: 手部关键点 CLI 演示工具"
```

---

### Task 11: README 完整文档 + 关键点编号示意图

**Files:**
- Create: `scripts/make_diagram.py`
- Create: `docs/images/hand_landmark_diagram.png`（脚本生成）
- Modify: `README.md`（替换占位）

- [ ] **Step 1: 创建 `scripts/make_diagram.py`（基于真实检测结果生成 21 点编号图）**

```python
"""生成 README 用的手部关键点编号示意图（基于真实检测结果）。

用法: python scripts/make_diagram.py
"""

from pathlib import Path

import cv2

from signbridge.core.landmarks import HAND_CONNECTIONS
from signbridge.hands.detector import HandDetector
from signbridge.hands.sources import ImageSource

ROOT = Path(__file__).resolve().parent.parent
ASSET = ROOT / "tests" / "assets" / "hand_open.jpg"
OUT = ROOT / "docs" / "images" / "hand_landmark_diagram.png"


def main() -> None:
    src = ImageSource(ASSET)
    frame, _, _ = next(iter(src))
    src.close()
    with HandDetector(max_num_hands=2) as detector:
        hand_frame = detector.detect(frame)
        if not hand_frame.hands:
            raise SystemExit("测试图片未检测到手，无法生成示意图")
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        hand = hand_frame.hands[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(canvas, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            cv2.circle(canvas, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(i),
                (pt[0] + 6, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT), canvas)
        print(f"示意图已生成: {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行生成示意图**

Run: `python scripts/make_diagram.py`
Expected: 输出 `示意图已生成: E:\SignBridge\docs\images\hand_landmark_diagram.png`

- [ ] **Step 3: 用完整 README 替换占位（`README.md`）**

```markdown
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
pytest -v
```

测试图片取自 Wikimedia Commons（CC0），来源见 `tests/assets/README.md`。

## 路线图

- [x] 第一步：手部关键点提取组件（本版本）
- [ ] 第二步：时序序列缓冲 → ST-GCN 图数据（滑动窗口 + 手腕原点归一化）
- [ ] 静态手势分类（MediaPipe 内置）
- [ ] 人体姿态组件 `signbridge.pose`
- [ ] ST-GCN 训练与推理（PyTorch）
- [ ] Qt 界面（PySide6）：摄像头预览 / 关键点叠加 / 翻译结果显示
```

- [ ] **Step 4: 跑全部测试确认无回归**

Run: `pytest -v`
Expected: 全部通过（errors 3 + landmarks 8 + model 4 + detector 7 + integration 3 + sources 7 + draw 4 + cli 4 = 40 passed）

- [ ] **Step 5: 提交**

```bash
git add README.md scripts/make_diagram.py docs/images/
git commit -m "docs: README 完整文档与 21 点编号示意图"
```

---

### Task 12: 全量验证与收尾

**Files:**
- Modify: 无（仅验证）

- [ ] **Step 1: 全量测试**

Run: `pytest -v`
Expected: 40 passed

- [ ] **Step 2: CLI 冒烟（三源各一次）**

Run: `python -m signbridge.hands.cli --source image --path tests/assets/hand_open.jpg --no-overlay`
Run: `python -m signbridge.hands.cli --source image --path tests/assets/fist.jpg --no-overlay`
Expected: 各打印一行摘要，退出码 0

- [ ] **Step 3: 公共 API 导入验证**

Run: `python -c "import signbridge; print(signbridge.__version__); from signbridge import HandDetector, HAND_CONNECTIONS, HandFrame; print(len(HAND_CONNECTIONS))"`
Expected: `0.1.0` 与 `21`

- [ ] **Step 4: git 状态检查**

Run: `git status --short && git log --oneline`
Expected: 工作区干净；日志含本计划各任务提交

- [ ] **Step 5: 完成声明**

实现完成。向用户报告：交付物清单、测试结果、使用示例、下一步（时序序列缓冲 → ST-GCN 图数据）。
