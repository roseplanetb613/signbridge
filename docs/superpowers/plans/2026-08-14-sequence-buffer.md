# SignBridge 时序序列缓冲组件实现计划（第二步）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现时序序列缓冲组件：`HandSequenceBuffer`（匈牙利匹配 + ID 生命周期 + 滑动窗口 + 腕点归一化），配套可插拔的平滑（OneEuro）与匹配（匈牙利）协议，输出可直接喂 ST-GCN 的 `HandSequence`。

**Architecture:** 复用第一步的 `HandFrame`。新增 `core/smoothing.py`（`LandmarkSmoother` 协议 + `OneEuroSmoother`）、`core/matching.py`（`Matcher` 协议 + `Matching` 结果 + `HungarianMatcher`）、`hands/sequence.py`（`HandSequence` 数据结构 + `HandSequenceBuffer` 状态机）。匹配与平滑均为可插拔协议；Buffer 只依赖协议，不依赖具体实现。

**Tech Stack:** Python 3.14、numpy 2.5（匈牙利算法手写，无 scipy）、pytest。全部纯 numpy 测试，不碰摄像头。

**关键环境事实：**
- 工作区 `E:\SignBridge`，第一步已完成（40+ 测试、`signbridge` 已 pip install -e）
- pytest 必须用 `python -m pytest`（裸 `pytest` 指向 Anaconda 旧环境）
- `HandFrame` / `Hand` / `Landmark` 来自 `signbridge.core.landmarks`（frozen dataclass）
- 规格：`docs/superpowers/specs/2026-08-14-sequence-buffer-design.md`（第 8 节已修订为「帧槽位占位推进」语义）

**执行约定：** 每步跑完测试再提交；全部测试通过后才 commit（上一步的教训）。

---

### Task 1: 平滑协议 + OneEuro 实现 `core/smoothing.py`（TDD）

**Files:**
- Create: `src/signbridge/core/smoothing.py`
- Test: `tests/test_smoothing.py`

- [ ] **Step 1: 写失败测试 `tests/test_smoothing.py`**

```python
import numpy as np
import pytest

from signbridge.core.smoothing import LandmarkSmoother, OneEuroSmoother


def _pts(v):
    return np.full((21, 3), v, dtype=np.float32)


def test_protocol_exposes_update_and_reset():
    assert hasattr(LandmarkSmoother, "update")
    assert hasattr(LandmarkSmoother, "reset")


def test_first_frame_passthrough():
    s = OneEuroSmoother()
    pts = np.random.default_rng(1).random((21, 3)).astype(np.float32)
    out = s.update(pts.copy())
    assert np.array_equal(out, pts)


def test_constant_sequence_converges():
    s = OneEuroSmoother()
    out = None
    for _ in range(200):
        out = s.update(_pts(0.5))
    assert out is not None
    assert np.allclose(out, _pts(0.5), atol=1e-3)


def test_step_response_smoothed_then_converges():
    s = OneEuroSmoother()
    for _ in range(50):
        s.update(_pts(0.0))
    first = s.update(_pts(1.0))
    assert np.all(first < 1.0)  # 阶跃被平滑，未立即跳满
    for _ in range(400):
        last = s.update(_pts(1.0))
    assert np.allclose(last, _pts(1.0), atol=1e-2)


def test_none_input_keeps_state():
    s = OneEuroSmoother()
    pts = np.random.default_rng(0).random((21, 3)).astype(np.float32)
    a = s.update(pts.copy())
    assert s.update(None) is None
    b = s.update(pts.copy())
    assert np.array_equal(a, b)  # None 不改变内部状态


def test_reset_clears_memory():
    s = OneEuroSmoother()
    for _ in range(100):
        s.update(_pts(1.0))
    s.reset()
    jump = _pts(0.0)
    out = s.update(jump.copy())
    assert np.array_equal(out, jump)  # reset 后无记忆，首帧直通
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_smoothing.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.core.smoothing'`）

- [ ] **Step 3: 实现 `src/signbridge/core/smoothing.py`**

```python
"""关键点平滑：抽象协议 + OneEuro 默认实现（可插拔，供 hands/pose 复用）。"""

from typing import Protocol

import numpy as np


class LandmarkSmoother(Protocol):
    """平滑协议：每帧喂入 (21,3) 点阵（或 None 表示该帧无数据），返回平滑结果。"""

    def update(self, points: np.ndarray | None) -> np.ndarray | None: ...

    def reset(self) -> None: ...


def _alpha(dt: float, cutoff) -> np.ndarray:
    """OneEuro 系数：alpha = 1 / (1 + tau/dt)，tau = 1/(2π·cutoff)。"""
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroSmoother:
    """运动自适应低通滤波（帧空间，dt=1）。

    静止时强平滑、快速运动时跟随（低延迟）。适合手部关键点抖动抑制。
    每个 track 独立持有一个实例；update(None) 保持内部状态。
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0 or d_cutoff <= 0 or beta < 0:
            raise ValueError("min_cutoff/d_cutoff 必须 > 0，beta 必须 >= 0")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    def update(self, points: np.ndarray | None) -> np.ndarray | None:
        if points is None:
            return None
        pts = np.asarray(points, dtype=np.float32)
        if self._x_prev is None:
            self._x_prev = pts.copy()
            self._dx_prev = np.zeros_like(pts)
            return pts.copy()
        dt = 1.0  # 帧空间
        dx = (pts - self._x_prev) / dt
        alpha_d = _alpha(dt, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        alpha = _alpha(dt, cutoff)
        x_hat = alpha * pts + (1.0 - alpha) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_smoothing.py -v`
Expected: 6 passed

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/core/smoothing.py tests/test_smoothing.py
git commit -m "feat: 关键点平滑协议 + OneEuro 实现（可插拔）"
```

---

### Task 2: 匹配协议 + 匈牙利实现 `core/matching.py`（TDD）

**Files:**
- Create: `src/signbridge/core/matching.py`
- Test: `tests/test_matching.py`

- [ ] **Step 1: 写失败测试 `tests/test_matching.py`**

```python
import numpy as np
import pytest

from signbridge.core.matching import HungarianMatcher, Matcher, Matching


def _pts(*xy):
    return np.array(xy, dtype=np.float32)


def test_protocol_exposes_match():
    assert hasattr(Matcher, "match")


def test_basic_assignment():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(_pts((0.0, 0.0), (1.0, 1.0)), _pts((0.1, 0.1), (0.9, 0.9)))
    assert set(res.matched) == {(0, 0), (1, 1)}
    assert res.unmatched_current == ()
    assert res.unmatched_previous == ()


def test_cross_swap_matches_by_nearest():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(_pts((0.9, 0.9), (0.1, 0.1)), _pts((0.1, 0.1), (0.9, 0.9)))
    assert set(res.matched) == {(0, 1), (1, 0)}  # 每只手仍绑定自己的轨迹


def test_distance_beyond_threshold_unmatched():
    m = HungarianMatcher(distance_threshold=0.3)
    res = m.match(_pts((0.0, 0.0)), _pts((0.9, 0.9)))
    assert res.matched == ()
    assert res.unmatched_current == (0,)
    assert res.unmatched_previous == (0,)


def test_asymmetric_counts():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(
        _pts((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)),  # 当前 3 手
        _pts((0.0, 0.0), (1.0, 1.0)),              # 轨迹 2 条
    )
    assert len(res.matched) == 2
    assert len(res.unmatched_current) == 1
    assert res.unmatched_previous == ()


def test_empty_side():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(np.zeros((0, 2), dtype=np.float32), _pts((0.0, 0.0)))
    assert res.matched == ()
    assert res.unmatched_current == ()
    assert res.unmatched_previous == (0,)


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        HungarianMatcher(distance_threshold=-0.1)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_matching.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.core.matching'`）

- [ ] **Step 3: 实现 `src/signbridge/core/matching.py`**

```python
"""帧间关联：抽象协议 + 匈牙利最小代价匹配默认实现（可插拔）。

Matcher 只做「谁跟谁」的关联决策，不管理 ID 生命周期（那是 Buffer 的职责）。
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Matching:
    """一次帧间匹配的结果。索引分别指向当前帧手与上一帧轨迹。"""

    matched: tuple[tuple[int, int], ...] = ()
    unmatched_current: tuple[int, ...] = ()
    unmatched_previous: tuple[int, ...] = ()


class Matcher(Protocol):
    """帧间关联协议。centroids 均为 (N,2) float32（当前帧 / 上一帧轨迹质心）。"""

    def match(
        self,
        current_centroids: np.ndarray,
        previous_centroids: np.ndarray,
    ) -> Matching: ...


def _hungarian_min(cost: np.ndarray) -> dict[int, int]:
    """经典匈牙利算法（最小化，O(n³)）。cost 为方阵，返回 {行索引: 列索引}。"""
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)
    inf = 1e18
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, inf)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return {i: p[i] for i in range(1, n + 1)}


class HungarianMatcher:
    """默认实现：匈牙利算法最小化质心欧氏距离，超过阈值视为不匹配。

    非方阵时以虚行/虚列（代价 1e6）补齐；虚对不参与匹配结果。
    """

    def __init__(self, distance_threshold: float = 0.15) -> None:
        if distance_threshold < 0:
            raise ValueError("distance_threshold 必须 >= 0")
        self.distance_threshold = distance_threshold

    def match(
        self,
        current_centroids: np.ndarray,
        previous_centroids: np.ndarray,
    ) -> Matching:
        cur = np.asarray(current_centroids, dtype=np.float32)
        prev = np.asarray(previous_centroids, dtype=np.float32)
        n, m = cur.shape[0], prev.shape[0]
        if n == 0 or m == 0:
            return Matching(
                matched=(),
                unmatched_current=tuple(range(n)),
                unmatched_previous=tuple(range(m)),
            )
        cost = np.linalg.norm(cur[:, None, :] - prev[None, :, :], axis=-1)
        size = max(n, m)
        padded = np.full((size, size), 1e6, dtype=np.float32)
        padded[:n, :m] = cost
        assign = _hungarian_min(padded)
        cur_used = [False] * n
        prev_used = [False] * m
        matched = []
        for i in range(1, size + 1):
            j = assign[i]
            if i <= n and j <= m and cost[i - 1, j - 1] < self.distance_threshold:
                matched.append((i - 1, j - 1))
                cur_used[i - 1] = True
                prev_used[j - 1] = True
        return Matching(
            matched=tuple(matched),
            unmatched_current=tuple(i for i in range(n) if not cur_used[i]),
            unmatched_previous=tuple(j for j in range(m) if not prev_used[j]),
        )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_matching.py -v`
Expected: 7 passed

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/core/matching.py tests/test_matching.py
git commit -m "feat: 帧间匹配协议 + 匈牙利实现（可插拔）"
```

---

### Task 3: 测试夹具扩展 + `HandSequence` 数据结构 + 缓冲窗口（TDD）

**Files:**
- Modify: `tests/conftest.py`（增加 make_hand_frame / hand_pts 工厂 fixture）
- Create: `src/signbridge/hands/sequence.py`
- Test: `tests/test_sequence_buffer.py`

- [ ] **Step 1: 扩展 `tests/conftest.py`（追加内容）**

```python
import numpy as np

from signbridge.core.landmarks import Hand, HandFrame, Landmark


def _make_hand_frame(hands, ts=0):
    """构造测试用 HandFrame。

    hands: [(handedness, pts(21,3) numpy 或 None), ...]
    pts 为图像归一化坐标（x,y∈[0,1]）。
    """
    out = []
    for handedness, pts in hands:
        lms = tuple(Landmark(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts)
        out.append(Hand(landmarks=lms, handedness=handedness, score=0.9))
    return HandFrame(hands=tuple(out), timestamp_ms=ts, frame_index=0)


@pytest.fixture
def make_hand_frame():
    return _make_hand_frame


@pytest.fixture
def hand_pts():
    def _factory(center=(0.5, 0.5), seed=0):
        rng = np.random.default_rng(seed)
        pts = rng.uniform(-0.1, 0.1, size=(21, 3)).astype(np.float32)
        pts[:, 0] += center[0]
        pts[:, 1] += center[1]
        pts[0] = (center[0], center[1], 0.0)  # WRIST 在中心
        return pts

    return _factory
```

（`conftest.py` 顶部需要 `import pytest` 与 `import numpy as np`——已有 `import pytest`，补 numpy import。）

- [ ] **Step 2: 写失败测试 `tests/test_sequence_buffer.py`**

```python
import numpy as np
import pytest

from signbridge.core.landmarks import HandFrame
from signbridge.hands.sequence import HandSequence, HandSequenceBuffer


def test_window_slides(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=60, coordinate="image", smoother=None)
    pts = hand_pts()
    for _ in range(70):
        buf.update(make_hand_frame([("Left", pts)]))
    seq = buf.update(make_hand_frame([("Left", pts)]))[0]
    assert seq.data.shape == (60, 21, 3)
    assert seq.valid_mask.all()
    assert seq.frame_indices[0] == 11   # 前 11 帧滑出
    assert seq.frame_indices[-1] == 70


def test_lost_frame_occupies_slot_with_nan(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=None,
                             max_lost_frames=5)
    pts = hand_pts()
    for _ in range(5):
        buf.update(make_hand_frame([("Left", pts)]))
    buf.update(HandFrame())                          # 第 6 帧：无手
    seq = buf.update(make_hand_frame([("Left", pts)]))[0]  # 第 7 帧：手回来
    assert seq.data.shape == (7, 21, 3)
    assert seq.valid_mask[5] == False
    assert np.isnan(seq.data[5]).all()
    assert seq.valid_mask[6] == True
    assert seq.frame_indices[-1] == 6


def test_wrist_normalized_to_origin(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=None)
    pts = hand_pts()
    for _ in range(5):
        buf.update(make_hand_frame([("Left", pts)]))
    seq = buf.update(make_hand_frame([("Left", pts)]))[0]
    assert np.allclose(seq.data[:, 0, :], 0.0, atol=1e-6)


def test_two_hands_independent_sequences(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=None)
    left = hand_pts(center=(0.2, 0.5), seed=0)
    right = hand_pts(center=(0.8, 0.5), seed=1)
    for _ in range(5):
        buf.update(make_hand_frame([("Left", left), ("Right", right)]))
    seqs = buf.update(make_hand_frame([("Left", left), ("Right", right)]))
    assert len(seqs) == 2
    assert [s.hand_id for s in seqs] == sorted(s.hand_id for s in seqs)
    lseq = next(s for s in seqs if s.handedness == "Left")
    rseq = next(s for s in seqs if s.handedness == "Right")
    assert not np.array_equal(lseq.data, rseq.data)


def test_smoother_called_per_valid_frame(make_hand_frame, hand_pts):
    class _Recorder:
        """记录调用的假平滑器。Buffer 会 deepcopy 实例，__deepcopy__ 返回自身以共享计数。"""

        def __init__(self):
            self.calls = []

        def update(self, pts):
            self.calls.append(None if pts is None else pts.copy())
            return pts

        def reset(self):
            self.calls.clear()

        def __deepcopy__(self, memo):
            return self

    smoother = _Recorder()
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=smoother)
    pts = hand_pts()
    for _ in range(5):
        buf.update(make_hand_frame([("Left", pts)]))
    buf.update(HandFrame())  # 无手帧 → smoother 收到 None
    assert len(smoother.calls) == 6
    assert smoother.calls[-1] is None


def test_reset_clears_state(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=None)
    pts = hand_pts()
    for _ in range(5):
        buf.update(make_hand_frame([("Left", pts)]))
    buf.reset()
    seq = buf.update(make_hand_frame([("Left", pts)]))[0]
    assert seq.frame_indices[0] == 0
    assert len(seq.data) == 1


def test_empty_frames_yield_no_sequences(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=8, coordinate="image", smoother=None)
    assert buf.update(HandFrame()) == ()
    assert buf.update(HandFrame()) == ()
```

- [ ] **Step 3: 跑测试确认失败**

Run: `python -m pytest tests/test_sequence_buffer.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.hands.sequence'`）

- [ ] **Step 4: 实现 `src/signbridge/hands/sequence.py`（完整实现，含追踪逻辑）**

```python
"""时序序列缓冲：帧间追踪 + ID 生命周期 + 滑动窗口 + 腕点归一化。

消费第一步的 HandFrame，输出按手 ID 稳定分离的 HandSequence
（ST-GCN 输入：data 为 (T, 21, 3) 腕点归一化坐标）。
匹配与平滑均为可插拔协议（core.matching / core.smoothing）。
"""

import copy
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from signbridge.core.landmarks import HandFrame
from signbridge.core.matching import HungarianMatcher, Matcher
from signbridge.core.smoothing import LandmarkSmoother


@dataclass(frozen=True)
class HandSequence:
    """一只手的时序序列（滑动窗口快照）。

    data: (T, 21, 3) float32，腕点(WRIST)归一化；丢失帧为 NaN 行。
    valid_mask: (T,) bool；timestamps/frame_indices 与 data 逐行对应。
    """

    hand_id: int
    handedness: str
    data: np.ndarray = field(repr=False)
    valid_mask: np.ndarray = field(repr=False)
    timestamps: np.ndarray = field(repr=False)
    frame_indices: np.ndarray = field(repr=False)


class _Track:
    __slots__ = ("hand_id", "handedness", "centroid", "lost_count",
                 "smoother", "slots", "timestamps", "frame_indices")

    def __init__(self, hand_id, handedness, centroid, smoother):
        self.hand_id = hand_id
        self.handedness = handedness
        self.centroid = centroid
        self.lost_count = 0
        self.smoother = smoother
        self.slots: deque = deque()
        self.timestamps: deque = deque()
        self.frame_indices: deque = deque()


class HandSequenceBuffer:
    """手部时序缓冲：每帧调用 update(hand_frame)，返回当前活动手的序列。

    参数：
        window_size: 滑动窗口帧数（槽位按帧推进，含丢失占位）
        max_hands: 手数上限（当前仅用于校验提示）
        max_lost_frames: ID 失联保留帧数，超过则回收
        matcher: 可插拔帧间匹配器（默认 HungarianMatcher）
        coordinate: "world"（米制 world_landmarks，默认）| "image"（归一化坐标）
        smoother: 可插拔平滑器实例（内部按手 deepcopy）；None 不平滑
    """

    def __init__(
        self,
        window_size: int = 60,
        max_hands: int = 2,
        max_lost_frames: int = 10,
        matcher: Matcher | None = None,
        coordinate: str = "world",
        smoother: LandmarkSmoother | None = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size 必须 > 0")
        if max_lost_frames < 0:
            raise ValueError("max_lost_frames 必须 >= 0")
        if coordinate not in ("world", "image"):
            raise ValueError("coordinate 必须是 'world' 或 'image'")
        self.window_size = window_size
        self.max_hands = max_hands
        self.max_lost_frames = max_lost_frames
        self.coordinate = coordinate
        self._matcher = matcher if matcher is not None else HungarianMatcher()
        self._smoother_factory = (
            (lambda: copy.deepcopy(smoother)) if smoother is not None else None
        )
        self._tracks: dict[int, _Track] = {}
        self._next_id = 0
        self._frame_index = 0

    @property
    def left_hand_id(self) -> int:
        """当前左手 ID；无左手时为 -1。"""
        return self._hand_id_for("Left")

    @property
    def right_hand_id(self) -> int:
        """当前右手 ID；无右手时为 -1。"""
        return self._hand_id_for("Right")

    def _hand_id_for(self, handedness: str) -> int:
        for t in self._tracks.values():
            if t.handedness == handedness:
                return t.hand_id
        return -1

    def update(self, hand_frame: HandFrame) -> tuple[HandSequence, ...]:
        """喂入一帧 HandFrame，返回当前所有活动手的 HandSequence（按 hand_id 升序）。"""
        cur = self._extract(hand_frame)
        cur_centroids = (
            np.array([c for _, c, _ in cur], dtype=np.float32) if cur
            else np.zeros((0, 2), dtype=np.float32)
        )
        prev_centroids = (
            np.array([t.centroid for t in self._tracks.values()], dtype=np.float32)
            if self._tracks else np.zeros((0, 2), dtype=np.float32)
        )
        matching = self._matcher.match(cur_centroids, prev_centroids)
        track_list = list(self._tracks.values())
        ts = hand_frame.timestamp_ms

        for ci, pi in matching.matched:                      # 匹配对 → 续用 ID
            track = track_list[pi]
            handedness, centroid, pts = cur[ci]
            track.lost_count = 0
            track.handedness = handedness
            track.centroid = centroid
            self._append_valid(track, pts, ts)

        for ci in matching.unmatched_current:                # 新手 → 新 ID
            handedness, centroid, pts = cur[ci]
            track = _Track(
                self._next_id, handedness, centroid,
                self._smoother_factory() if self._smoother_factory else None,
            )
            self._next_id += 1
            self._tracks[track.hand_id] = track
            self._append_valid(track, pts, ts)

        for pi in matching.unmatched_previous:               # 失联 → lost 计数 + 占位
            track = track_list[pi]
            track.lost_count += 1
            self._append_invalid(track, ts)

        for hand_id in [
            t.hand_id for t in self._tracks.values()
            if t.lost_count > self.max_lost_frames
        ]:
            del self._tracks[hand_id]

        self._frame_index += 1
        return tuple(
            self._to_sequence(t)
            for t in sorted(self._tracks.values(), key=lambda t: t.hand_id)
        )

    def reset(self) -> None:
        """清空所有轨迹与窗口，帧计数归零。"""
        self._tracks.clear()
        self._next_id = 0
        self._frame_index = 0

    # ---- 内部 ----

    def _extract(self, hand_frame: HandFrame):
        """提取当前帧每只手：[(handedness, 质心(x,y), pts(21,3))]。"""
        out = []
        for hand in hand_frame.hands:
            lms = hand.world_landmarks if self.coordinate == "world" else hand.landmarks
            if len(lms) < 21:
                continue
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            out.append((hand.handedness, pts[:, :2].mean(axis=0), pts))
        return out

    def _append_valid(self, track: _Track, pts: np.ndarray, ts: int) -> None:
        pts = pts - pts[0]  # 腕点归一化（WRIST 为原点）
        if track.smoother is not None:
            pts = track.smoother.update(pts)
        track.slots.append(pts.copy())
        track.timestamps.append(ts)
        track.frame_indices.append(self._frame_index)
        self._trim(track)

    def _append_invalid(self, track: _Track, ts: int) -> None:
        track.slots.append(None)
        track.timestamps.append(ts)
        track.frame_indices.append(self._frame_index)
        if track.smoother is not None:
            track.smoother.update(None)
        self._trim(track)

    def _trim(self, track: _Track) -> None:
        while len(track.slots) > self.window_size:
            track.slots.popleft()
            track.timestamps.popleft()
            track.frame_indices.popleft()

    def _to_sequence(self, track: _Track) -> HandSequence:
        t = len(track.slots)
        data = np.full((t, 21, 3), np.nan, dtype=np.float32)
        valid = np.zeros(t, dtype=bool)
        for i, slot in enumerate(track.slots):
            if slot is not None:
                data[i] = slot
                valid[i] = True
        return HandSequence(
            hand_id=track.hand_id,
            handedness=track.handedness,
            data=data,
            valid_mask=valid,
            timestamps=np.array(track.timestamps, dtype=np.int64),
            frame_indices=np.array(track.frame_indices, dtype=np.int64),
        )
```

- [ ] **Step 5: 跑测试确认通过**

Run: `python -m pytest tests/test_sequence_buffer.py -v`
Expected: 7 passed

- [ ] **Step 6: 提交**

```bash
git add src/signbridge/hands/sequence.py tests/conftest.py tests/test_sequence_buffer.py
git commit -m "feat: HandSequence 数据结构与滑动窗口缓冲（槽位占位、腕点归一化）"
```

---

### Task 4: ID 生命周期与可插拔验证 `tests/test_tracker.py`（TDD）

**Files:**
- Create: `tests/test_tracker.py`

- [ ] **Step 1: 写失败测试 `tests/test_tracker.py`**

```python
import numpy as np
import pytest

from signbridge.core.landmarks import HandFrame
from signbridge.core.matching import Matching
from signbridge.hands.sequence import HandSequenceBuffer


def test_single_hand_stable_id(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None)
    pts = hand_pts()
    ids = set()
    for _ in range(10):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            ids.add(s.hand_id)
    assert len(ids) == 1


def test_cross_swap_keeps_identity(make_hand_frame, hand_pts):
    """双手互换位置：ID 数稳定、序列完整、无 NaN 丢失帧。"""
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None)
    a_left = hand_pts(center=(0.2, 0.5), seed=0)   # 手 X（形 A）初始在左
    b_right = hand_pts(center=(0.8, 0.5), seed=1)  # 手 Y（形 B）初始在右
    a_right = hand_pts(center=(0.8, 0.5), seed=0)  # X 移到右
    b_left = hand_pts(center=(0.2, 0.5), seed=1)   # Y 移到左
    frames = []
    for _ in range(5):
        frames.append(make_hand_frame([("Left", a_left), ("Right", b_right)]))
    for _ in range(5):  # 位置互换（标记随位置走，模拟镜像/误判）
        frames.append(make_hand_frame([("Left", b_left), ("Right", a_right)]))
    ids = set()
    for hf in frames:
        for s in buf.update(hf):
            ids.add(s.hand_id)
    assert len(ids) == 2  # 交换不产生新 ID、不丢 ID
    seqs = buf.update(frames[-1])
    assert len(seqs) == 2
    for s in seqs:
        assert s.data.shape == (10, 21, 3)
        assert s.valid_mask.all()       # 全程无丢失帧
        assert not np.isnan(s.data).any()


def test_lost_within_k_frames_keeps_id(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None,
                             max_lost_frames=10)
    pts = hand_pts()
    ids = set()
    for _ in range(5):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            ids.add(s.hand_id)
    for _ in range(5):  # 消失 5 帧（<= 10）
        buf.update(HandFrame())
    for _ in range(5):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            ids.add(s.hand_id)
    assert len(ids) == 1


def test_lost_beyond_k_frames_recycles_id(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=60, coordinate="image", smoother=None,
                             max_lost_frames=10)
    pts = hand_pts()
    first = set()
    for _ in range(5):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            first.add(s.hand_id)
    for _ in range(15):  # 消失 15 帧（> 10）
        buf.update(HandFrame())
    second = set()
    for _ in range(5):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            second.add(s.hand_id)
    assert first.isdisjoint(second)  # 回收后不复用


def test_new_hand_gets_new_id(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None)
    pts = hand_pts()
    for _ in range(3):
        buf.update(make_hand_frame([("Left", pts)]))
    seqs = buf.update(make_hand_frame(
        [("Left", pts), ("Right", hand_pts(seed=1))]))
    assert len(seqs) == 2
    assert len({s.hand_id for s in seqs}) == 2


def test_left_right_hand_id_properties(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None)
    pts = hand_pts()
    buf.update(make_hand_frame([("Left", pts)]))
    assert buf.left_hand_id >= 0 and buf.right_hand_id == -1
    buf.update(make_hand_frame([("Left", pts), ("Right", hand_pts(seed=1))]))
    assert buf.left_hand_id >= 0 and buf.right_hand_id >= 0
    assert buf.left_hand_id != buf.right_hand_id


class _FixedMatcher:
    """固定返回指定匹配结果的假匹配器（可插拔验证）。"""

    def __init__(self, matching: Matching):
        self._m = matching
        self.calls = 0

    def match(self, current_centroids, previous_centroids):
        self.calls += 1
        return self._m


def test_custom_matcher_full_match_reuses_id(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(
        window_size=10, coordinate="image", smoother=None,
        matcher=_FixedMatcher(Matching(matched=((0, 0),),
                                       unmatched_current=(), unmatched_previous=())),
    )
    pts = hand_pts()
    ids = set()
    for _ in range(4):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            ids.add(s.hand_id)
    assert len(ids) == 1  # 匹配对驱动 ID 续用


def test_custom_matcher_full_unmatched_creates_new_ids(make_hand_frame, hand_pts):
    buf = HandSequenceBuffer(
        window_size=10, coordinate="image", smoother=None,
        matcher=_FixedMatcher(Matching(matched=(),
                                       unmatched_current=(0,), unmatched_previous=(0,))),
    )
    pts = hand_pts()
    ids = set()
    for _ in range(4):
        for s in buf.update(make_hand_frame([("Left", pts)])):
            ids.add(s.hand_id)
    assert len(ids) == 4  # 每次都不匹配 → 每帧新 ID
```

- [ ] **Step 2: 跑测试确认通过**

Run: `python -m pytest tests/test_tracker.py -v`
Expected: 8 passed（Task 3 的实现已包含追踪逻辑；若失败按失败原因修复实现）

- [ ] **Step 3: 全量回归**

Run: `python -m pytest -v`
Expected: 全部通过（原 45 + smoothing 6 + matching 7 + sequence 7 + tracker 8 = 73）

- [ ] **Step 4: 提交**

```bash
git add tests/test_tracker.py
git commit -m "test: ID 生命周期与可插拔匹配验证"
```

---

### Task 5: 导出 + README + 全量验证

**Files:**
- Modify: `src/signbridge/core/__init__.py`
- Modify: `src/signbridge/hands/__init__.py`
- Modify: `src/signbridge/__init__.py`
- Modify: `README.md`

- [ ] **Step 1: 更新 `src/signbridge/core/__init__.py`**

```python
"""signbridge.core: 各组件共享的基础设施（数据结构、图谱常量、异常体系）。"""

from signbridge.core.landmarks import (
    HAND_CONNECTIONS,
    HAND_LANDMARK_NAMES,
    Hand,
    HandFrame,
    Landmark,
)
from signbridge.core.matching import HungarianMatcher, Matcher, Matching
from signbridge.core.smoothing import LandmarkSmoother, OneEuroSmoother

__all__ = [
    "HAND_CONNECTIONS",
    "HAND_LANDMARK_NAMES",
    "Hand",
    "HandFrame",
    "Landmark",
    "HungarianMatcher",
    "Matcher",
    "Matching",
    "LandmarkSmoother",
    "OneEuroSmoother",
]
```

- [ ] **Step 2: 更新 `src/signbridge/hands/__init__.py`**

```python
"""signbridge.hands: 手部关键点提取与时序缓冲组件。"""

from signbridge.hands.detector import HandDetector
from signbridge.hands.draw import draw_landmarks, draw_landmarks_depth
from signbridge.hands.sequence import HandSequence, HandSequenceBuffer
from signbridge.hands.sources import CameraSource, ImageSource, VideoSource

__all__ = [
    "HandDetector",
    "draw_landmarks",
    "draw_landmarks_depth",
    "HandSequence",
    "HandSequenceBuffer",
    "CameraSource",
    "ImageSource",
    "VideoSource",
]
```

- [ ] **Step 3: 更新 `src/signbridge/__init__.py`（在 hands 导出区追加）**

```python
from signbridge.hands.sequence import HandSequence, HandSequenceBuffer
```

并在 `__all__` 中追加 `"HandSequence"`、`"HandSequenceBuffer"`。

- [ ] **Step 4: README 增加第二步章节（在「快速上手」之后插入）**

```markdown
## 时序序列缓冲（第二步）

把连续帧的手部关键点缓冲成按手 ID 稳定分离的时序序列，直接作为 ST-GCN 输入：

```python
import cv2
from signbridge import HandDetector, ImageSource, HandSequenceBuffer, OneEuroSmoother, HungarianMatcher

buf = HandSequenceBuffer(
    window_size=60,                       # 滑动窗口（帧）
    max_lost_frames=10,                   # 手失联保留帧数，超时回收 ID
    matcher=HungarianMatcher(distance_threshold=0.15),  # 可插拔：帧间匹配
    coordinate="world",                   # 米制 3D（ST-GCN 首选）
    smoother=OneEuroSmoother(),           # 可插拔：关键点平滑（None 关闭）
)

detector = HandDetector(max_num_hands=2)
src = ImageSource("hand.jpg", repeat=True)   # 单帧循环演示
for frame, _, _ in src:
    sequences = buf.update(detector.detect(frame))   # -> tuple[HandSequence, ...]
    for seq in sequences:
        print(seq.hand_id, seq.handedness)           # 0 "Left"
        print(seq.data.shape)                        # (60, 21, 3) 腕点归一化
        print(seq.valid_mask)                        # 丢失帧为 False（data 行 NaN）
    if buf.left_hand_id >= 0 and buf.right_hand_id >= 0:
        break
src.close()
detector.close()
```

`HandSequence.data` 为 (T, 21, 3) float32：每帧减去腕点坐标（腕点即原点），
`valid_mask` 标记真实数据（丢失帧 NaN 占位），可直接拼接为 (T, 2, 21, 3) 双通道
张量或按手独立喂给 ST-GCN。匹配与平滑均为可插拔协议：实现 `Matcher` /
`LandmarkSmoother` 协议即可替换（如光流匹配、卡尔曼平滑）。
```

- [ ] **Step 5: 全量测试 + 公共 API 验证**

Run: `python -m pytest -v`
Expected: 73 passed

Run: `python -c "from signbridge import HandSequenceBuffer, OneEuroSmoother, HungarianMatcher; print('ok')"`
Expected: `ok`

- [ ] **Step 6: 提交**

```bash
git add src/signbridge/ README.md
git commit -m "docs: README 第二步时序缓冲文档与公共 API 导出"
```

---

### Task 6: 端到端冒烟（真实模型 + 真实图片）

**Files:**
- 无（验证）

- [ ] **Step 1: 用 test.png 跑端到端（图片源 repeat + 序列缓冲）**

Run:

```bash
python -c "
import cv2
from signbridge import HandDetector, ImageSource, HandSequenceBuffer, OneEuroSmoother

src = ImageSource(r'C:\Users\Inspiration\Desktop\test.png', repeat=True)
buf = HandSequenceBuffer(window_size=30, coordinate='world', smoother=OneEuroSmoother())
with HandDetector(max_num_hands=2) as detector:
    for i, (frame, _, _) in enumerate(src):
        seqs = buf.update(detector.detect(frame))
        if i == 5:
            for s in seqs:
                print(f'id={s.hand_id} {s.handedness} data={s.data.shape} valid={s.valid_mask.sum()}/{len(s.valid_mask)}')
            break
src.close()
"
```

Expected: 打印两个序列（Left/Right 各一），data 形状 (6, 21, 3) 或接近，valid 全 True

- [ ] **Step 2: 工作区与历史检查**

Run: `git status --short && git log --oneline -8`
Expected: 工作区干净；日志含本计划各任务提交

- [ ] **Step 3: 完成声明**

实现完成。向用户报告：交付物清单、测试结果、端到端输出示例、下一步（ST-GCN 训练或手势分段）。
