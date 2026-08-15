# SignBridge 特征增强帧间匹配实现计划（0.3.0）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现特征增强的分层匹配：`HandShapeFeature` 距离矩阵特征 + `FeatureVerifier` 同一性判定协议（未来 transformer/GCN 接入点）+ `FeatureHungarianMatcher` 分层匹配器 + Buffer 集成（跨位置丢失恢复）。

**Architecture:** 新增 `core/features.py`（双协议 + 两个默认实现）；`core/matching.py` 协议升级为 `HandDescriptor`（centroid + feature），位置匈牙利逻辑抽成共享函数，新增 `FeatureHungarianMatcher`（第一层位置匈牙利、第二层特征恢复贪心匹配）；`HandSequenceBuffer` 新增 `feature_extractor` 参数并维护 `_Track.last_feature`。纯位置 `HungarianMatcher` 保留。

**Tech Stack:** Python 3.14、numpy 2.5、pytest。全部纯 numpy 测试。

**关键环境事实：**
- pytest 用 `python -m pytest`；当前 73 测试全绿；版本 0.2.0（本计划 bump 0.3.0）
- 协议 v2 是破坏性变更（`Matcher.match` 签名从质心数组升级为 `HandDescriptor` 序列）——现有测试同步更新
- 规格：`docs/superpowers/specs/2026-08-14-feature-matching-design.md`

**执行约定：** 每步跑完测试再提交；全部测试通过后才 commit。

---

### Task 1: 特征提取与同一性判定 `core/features.py`（TDD）

**Files:**
- Create: `src/signbridge/core/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: 写失败测试 `tests/test_features.py`**

```python
import numpy as np
import pytest

from signbridge.core.features import (
    DistanceFeatureVerifier,
    FeatureExtractor,
    FeatureVerifier,
    HandShapeFeature,
)


def _pts(seed=0, center=(0.5, 0.5)):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.1, 0.1, size=(21, 3)).astype(np.float32)
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    pts[0] = (center[0], center[1], 0.0)
    return pts


def test_protocols_expose_methods():
    assert hasattr(FeatureExtractor, "extract")
    assert hasattr(FeatureVerifier, "verify")


def test_output_is_210_dim():
    f = HandShapeFeature()
    vec = f.extract(_pts())
    assert vec.shape == (210,)
    assert vec.dtype == np.float32


def test_translation_invariant():
    f = HandShapeFeature()
    a = f.extract(_pts(center=(0.2, 0.5)))
    b = f.extract(_pts(center=(0.8, 0.5)))
    assert np.allclose(a, b, atol=1e-5)


def test_rotation_invariant():
    f = HandShapeFeature()
    pts = _pts()
    theta = np.pi / 4
    rot_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]], dtype=np.float32)
    rotated = pts @ rot_z.T
    a = f.extract(pts)
    b = f.extract(rotated)
    assert np.allclose(a, b, atol=1e-4)


def test_scale_invariant():
    f = HandShapeFeature()
    pts = _pts()
    a = f.extract(pts)
    b = f.extract(pts * 0.5)  # 整体缩放（腕点保持原点附近）
    assert np.allclose(a, b, atol=1e-4)


def test_different_shapes_are_far_apart():
    f = HandShapeFeature()
    a = f.extract(_pts(seed=0))
    b = f.extract(_pts(seed=1))
    d_ab = float(np.linalg.norm(a - b))
    d_aa = float(np.linalg.norm(a - f.extract(_pts(seed=0))))
    assert d_ab > 5 * d_aa


def test_verifier_same_feature_is_1():
    v = DistanceFeatureVerifier()
    fvec = HandShapeFeature().extract(_pts())
    assert v.verify(fvec, fvec) == pytest.approx(1.0)


def test_verifier_monotonic():
    v = DistanceFeatureVerifier(sigma=0.3)
    fvec = HandShapeFeature().extract(_pts())
    close = v.verify(fvec, fvec + 1e-3)
    far = v.verify(fvec, fvec + 0.5)
    assert close > far
    assert 0.0 <= close <= 1.0 and 0.0 <= far <= 1.0


def test_verifier_invalid_sigma_raises():
    with pytest.raises(ValueError):
        DistanceFeatureVerifier(sigma=0.0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_features.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.core.features'`）

- [ ] **Step 3: 实现 `src/signbridge/core/features.py`**

```python
"""手部特征提取与同一性判定（均为可插拔协议）。

FeatureExtractor：21×3 点阵 → 特征向量。
FeatureVerifier：两特征 → 置信度 [0,1]（未来 transformer / GCN 模型实现此协议接入）。
"""

from typing import Protocol

import numpy as np


class FeatureExtractor(Protocol):
    """特征提取协议：21×3 点阵 → 特征向量。"""

    def extract(self, pts: np.ndarray) -> np.ndarray: ...


class FeatureVerifier(Protocol):
    """同一性判定协议：两特征向量 → 置信度 ∈ [0,1]（1=同一只手）。

    ★ 学习型模型的接入点：transformer 双塔 / GCN 相似度网络
      训练后实现此协议即可无缝替换默认判定。
    """

    def verify(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float: ...


class HandShapeFeature:
    """归一化距离矩阵特征（210 维 = 21×21 距离矩阵上三角）。

    腕点归一化 → 点间欧氏距离 → 上三角向量 → 除以平均距离。
    旋转不变、尺度不变、平移无关——手在画面任意位置/角度出现都能比对手形。
    """

    def extract(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32)
        if pts.shape != (21, 3):
            raise ValueError(f"输入必须是 (21,3) 点阵，收到 {pts.shape}")
        centered = pts - pts[0]                      # 腕点归一化（平移无关）
        diff = centered[:, None, :] - centered[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)         # (21,21) 欧氏距离矩阵
        vec = dist[np.triu_indices(21, k=1)].copy()  # 210 维上三角
        scale = float(vec.mean())
        if scale > 1e-9:
            vec = vec / scale                        # 尺度归一化
        return vec


class DistanceFeatureVerifier:
    """L2 距离 → 置信度：exp(-d² / 2σ²)（高斯核）。

    d 越小置信度越高；σ 控制衰减速度（σ 越大判定越宽松）。
    """

    def __init__(self, sigma: float = 0.3) -> None:
        if sigma <= 0:
            raise ValueError("sigma 必须 > 0")
        self.sigma = sigma

    def verify(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float:
        a = np.asarray(feature_a, dtype=np.float32)
        b = np.asarray(feature_b, dtype=np.float32)
        d = float(np.linalg.norm(a - b))
        return float(np.exp(-(d * d) / (2.0 * self.sigma * self.sigma)))
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_features.py -v`
Expected: 9 passed

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/core/features.py tests/test_features.py
git commit -m "feat: 特征提取与同一性判定协议（HandShapeFeature + DistanceFeatureVerifier）"
```

---

### Task 2: 匹配协议 v2 + 分层匹配器 `core/matching.py`（TDD）

**Files:**
- Modify: `src/signbridge/core/matching.py`（重构：HandDescriptor、协议 v2、FeatureHungarianMatcher）
- Test: `tests/test_matching.py`（升级 descriptor + 新增分层测试）

- [ ] **Step 1: 更新测试 `tests/test_matching.py`（descriptor 化 + 新增）**

```python
import numpy as np
import pytest

from signbridge.core.features import DistanceFeatureVerifier, HandShapeFeature
from signbridge.core.matching import (
    FeatureHungarianMatcher,
    HandDescriptor,
    HungarianMatcher,
    Matcher,
    Matching,
)


def _desc(*xy, feature=None):
    """构造 HandDescriptor 列表：_desc((x0,y0),(x1,y1), feature=...)"""
    out = []
    for i, p in enumerate(xy):
        feat = None if feature is None else feature[i]
        out.append(HandDescriptor(centroid=np.array(p, dtype=np.float32), feature=feat))
    return out


def _features(seed_a=0, seed_b=1):
    f = HandShapeFeature()
    a = f.extract(_pts21(seed_a))
    b = f.extract(_pts21(seed_b))
    return [a, b]


def _pts21(seed):
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.1, 0.1, size=(21, 3)).astype(np.float32)
    pts[0] = (0.5, 0.5, 0.0)
    return pts


def test_protocol_exposes_match():
    assert hasattr(Matcher, "match")


def test_basic_assignment():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(_desc((0.0, 0.0), (1.0, 1.0)), _desc((0.1, 0.1), (0.9, 0.9)))
    assert set(res.matched) == {(0, 0), (1, 1)}
    assert res.unmatched_current == ()
    assert res.unmatched_previous == ()


def test_cross_swap_matches_by_nearest():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(_desc((0.9, 0.9), (0.1, 0.1)), _desc((0.1, 0.1), (0.9, 0.9)))
    assert set(res.matched) == {(0, 1), (1, 0)}


def test_distance_beyond_threshold_unmatched():
    m = HungarianMatcher(distance_threshold=0.3)
    res = m.match(_desc((0.0, 0.0)), _desc((0.9, 0.9)))
    assert res.matched == ()
    assert res.unmatched_current == (0,)
    assert res.unmatched_previous == (0,)


def test_asymmetric_counts():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match(
        _desc((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)),
        _desc((0.0, 0.0), (1.0, 1.0)),
    )
    assert len(res.matched) == 2
    assert len(res.unmatched_current) == 1
    assert res.unmatched_previous == ()


def test_empty_side():
    m = HungarianMatcher(distance_threshold=0.5)
    res = m.match([], _desc((0.0, 0.0)))
    assert res.matched == ()
    assert res.unmatched_current == ()
    assert res.unmatched_previous == (0,)


def test_invalid_threshold_raises():
    with pytest.raises(ValueError):
        HungarianMatcher(distance_threshold=-0.1)


# ---- FeatureHungarianMatcher ----

def test_feature_matcher_position_near_behaves_like_position_only():
    m = FeatureHungarianMatcher(distance_threshold=0.5)
    res = m.match(
        _desc((0.0, 0.0), (1.0, 1.0), feature=_features()),
        _desc((0.1, 0.1), (0.9, 0.9), feature=_features()),
    )
    assert set(res.matched) == {(0, 0), (1, 1)}


def test_feature_recovery_far_but_same_shape():
    m = FeatureHungarianMatcher(confidence_threshold=0.85)
    fa = HandShapeFeature().extract(_pts21(0))
    cur = _desc((0.8, 0.5), feature=[fa])      # 画面另一侧
    prev = _desc((0.2, 0.5), feature=[fa])     # 同手形
    res = m.match(cur, prev)
    assert set(res.matched) == {(0, 0)}        # 特征恢复匹配
    assert res.unmatched_current == ()
    assert res.unmatched_previous == ()


def test_feature_recovery_rejects_different_shape():
    m = FeatureHungarianMatcher(confidence_threshold=0.85)
    fa = HandShapeFeature().extract(_pts21(0))
    fb = HandShapeFeature().extract(_pts21(1))
    cur = _desc((0.8, 0.5), feature=[fb])      # 异手形
    prev = _desc((0.2, 0.5), feature=[fa])
    res = m.match(cur, prev)
    assert res.matched == ()
    assert res.unmatched_current == (0,)
    assert res.unmatched_previous == (0,)


def test_feature_recovery_requires_confidence_threshold():
    m = FeatureHungarianMatcher(confidence_threshold=0.999)
    fa = HandShapeFeature().extract(_pts21(0))
    cur = _desc((0.8, 0.5), feature=[fa])
    prev = _desc((0.2, 0.5), feature=[fa + 0.1])  # 轻微扰动但低于阈值
    res = m.match(cur, prev)
    assert res.matched == ()


def test_feature_recovery_skipped_when_feature_none():
    m = FeatureHungarianMatcher(confidence_threshold=0.85)
    cur = _desc((0.8, 0.5))   # feature=None
    prev = _desc((0.2, 0.5))
    res = m.match(cur, prev)
    assert res.matched == ()


def test_feature_recovery_greedy_no_double_match():
    m = FeatureHungarianMatcher(confidence_threshold=0.85)
    fa = HandShapeFeature().extract(_pts21(0))
    cur = _desc((0.8, 0.5), (0.85, 0.5), feature=[fa, fa])   # 两只同形新手
    prev = _desc((0.2, 0.5), feature=[fa])                    # 一条轨迹
    res = m.match(cur, prev)
    assert len(res.matched) == 1                              # 每边最多一次
    assert len(res.unmatched_current) == 1
    assert res.unmatched_previous == ()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_matching.py -v`
Expected: FAIL（`HandDescriptor` / `FeatureHungarianMatcher` 不存在）

- [ ] **Step 3: 重构 `src/signbridge/core/matching.py`**

```python
"""帧间关联：抽象协议 + 匹配实现（可插拔）。

Matcher 只做「谁跟谁」的关联决策，不管理 ID 生命周期（那是 Buffer 的职责）。
v2：输入升级为 HandDescriptor（位置 + 特征）；纯位置 HungarianMatcher 保留。
"""

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from signbridge.core.features import DistanceFeatureVerifier, FeatureVerifier


@dataclass(frozen=True)
class Matching:
    """一次帧间匹配的结果。索引分别指向当前帧手与上一帧轨迹。"""

    matched: tuple[tuple[int, int], ...] = ()
    unmatched_current: tuple[int, ...] = ()
    unmatched_previous: tuple[int, ...] = ()


@dataclass(frozen=True)
class HandDescriptor:
    """参与帧间关联的手单元：位置 + 特征（feature 可为 None，关闭特征）。"""

    centroid: np.ndarray
    feature: np.ndarray | None = None


class Matcher(Protocol):
    """帧间关联协议 v2。current / previous 均为 HandDescriptor 序列。"""

    def match(
        self,
        current: Sequence[HandDescriptor],
        previous: Sequence[HandDescriptor],
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


def _centroids(descriptors: Sequence[HandDescriptor]) -> np.ndarray:
    if not descriptors:
        return np.zeros((0, 2), dtype=np.float32)
    return np.array([d.centroid for d in descriptors], dtype=np.float32)


def _position_match(
    cur: np.ndarray, prev: np.ndarray, distance_threshold: float
) -> Matching:
    """纯位置匈牙利匹配（两匹配器共用）。"""
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
        if i <= n and j <= m and cost[i - 1, j - 1] < distance_threshold:
            matched.append((i - 1, j - 1))
            cur_used[i - 1] = True
            prev_used[j - 1] = True
    return Matching(
        matched=tuple(matched),
        unmatched_current=tuple(i for i in range(n) if not cur_used[i]),
        unmatched_previous=tuple(j for j in range(m) if not prev_used[j]),
    )


class HungarianMatcher:
    """纯位置匹配（向后兼容路径）：只用 centroid，忽略 feature。"""

    def __init__(self, distance_threshold: float = 0.15) -> None:
        if distance_threshold < 0:
            raise ValueError("distance_threshold 必须 >= 0")
        self.distance_threshold = distance_threshold

    def match(
        self,
        current: Sequence[HandDescriptor],
        previous: Sequence[HandDescriptor],
    ) -> Matching:
        return _position_match(
            _centroids(current), _centroids(previous), self.distance_threshold
        )


class FeatureHungarianMatcher:
    """分层匹配（新默认）：位置匈牙利为主 + 特征恢复。

    第一层：位置匈牙利（distance_threshold，正常跟踪路径）。
    第二层：位置未匹配对用特征置信度判定同一性（贪心按置信度降序，
            每边最多匹配一次）——用于跨位置丢失恢复。
    """

    def __init__(
        self,
        feature_verifier: FeatureVerifier | None = None,
        confidence_threshold: float = 0.85,
        distance_threshold: float = 0.15,
    ) -> None:
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold 必须在 [0,1]")
        if distance_threshold < 0:
            raise ValueError("distance_threshold 必须 >= 0")
        self._verifier = (
            feature_verifier if feature_verifier is not None
            else DistanceFeatureVerifier()
        )
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold

    def match(
        self,
        current: Sequence[HandDescriptor],
        previous: Sequence[HandDescriptor],
    ) -> Matching:
        pos = _position_match(
            _centroids(current), _centroids(previous), self.distance_threshold
        )
        matched = list(pos.matched)
        cur_un = list(pos.unmatched_current)
        prev_un = list(pos.unmatched_previous)

        # 第二层：特征恢复（贪心，置信度降序，每边一次）
        if cur_un and prev_un:
            candidates = []
            for i in cur_un:
                fc = current[i].feature
                if fc is None:
                    continue
                for j in prev_un:
                    fp = previous[j].feature
                    if fp is None:
                        continue
                    conf = self._verifier.verify(fc, fp)
                    if conf >= self.confidence_threshold:
                        candidates.append((conf, i, j))
            candidates.sort(key=lambda t: t[0], reverse=True)
            used_cur: set[int] = set()
            used_prev: set[int] = set()
            for _conf, i, j in candidates:
                if i in used_cur or j in used_prev:
                    continue
                matched.append((i, j))
                used_cur.add(i)
                used_prev.add(j)
            cur_un = [i for i in cur_un if i not in used_cur]
            prev_un = [j for j in prev_un if j not in used_prev]

        return Matching(
            matched=tuple(sorted(matched)),
            unmatched_current=tuple(cur_un),
            unmatched_previous=tuple(prev_un),
        )
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_matching.py -v`
Expected: 14 passed（7 旧 + 7 新）

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/core/matching.py tests/test_matching.py
git commit -m "feat: 匹配协议 v2（HandDescriptor）+ 分层匹配 FeatureHungarianMatcher"
```

---

### Task 3: Buffer 集成（feature_extractor + 跨位置恢复）TDD

**Files:**
- Modify: `src/signbridge/hands/sequence.py`
- Test: `tests/test_tracker.py`（新增跨位置恢复测试；_FixedMatcher 适配 v2）

- [ ] **Step 1: 更新 `tests/test_tracker.py`（_FixedMatcher 适配协议 v2 + 新增恢复测试）**

把 `_FixedMatcher.match` 签名改为接收 descriptor 序列（长度判断不变）：

```python
class _FixedMatcher:
    """固定返回指定匹配结果的假匹配器（可插拔验证）。

    首帧（上一帧轨迹为空）时返回全不匹配——这是任何匹配器的合理行为。
    """

    def __init__(self, matching: Matching):
        self._m = matching
        self.calls = 0

    def match(self, current, previous):
        self.calls += 1
        if len(previous) == 0:
            return Matching(
                matched=(),
                unmatched_current=tuple(range(len(current))),
                unmatched_previous=(),
            )
        return self._m
```

追加跨位置恢复测试：

```python
def test_recovery_across_position_same_shape(make_hand_frame, hand_pts):
    """手消失后从画面另一侧以同手形出现 → 特征恢复原 ID。"""
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None,
                             max_lost_frames=10)
    pts_a = hand_pts(center=(0.2, 0.5), seed=0)   # 左手形 A，位置左
    pts_b = hand_pts(center=(0.8, 0.5), seed=0)   # 同手形 A，位置右（跨位置）
    ids = set()
    for _ in range(5):
        for s in buf.update(make_hand_frame([("Left", pts_a)])):
            ids.add(s.hand_id)
    for _ in range(5):  # 消失 5 帧（<= max_lost_frames）
        buf.update(HandFrame())
    for _ in range(5):  # 画面另一侧同手形出现
        for s in buf.update(make_hand_frame([("Left", pts_b)])):
            ids.add(s.hand_id)
    assert len(ids) == 1  # 特征判定为同一只手 → ID 恢复


def test_recovery_across_position_different_shape_gets_new_id(make_hand_frame, hand_pts):
    """消失后另一侧出现异手形 → 特征判定拒绝 → 新 ID。"""
    buf = HandSequenceBuffer(window_size=30, coordinate="image", smoother=None,
                             max_lost_frames=10)
    pts_a = hand_pts(center=(0.2, 0.5), seed=0)
    pts_other = hand_pts(center=(0.8, 0.5), seed=1)  # 异手形
    first = set()
    for _ in range(5):
        for s in buf.update(make_hand_frame([("Left", pts_a)])):
            first.add(s.hand_id)
    for _ in range(5):
        buf.update(HandFrame())
    second = set()
    for _ in range(5):
        for s in buf.update(make_hand_frame([("Left", pts_other)])):
            second.add(s.hand_id)
    assert first.isdisjoint(second)  # 新 ID
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_tracker.py -v`
Expected: `test_recovery_across_position_same_shape` FAIL（旧 matcher 纯位置，跨位置不匹配 → 新 ID）

- [ ] **Step 3: 集成 `src/signbridge/hands/sequence.py`**

修改点：

```python
# import 区追加
from signbridge.core.features import FeatureExtractor, HandShapeFeature
from signbridge.core.matching import (
    FeatureHungarianMatcher, HandDescriptor, HungarianMatcher, Matcher,
)
```

`__init__` 签名与默认值：

```python
    def __init__(
        self,
        window_size: int = 60,
        max_hands: int = 2,
        max_lost_frames: int = 10,
        matcher: Matcher | None = None,
        coordinate: str = "world",
        smoother: LandmarkSmoother | None = None,
        feature_extractor: FeatureExtractor | None = None,
    ) -> None:
        ...
        self._matcher = (
            matcher if matcher is not None
            else FeatureHungarianMatcher()          # 新默认：分层匹配
        )
        self._feature_extractor = (
            feature_extractor if feature_extractor is not None
            else HandShapeFeature()
        )
```

`_extract` 返回四元组（加 feature）：

```python
    def _extract(self, hand_frame: HandFrame):
        """提取当前帧每只手：[(handedness, 质心, pts(21,3), feature|None)]。"""
        out = []
        for hand in hand_frame.hands:
            lms = hand.world_landmarks if self.coordinate == "world" else hand.landmarks
            if len(lms) < 21:
                continue
            pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
            feature = (
                self._feature_extractor.extract(pts)
                if self._feature_extractor is not None else None
            )
            out.append((hand.handedness, pts[:, :2].mean(axis=0), pts, feature))
        return out
```

`update` 中构造 descriptor 并维护 `last_feature`：

```python
        cur = self._extract(hand_frame)
        cur_descriptors = [
            HandDescriptor(centroid=c, feature=feat) for _, c, _, feat in cur
        ]
        prev_descriptors = [
            HandDescriptor(centroid=t.centroid, feature=t.last_feature)
            for t in self._tracks.values()
        ]
        matching = self._matcher.match(cur_descriptors, prev_descriptors)
        track_list = list(self._tracks.values())

        for ci, pi in matching.matched:
            track = track_list[pi]
            handedness, centroid, pts, feature = cur[ci]
            track.lost_count = 0
            track.handedness = handedness
            track.centroid = centroid
            track.last_feature = feature
            self._append_valid(track, pts, ts)

        for ci in matching.unmatched_current:
            handedness, centroid, pts, feature = cur[ci]
            track = _Track(
                self._next_id, handedness, centroid,
                self._smoother_factory() if self._smoother_factory else None,
            )
            track.last_feature = feature
            self._next_id += 1
            self._tracks[track.hand_id] = track
            self._append_valid(track, pts, ts)
```

`_Track.__slots__` 增加 `last_feature`，`__init__` 初始化 `self.last_feature = None`。

`reset()` 不变。`test_sequence_buffer.py` 无需改动（位置路径行为不变）。

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_tracker.py tests/test_sequence_buffer.py -v`
Expected: 全部通过（tracker 10 = 8 旧 + 2 新）

- [ ] **Step 5: 全量回归**

Run: `python -m pytest -v`
Expected: 全部通过（73 + features 9 + matching 新 7 + tracker 新 2 = 91，matching 旧 8 保留 = 73 - 8 + 8 + 9 + 7 + 2 = 91）

- [ ] **Step 6: 提交**

```bash
git add src/signbridge/hands/sequence.py tests/test_tracker.py
git commit -m "feat: Buffer 集成特征提取与跨位置丢失恢复"
```

---

### Task 4: 版本 0.3.0 + README + 全量验证 + 冒烟

**Files:**
- Modify: `pyproject.toml`、`src/signbridge/__init__.py`（0.2.0 → 0.3.0）
- Modify: `src/signbridge/core/__init__.py`（导出 features）
- Modify: `src/signbridge/__init__.py`（顶层导出 features 相关）
- Modify: `README.md`

- [ ] **Step 1: 版本 bump**

`pyproject.toml`: `version = "0.3.0"`
`src/signbridge/__init__.py`: `__version__ = "0.3.0"`

- [ ] **Step 2: `src/signbridge/core/__init__.py` 追加导出**

```python
from signbridge.core.features import (
    DistanceFeatureVerifier,
    FeatureExtractor,
    FeatureVerifier,
    HandShapeFeature,
)
```

`__all__` 追加 `"DistanceFeatureVerifier"`, `"FeatureExtractor"`, `"FeatureVerifier"`, `"HandShapeFeature"`。

- [ ] **Step 3: `src/signbridge/__init__.py` 顶层追加**

```python
from signbridge.core.features import (
    DistanceFeatureVerifier,
    FeatureExtractor,
    FeatureVerifier,
    HandShapeFeature,
)
from signbridge.core.matching import FeatureHungarianMatcher, HungarianMatcher
```

`__all__` 追加对应名称。

- [ ] **Step 4: README 更新**

- 版本描述改 0.3.0：手部关键点提取 + 时序序列缓冲 + **特征增强匹配（跨位置恢复）**
- API 速览表 `core.features` 行：

```markdown
| `signbridge.core.features` | `FeatureExtractor` / `FeatureVerifier` 协议 + `HandShapeFeature`（210 维距离矩阵）/ `DistanceFeatureVerifier`（高斯核置信度） |
```

- `core.matching` 行更新为：

```markdown
| `signbridge.core.matching` | `Matcher` 协议 v2（`HandDescriptor`）+ `Matching` + `HungarianMatcher`（纯位置）/ `FeatureHungarianMatcher`（分层：位置 + 特征恢复） |
```

- 时序缓冲示例中 `matcher` 行注释更新：

```python
    matcher=FeatureHungarianMatcher(confidence_threshold=0.85),  # 可插拔：位置匹配 + 特征恢复
```

- 新增小节「特征增强匹配（0.3.0）」说明分层逻辑与扩展位：

```markdown
## 特征增强匹配（0.3.0）

正常跟踪走位置匈牙利匹配；手短暂消失后从画面**另一侧**重新出现时，
用**手形特征**（210 维归一化距离矩阵，旋转/尺度/平移不变）做同一性判定：
置信度 ≥ `confidence_threshold`（默认 0.85）→ 恢复原 ID，否则视为新手。

两个可插拔协议：
- `FeatureExtractor`：21×3 点阵 → 特征向量（默认 `HandShapeFeature`）
- `FeatureVerifier`：两特征 → 置信度 [0,1]（默认 `DistanceFeatureVerifier`；
  **未来 transformer / GCN 相似度模型实现此协议即可替换**）

传 `matcher=HungarianMatcher()` 或 `feature_extractor=None` 可退回纯位置模式。
```

- [ ] **Step 5: 全量测试 + 公共 API 验证**

Run: `python -m pytest -v`
Expected: 91 passed

Run: `python -c "from signbridge import FeatureHungarianMatcher, HandShapeFeature, DistanceFeatureVerifier, FeatureVerifier; print('ok')"`
Expected: `ok`

- [ ] **Step 6: 真实视频冒烟（特征恢复路径实测）**

Run: `python scripts/verify_tracking.py "D:\data\Video\OBS\2026-08-14 22-42-14.mp4" --max-frames 120`
Expected: 正常输出跟踪统计（新默认匹配器工作；断续出现场景 ID 行为与之前一致或更好）

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml src/signbridge/ README.md
git commit -m "release: 0.3.0 特征增强匹配（分层位置+特征恢复）"
```

- [ ] **Step 8: 完成声明**

向用户报告：交付物、测试结果（91 passed）、冒烟结果、扩展位说明（transformer/GCN 接入 FeatureVerifier）、可退回纯位置模式。
