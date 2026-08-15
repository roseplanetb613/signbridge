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
