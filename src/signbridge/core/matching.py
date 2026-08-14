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
