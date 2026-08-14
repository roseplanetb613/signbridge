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
