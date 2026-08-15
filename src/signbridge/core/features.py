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
