"""骨架时序分类模型协议（整模型级可插拔）。

任何实现此协议的模型（ST-GCN 默认 / 未来 transformer / GCN 变体）
可直接替换，训练/推理/部署管线无需改动。
"""

from typing import Protocol

import torch
import torch.nn as nn


class SkeletonClassifier(nn.Module, Protocol):
    """骨架时序分类模型协议。

    输入约定：(N, C, T, V) —— N 批、C 特征通道（x,y,z=3）、T 时间、V 节点。
    """

    num_classes: int
    num_nodes: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, T, V) 骨架张量 → (N, num_classes) logits。"""

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, T, V) → (N,) 类别索引（argmax 包装 forward）。"""
