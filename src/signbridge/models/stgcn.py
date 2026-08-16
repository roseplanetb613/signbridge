"""经典 ST-GCN（Yan et al. 2018）参数化实现。

模块划分：GraphConv（图卷积 + 可选自适应图残差 B）/ TemporalConv（时间卷积）/
STGCNBlock（残差 + BN + Dropout）。全部参数可配置，便于微调。
图（邻接矩阵）为构造参数，num_nodes 由矩阵推导——不写死 21/33/42。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signbridge.core.graphs import normalize_adjacency


class GraphConv(nn.Module):
    """单层图卷积：Y = A_hat @ X @ W。

    A_hat：归一化邻接（+自环、D^-1/2 对称归一化），作为 buffer；
    adaptive=True 时叠加可学习图残差 B（A-Link Inference）；W 为 1×1 卷积。
    """

    def __init__(self, in_channels, out_channels, adjacency, adaptive=True):
        super().__init__()
        adj = np.asarray(adjacency, dtype=np.float32)
        norm = normalize_adjacency(adj, include_self=True)
        self.register_buffer("adjacency", torch.from_numpy(norm))
        self.adaptive = adaptive
        num_nodes = int(adj.shape[0])
        if adaptive:
            self.B = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V)
        a = self.adjacency
        if self.adaptive:
            a = a + self.B
        x = torch.einsum("nctv,vw->nctw", x, a)
        return self.conv(x)


class TemporalConv(nn.Module):
    """时间卷积：kernel_size 默认 9；stride 支持时间下采样。"""

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(kernel_size // 2, 0),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class STGCNBlock(nn.Module):
    """图卷积 + 时间卷积 + 残差 + BN/ReLU/Dropout。

    残差：通道变化或 stride>1 时用 1×1 卷积投影对齐。
    """

    def __init__(self, in_channels, out_channels, adjacency, stride=1,
                 kernel_size=9, adaptive=True, dropout=0.0):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, adjacency,
                             adaptive=adaptive)
        self.tcn = TemporalConv(out_channels, out_channels,
                                kernel_size=kernel_size, stride=stride)
        self.residual = None
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.tcn(self.gcn(x))
        res = self.residual(x) if self.residual is not None else x
        return self.dropout(F.relu(out + res))


class STGCN(nn.Module):
    """经典 9 层 ST-GCN（默认 = Yan 2018 配置），全部参数化。

    输入 (N, C, T, V) → 全局平均池化 → 线性分类头 → (N, num_classes)。
    """

    def __init__(self, num_classes, adjacency, in_channels=3,
                 channels=(64, 64, 64, 128, 128, 128, 256, 256, 256),
                 strides=(1, 1, 1, 2, 1, 1, 2, 1, 1),
                 kernel_size=9, adaptive=True, dropout=0.5):
        super().__init__()
        adj = np.asarray(adjacency, dtype=np.float32)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("adjacency 必须是方阵")
        if len(channels) != len(strides):
            raise ValueError("channels 与 strides 长度必须一致")
        self.num_classes = int(num_classes)
        self.num_nodes = int(adj.shape[0])
        self.in_channels = int(in_channels)
        self.kernel_size = int(kernel_size)
        self.blocks = nn.ModuleList()
        in_ch = self.in_channels
        for out_ch, stride in zip(channels, strides):
            self.blocks.append(STGCNBlock(
                in_ch, out_ch, adj, stride=stride,
                kernel_size=kernel_size, adaptive=adaptive, dropout=dropout,
            ))
            in_ch = out_ch
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_ch, self.num_classes)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"输入必须是 4 维 (N,C,T,V)，收到 {x.dim()} 维")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"通道数应为 {self.in_channels} (x,y,z)，收到 {x.shape[1]}")
        if x.shape[3] != self.num_nodes:
            raise ValueError(f"节点数应为 {self.num_nodes}，收到 {x.shape[3]}")
        if x.shape[2] < self.kernel_size:
            raise ValueError(
                f"时间长度 T={x.shape[2]} 必须 >= kernel_size={self.kernel_size}")
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

    def predict(self, x):
        """(N, C, T, V) → (N,) 类别索引。"""
        return self.forward(x).argmax(dim=1)
