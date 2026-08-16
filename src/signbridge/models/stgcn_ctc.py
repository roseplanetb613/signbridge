"""ST-GCN 的 CTC 变体：每时间步输出类别分布（连续手语识别用）。

复用 STGCNBlock；去掉全局平均池化，改为帧级共享分类头：
blocks → (N, C_last, T', V) → Conv2d(1x1, K+1) → 节点维均值 → (N, T', K+1)。
"""

import numpy as np
import torch
import torch.nn as nn

from signbridge.models.decoding import ctc_beam_search
from signbridge.models.stgcn import STGCNBlock


class STGCNCTC(nn.Module):
    """CTC 输出的 ST-GCN：logits (N, T', K+1)；0 为 blank。"""

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
        self.num_classes = int(num_classes)   # K（不含 blank）
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
        self.head = nn.Conv2d(in_ch, self.num_classes + 1, kernel_size=1)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"输入必须是 4 维 (N,C,T,V)，收到 {x.dim()} 维")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"通道数应为 {self.in_channels}，收到 {x.shape[1]}")
        if x.shape[3] != self.num_nodes:
            raise ValueError(f"节点数应为 {self.num_nodes}，收到 {x.shape[3]}")
        if x.shape[2] < self.kernel_size:
            raise ValueError(
                f"时间长度 T={x.shape[2]} 必须 >= kernel_size={self.kernel_size}")
        for block in self.blocks:
            x = block(x)
        x = self.head(x)                       # (N, K+1, T', V)
        x = x.mean(dim=3)                      # 节点维均值 → (N, K+1, T')
        return x.permute(0, 2, 1)              # (N, T', K+1)

    def log_probs(self, x):
        """(N, C, T, V) → (T', N, K+1) log-softmax（CTCLoss 标准输入）。"""
        logits = self.forward(x)               # (N, T', K+1)
        return torch.log_softmax(logits, dim=2).permute(1, 0, 2)

    def decode(self, logits):
        """贪心解码：(N, T', K+1) logits → list[list[int]] 词 id 序列。

        连续重复合并 + 去 blank(0)。
        """
        pred = logits.argmax(dim=2)            # (N, T')
        out = []
        for row in pred:
            seq = []
            prev = -1
            for c in row.tolist():
                if c != prev and c != 0:
                    seq.append(c)
                prev = c
            out.append(seq)
        return out

    def beam_decode(self, logits, beam_width: int = 10,
                    top_tokens: int = 20) -> list[list[int]]:
        """束搜索解码：(N, T', K+1) logits → list[list[int]] 词 id 序列。

        合并同一前缀的多条对齐路径，优于贪心（相邻重复/blank 竞争场景）。
        """
        log_probs = torch.log_softmax(logits, dim=2).cpu().numpy()
        return [ctc_beam_search(lp, blank=0, beam_width=beam_width,
                                top_tokens=top_tokens)
                for lp in log_probs]
