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
                    top_tokens: int = 20, length_bonus: float = 0.0
                    ) -> list[list[int]]:
        """束搜索解码：(N, T', K+1) logits → list[list[int]] 词 id 序列。

        合并同一前缀的多条对齐路径，优于贪心（相邻重复/blank 竞争场景）。
        length_bonus: 每输出一个词乘 (1+length_bonus)，缓解 CTC 欠预测。
        """
        log_probs = torch.log_softmax(logits, dim=2).cpu().numpy()
        return [ctc_beam_search(lp, blank=0, beam_width=beam_width,
                                top_tokens=top_tokens,
                                length_bonus=length_bonus)
                for lp in log_probs]

    def embed(self, x) -> torch.Tensor:
        """段 → 嵌入向量 (N, D)。

        blocks 输出 (N, C, T', V) → 节点均值 → 时间均值 → (N, C_last)。
        嵌入空间由 CTC 监督塑造，可直接用于骨架段相似度检索/词识别。
        """
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
        x = x.mean(dim=3)          # 节点均值 → (N, C, T')
        x = x.mean(dim=2)          # 时间均值 → (N, C)
        return x


class STGCNCTCEmb(STGCNCTC):
    """嵌入头版 STGCNCTC：特征投影到 D 维后与词嵌入表点积（方案 A'）。

    与 one-hot 头（STGCNCTC）的区别仅在分类层：
      one-hot: head Conv2d(C_last, K+1) → 1321 个相互独立的类别
      嵌入头:  head Conv2d(C_last, D) → f (N,T',D)；logits = f @ word_emb^T
               word_emb (K+1, D) 随机初始化、随 CTC 训练学习
    词类共享 D 维空间——语义相近的词 logits 相关，梯度可沿嵌入
    空间在词间"溢出"（低频词从高频词借力）。blank 是 word_emb[0]。

    logits/decode/beam_decode 接口与 STGCNCTC 完全一致。
    """

    def __init__(self, num_classes, adjacency, embed_dim: int = 256,
                 emb_std: float = 0.1, in_channels=3,
                 channels=(64, 64, 64, 128, 128, 128, 256, 256, 256),
                 strides=(1, 1, 1, 2, 1, 1, 2, 1, 1),
                 kernel_size=9, adaptive=True, dropout=0.5):
        super().__init__(
            num_classes, adjacency, in_channels=in_channels,
            channels=channels, strides=strides, kernel_size=kernel_size,
            adaptive=adaptive, dropout=dropout)
        self.embed_dim = int(embed_dim)
        # 替换 one-hot 分类头：特征投影 + 词嵌入表（E[0] = blank）
        self.head = nn.Conv2d(channels[-1], self.embed_dim, kernel_size=1)
        self.word_emb = nn.Parameter(
            torch.randn(self.num_classes + 1, self.embed_dim) * emb_std)

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
        f = self.head(x).mean(dim=3)        # (N, D, T')
        f = f.permute(0, 2, 1)              # (N, T', D)
        return torch.matmul(f, self.word_emb.t())   # (N, T', K+1)
