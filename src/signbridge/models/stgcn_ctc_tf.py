"""骨架版 TFNet（方案 A）：STGCN 主干 + 频域 FFT 分支 + 多 CTC + SeqKD。

结构（增量式，复用 STGCNCTC 的 blocks）：
    blocks (9 层 STGCN) → 节点均值 → f (N, 256, T')
      ├─ 时域: head_t(f) → logits_t        （可由 STGCNCTC checkpoint 初始化）
      ├─ 频域: |FFT(f, dim=T')| → 1D Conv+BN+ReLU → head_f → logits_f
      └─ 融合: f + f_freq → head_fusion → logits_fusion   （评估用）

Loss（对齐 TFNet 论文 7-loss 的骨架版）：
    CTC(logits_t) + CTC(logits_f) + CTC(logits_fusion)
    + 25 × SeqKD(prediction=logits_f, ref=logits_t)   （频域向时域对齐）

FFT 细节与官方一致：torch.fft.fft(dim=-1, norm="forward") + abs（幅度谱）。
评估时只用 logits_fusion。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signbridge.models.decoding import ctc_beam_search
from signbridge.models.stgcn import STGCNBlock


class SeqKD(nn.Module):
    """分支间输出分布 KL 一致性（官方 TFNet 同款，T=1 去 blank）。"""

    def __init__(self, T: float = 1.0):
        super().__init__()
        self.kdloss = nn.KLDivLoss(reduction="batchmean")
        self.T = T

    def forward(self, prediction_logits, ref_logits, use_blank=False):
        start_idx = 0 if use_blank else 1
        pred = F.log_softmax(
            prediction_logits[:, :, start_idx:] / self.T, dim=-1
        ).reshape(-1, ref_logits.shape[2] - start_idx)
        ref = F.softmax(
            ref_logits[:, :, start_idx:] / self.T, dim=-1
        ).reshape(-1, ref_logits.shape[2] - start_idx)
        return self.kdloss(pred, ref) * self.T * self.T


class STGCNCTCTF(nn.Module):
    """STGCN + 时频双分支 CTC（骨架版 TFNet）。

    输入 hand (N,3,T,V) → (logits_t, logits_f, logits_fusion)，
    三者均为 (N, T', K+1)；评估用 logits_fusion。
    """

    def __init__(self, num_classes, adjacency, in_channels=3,
                 channels=(64, 64, 64, 128, 128, 128, 256, 256, 256),
                 strides=(1, 1, 1, 2, 1, 1, 2, 1, 1),
                 kernel_size=9, adaptive=True, dropout=0.5,
                 freq_kernel=5):
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
                kernel_size=kernel_size, adaptive=adaptive, dropout=dropout))
            in_ch = out_ch
        self.feat_dim = in_ch          # 256

        # 时域头（与 STGCNCTC.head 等价，Conv1d 便于拼接）
        self.head_t = nn.Conv1d(self.feat_dim, self.num_classes + 1, 1)
        # 频域分支：幅度谱 → 1D Conv+GroupNorm+ReLU（保持 T'，不再下采样）
        # 用 GroupNorm 而非 BatchNorm：|FFT| 幅度谱结构化（DC 大/高频趋 0），
        # BN 在 batch 方差为 0 的通道上除 0 → 梯度 nan → 权重污染（实测 batch
        # 63 起全 nan 的元凶）
        self.freq_conv = nn.Sequential(
            nn.Conv1d(self.feat_dim, self.feat_dim, freq_kernel,
                      padding=freq_kernel // 2),
            nn.GroupNorm(num_groups=4, num_channels=self.feat_dim),
            nn.ReLU(inplace=True),
        )
        self.head_f = nn.Conv1d(self.feat_dim, self.num_classes + 1, 1)
        # 融合头
        self.head_fusion = nn.Conv1d(self.feat_dim, self.num_classes + 1, 1)

    def _validate(self, x):
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

    def forward(self, x):
        """→ (logits_t, logits_f, logits_fusion)，均为 (N, T', K+1)。"""
        self._validate(x)
        for block in self.blocks:
            x = block(x)
        f = x.mean(dim=3)                       # (N, 256, T')

        logits_t = self.head_t(f).permute(0, 2, 1)          # (N, T', K+1)

        spec = torch.fft.fft(f, dim=-1, norm="forward")
        spec = torch.abs(spec)                               # (N, 256, T')
        f_freq = self.freq_conv(spec)                        # (N, 256, T')
        logits_f = self.head_f(f_freq).permute(0, 2, 1)

        f_fusion = f + f_freq
        logits_fusion = self.head_fusion(f_fusion).permute(0, 2, 1)
        return logits_t, logits_f, logits_fusion

    def log_probs(self, logits):
        """(N, T', K+1) → (T', N, K+1)（CTCLoss 标准输入）。"""
        return torch.log_softmax(logits, dim=2).permute(1, 0, 2)

    def decode(self, logits):
        pred = logits.argmax(dim=2)
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

    def beam_decode(self, logits, beam_width=10, top_tokens=20,
                    length_bonus: float = 0.0):
        lp = torch.log_softmax(logits, dim=2).cpu().numpy()
        return [ctc_beam_search(x, blank=0, beam_width=beam_width,
                                top_tokens=top_tokens,
                                length_bonus=length_bonus)
                for x in lp]

    def load_temporal_state(self, ckpt_state: dict) -> int:
        """从 STGCNCTC checkpoint 加载 blocks + head_t（时域路径初始化）。

        STGCNCTC.head 是 Conv2d(256, K+1, 1)，权重 (K+1,256,1,1)
        → head_t Conv1d 权重 (K+1,256,1)。返回加载的参数组数。
        """
        own = self.state_dict()
        n = 0
        for k, v in ckpt_state.items():
            if k.startswith("blocks.") and k in own:
                own[k] = v
                n += 1
            elif k == "head.weight" and "head_t.weight" in own:
                own["head_t.weight"] = v.squeeze(-1)     # (K+1,256,1,1)→(K+1,256,1)
                own["head_t.bias"] = ckpt_state["head.bias"]
                n += 2
        self.load_state_dict(own)
        return n
