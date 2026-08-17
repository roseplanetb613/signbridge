"""三流融合 CTC 模型：hand 骨架 + pose 姿态 + ROI 手部图像。

骨架流（hand）：STGCNBlock 9 层（复用）→ 节点均值 → (N, 256, T')
姿态流（pose）：STGCNBlock 4 层（浅）→ 节点均值 → (N, 256, T')
RGB 流（roi）：ResNet18（ImageNet 预训练）逐帧特征 → 时间下采样 ×2 → (N, 512, T')
融合：concat (N, 1024, T') → 1×1 分类头 → (N, T', K+1) → CTC

输入约定：T 统一（默认 128 → T'=32）。
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision

from signbridge.models.decoding import ctc_beam_search
from signbridge.models.stgcn import STGCNBlock


class _TemporalDownsample(nn.Module):
    """时间维下采样（stride 2 卷积 + BN + ReLU）。"""

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels,
                              kernel_size=(kernel_size, 1),
                              stride=(2, 1),
                              padding=(kernel_size // 2, 0))
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)))


class FusionSTGCNCTC(nn.Module):
    """三流融合 CTC 模型（连续手语识别）。

    注意：三输入 forward（hand/pose/roi），不属于 SkeletonClassifier
    单输入协议；decode/beam_decode 复用（输入为 logits）。
    """

    def __init__(self, num_classes, hand_adjacency, pose_adjacency,
                 in_channels=3,
                 hand_channels=(64, 64, 64, 128, 128, 128, 256, 256, 256),
                 hand_strides=(1, 1, 1, 2, 1, 1, 2, 1, 1),
                 pose_channels=(64, 128, 256, 256),
                 pose_strides=(1, 2, 1, 2),
                 kernel_size=9, adaptive=True, dropout=0.5,
                 resnet_pretrained=True, roi_input_size=112):
        super().__init__()
        h_adj = np.asarray(hand_adjacency, dtype=np.float32)
        p_adj = np.asarray(pose_adjacency, dtype=np.float32)
        if h_adj.ndim != 2 or p_adj.ndim != 2:
            raise ValueError("邻接矩阵必须是方阵")
        if len(hand_channels) != len(hand_strides):
            raise ValueError("hand_channels 与 hand_strides 长度必须一致")
        if len(pose_channels) != len(pose_strides):
            raise ValueError("pose_channels 与 pose_strides 长度必须一致")

        self.num_classes = int(num_classes)
        self.hand_nodes = int(h_adj.shape[0])
        self.pose_nodes = int(p_adj.shape[0])
        self.in_channels = int(in_channels)
        self.kernel_size = int(kernel_size)
        self.roi_input_size = int(roi_input_size)

        # hand 流（9 层，与 STGCNCTC 一致）
        self.hand_blocks = nn.ModuleList()
        in_ch = self.in_channels
        for out_ch, stride in zip(hand_channels, hand_strides):
            self.hand_blocks.append(STGCNBlock(
                in_ch, out_ch, h_adj, stride=stride,
                kernel_size=kernel_size, adaptive=adaptive, dropout=dropout))
            in_ch = out_ch
        self.hand_dim = in_ch

        # pose 流（4 层浅配置）
        self.pose_blocks = nn.ModuleList()
        in_ch = self.in_channels
        for out_ch, stride in zip(pose_channels, pose_strides):
            self.pose_blocks.append(STGCNBlock(
                in_ch, out_ch, p_adj, stride=stride,
                kernel_size=kernel_size, adaptive=adaptive, dropout=dropout))
            in_ch = out_ch
        self.pose_dim = in_ch

        # ROI 流：ResNet18 去 avgpool/fc
        weights = (torchvision.models.ResNet18_Weights.DEFAULT
                   if resnet_pretrained else None)
        resnet = torchvision.models.resnet18(weights=weights)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # → (B,512,1,1)
        self.resnet_dim = 512
        self.roi_tconv1 = _TemporalDownsample(self.resnet_dim)
        self.roi_tconv2 = _TemporalDownsample(self.resnet_dim)

        # 融合头
        fusion_in = self.hand_dim + self.pose_dim + self.resnet_dim
        self.head = nn.Conv2d(fusion_in, self.num_classes + 1, kernel_size=1)

    def _validate(self, hand, pose, roi):
        if hand.dim() != 4 or pose.dim() != 4 or roi.dim() != 5:
            raise ValueError("hand/pose 必须为 4 维 (N,C,T,V)，roi 必须为 5 维 "
                             "(N,T,3,S,S)")
        if hand.shape[1] != self.in_channels or pose.shape[1] != self.in_channels:
            raise ValueError("hand/pose 通道数应为 in_channels")
        if hand.shape[3] != self.hand_nodes:
            raise ValueError(f"hand 节点数应为 {self.hand_nodes}")
        if pose.shape[3] != self.pose_nodes:
            raise ValueError(f"pose 节点数应为 {self.pose_nodes}")
        if not (hand.shape[2] == pose.shape[2]):
            raise ValueError("hand 与 pose 时间长度必须一致")
        if hand.shape[2] < self.kernel_size:
            raise ValueError(f"T={hand.shape[2]} 必须 >= kernel_size")
        if roi.shape[1] != hand.shape[2]:
            raise ValueError("roi 时间长度必须与 hand 一致")
        if roi.shape[2:] != (3, self.roi_input_size, self.roi_input_size):
            raise ValueError(f"roi 形状应为 (N,T,3,{self.roi_input_size},"
                             f"{self.roi_input_size})，收到 {tuple(roi.shape)}")

    def forward(self, hand, pose, roi):
        """hand (N,3,T,Vh) + pose (N,3,T,Vp) + roi (N,T,3,S,S)
        → logits (N, T', K+1)。"""
        self._validate(hand, pose, roi)
        n, _, t, _ = hand.shape

        # hand 流
        h = hand
        for block in self.hand_blocks:
            h = block(h)
        h = h.mean(dim=3)                       # (N, 256, T')

        # pose 流
        p = pose
        for block in self.pose_blocks:
            p = block(p)
        p = p.mean(dim=3)                       # (N, 256, T')

        # ROI 流
        r = roi.float() / 255.0                 # (N, T, 3, S, S)
        r = r.reshape(n * t, 3, self.roi_input_size, self.roi_input_size)
        r = self.resnet(r).flatten(1)           # (N·T, 512)
        r = r.reshape(n, t, self.resnet_dim).permute(0, 2, 1)  # (N, 512, T)
        r = r.unsqueeze(-1)                     # (N, 512, T, 1)
        r = self.roi_tconv1(r)
        r = self.roi_tconv2(r)                  # (N, 512, T', 1)
        r = r.squeeze(-1)                       # (N, 512, T')

        # 融合
        x = torch.cat([h, p, r], dim=1)         # (N, 1024, T')
        x = x.unsqueeze(-1)                     # (N, 1024, T', 1)
        x = self.head(x).squeeze(-1)            # (N, K+1, T')
        return x.permute(0, 2, 1)               # (N, T', K+1)

    def log_probs(self, hand, pose, roi):
        """→ (T', N, K+1) log-softmax（CTCLoss 标准输入）。"""
        logits = self.forward(hand, pose, roi)
        return torch.log_softmax(logits, dim=2).permute(1, 0, 2)

    def decode(self, logits):
        """贪心解码：(N, T', K+1) logits → list[list[int]]。"""
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
        """束搜索解码：(N, T', K+1) logits → list[list[int]]。

        length_bonus: 每输出一个词乘 (1+length_bonus)，缓解 CTC
        欠预测（与 STGCNCTC.beam_decode 一致）。0=不启用。
        """
        lp = torch.log_softmax(logits, dim=2).cpu().numpy()
        return [ctc_beam_search(x, blank=0, beam_width=beam_width,
                                top_tokens=top_tokens,
                                length_bonus=length_bonus)
                for x in lp]
