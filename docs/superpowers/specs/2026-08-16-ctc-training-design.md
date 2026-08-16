# SignBridge CTC 训练链路设计（小样本验证阶段）

日期：2026-08-16
状态：设计已获用户确认（新类 STGCNCTC / 先小样本验证链路）

## 1. 背景与目标

CE-CSL 是**连续手语**数据集（句子 = 手势词序列，无帧级时间戳标注）。
标准训练方案：**CTC（连接时序分类）**——模型每时间步输出词类别分布，
CTC 损失自动对齐词序列与时间步（容忍重复与 blank）。

本步目标：**小样本验证 CTC 全链路**——`STGCNCTC` 模型（帧级输出）+
句子词序列标签 + CTCLoss 训练 + 贪心解码，用现有 30 段样本跑通。
全量提取（5988 视频）与正式训练为后续步骤。

## 2. 决策（用户已确认）

| 决策点 | 选择 |
| --- | --- |
| 模型改造 | 新类 `STGCNCTC`（复用 GraphConv/STGCNBlock；现有 STGCN 不动） |
| 输出形态 | 去掉全局池化 → 帧级共享分类头 → `(N, T', K+1)`（K=词表，0=blank） |
| 标签 | gloss 词序列 → 词表 → `[词id+1]`（0 保留给 blank） |
| 损失 | `torch.nn.CTCLoss` |
| 解码 | 贪心（argmax → 去重 → 去 blank）；束搜索后续 |
| 验证范围 | 30 样本：loss 下降、解码合法、无 NaN |

## 3. 模型 `models/stgcn_ctc.py`

```python
class STGCNCTC(nn.Module):
    """ST-GCN 的 CTC 变体：每时间步输出类别分布。

    复用 STGCNBlock 9 层（默认配置）；T=128 → T'=32（两次 stride=2 下采样）。
    """

    def __init__(self, num_classes, adjacency, in_channels=3,
                 channels=(64,64,64,128,128,128,256,256,256),
                 strides=(1,1,1,2,1,1,2,1,1),
                 kernel_size=9, adaptive=True, dropout=0.5): ...

    def forward(self, x) -> torch.Tensor:
        """(N, C, T, V) → (N, T', K+1) logits。"""
        # blocks → (N, C_last, T', V) → Conv2d(1x1 → K+1) → 节点维均值 → permute

    def log_probs(self, x) -> torch.Tensor:
        """(N, C, T, V) → (T', N, K+1) log-softmax（CTCLoss 标准输入）。"""

    def decode(self, logits, greedy=True) -> list[list[int]]:
        """argmax → 合并连续重复 → 去 blank(0) → 词 id 序列。"""
```

- 输入校验与 STGCN 一致（维度/通道/节点数/T ≥ kernel_size）
- 帧级分类头：`nn.Conv2d(C_last, K+1, 1)` 共享权重；节点维用均值池化
- `num_nodes` 从邻接矩阵推导

## 4. 标签构建

- gloss 过滤标点（。，？！、；：等）→ 词序列
- 词表：全量句子词频排序（本阶段用当前 30 样本词表）
- 每句标签：`[vocab[word]+1 for word in 词序列]`（0 = blank，不占词表）

## 5. 训练脚本 `scripts/train_ctc.py`

- 加载 `data/extracted/segments.npz` → 张量化 `(N, 3, 128, 42)`（现有对齐逻辑）
- 标签：gloss → 词序列 → id 序列；`target_lengths`（≤ T'=32）
- `CTCLoss(log_probs, targets, input_lengths=[T']*N, target_lengths=...)`
- 60 epochs / Adam / lr 1e-3；每 5 epoch 打印 loss + 2 样本
  贪心解码 vs 真实词序列
- 成功标准：loss 下降、解码合法、无 NaN——链路验证

## 6. 测试策略（`tests/test_stgcn_ctc.py`，纯 CPU）

- 前向形状 `(N, T', K+1)`（单/双手图）
- `log_probs` 形状 `(T', N, K+1)` 且 log-softmax 归一（行和 ≈ 1）
- `decode`：连续重复合并、blank 去除、空序列、单字符
- 反向梯度存在（CTCLoss 前向+backward 可跑）
- 非默认配置（channels/strides）可跑
- 输入校验（维度/节点数/T 不足）抛 ValueError

## 7. 范围外（后续步骤）

- 全量骨架提取管线（5988 视频）与正式训练
- 束搜索解码与 WER/句准确率评估
- 词表统一构建（全量数据）
- 数据增强（时间缩放/扰动）

## 8. 交付物

1. `src/signbridge/models/stgcn_ctc.py`（STGCNCTC + 导出）
2. `tests/test_stgcn_ctc.py`
3. `scripts/train_ctc.py`（小样本 CTC 训练验证）
4. README 更新；版本 bump **0.5.0**
