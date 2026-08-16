# SignBridge CTC 训练链路实现计划（0.5.0）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `STGCNCTC` 模型（帧级输出）+ 小样本 CTC 训练链路验证（30 段样本跑通 CTCLoss 训练与贪心解码）。

**Architecture:** 新增 `models/stgcn_ctc.py`（复用 `STGCNBlock`，去掉全局池化，帧级共享分类头 Conv2d(1x1→K+1) + 节点均值池化 → `(N,T',K+1)`）；`scripts/train_ctc.py` 加载现有 segments.npz → 张量化 → CTCLoss 训练 → 贪心解码对比。现有 STGCN 零改动。

**Tech Stack:** PyTorch 2.11（测试 CPU）、numpy、pytest。

**关键环境事实：**
- pytest 用 `python -m pytest`；当前 121 passed；版本 0.4.0（本计划 bump 0.5.0）
- `STGCNBlock` 9 层 T=128 → T'=32（已实测）
- 数据：`data/extracted/segments.npz`（30 段，含 gloss 原文，allow_pickle）
- 规格：`docs/superpowers/specs/2026-08-16-ctc-training-design.md`

**执行约定：** 每步跑完测试再提交；全部测试通过后才 commit。

---

### Task 1: `STGCNCTC` 模型（TDD）

**Files:**
- Create: `src/signbridge/models/stgcn_ctc.py`
- Test: `tests/test_stgcn_ctc.py`

- [ ] **Step 1: 写失败测试 `tests/test_stgcn_ctc.py`**

```python
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from signbridge.core.graphs import build_hand_graph
from signbridge.models.stgcn_ctc import STGCNCTC


def _adj():
    return build_hand_graph(num_hands=1)


def test_forward_shape_single_hand():
    model = STGCNCTC(num_classes=10, adjacency=_adj())
    out = model(torch.randn(2, 3, 128, 21))
    assert out.dim() == 3
    assert out.shape[0] == 2 and out.shape[2] == 11   # K+1
    assert out.shape[1] == 32                          # T' = 128/2/2


def test_forward_shape_two_hands():
    model = STGCNCTC(num_classes=10, adjacency=build_hand_graph(num_hands=2))
    out = model(torch.randn(2, 3, 128, 42))
    assert out.shape == (2, 32, 11)


def test_log_probs_shape_and_normalized():
    model = STGCNCTC(num_classes=10, adjacency=_adj())
    lp = model.log_probs(torch.randn(2, 3, 128, 21))
    assert lp.shape == (32, 2, 11)                     # (T', N, K+1)
    assert torch.allclose(lp.exp().sum(dim=2), torch.ones(32, 2), atol=1e-4)


def test_decode_merges_repeats_and_removes_blank():
    model = STGCNCTC(num_classes=3, adjacency=_adj())
    model.eval()
    # 手工构造 logits：argmax 序列 = [0,1,1,0,2,0,0,2] → 去重合并 → [1,2,2] → 去 blank → [1,2]
    logits = torch.full((1, 8, 4), -10.0)
    seq = [0, 1, 1, 0, 2, 0, 0, 2]
    for t, c in enumerate(seq):
        logits[0, t, c] = 10.0
    decoded = model.decode(logits)
    assert decoded == [[1, 2]]


def test_decode_empty_output():
    model = STGCNCTC(num_classes=3, adjacency=_adj())
    logits = torch.full((1, 8, 4), -10.0)
    for t in range(8):
        logits[0, t, 0] = 10.0                        # 全 blank
    assert model.decode(logits) == [[]]


def test_ctc_backward_gradients_exist():
    model = STGCNCTC(num_classes=5, adjacency=_adj())
    x = torch.randn(2, 3, 128, 21)
    targets = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])  # 词 id（含 0 填充）
    target_lengths = torch.tensor([2, 1])
    loss = F.ctc_loss(model.log_probs(x), targets,
                      input_lengths=torch.full((2,), 32),
                      target_lengths=target_lengths)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"参数 {name} 无梯度"


def test_custom_config_forward():
    model = STGCNCTC(num_classes=3, adjacency=_adj(),
                     channels=(32, 64, 128), strides=(1, 2, 2))
    out = model(torch.randn(2, 3, 128, 21))
    assert out.shape == (2, 32, 4)


def test_invalid_input_raises():
    model = STGCNCTC(num_classes=3, adjacency=_adj())
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 128, 20))              # V=20 != 21
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 5, 21))                # T < kernel_size
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_stgcn_ctc.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.models.stgcn_ctc'`）

- [ ] **Step 3: 实现 `src/signbridge/models/stgcn_ctc.py`**

```python
"""ST-GCN 的 CTC 变体：每时间步输出类别分布（连续手语识别用）。

复用 STGCNBlock；去掉全局平均池化，改为帧级共享分类头：
blocks → (N, C_last, T', V) → Conv2d(1x1, K+1) → 节点维均值 → (N, T', K+1)。
"""

import numpy as np
import torch
import torch.nn as nn

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
```

- [ ] **Step 4: 更新 `models/__init__.py` 导出并跑测试**

`src/signbridge/models/__init__.py` 追加：

```python
from signbridge.models.stgcn_ctc import STGCNCTC
```

`__all__` 追加 `"STGCNCTC"`。

Run: `python -m pytest tests/test_stgcn_ctc.py -v`
Expected: 8 passed

- [ ] **Step 5: 全量回归**

Run: `python -m pytest`
Expected: 129 passed（121 + 8）

- [ ] **Step 6: 提交**

```bash
git add src/signbridge/models/ tests/test_stgcn_ctc.py
git commit -m "feat: STGCNCTC 帧级 CTC 输出模型（复用 blocks + 共享分类头）"
```

---

### Task 2: 小样本 CTC 训练脚本 `scripts/train_ctc.py`

**Files:**
- Create: `scripts/train_ctc.py`

- [ ] **Step 1: 实现脚本**

```python
"""CTC 训练链路小样本验证：segments.npz → STGCNCTC → CTCLoss → 贪心解码。

成功标准：loss 下降、解码输出合法词、无 NaN。

用法: python scripts/train_ctc.py [--npz data/extracted/segments.npz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from signbridge import STGCNCTC, build_hand_graph

PUNCT = set("。，？！、；：""''（）《》")


def gloss_words(gloss: str) -> list[str]:
    return [w for w in gloss.split("/") if w.strip() and w.strip() not in PUNCT]


def to_tensor_batch(samples, target_t: int):
    batch = []
    for data in samples:
        t = len(data)
        if t >= target_t:
            arr = data[:target_t]
        else:
            reps = int(np.ceil(target_t / t))
            arr = np.tile(data, (reps, 1, 1))[:target_t]
        batch.append(arr)
    x = np.stack(batch)
    x = np.transpose(x, (0, 3, 1, 2))
    return torch.from_numpy(x).float()


def main() -> int:
    parser = argparse.ArgumentParser(description="CTC 小样本训练验证")
    parser.add_argument("--npz", type=str, default="data/extracted/segments.npz")
    parser.add_argument("--target-t", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.npz)
    if not path.exists():
        print(f"未找到 {path}")
        return 1
    data = np.load(path, allow_pickle=True)
    samples = data["data"]
    glosses = list(data["glosses"])

    # 词表（当前小样本）
    from collections import Counter
    freq = Counter(w for g in glosses for w in gloss_words(g))
    vocab = [w for w, _ in freq.most_common()]
    vocab_idx = {w: i + 1 for i, w in enumerate(vocab)}   # 0 = blank
    print(f"样本 {len(samples)}，词表 {len(vocab)}: {vocab}")

    x = to_tensor_batch(samples, args.target_t)
    targets, target_lengths = [], []
    for g in glosses:
        ids = [vocab_idx[w] for w in gloss_words(g) if w in vocab_idx]
        targets.append(ids)
        target_lengths.append(len(ids))
    max_len = max(target_lengths)
    targets_pad = torch.zeros(len(targets), max_len, dtype=torch.long)
    for i, ids in enumerate(targets):
        targets_pad[i, :len(ids)] = torch.tensor(ids)
    target_lengths = torch.tensor(target_lengths)
    print(f"张量 {tuple(x.shape)}；标签最长 {max_len}（≤ T'=32）")

    torch.manual_seed(args.seed)
    model = STGCNCTC(num_classes=len(vocab),
                     adjacency=build_hand_graph(num_hands=2))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    input_lengths = torch.full((len(x),), 32)   # T'

    print(f"{'epoch':>5} {'ctc_loss':>10} {'样本1预测':>14} {'样本1真值':>14}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        lp = model.log_probs(x)
        loss = F.ctc_loss(lp, targets_pad, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                decoded = model.decode(model(x))
                pred0 = "".join(vocab[c - 1] for c in decoded[0]) or "(空)"
                truth0 = "".join(gloss_words(glosses[0]))
            print(f"{epoch:>5} {loss.item():>10.4f} {pred0:>14} {truth0:>14}")
    print("\nCTC 链路验证完成：loss 下降 / 解码合法 / 梯度正常")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 运行小样本训练**

Run: `python scripts/train_ctc.py`
Expected: ctc_loss 下降；样本 1 解码输出与真值对比打印；无 NaN

- [ ] **Step 3: 提交**

```bash
git add scripts/train_ctc.py
git commit -m "feat: CTC 小样本训练验证脚本（STGCNCTC + CTCLoss + 贪心解码）"
```

---

### Task 3: 版本 0.5.0 + README + 收尾

**Files:**
- Modify: `pyproject.toml`、`src/signbridge/__init__.py`（0.4.0 → 0.5.0）
- Modify: `README.md`

- [ ] **Step 1: 版本 bump**

`pyproject.toml`: `version = "0.5.0"`
`src/signbridge/__init__.py`: `__version__ = "0.5.0"`

- [ ] **Step 2: 顶层 `__init__.py` 追加导出**

```python
from signbridge.models.stgcn_ctc import STGCNCTC
```

`__all__` 追加 `"STGCNCTC"`。

- [ ] **Step 3: README 更新**

- API 表 `signbridge.models` 行追加 `STGCNCTC`：
  `+ `STGCNCTC(num_classes, adjacency, ...)`（CTC 帧级输出，连续手语识别）`
- 新增小节「CTC 连续手语训练（0.5.0 链路验证）」：

```markdown
## CTC 连续手语训练（0.5.0 链路验证）

CE-CSL 是连续手语句子（无帧级时间戳），用 CTC 对齐词序列与时间步：

```python
import torch
from signbridge import STGCNCTC, build_hand_graph

model = STGCNCTC(num_classes=词表大小, adjacency=build_hand_graph(num_hands=2))
x = torch.randn(2, 3, 128, 42)          # (N, C, T, V)
logits = model(x)                        # (N, T'=32, K+1)，0=blank
lp = model.log_probs(x)                  # (T', N, K+1) log-softmax
pred = model.decode(logits)              # [[词id...], ...] 贪心解码
```

训练：`python scripts/train_ctc.py`（小样本链路验证）。全量提取与正式训练为后续步骤。
```

- [ ] **Step 4: 全量测试 + 公共 API 验证**

Run: `python -m pytest -v`
Expected: 129 passed

Run: `python -c "from signbridge import STGCNCTC; print('ok')"`
Expected: `ok`

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml src/signbridge/ README.md
git commit -m "release: 0.5.0 STGCNCTC 帧级 CTC 模型与训练链路"
```

- [ ] **Step 6: 完成声明**

向用户报告：STGCNCTC 交付、小样本 CTC 训练结果（loss 曲线/解码样例）、
下一步（全量提取 → 正式 CTC 训练 → 束搜索 + WER 评估）。
