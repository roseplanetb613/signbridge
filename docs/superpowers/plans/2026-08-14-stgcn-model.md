# SignBridge ST-GCN 模型组件实现计划（0.4.0）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可插拔的骨架时序分类模型组件：`SkeletonClassifier` 协议（整模型级）+ 经典 ST-GCN 参数化实现（GraphConv/TemporalConv/Block 模块化）+ `core/graphs.py` 图工具（邻接构造/归一化/多手分块/姿态扩展位）。

**Architecture:** 新增 `core/graphs.py`（图工具，core 级共用）；新增 `models/` 子包（`protocol.py` 协议 + `stgcn.py` 实现）。模型只认识邻接矩阵（`num_nodes` 从矩阵推导），与手/姿态领域解耦；图归一化在模型内部完成。

**Tech Stack:** Python 3.14、PyTorch 2.11（测试固定 CPU）、numpy 2.5、pytest。

**关键环境事实：**
- pytest 用 `python -m pytest`；当前 90 测试全绿；版本 0.3.0（本计划 bump 0.4.0）
- torch 2.11.0+cu128 已装，CUDA 可用；**测试固定 CPU 张量**（环境无关、确定性）
- 规格：`docs/superpowers/specs/2026-08-14-stgcn-model-design.md`

**执行约定：** 每步跑完测试再提交；全部测试通过后才 commit。

---

### Task 1: 图工具 `core/graphs.py`（TDD）

**Files:**
- Create: `src/signbridge/core/graphs.py`
- Test: `tests/test_graphs.py`

- [ ] **Step 1: 写失败测试 `tests/test_graphs.py`**

```python
import numpy as np
import pytest

from signbridge.core.graphs import (
    build_adjacency,
    build_block_diagonal_graph,
    build_hand_graph,
    normalize_adjacency,
)
from signbridge.core.landmarks import HAND_CONNECTIONS


def test_build_adjacency_shape_and_symmetry():
    adj = build_adjacency([(0, 1), (1, 2)], 3)
    assert adj.shape == (3, 3)
    assert np.allclose(adj, adj.T)          # 对称
    assert np.allclose(np.diag(adj), 0)     # 无自环


def test_build_adjacency_maps_edges():
    adj = build_adjacency([(0, 1), (2, 3)], 4)
    assert adj[0, 1] == 1 and adj[1, 0] == 1
    assert adj[2, 3] == 1 and adj[3, 2] == 1
    assert adj[0, 2] == 0


def test_build_adjacency_invalid_index_raises():
    with pytest.raises(ValueError):
        build_adjacency([(0, 21)], 21)


def test_normalize_adjacency_symmetric_row_sum_le_one():
    adj = build_adjacency([(0, 1), (1, 2)], 3)
    norm = normalize_adjacency(adj)
    assert np.allclose(norm, norm.T, atol=1e-6)
    assert norm.sum(axis=1).max() <= 1.0 + 1e-5


def test_normalize_adjacency_with_self_loop_rows_sum_one():
    adj = build_adjacency([(0, 1)], 2)
    norm = normalize_adjacency(adj, include_self=True)
    assert np.allclose(norm.sum(axis=1), 1.0, atol=1e-5)


def test_normalize_adjacency_isolated_node_row_zero():
    adj = build_adjacency([(0, 1)], 3)   # 节点 2 孤立
    norm = normalize_adjacency(adj, include_self=False)
    assert np.allclose(norm[2], 0.0)


def test_block_diagonal_42_from_two_21_blocks():
    single = build_hand_graph(num_hands=1)
    dual = build_hand_graph(num_hands=2)
    assert dual.shape == (42, 42)
    assert np.allclose(dual[:21, :21], single)
    assert np.allclose(dual[21:, 21:], single)
    assert np.allclose(dual[:21, 21:], 0)   # 跨块无边
    assert np.allclose(dual[21:, :21], 0)


def test_block_diagonal_generic():
    block = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    out = build_block_diagonal_graph(block, 3)
    assert out.shape == (6, 6)
    assert np.allclose(out[:2, :2], block)
    assert np.allclose(out[2:4, 2:4], block)
    assert np.allclose(out[0, 4], 0)


def test_hand_graph_matches_connections():
    adj = build_hand_graph(num_hands=1)
    n_edges = sum(1 for a, b in HAND_CONNECTIONS if adj[a, b] == 1)
    assert n_edges == len(HAND_CONNECTIONS)
    assert adj.sum() == 2 * len(HAND_CONNECTIONS)  # 对称


def test_invalid_num_hands_raises():
    with pytest.raises(ValueError):
        build_hand_graph(num_hands=0)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_graphs.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.core.graphs'`）

- [ ] **Step 3: 实现 `src/signbridge/core/graphs.py`**

```python
"""骨架图工具：邻接矩阵构造与归一化。

模型只认识邻接矩阵，不认识「手/姿态」——图来源与本领域解耦。
扩展位：人体姿态图（POSE_CONNECTIONS）、手+姿态联合图用同一工具族组合。
"""

import numpy as np

from signbridge.core.landmarks import HAND_CONNECTIONS


def build_adjacency(connections, num_nodes: int) -> np.ndarray:
    """边表 (a,b) 列表 → V×V 对称 0/1 邻接矩阵。"""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for a, b in connections:
        if not (0 <= a < num_nodes and 0 <= b < num_nodes):
            raise ValueError(f"边 ({a},{b}) 超出节点范围 0..{num_nodes - 1}")
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    return adj


def normalize_adjacency(adj, include_self: bool = True) -> np.ndarray:
    """对称归一化：可选加自环，返回 D^-1/2 A_hat D^-1/2。

    include_self=True：A_hat = A + I（图卷积标准做法）。
    孤立节点行保持 0。
    """
    a = np.asarray(adj, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("adjacency 必须是方阵")
    if include_self:
        a = a + np.eye(a.shape[0], dtype=np.float32)
    degree = a.sum(axis=1)
    d_inv_sqrt = np.zeros_like(degree)
    mask = degree > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(degree[mask])
    return d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]


def build_block_diagonal_graph(block_adjacency, num_blocks: int) -> np.ndarray:
    """分块对角图：多手/多实例泛化（跨块无边）。"""
    block = np.asarray(block_adjacency, dtype=np.float32)
    if block.ndim != 2 or block.shape[0] != block.shape[1]:
        raise ValueError("block_adjacency 必须是方阵")
    if num_blocks < 1:
        raise ValueError("num_blocks 必须 >= 1")
    n = block.shape[0]
    out = np.zeros((n * num_blocks, n * num_blocks), dtype=np.float32)
    for i in range(num_blocks):
        s = i * n
        out[s:s + n, s:s + n] = block
    return out


def build_hand_graph(num_hands: int = 1) -> np.ndarray:
    """基于 HAND_CONNECTIONS 构造手部图（原始 0/1 邻接，模型内部归一化）。

    num_hands=1 → 21×21；num_hands=2 → 42×42 分块对角。
    """
    if num_hands < 1:
        raise ValueError("num_hands 必须 >= 1")
    single = build_adjacency(HAND_CONNECTIONS, 21)
    return build_block_diagonal_graph(single, num_hands)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_graphs.py -v`
Expected: 10 passed

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/core/graphs.py tests/test_graphs.py
git commit -m "feat: 骨架图工具（邻接构造/归一化/多手分块图）"
```

---

### Task 2: 模型协议 `models/protocol.py`（TDD）

**Files:**
- Create: `src/signbridge/models/__init__.py`
- Create: `src/signbridge/models/protocol.py`
- Test: `tests/test_protocol.py`

- [ ] **Step 1: 写失败测试 `tests/test_protocol.py`**

```python
import torch.nn as nn

from signbridge.models.protocol import SkeletonClassifier


def test_protocol_is_nn_module_compatible():
    assert issubclass(SkeletonClassifier, nn.Module)


def test_protocol_exposes_interface():
    assert hasattr(SkeletonClassifier, "forward")
    assert hasattr(SkeletonClassifier, "predict")
    assert hasattr(SkeletonClassifier, "num_classes")
    assert hasattr(SkeletonClassifier, "num_nodes")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.models.protocol'`）

- [ ] **Step 3: 实现 `src/signbridge/models/protocol.py` 与 `__init__.py`**

`src/signbridge/models/protocol.py`:
```python
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
```

`src/signbridge/models/__init__.py`:
```python
"""signbridge.models: 骨架时序分类模型组件（可插拔）。"""

from signbridge.models.protocol import SkeletonClassifier

__all__ = ["SkeletonClassifier"]
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python -m pytest tests/test_protocol.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/models/ tests/test_protocol.py
git commit -m "feat: SkeletonClassifier 整模型级协议"
```

---

### Task 3: ST-GCN 实现 `models/stgcn.py`（TDD）

**Files:**
- Create: `src/signbridge/models/stgcn.py`
- Test: `tests/test_stgcn.py`

- [ ] **Step 1: 写失败测试 `tests/test_stgcn.py`**

```python
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from signbridge.core.graphs import build_hand_graph
from signbridge.models.stgcn import STGCN


def _adj():
    return build_hand_graph(num_hands=1)


def test_forward_shape_single_hand():
    model = STGCN(num_classes=5, adjacency=_adj())
    out = model(torch.randn(2, 3, 64, 21))
    assert out.shape == (2, 5)


def test_forward_shape_two_hands():
    model = STGCN(num_classes=5, adjacency=build_hand_graph(num_hands=2))
    out = model(torch.randn(2, 3, 64, 42))
    assert out.shape == (2, 5)


def test_backward_gradients_exist():
    model = STGCN(num_classes=5, adjacency=_adj())
    y = torch.tensor([1, 3])
    loss = F.cross_entropy(model(torch.randn(2, 3, 64, 21)), y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"参数 {name} 无梯度"


def test_adaptive_false_forward():
    model = STGCN(num_classes=5, adjacency=_adj(), adaptive=False)
    out = model(torch.randn(2, 3, 64, 21))
    assert out.shape == (2, 5)
    assert not any("B" in n for n, _ in model.named_parameters())


def test_custom_config_forward():
    model = STGCN(num_classes=3, adjacency=_adj(),
                  channels=(32, 64, 128), strides=(1, 2, 2), dropout=0.1)
    out = model(torch.randn(2, 3, 64, 21))
    assert out.shape == (2, 3)


def test_predict_returns_argmax_indices():
    model = STGCN(num_classes=5, adjacency=_adj())
    model.eval()
    pred = model.predict(torch.randn(4, 3, 64, 21))
    assert pred.shape == (4,)
    assert pred.dtype == torch.int64
    assert pred.min() >= 0 and pred.max() < 5


def test_short_t_raises():
    model = STGCN(num_classes=5, adjacency=_adj(), kernel_size=9)
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 5, 21))   # T=5 < kernel_size=9


def test_normalized_adjacency_buffer():
    model = STGCN(num_classes=5, adjacency=_adj())
    a = model.blocks[0].gcn.adjacency
    assert a.shape == (21, 21)
    assert torch.allclose(a, a.T, atol=1e-6)      # 对称
    assert a.sum(dim=1).max() <= 1.0 + 1e-5       # 行和 ≤ 1


def test_wrong_node_count_raises():
    model = STGCN(num_classes=5, adjacency=_adj())
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 64, 20))          # V=20 != 21


def test_channels_strides_mismatch_raises():
    with pytest.raises(ValueError):
        STGCN(num_classes=5, adjacency=_adj(),
              channels=(64, 64), strides=(1, 1, 1))
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python -m pytest tests/test_stgcn.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'signbridge.models.stgcn'`）

- [ ] **Step 3: 实现 `src/signbridge/models/stgcn.py`**

```python
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
```

- [ ] **Step 4: 更新 `src/signbridge/models/__init__.py` 并跑测试**

```python
"""signbridge.models: 骨架时序分类模型组件（可插拔）。"""

from signbridge.models.protocol import SkeletonClassifier
from signbridge.models.stgcn import STGCN

__all__ = ["SkeletonClassifier", "STGCN"]
```

Run: `python -m pytest tests/test_stgcn.py tests/test_protocol.py -v`
Expected: 12 passed（stgcn 10 + protocol 2）

- [ ] **Step 5: 提交**

```bash
git add src/signbridge/models/ tests/test_stgcn.py
git commit -m "feat: ST-GCN 参数化实现（GraphConv/TemporalConv/Block 模块化）"
```

---

### Task 4: 导出 + README + 版本 0.4.0 + 全量验证

**Files:**
- Modify: `pyproject.toml`、`src/signbridge/__init__.py`（0.3.0 → 0.4.0）
- Modify: `src/signbridge/core/__init__.py`（导出 graphs）
- Modify: `README.md`

- [ ] **Step 1: 版本 bump**

`pyproject.toml`: `version = "0.4.0"`
`src/signbridge/__init__.py`: `__version__ = "0.4.0"`

- [ ] **Step 2: `core/__init__.py` 追加导出**

```python
from signbridge.core.graphs import (
    build_adjacency,
    build_block_diagonal_graph,
    build_hand_graph,
    normalize_adjacency,
)
```

`__all__` 追加四个函数名。

- [ ] **Step 3: 顶层 `__init__.py` 追加**

```python
from signbridge.core.graphs import build_hand_graph
from signbridge.models.stgcn import STGCN
```

`__all__` 追加 `"build_hand_graph"`, `"STGCN"`。

- [ ] **Step 4: README 更新**

- 版本描述改 0.4.0：加入「ST-GCN 模型组件（可插拔协议）」
- API 速览表追加两行：

```markdown
| `signbridge.core.graphs` | `build_adjacency` / `normalize_adjacency` / `build_block_diagonal_graph` / `build_hand_graph(num_hands=1\|2)`（图工具，姿态图扩展位） |
| `signbridge.models` | `SkeletonClassifier` 协议（整模型级可插拔）+ `STGCN(num_classes, adjacency, channels, strides, kernel_size, adaptive, dropout)` |
```

- 新增章节「ST-GCN 模型组件（0.4.0）」：

```markdown
## ST-GCN 模型组件（0.4.0）

骨架时序分类模型，消费时序缓冲输出（`HandSequence.data`，`(T,21,3)` 腕点归一化）：

```python
import torch
from signbridge import STGCN, build_hand_graph

adj = build_hand_graph(num_hands=1)        # 21 节点单图；num_hands=2 → 42 双手分块图
model = STGCN(num_classes=10, adjacency=adj)  # 经典 9 层参数化

x = torch.randn(2, 3, 64, 21)              # (N, C=xyz, T, V)
logits = model(x)                          # (2, 10)
pred = model.predict(x)                    # (2,) 类别索引
```

- 输入 `(N, C, T, V)`；`num_nodes` 由邻接矩阵推导（换姿态图无需改模型）
- 模块化：`GraphConv`（图卷积 + 自适应图残差 B）/ `TemporalConv` / `STGCNBlock`（残差+BN）
- **可插拔**：任何模型实现 `SkeletonClassifier` 协议（`forward` + `predict`）即可替换 ST-GCN
- 训练管线（数据张量化、训练循环、数据集）为后续步骤
```

- [ ] **Step 5: 全量测试 + 公共 API 验证**

Run: `python -m pytest -v`
Expected: 全部通过（90 + graphs 10 + protocol 2 + stgcn 10 = 112）

Run: `python -c "from signbridge import STGCN, build_hand_graph, SkeletonClassifier; print('ok')"`
Expected: `ok`

- [ ] **Step 6: 冒烟（合成数据前向 + 一步反向）**

Run:

```bash
python -c "
import torch, torch.nn.functional as F
from signbridge import STGCN, build_hand_graph

model = STGCN(num_classes=5, adjacency=build_hand_graph(num_hands=2))
x = torch.randn(4, 3, 64, 42)              # 双手骨架序列
y = torch.tensor([0, 2, 1, 3])
loss = F.cross_entropy(model(x), y)
loss.backward()
print('loss =', loss.item())
print('grad B[0,0] =', model.blocks[0].gcn.B.grad.flatten()[0].item())
"
```

Expected: 打印 loss 数值与自适应图残差 B 的梯度（图结构可学习验证）

- [ ] **Step 7: 提交**

```bash
git add pyproject.toml src/signbridge/ README.md
git commit -m "release: 0.4.0 ST-GCN 模型组件（可插拔协议 + 参数化实现）"
```

- [ ] **Step 8: 完成声明**

向用户报告：交付物、测试结果、冒烟输出（自适应图 B 梯度存在）、扩展位（换模型实现协议 / 姿态图接入 / 训练管线后续）。
