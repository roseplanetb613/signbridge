# SignBridge ST-GCN 模型组件设计（0.4.0）

日期：2026-08-14
状态：设计已获用户确认（抽象层次 / 范围 / 架构基准 / 图支持）

## 1. 背景与目标

第三步：**ST-GCN 图卷积模型组件**。消费时序缓冲的输出（`HandSequence.data`，
`(T, 21, 3)` 腕点归一化骨架序列），对手势/手语类别做分类。

用户已确认的要求：

1. **整模型级可插拔协议**：定义 `SkeletonClassifier` 协议，ST-GCN 为默认实现，
   未来任何模型（transformer / GCN 变体）实现同一协议即可无缝替换
2. **本步范围**：仅模型架构 + 协议接口（不做训练循环/数据集）；用户将自行微调架构
3. **架构基准**：经典 ST-GCN（Yan et al. 2018）参数化 + 自适应图开关
4. **图参数化**：邻接矩阵作为构造参数传入，模型与图解耦；
   支持多手（21 单图 / 42 双手分块图）与未来人体姿态节点

## 2. 模块结构

```
src/signbridge/
├── core/
│   └── graphs.py        # 图工具：邻接矩阵构造与归一化（core 级，hands/pose/models 共用）
└── models/              # 模型组件（新子包）
    ├── __init__.py
    ├── protocol.py      # SkeletonClassifier 协议
    └── stgcn.py         # ST-GCN 实现
```

## 3. 图工具 `core/graphs.py`

```python
def build_adjacency(connections, num_nodes) -> np.ndarray:
    """边表 (a,b) 列表 → V×V 对称 0/1 邻接矩阵。"""

def normalize_adjacency(adj, include_self=True) -> np.ndarray:
    """对称归一化：A_hat = (A + I)；返回 D^-1/2 A_hat D^-1/2。"""

def build_block_diagonal_graph(block_adjacency, num_blocks) -> np.ndarray:
    """分块对角图：多手/多实例泛化（跨块无边）。"""

def build_hand_graph(num_hands=1) -> np.ndarray:
    """基于 HAND_CONNECTIONS 构造手部图。
    num_hands=1 → 21×21；num_hands=2 → 42×42 分块对角。
    原始 0/1 邻接（模型内部做归一化）。
    """
```

- 模型只认识邻接矩阵，不认识"手/姿态"：`num_nodes = adjacency.shape[0]` 推导
- 扩展位（后续步骤）：`build_pose_graph(pose_connections)`（MediaPipe Pose 33 点）、
  手+姿态联合图（分块组合）——同一工具族

## 4. 模型协议 `models/protocol.py`

```python
class SkeletonClassifier(Protocol):   # runtime_checkable 纯协议
    """骨架时序分类模型协议（整模型级可插拔）。

    任何实现此协议的模型（ST-GCN 默认 / 未来 transformer / GCN 变体）
    可直接替换，训练/推理/部署管线无需改动。
    """

    num_classes: int
    num_nodes: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, T, V) 骨架张量 → (N, num_classes) logits。"""

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """(N, C, T, V) → (N,) 类别索引（argmax 包装 forward）。"""
```

- 纯 `Protocol`（Python 中 `nn.Module` 与 `Protocol` 多重继承运行时非法）：
  实现类继承 `nn.Module` 并实现 `forward` / `predict` 即符合协议（结构性检查）

## 5. ST-GCN 实现 `models/stgcn.py`

### 模块划分（便于微调，每个类独立可改）

```python
class GraphConv(nn.Module):
    """单层图卷积：Y = A_hat @ X @ W。

    A_hat：归一化邻接（+自环、D^-1/2 对称归一化）作为 buffer；
    adaptive=True 时叠加可学习图残差 B（A-Link Inference，Yan 2018）；
    W 为 1×1 卷积（通道变换）。
    """

class TemporalConv(nn.Module):
    """时间卷积：kernel_size 默认 9；stride 支持时间下采样。"""

class STGCNBlock(nn.Module):
    """图卷积 + 时间卷积 + 残差 + BN/ReLU/Dropout。

    残差：通道变化或 stride>1 时用 1×1 卷积投影对齐。
    """

class STGCN:
    """经典 9 层 ST-GCN，全部参数化（默认=Yan 2018 配置）。

    STGCN(num_classes, adjacency, in_channels=3,
          channels=(64,64,64,128,128,128,256,256,256),
          strides=(1,1,1,2,1,1,2,1,1),
          kernel_size=9, adaptive=True, dropout=0.5)
    """
```

### 关键设计点

- `num_nodes = adjacency.shape[0]`——不写死 21/33/42
- 图归一化在模型内部完成（`normalize_adjacency(adj, include_self=True)` → buffer）
- 全局平均池化（时间×空间）→ `nn.Linear` 分类头 → `(N, num_classes)`
- `predict(x) = forward(x).argmax(dim=1)`
- 输入校验：`T < kernel_size` 时抛 `ValueError`（友好提示，避免晦涩的卷积错误）
- 权重初始化：默认 PyTorch 初始化即可（用户微调时自行决定）

## 6. 测试策略（纯 torch CPU，无 GPU 依赖）

- `test_graphs.py`：
  - `build_adjacency`：形状 `(V,V)`、对称、对角线为 0、边表映射正确
  - `normalize_adjacency`：对称、行和 ≤ 1、`include_self` 语义
  - `build_block_diagonal_graph`：42 节点 = 两个 21 块、跨块全 0
  - `build_hand_graph(1/2)`：与 `HAND_CONNECTIONS` 一致（21 条边 + 分块）
- `test_stgcn.py`：
  - 前向形状：`(2,3,64,21) → (2,num_classes)`；双手图 `(2,3,64,42)` 前向
  - 反向传播：`loss.backward()` 后所有可学习参数梯度非 None
  - `adaptive=False` 前向正常
  - 非默认配置（不同 channels/strides）前向正常
  - `predict` 返回 argmax 类别索引
  - `T < kernel_size` 抛 `ValueError`
  - 图归一化 buffer：`A_hat` 对称、行和合理（随机图）

## 7. 范围外（本步不实现）

- 训练循环 / 数据加载 / 数据集采集
- `(T,21,3) → (C,T,V)` 张量转换工具（训练管线阶段提供）
- `build_pose_graph`（姿态组件阶段）
- checkpoint / ONNX 导出 / 推理部署

## 8. 交付物

1. `src/signbridge/core/graphs.py` + `tests/test_graphs.py`
2. `src/signbridge/models/`（`protocol.py` + `stgcn.py` + `__init__.py`）+ `tests/test_stgcn.py`
3. 导出：`core/__init__.py`、`models/__init__.py`、顶层 `__init__.py`
4. README 更新（models 章节 + API 表）；版本 **0.4.0**
