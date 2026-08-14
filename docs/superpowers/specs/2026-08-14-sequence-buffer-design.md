# SignBridge 时序序列缓冲组件设计（第二步）

日期：2026-08-14
状态：设计已获用户确认（两节 + 可插拔修订）

## 1. 背景与目标

手语翻译项目第二步：**时序序列缓冲 → ST-GCN 图数据**。

消费第一步的手部关键点组件（`HandFrame`），输出**按手 ID 稳定分离**的时序序列
（滑动窗口 + 腕点归一化），为 ST-GCN 训练与推理提供输入张量。

## 2. 范围

### 本步实现（In Scope）

- `core/smoothing.py`：平滑抽象协议 + `OneEuroSmoother` 默认实现（可插拔）
- `core/matching.py`：帧间匹配抽象协议 + `HungarianMatcher` 默认实现（可插拔）
- `hands/sequence.py`：`HandSequenceBuffer`（ID 生命周期 + 滑动窗口 + 间隙语义）
- `HandSequence` 数据结构
- 测试（纯 numpy，不碰摄像头）+ README 更新

### 本步不实现（Out of Scope）

- ST-GCN 模型、训练与推理
- 静态手势分类
- 人体姿态组件
- 手势分段 / 单词边界切分
- Qt 界面

## 3. 关键决策（用户已确认）

| 决策点 | 选择 |
| --- | --- |
| 多手区分 | 方案 B：完整追踪（匈牙利匹配 + ID 生命周期 + 丢失恢复） |
| 检测间隙策略 | 窗口暂停推进 + ID 保留 K 帧（默认 10 帧 ≈ 0.3s），超时回收 |
| 关键点平滑 | 可插拔协议，默认 OneEuro（运动自适应低通） |
| 帧间匹配 | 可插拔协议，默认匈牙利最小化质心距离（阈值截断） |

## 4. 数据流

```
HandFrame（每帧，来自第一步） 
        │
        ▼
HandSequenceBuffer.update(hand_frame)   （每帧调用一次）
        │
        ▼
tuple[HandSequence, ...]   （当前所有活动手，按 hand_id 升序）
```

## 5. 核心数据结构

```python
@dataclass(frozen=True)
class HandSequence:
    hand_id: int              # 追踪器分配的稳定 ID（跨帧不变，回收后不复用）
    handedness: str           # "Left" / "Right"
    data: np.ndarray          # (T, 21, 3) float32 —— 腕点(WRIST=0)原点归一化坐标
    valid_mask: np.ndarray    # (T,) bool —— 该帧该手是否有真实数据
    timestamps: np.ndarray    # (T,) 毫秒
    frame_indices: np.ndarray # (T,) 帧序号
```

- `data` 默认取 `world_landmarks`（米制 3D，不受镜头距离/手大小影响），每帧减去
  腕点坐标 → 腕点即原点。`coordinate="image"` 时取归一化图像坐标并同样腕点归一化。
- `data` 直接可作为 ST-GCN 输入 `(T, 21, 3)`。

## 6. 组件 API

```python
buf = HandSequenceBuffer(
    window_size=60,                 # 滑动窗口帧数（≈2s @30fps）
    max_hands=2,
    max_lost_frames=10,             # ID 失联保留帧数，超时回收
    matcher=HungarianMatcher(distance_threshold=0.15),   # 可插拔
    coordinate="world",             # "world" | "image"
    smoother=OneEuroSmoother(),     # 可插拔；None 表示不平滑
)
sequences = buf.update(hand_frame)  # -> tuple[HandSequence, ...]
buf.reset()                         # 清空所有轨迹与窗口
buf.left_hand_id / buf.right_hand_id  # 公开只读属性（-1 表示当前无该手）
```

## 7. 帧间匹配与 ID 生命周期（方案 B）

### 匹配协议（`core/matching.py`）

```python
@dataclass(frozen=True)
class Matching:
    matched: tuple[tuple[int, int], ...]         # (当前帧索引, 上一帧轨迹索引) 对
    unmatched_current: tuple[int, ...]           # 当前帧未匹配索引
    unmatched_previous: tuple[int, ...]          # 上一帧未匹配轨迹索引

class Matcher(Protocol):
    """帧间关联协议：只做「谁跟谁」决策，不管理 ID。"""
    def match(self, current_centroids: np.ndarray,    # (N,2)
              previous_centroids: np.ndarray,         # (M,2)
              ) -> Matching: ...

class HungarianMatcher:
    """默认实现：匈牙利算法最小化质心欧氏距离，超过阈值不匹配。"""
    def __init__(self, distance_threshold: float = 0.15): ...
```

- 质心 = 21 点均值，坐标空间与 `coordinate` 一致（统一可比）
- 匈牙利算法 numpy 手写实现（~40 行），不引入 scipy 依赖

### ID 生命周期（`HandSequenceBuffer` 固定逻辑）

| 事件 | 行为 |
| --- | --- |
| 匹配对 | 续用上一帧 ID；handedness 以当前帧为准刷新 |
| 当前帧未匹配手 | 分配新 ID（自增，回收的 ID 不复用） |
| 上一帧未匹配轨迹 | `lost_count += 1`；超过 `max_lost_frames` 回收 ID 并从输出移除 |
| 手重新出现（≤K 帧内） | 匹配成功 → 恢复原 ID 继续写入序列 |

- **丢失中的轨迹继续参与匹配**：lost 中的手保留最后已知质心作为匹配候选，
  重新出现时才能恢复原 ID；超过 K 帧才从候选移除并回收
- ID 稳定性：手不出镜超过 K 帧则 ID 跨帧不变（双手交叉也能保持身份）
- 输出按 `hand_id` 升序排序（确定性）

## 8. 窗口 / 间隙语义

- `update()` 每帧调用：新帧入窗，超出 `window_size` 的旧帧滑出
- **无手 / 匹配失败帧**：窗口不推进（`data` 不追加该帧），该手 `valid_mask`
  对应位为 False，时间戳继续走；该手进入 lost 计数
- **ID 回收后**：其 `HandSequence` 从输出移除（调用方自行保存已结束序列）
- **双手全无**：缓冲空转，无输出

## 9. 平滑接口（`core/smoothing.py`，可插拔）

```python
class LandmarkSmoother(Protocol):
    def update(self, points: np.ndarray | None) -> np.ndarray | None: ...
    def reset(self) -> None: ...

class OneEuroSmoother:
    """运动自适应低通：静止时强平滑、快速运动时跟随。"""
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05,
                 d_cutoff: float = 1.0): ...
```

- 每只手独立一个平滑器实例；`None` 输入（该帧无手）→ 保持内部状态不更新
- 后续换卡尔曼 / EMA 只需实现同一协议

## 10. 测试策略（pytest，纯 numpy）

- `test_smoothing.py`：OneEuro 静态序列收敛、阶跃响应、reset 行为
- `test_tracker.py`：
  - 单手连续帧 → 同一 ID 稳定
  - 双手互换位置（交叉）→ ID 仍绑定原手（已知轨迹断言）
  - 手短暂消失 ≤ K 帧 → ID 保留；> K 帧 → 回收
  - 新手出现 → 新 ID；回收 ID 不复用
  - fake matcher 驱动 Buffer → 证明 Buffer 不依赖具体匹配实现（可插拔验证）
- `test_sequence_buffer.py`：
  - 窗口滑动：30 帧喂入 window=60 → data 长度 30；70 帧 → 长度 60
  - 无手帧不推进窗口（valid_mask 语义）
  - 腕点归一化正确性：每帧 data 的 WRIST 行 ≈ (0,0,0)
  - 双手两路序列独立、ID 排序稳定
  - fake smoother 断言被正确调用

## 11. 依赖

- 无新增运行时依赖（numpy 手写匈牙利）
- 无新增开发依赖

## 12. 交付物

1. `src/signbridge/core/smoothing.py`
2. `src/signbridge/core/matching.py`
3. `src/signbridge/hands/sequence.py`
4. `tests/test_smoothing.py` / `tests/test_tracker.py` / `tests/test_sequence_buffer.py`
5. README 第二步文档与 API 表更新

## 13. 后续步骤

- 手势分段 / 单词边界切分（消费 HandSequence）
- ST-GCN 训练与推理（消费 `HandSequence.data`）
- 人体姿态组件 `signbridge.pose`（复用 matching/smoothing 协议）
- Qt 界面
