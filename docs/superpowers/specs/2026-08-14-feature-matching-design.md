# SignBridge 特征增强帧间匹配设计（0.3.0）

日期：2026-08-14
状态：设计已获用户确认（融合策略 / 特征内容 / 判定器协议化 / 分层匹配）

## 1. 背景与目标

现状：`HungarianMatcher` 纯位置（质心）匹配 + Buffer ID 生命周期，两个已知局限：

1. **交叉错配**：双手位置重合/靠近时，匈牙利只能靠距离区分
2. **丢失恢复只看位置**：手消失后从画面别处重新出现时，无法确认是否同一只手

目标：**特征增强的分层匹配**——位置匹配为主（正常跟踪路径不变），手形特征
用于丢失恢复判定（位置未匹配时用特征置信度判断是否同一 ID）。全部可插拔：
特征提取、同一性判定均为协议，未来可替换为 transformer / GCN 学习模型。

## 2. 关键决策（用户已确认）

| 决策点 | 选择 |
| --- | --- |
| 融合策略 | 分层：位置匈牙利为主 + 特征恢复（不做统一联合代价） |
| 特征内容 | 归一化距离矩阵（210 维，旋转/尺度/平移不变） |
| 同一性判定 | 协议化 `FeatureVerifier`（未来 transformer/GCN 接入点） |
| 匹配器 | `FeatureHungarianMatcher`（组合 extractor 特征 + verifier 判定） |
| 兼容 | 纯位置 `HungarianMatcher` 保留，零回归路径 |

## 3. 新模块 `core/features.py`

```python
class FeatureExtractor(Protocol):
    """特征提取协议：21×3 点阵 → 特征向量（可插拔）。"""

    def extract(self, pts: np.ndarray) -> np.ndarray: ...


class FeatureVerifier(Protocol):
    """同一性判定协议：两特征向量 → 置信度 ∈ [0,1]（1=同一只手）。

    ★ 学习型模型的接入点：transformer 双塔 / GCN 相似度网络
      训练后实现此协议即可无缝替换默认判定。
    """

    def verify(self, feature_a: np.ndarray, feature_b: np.ndarray) -> float: ...


class HandShapeFeature:
    """归一化距离矩阵特征（210 维）。

    腕点归一化 → 点间欧氏距离矩阵 → 上三角向量 → 除以平均距离。
    旋转不变、尺度不变、平移无关。
    """


class DistanceFeatureVerifier:
    """L2 距离 → 置信度：exp(-d² / 2σ²)（高斯核，σ 默认 0.3）。"""

    def __init__(self, sigma: float = 0.3): ...
```

## 4. 匹配协议升级 `core/matching.py`

```python
@dataclass(frozen=True)
class HandDescriptor:
    """参与帧间关联的手单元：位置 + 特征。feature 可为 None（关闭特征）。"""

    centroid: np.ndarray   # (2,)
    feature: np.ndarray | None  # (D,)

class Matcher(Protocol):
    """协议 v2：输入升级为 HandDescriptor 序列（feature 由 Buffer 提取注入）。"""

    def match(self, current: Sequence[HandDescriptor],
              previous: Sequence[HandDescriptor]) -> Matching: ...

class HungarianMatcher:
    """纯位置匹配（向后兼容路径）：只用 centroid，忽略 feature。"""

class FeatureHungarianMatcher:
    """分层匹配（新默认）：
    第一层 位置匈牙利（distance_threshold，现有逻辑）；
    第二层 特征恢复（见下）。
    """
    def __init__(self, feature_verifier: FeatureVerifier | None = None,
                 confidence_threshold: float = 0.85,
                 distance_threshold: float = 0.15): ...
```

### 第二层：特征恢复（分层逻辑）

```
第一层 位置匈牙利匹配后：
    ├── 匹配对 → 正常（进入 matched）
    └── 位置未匹配的 current 手 × 位置未匹配的 previous 轨迹（含 lost tracks）
第二层 对每对（双方 feature 均非 None）：
    conf = verifier.verify(feature_cur, feature_prev)
    conf >= confidence_threshold → 判定同一只手 → 进入 matched（恢复原 ID）
    （贪心按置信度降序，每边最多匹配一次，避免一对多）
其余 → unmatched_current / unmatched_previous
```

- 置信度语义：特征距离越小置信度越高；`confidence_threshold` 决定"是否同一 ID"
  的保守程度（越高越严）
- lost track 保留最后特征参与第二层——**跨位置恢复**（画面另一侧同手形 → 恢复 ID）

## 5. `HandSequenceBuffer` 集成

```python
buf = HandSequenceBuffer(
    window_size=60,
    max_lost_frames=10,
    matcher=FeatureHungarianMatcher(),        # 新默认（含 DistanceFeatureVerifier）
    feature_extractor=HandShapeFeature(),     # 新参数：特征提取（None 关闭）
    coordinate="world",
    smoother=OneEuroSmoother(),
)
```

- **职责分离**：Buffer 负责特征**提取**（可插拔 extractor，逐手注入 descriptor）；
  matcher 负责匹配**决策**（可插拔 verifier）
- `_Track` 新增 `last_feature`：匹配/新建时更新，lost 期间保留
- 正常跟踪路径（位置匹配）行为与现版本完全一致——特征只在恢复分支生效
- 传 `matcher=HungarianMatcher()` 或 `feature_extractor=None` 可退回纯位置模式

## 6. 测试策略

- `test_features.py`：
  - 输出 210 维；平移/旋转/尺度不变性（变换前后特征近似相等）
  - 不同手形（不同 seed 点阵）特征距离显著大于同手形
  - `DistanceFeatureVerifier`：距离小 → 置信度高（单调）；σ 行为；边界 [0,1]
- `test_matching.py`（升级 descriptor + 新增）：
  - 位置近：`FeatureHungarianMatcher` 行为与纯位置一致
  - 位置远 + 同手形 → 特征恢复匹配
  - 位置远 + 异手形 → 不匹配
  - 阈值之下不恢复；贪心每边最多匹配一次
- `test_tracker.py` 新增：
  - **跨位置丢失恢复**：手消失 ≤K 帧，从画面另一侧以同手形出现 → ID 恢复
  - 异手形出现 → 新 ID（特征判定正确拒绝）
- 回归：现有全部测试通过（descriptor 化接口同步更新）
- 版本 bump 0.3.0，README 更新（API 表 + 特征匹配说明）

## 7. 范围外（本步只留协议，不实现）

- transformer / GCN 同一性判定模型（实现 `FeatureVerifier` 即可接入）
- 多特征组合 `FeatureSet`（距离矩阵 + 姿态 + 颜色加权）
- `Matching` 增加 scores 置信度透出（接口已天然可扩展）

## 8. 交付物

1. `src/signbridge/core/features.py`（两个协议 + 两个默认实现）
2. `src/signbridge/core/matching.py`（HandDescriptor + 协议 v2 + FeatureHungarianMatcher）
3. `src/signbridge/hands/sequence.py`（feature_extractor 集成 + _Track.last_feature）
4. `tests/test_features.py` 新增；`test_matching.py` / `test_tracker.py` 扩展
5. README 更新；版本 0.3.0
