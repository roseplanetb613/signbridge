# STGCNCTCTF 训练梯度 NaN 完整排查记录（Postmortem）

> 项目：SignBridge 中文手语翻译（Qt + PyTorch + MediaPipe + ST-GCN）
> 日期：2026-08-18 · 定位人：会话调试（复现 → 二分定位 → 一行修复）
> 结论先行：**元凶是 SeqKD（KL 蒸馏 loss）对 ref 分支的梯度在 logits 极端值下产生 NaN**；修复 = ref 侧 detach（一行），且符合知识蒸馏语义。

---

## 1. 背景：骨架版 TFNet（方案 A）

为复现 TFNet 论文（arXiv 2409.11960，CE-CSL dev/test WER 42.1%/41.9%），在骨架模态实现了增量版：

```
blocks (9 层 STGCN) → 节点均值 → f (N, 256, T'=32)
  ├─ 时域: head_t(f) → logits_t          ← 由 best.pt（STGCNCTC，0.740）迁移初始化
  ├─ 频域: |FFT(f, dim=T')| → 1D Conv+GroupNorm+ReLU → head_f → logits_f
  └─ 融合: f + f_freq → head_fusion → logits_fusion   （评估用）
```

训练 loss（7-loss 的骨架化）：

```
loss = CTC(logits_t) + CTC(logits_f) + CTC(logits_fusion)
     + 25 × SeqKD(prediction=logits_f, ref=logits_t)
```

SeqKD 实现（官方 TFNet 同款，`DataProcessMoudle.py`）：

```python
pred = F.log_softmax(prediction_logits[:, :, 1:] / T, dim=-1)  # 学生，去 blank
ref  = F.softmax(ref_logits[:, :, 1:] / T, dim=-1)             # 老师
loss = KLDivLoss(reduction='batchmean')(pred, ref) * T * T
```

训练环境：Windows + PyTorch 2.x（GPU RTX 8GB）+ numpy 2.5.1，lr 1e-3、batch 32、55 epochs、时间/空间增强。

---

## 2. 现象：三轮排障

### 第一轮：loss 全 NaN（前向污染型）

```
[警告] epoch 1 batch 63: loss 非有限（nan），跳过   ← 之后每个 batch 都 nan
epoch 1: train_loss nan  dev_loss nan  dev_WER 7.891
```

特征：**loss 本身 NaN**（前向已污染），从某个 batch 起**持续**。dev_WER 7.891 是 NaN 权重下 argmax 乱输出。

**初步排查（全部排除）**：
- 数据含 NaN？→ 扫描 train/dev 全部 4972/507 段 `np.isfinite`：**0 段异常**
- CPU 64 段复现？→ 无 NaN，loss 正常
- GPU 32 段复现？→ 无 NaN，loss 正常

**当时归因**：频域分支 BatchNorm。`|FFT|` 幅度谱是结构化输入（DC 分量大、高频趋 0），某 batch 某通道方差为 0 → BN 除 0 → inf → 梯度 NaN → 权重污染 → 持续 NaN。

**处理**：① `freq_conv` 的 BN → **GroupNorm**（不依赖 batch 统计）；② 加官方 TFNet 同款防护：`loss` 非有限 → 跳过该 batch。

### 第二轮：梯度 NaN（loss 有限，模型静默失效）

```
epoch 1: batch 37 起 "梯度非有限，跳过 step"（持续）
epoch 2: train_loss 0.0000   ← 全部 batch 被跳过，模型完全没学
epoch 3: train_loss 120.1    ← 部分 batch 正常
epoch 4: train_loss 0.0000
dev_WER 0.974~0.982（不下降）
```

特征：**loss 有限（133-144），backward 后梯度 NaN**——比第一轮更隐蔽：训练"看起来在跑"，实际几乎不更新。

---

## 3. 复现：目标环境（GPU）完整循环复现

CPU dry-run（60 段 4 batch）无法复现——**数值问题必须在对的环境（GPU + 全量数据 + 增强随机路径）复现**。

写了 `grad_nan_repro.py`：GPU + 全量 4882 段 + 与训练脚本完全一致的增强/迁移/7-loss/优化器，逐 batch 检查每个参数的梯度。

**复现成功，拿到三个关键事实**：

```
step 0:  loss 281.7（正常）
step 11: 坏参数 ['blocks.0.gcn.B', 'blocks.0.gcn.conv.weight', ...]  ← 之后持续
```

1. **坏梯度集中在 `blocks.0.*`（第一层 STGCN）**——NaN 不是频域分支本地产生，而是从深层 loss 链路回传到底层主干
2. **梯度是 NaN 而非 Inf**（`nan=True, inf=False`）——不是"梯度爆炸"（数值溢出），而是**前向图中某算子的 backward 出现 0/0 型未定义值**
3. **模型参数全程 isfinite（从未污染）**——跳过逻辑有效，也证明 NaN 在**每个受影响 batch 独立复现**（不是一次污染后的连锁反应）

---

## 4. 定位：排除法 + 二分

| 假设 | 验证 | 结果 |
|---|---|---|
| 数据含 NaN/Inf | 全量扫描 4972 段 | ❌ 0 段异常 |
| 增强路径产生 NaN | `nan_to_num` 防御后依旧 | ❌ |
| `torch.abs(FFT)` 复数梯度 | 单元测试：z=0 处梯度 | ❌ 有限（PyTorch 返回 0） |
| BN 除 0（第一轮归因） | GroupNorm 替换后 | ❌ 问题转移为梯度 NaN |
| 梯度爆炸（Inf） | 梯度值检查 | ❌ 是 NaN 非 Inf |
| 参数污染连锁 | 全程 isfinite 检查 | ❌ 参数干净 |
| cufft（GPU FFT）backward | CPU/GPU 对照 + 排除 | ❌（bad 参数在 blocks.0 而非频域） |
| **SeqKD 对 ref 的梯度** | **`kld_fn(logits_f, logits_t.detach())` 后跑 120 batch** | ✅ **全程无坏梯度，loss 281.9→83-98 正常下降** |

---

## 5. 根因分析

**触发条件链**：

1. 时域路径从 `best.pt` 迁移初始化，其输出 **`logits_t` 的极端值高达 max ≈ 100.83**（迁移模型对增强后输入分布略有偏移，输出 logits 尺度大——本身不致命，CTC 对 logits 尺度鲁棒）
2. `SeqKD.forward` 对 ref 分支做 `F.softmax(ref_logits / T)` 并参与反向
3. **KLDivLoss 反向穿过 ref 分支（logits_t → head_t → blocks）时，在 float32 数值路径下产生 0/0 型 NaN**（logits 极端值附近 softmax/exp 的导数路径）
4. 该 NaN 梯度沿共享主干回传，污染 `blocks.0` 全部参数梯度 → 训练防护触发跳过 → 模型静默失效

**为什么 CPU 复现不了**：CPU 与 GPU 的算子数值实现（数学库、约简顺序、fp32 路径）不同，极端值附近的舍入行为不同——**数值 bug 必须在对的环境复现**。

**为什么第一轮（BN 版）是 loss NaN、第二轮（GroupNorm 版）是梯度 NaN**：GroupNorm 修复了频域分支的前向 inf，暴露了深层隐藏的 backward NaN——两层问题叠加。

---

## 6. 解决方案：一行修复 + 语义佐证

```python
# 修复前
loss += args.kld_weight * kld_fn(logits_f, logits_t)
# 修复后
loss += args.kld_weight * kld_fn(logits_f, logits_t.detach())
```

**语义合理性**：SeqKD 是知识蒸馏——`ref`（时域分支）是"老师"，`prediction`（频域分支）是"学生"。蒸馏的标准做法就是**老师不收学生梯度**（teacher 输出 detach）。修复既消除 NaN，又比原写法更符合论文本意。

**配套防护（保留，防御未来偶发）**：
- `loss` 非有限 → 跳过 batch（官方 TFNet 同款逻辑）
- 梯度非有限 → 跳过 step（`any(not isfinite(p.grad))`）
- 增强后 `np.nan_to_num`
- 警告限频（每 epoch ≤3 条）

**验证**：GPU 120 batch 全程无坏梯度、loss 正常下降（281.9 → 83-98）；单元测试 6 个全过。

---

## 7. 方法论沉淀：数值问题调试清单

1. **先区分"loss NaN"与"梯度 NaN"**：前者是前向污染（传播型），后者是算子 backward 数值问题（每步独立复现型）——两者的排查方向完全不同
2. **区分 Inf 与 NaN**：Inf = 溢出/爆炸（梯度裁剪、归一化方向）；NaN = 0/0 型未定义（某个算子的奇点）——打印 `torch.isnan`/`torch.isinf` 分开看
3. **参数 vs 梯度污染检查**：skip 后问题是否持续？参数一直干净 = 独立复现的算子问题，不是连锁反应
4. **坏梯度参数的位置是定位信标**：集中在最深层（blocks.0）= NaN 从 loss 链路回传；集中在某一分支 = 该分支本地问题
5. **CPU 正常 ≠ GPU 正常**：数值实现（cufft/数学库/fp32 路径）不同——**必须在目标环境复现**
6. **loss 组件二分**：对可疑组件做 detach 实验（kld 的 pred/ref 分别 detach）——一次实验就能锁定
7. **防护先行、根因后置**：先加 skip/防御让训练不崩溃，再逐步定位根因；官方代码里的防护逻辑（TFNet 的 `if loss is nan: continue`）是重要线索
8. **排查工具**：逐参数梯度检查脚本（named_parameters + isfinite）、全量数据复现循环（勿用小样本——样本路径可能触发不了）

---

## 8. 遗留与后续

- `logits_t` 极端值（max ~100）偏大，来自迁移模型对增强后输入的分布偏移——训练已正常，观察收敛即可；若影响稳定可加 logits 温度/缩放
- 第一轮的 BN→GroupNorm 改动保留（对 `|FFT|` 结构化输入更稳）
- **完整 55 epochs 训练结果待跑**：预期时域路径继承 0.740 起点，频域分支 + 融合头 + SeqKD 贡献增量，目标突破骨架天花板
- 本案例应同步进 `docs/session-continuation-prompt.md` 的"别踩坑"清单（蒸馏类 loss 的 ref 梯度数值风险）

---

## 附：关键文件

- `src/signbridge/models/stgcn_ctc_tf.py`：STGCNCTCTF 模型 + SeqKD
- `scripts/train/train_full_tf.py`：训练脚本（含防护逻辑）
- 官方参考：`github.com/woshisad159/TFNet`（Net.py / Train.py / DataProcessMoudle.py）
