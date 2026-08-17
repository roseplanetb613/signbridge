# 会话延续提示词（新会话直接粘贴）

你正在继续 SignBridge 项目（中文手语翻译：Qt + PyTorch + MediaPipe + ST-GCN）。工作目录 `E:\SignBridge`，用中文回复。以下是项目现状与任务，先读这些文件再动手。

## 项目背景
- 流程：MediaPipe 手势/姿态提取 → 时序段数据集 → ST-GCN（STGCNCTC / FusionSTGCNCTC）CTC 训练 → WER 评估 → Qt 可视化
- 数据：`data/CE-CSL`（4973 train/515 dev/500 test 视频，12 手语者 A-L）+ `data/SpreadTheSign中文手语词汇集`（1098 词视频）
- 已提取特征：`data/dataset/{train,dev,test}.npz`（hand 42 节点，含 detection_rates/translators/videos 元数据）+ `_pose.npz`（33 点）+ `_roi.npz`（JPEG）+ vocab.npz（3836 词）
- GitHub：roseplanetb613/signbridge（私有）。push 前需代理 127.0.0.1:7893 运行中，否则 fatal 连接失败。data/、checkpoints/、reports/ 中 results.json 已提交，其余按 .gitignore
- 测试：`python -m pytest`（当前 167 passed；裸 pytest 是 Anaconda 的，别用）
- 环境：Windows + Anaconda Python，16GB RAM（训练时用户需关 WSL/浏览器等），GPU 8GB（用户正用 GPU 跑融合训练——分析推理请用 --device cpu，勿抢显存）
- 用户偏好：质量优先（1080p full pose）、命令自己跑（带进度条+断点续跑）、每完成一步推 GitHub、WER 评估标准：可用演示 ≈0.4-0.5，商用 ≈0.15-0.2

## 当前进度（全部已提交）
- 0.1.0 手关键点提取（HandDetector refine_roi、防抖、OneEuro 平滑）→ 0.2/0.3 时序缓冲+匹配 → 0.4 ST-GCN+SkeletonClassifier 协议 → 0.5 STGCNCTC+CTC 训练
- 骨架 baseline：dev WER 0.740（checkpoints/best.pt，词表 1321，min_count=3 过滤）
- FusionSTGCNCTC（三流：hand 9 层 STGCNBlock + pose 4 层 + ROI ResNet18，concat 1024 → CTC 头）已实现，`checkpoints/fusion_best.pt` 训练中
- 词级 trigram LM 重打分（train_lm.py/lm_score.py，--lm 集成）已实现；骨架模型上验证无效（0.762→0.786），保留代码待融合模型复测
- WER 分桶分析（scripts/analyze/wer_buckets.py，commit 5cda287）：句长/signer/词频三桶 + 词级 S/D/I 对齐 + 图表。关键发现：①删除错误主导（模型欠预测：hyp 均长 2.1-2.4 词 vs ref 4.5-5.1，34% 样本 ≤1 词）②低频词错误率 99.5%（train 3836 词中 2515 词仅出现 1-2 次）③句长影响温和 ④signer 差异大（E/G 好，I/H 差但样本仅 9）
- 解码长度偏置已实现（decoding.py `length_bonus` 参数 + STGCNCTC.beam_decode 透传 + wer_buckets.py --length-bonus，commit 待推）：每输出一词乘 (1+bonus) 缓解欠预测，tests 已加 3 个

## 正在进行
1. **长度偏置实验已完成**（commit 待推）：扫描结果 bonus=1.0 最优，WER 0.762→0.749（dev，beam=5），del 1216→930（-23%），hyp 均长 2.09→2.77；bonus≥2.0 劣化（噪声插入）。结论：欠预测部分可解码侧缓解，根本瓶颈是低频词长尾。融合模型评估时复测 bonus=1.0
2. **融合训练**（用户机器上跑）：`python scripts/train/train_fusion.py --augment --beam-width 5`，epoch 1-3 WER 0.954/0.938/0.949（正常波动），~1-1.5h/epoch，30 epochs≈30-45h，断点自动恢复（fusion_latest.pt > fusion_best.pt）

## 下一步任务（按优先级）
1. push 长度偏置 commit（先确认代理 127.0.0.1:7893）：改动是 decoding.py length_bonus、stgcn_ctc.py beam_decode 透传、wer_buckets.py --length-bonus、test_decoding.py +3、_scan_length_bonus.py、worklog
2. 融合训练完成后（用户告知）：跑 `python scripts/train/train_fusion.py --eval-only --checkpoint checkpoints/fusion_best.pt` 对比骨架 baseline 0.740；跑 wer_buckets.py 三桶对比（看欠预测/长尾是否改善，--length-bonus 1.0 复测）；LM 重打分复测（--lm 参数）；报告+push
3. **注意**：wer_buckets.py 目前只支持 STGCNCTC 骨架模型（硬编码 build_hand_graph）；要分析融合模型需先扩展 --model-type（fusion 三流加载 FusionSTGCNCTC + FusionDataset 数据加载），或给 train_fusion.py 加 --length-bonus 参数（该文件用户正在运行，改完等训练结束再让用户重启验证）
4. 候选优化（用户选过 B，曾推荐 C）：C 实时翻译演示（Qt+best.pt 摄像头→分段→识别→文字）、D 可学习嵌入词识别（SpreadTheSign 词识别）、E 训练节奏优化、CSL-Daily 数据扩充（解决低频词长尾）

## 关键技术点（别踩坑）
- np.load 是惰性的，真实耗时在数组访问处；npz object 数组反序列化后要显式 float32 转换
- Windows spawn：后台任务运行中不要移动/编辑脚本文件；DataLoader 用 num_workers=0
- mediapipe 日志抑制：GLOG_minloglevel=2 + ABSL_MIN_LOG_LEVEL=2（脚本里已设）
- 提取/训练脚本已有 chunked pool（每 400 视频重建）、batch 断点（每 100 batch）、tqdm 进度条——别回退
- 命令用 PowerShell：不支持 &&（用 ; 分隔）；python 用 `python -m pytest` 跑测试
- 模型输入格式：hand (N,3,T,42)、pose (N,3,T,33)、roi (N,T,3,112,112)，CTC 输出 (N,T'=32,K+1)
- 训练脚本不能动：用户正在用 train_fusion.py 跑着（Windows spawn 按路径重载主脚本）
