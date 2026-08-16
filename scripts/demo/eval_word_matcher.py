"""词识别评估：命中率 + 词间可分性 + 最易混淆词对。

数据限制：每词仅 1 个示范视频 → 无法做跨人训练/测试划分。
本评估回答：1) 检索管线正确性（自查询 top-k） 2) 词间可分性
（异类最小距离 vs 0——若异类距离远大于 0 则真实场景可识别）
3) 最易混淆的词对（可分性最差的词）。

用法: python scripts/demo/eval_word_matcher.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from word_matcher import build_templates, segment_feature  # noqa: E402


def main() -> int:
    npz = Path("data/dataset/spreadthesign.npz")
    if not npz.exists():
        print("未找到 spreadthesign.npz")
        return 1
    d = np.load(npz, allow_pickle=True)
    n = len(d["data"])
    words = [str(w) for w in d["words"]]
    feats = [segment_feature(np.asarray(d["data"][i], dtype=np.float32))
             for i in range(n)]
    mat = np.stack(feats)
    print(f"样本 {n}，词表 {len(set(words))}")

    # 注意：每词基本只有 1 个独立示范视频（basic/common 同名多为同源副本），
    # 无法在数据内做真实的训练/测试划分——以下指标说明可分性与管线，
    # 真实泛化准确率只能用摄像头/新视频实测。

    # 1) 自查询命中（管线性验证，查询含自身 → 必然命中但验证管线）
    dists = np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)
    order = np.argsort(dists, axis=1)
    hits = {1: 0, 3: 0, 5: 0}
    for i in range(n):
        for k in hits:
            topk = order[i, :k]
            if words[i] in {words[j] for j in topk}:
                hits[k] += 1
    print("\n排除自身后 top-k 命中率（仅 41 个同名词可命中，其余词不在库中）:")
    for k in (1, 3, 5):
        print(f"  top-{k}: {hits[k]}/{n} = {hits[k] / n:.1%}")

    # 2) 可分性：每样本最近异类距离（>0 越大词间越可分）
    nearest_other = dists.min(axis=1)
    print(f"\n可分性: 最近异类距离中位 {np.median(nearest_other):.3f} "
          f"（P10 {np.percentile(nearest_other, 10):.3f}）")
    print("  异类距离越大 → 词与词越可分（特征质量好）")
    print("  ⚠️ 数据内无法测同类距离（每词 1 独立样本）→ 真实识别准确率")
    print("     需摄像头/新视频实测")

    # 3) 最易混淆词对（最近异类距离最小的样本）
    print("\n最易混淆（最近异类距离最小的 10 个样本）:")
    worst = np.argsort(nearest_other)[:10]
    for i in worst:
        j = int(np.argmin(dists[i]))
        print(f"  {words[i]:>6} ↔ {words[j]:>6}  距离 {nearest_other[i]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
