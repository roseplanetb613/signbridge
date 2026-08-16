"""把 data/extracted/segments.npz（30 段小样本）转为 train_full 标准格式。

产出 data/dataset_mini/{train,dev}.npz + vocab.npz，供 train_full 直接训练。

用法: python scripts/train/prep_mini_dataset.py [--out data/dataset_mini]
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PUNCT = set("。，？！、；：""''（）《》")


def gloss_words(gloss: str) -> list[str]:
    return [w for w in gloss.split("/") if w.strip() and w.strip() not in PUNCT]


def main() -> int:
    parser = argparse.ArgumentParser(description="小样本数据转标准格式")
    parser.add_argument("--src", type=str,
                        default="data/extracted/segments.npz")
    parser.add_argument("--out", type=str, default="data/dataset_mini")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"未找到 {src}")
        return 1
    d = np.load(src, allow_pickle=True)
    samples = [np.asarray(s, dtype=np.float32) for s in d["data"]]
    glosses = [str(g) for g in d["glosses"]]
    videos = [str(v) for v in d["videos"]]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(samples))
    n_val = max(int(len(samples) * args.val_ratio), 1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx in (("train", perm[n_val:]), ("dev", perm[:n_val])):
        np.savez_compressed(
            out_dir / f"{name}.npz",
            data=np.array([samples[i] for i in idx], dtype=object),
            glosses=np.array([glosses[i] for i in idx], dtype=object),
            videos=np.array([videos[i] for i in idx], dtype=object),
            translators=np.array([""] * len(idx), dtype=object),
            detection_rates=np.ones(len(idx)),
            avg_bboxes=np.ones(len(idx)),
            spans=np.array([(0, 0)] * len(idx), dtype=object),
        )
        print(f"[{name}] {len(idx)} 段 → {out_dir / (name + '.npz')}")

    train_glosses = [glosses[i] for i in perm[n_val:]]
    freq = Counter(w for g in train_glosses for w in gloss_words(g))
    words = [w for w, _ in freq.most_common()]
    np.savez_compressed(out_dir / "vocab.npz",
                        words=np.array(words, dtype=object))
    print(f"词表 {len(words)} 词 → {out_dir / 'vocab.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
