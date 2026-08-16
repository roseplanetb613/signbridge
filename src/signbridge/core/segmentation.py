"""时序段切分工具：从帧有效性标记中提取连续有效段（core 级通用）。"""

import numpy as np


def extract_segments(valid, min_len: int = 9, merge_gap: int = 2):
    """连续有效段提取：返回 [(start, length), ...]。

    merge_gap: 有效段之间 ≤ merge_gap 帧的缝隙合并（容忍偶发漏检）。
    段长 < min_len 的段被丢弃。
    """
    n = len(valid)
    if n == 0:
        return []
    starts, ends = [], []
    in_seg = False
    for i in range(n):
        if valid[i] and not in_seg:
            starts.append(i)
            in_seg = True
        elif not valid[i] and in_seg:
            ends.append(i)
            in_seg = False
    if in_seg:
        ends.append(n)
    segs = []
    cur_start, cur_end = None, None
    for s, e in zip(starts, ends):
        if cur_start is None:
            cur_start, cur_end = s, e
        elif s - cur_end <= merge_gap:
            cur_end = e
        else:
            if cur_end - cur_start >= min_len:
                segs.append((cur_start, cur_end - cur_start))
            cur_start, cur_end = s, e
    if cur_start is not None and cur_end - cur_start >= min_len:
        segs.append((cur_start, cur_end - cur_start))
    return segs
