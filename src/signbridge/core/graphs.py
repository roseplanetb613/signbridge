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


def build_hand_pose_graph(hand_adjacency, pose_adjacency) -> np.ndarray:
    """双手图 + 人体姿态图 → 分块对角拼接图（hand 在前、pose 在后）。

    用于双流骨架拼接（hand 42 + pose 33 = 75 节点）：两个子图内部
    连通、跨子图无边（跨模态关联由 STGCN 的图卷积后续层学习）。
    """
    h = np.asarray(hand_adjacency, dtype=np.float32)
    p = np.asarray(pose_adjacency, dtype=np.float32)
    if h.ndim != 2 or p.ndim != 2:
        raise ValueError("邻接矩阵必须是方阵")
    n = h.shape[0] + p.shape[0]
    out = np.zeros((n, n), dtype=np.float32)
    out[:h.shape[0], :h.shape[0]] = h
    out[h.shape[0]:, h.shape[0]:] = p
    return out
