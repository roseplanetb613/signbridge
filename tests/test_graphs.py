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


def test_normalize_adjacency_symmetric_and_diag_inverse_degree():
    adj = build_adjacency([(0, 1), (1, 2)], 3)
    norm = normalize_adjacency(adj)   # include_self=True
    assert np.allclose(norm, norm.T, atol=1e-6)   # 对称归一化保持对称
    # 自环归一化后对角 = 1/d_i（d_i = 邻居数 + 1）
    assert np.allclose(norm[0, 0], 1.0 / 2.0, atol=1e-5)
    assert np.allclose(norm[1, 1], 1.0 / 3.0, atol=1e-5)
    assert np.allclose(norm[2, 2], 1.0 / 2.0, atol=1e-5)


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


def test_hand_pose_graph_block_diagonal():
    """hand 42 + pose 33 → 75×75 分块对角（跨子图无边）。"""
    from signbridge import build_adjacency, build_hand_pose_graph
    hand = build_hand_graph(num_hands=2)
    pose = build_adjacency(
        ((0, 1), (1, 2)), 5)          # 小 pose 图便于验证
    g = build_hand_pose_graph(hand, pose)
    assert g.shape == (47, 47)
    # 内部子图保持
    assert np.array_equal(g[:42, :42], hand)
    assert np.array_equal(g[42:, 42:], pose)
    # 跨子图无边
    assert g[:42, 42:].sum() == 0
    assert g[42:, :42].sum() == 0
    # 对称
    assert np.array_equal(g, g.T)
