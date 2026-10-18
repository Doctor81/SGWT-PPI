"""Загрузка SHS27k и разбиение рёбер для link prediction.

Повторяет исходный протокол GNN-PPI/HGCN (utils/data_utils.py) с одним
переключателем --neg-mode:

  legacy — как в статье: обучающие негативы = все несуществующие пары
           + валидационные и тестовые ПОЛОЖИТЕЛЬНЫЕ рёбра (ошибка из HGCN);
           тестовые отрицательные пары тоже входят в обучение.
  clean  — обучающие негативы = все несуществующие пары, кроме
           валидационных и тестовых отрицательных; val/test-рёбра
           в обучение не попадают.

Всё остальное (порядок перемешивания, размеры выборок, нормировка
признаков) совпадает с исходным кодом, поэтому при --neg-mode legacy
разбиение идентично использованному в статье.
"""
import os

import numpy as np
import scipy.sparse as sp

PPI_MODES = ["reaction", "binding", "ptmod", "activation",
             "inhibition", "catalysis", "expression"]


def load_ppi(data_dir, ppitype, ppimode):
    features = np.load(os.path.join(data_dir, f"features_{ppitype}.npy"))
    edges = np.loadtxt(os.path.join(data_dir, f"{ppitype}_{ppimode}.txt"))
    n = len(features)
    adj = np.zeros((n, n))
    for a, b in edges:
        adj[int(a), int(b)] = 1.0
        adj[int(b), int(a)] = 1.0
    return sp.csr_matrix(adj), features


def row_normalize(mx):
    """Построчная нормировка признаков, как normalize() в исходном коде."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    return sp.diags(r_inv).dot(mx)


def split_edges(adj, val_prop, test_prop, split_seed, neg_mode):
    if neg_mode not in ("legacy", "clean"):
        raise ValueError(f"neg_mode must be legacy or clean, got {neg_mode!r}")

    np.random.seed(split_seed)
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)

    x, y = sp.triu(sp.csr_matrix(1.0 - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)

    val_edges = pos_edges[:n_val]
    test_edges = pos_edges[n_val:n_val + n_test]
    train_edges = pos_edges[n_val + n_test:]

    val_edges_false = neg_edges[:n_val]
    test_edges_false = neg_edges[n_val:n_val + n_test]

    if neg_mode == "legacy":
        train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    else:
        train_edges_false = neg_edges[n_val + n_test:]

    return {
        "train_edges": train_edges, "train_edges_false": train_edges_false,
        "val_edges": val_edges, "val_edges_false": val_edges_false,
        "test_edges": test_edges, "test_edges_false": test_edges_false,
    }


def load_data(data_dir, ppitype, ppimode, split_seed=1234, neg_mode="clean",
              val_prop=0.05, test_prop=0.10):
    adj, features = load_ppi(data_dir, ppitype, ppimode)
    data = split_edges(adj, val_prop, test_prop, split_seed, neg_mode)
    data["features"] = np.asarray(row_normalize(features), dtype=np.float32)
    return data
