"""Graph embeddings for HRP.

Build a sparse correlation graph from the sample correlation matrix
(kNN edges or threshold edges), run node2vec random walks + skip-gram
to learn per-asset embeddings, then take cosine distance for HRP's
dendrogram.

References:
    Grover & Leskovec (2016). node2vec: Scalable feature learning for networks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec


def build_corr_knn_graph(
    corr: pd.DataFrame,
    k: int = 10,
    weight_mode: str = "absolute_corr",
) -> nx.Graph:
    """Build a kNN graph from a correlation matrix.

    For each asset, keep edges to the k nearest neighbours by |correlation|.
    Edges are weighted by the chosen `weight_mode`:
      - "absolute_corr": |ρ|
      - "positive_corr": max(ρ, 0)
      - "distance":      max(1 - |ρ|, 1e-3)

    Args:
        corr: N x N correlation DataFrame (must be symmetric).
        k:    number of neighbours per node.
        weight_mode: see above.

    Returns:
        A NetworkX undirected graph with N nodes (one per asset).
    """
    C = corr.values.copy()
    np.fill_diagonal(C, 0.0)
    N = C.shape[0]
    g = nx.Graph()
    g.add_nodes_from(list(corr.index))

    abs_c = np.abs(C)
    for i in range(N):
        # k nearest neighbours by |corr|
        order = np.argsort(-abs_c[i])
        for j in order[:k]:
            if j == i:
                continue
            if weight_mode == "absolute_corr":
                w = float(abs_c[i, j])
            elif weight_mode == "positive_corr":
                w = float(max(C[i, j], 0.0))
            elif weight_mode == "distance":
                w = float(max(1.0 - abs_c[i, j], 1e-3))
            else:
                raise ValueError(f"unknown weight_mode: {weight_mode}")
            if w > 0:
                g.add_edge(corr.index[i], corr.index[j], weight=w)
    return g


def node2vec_embeddings(
    graph: nx.Graph,
    dimensions: int = 32,
    walk_length: int = 20,
    num_walks: int = 40,
    p: float = 1.0,
    q: float = 1.0,
    seed: int = 42,
    quiet: bool = True,
) -> pd.DataFrame:
    """Train node2vec on a graph and return per-node embeddings.

    Args:
        graph:        NetworkX graph with N nodes labelled by asset name.
        dimensions:   embedding dimension (default 32).
        walk_length:  steps per random walk (default 20).
        num_walks:    walks per node (default 40).
        p, q:         node2vec return/in-out hyperparameters.
        seed:         RNG seed for reproducibility.
        quiet:        suppress node2vec's tqdm progress bar.

    Returns:
        DataFrame indexed by node, columns = embedding dimensions.
    """
    if graph.number_of_edges() == 0:
        raise RuntimeError("graph has no edges — cannot run node2vec")
    n2v = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=1,
        quiet=quiet,
        seed=seed,
    )
    model = n2v.fit(window=10, min_count=1, batch_words=4, seed=seed)
    nodes = list(graph.nodes())
    rows = np.vstack([model.wv[str(n)] for n in nodes])
    return pd.DataFrame(rows, index=nodes,
                        columns=[f"emb_{i}" for i in range(dimensions)])


def embeddings_to_distance(emb_df: pd.DataFrame) -> pd.DataFrame:
    """Convert embeddings to pairwise cosine-distance matrix in [0, sqrt(2)]."""
    X = emb_df.values.astype(float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1e-12, norms)
    Xn = X / norms
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = np.sqrt(np.maximum(1.0 - sim, 0.0))
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    return pd.DataFrame(dist, index=emb_df.index, columns=emb_df.index)
