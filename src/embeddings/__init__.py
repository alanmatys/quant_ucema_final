"""HRP embedding-distance variants (feature/embedding-hrp-variants).

Four embedding approaches, all converting per-asset representations into
cosine-distance matrices that feed the standard HRP pipeline:

- path_signatures: deterministic geometric path embeddings (esig).
- graph_emb:       node2vec on a kNN correlation graph.
- contrastive:     PyTorch contrastive SSL (Gaussian-jitter augmentation).
- ts2vec_lite:     PyTorch contrastive SSL (timestamp-mask augmentation).
"""

from src.embeddings.path_signatures import (
    asset_path_signatures, signatures_to_distance,
)
from src.embeddings.graph_emb import (
    build_corr_knn_graph, node2vec_embeddings,
    embeddings_to_distance as nodevec_distance,
)
from src.embeddings.contrastive import (
    train_contrastive_encoder,
    asset_embeddings_from_encoder as contrastive_asset_emb,
    embeddings_to_distance as contrastive_distance,
)
from src.embeddings.ts2vec_lite import (
    train_ts2vec_lite_encoder,
    asset_embeddings_from_encoder as ts2vec_asset_emb,
    embeddings_to_distance as ts2vec_distance,
)

__all__ = [
    "asset_path_signatures", "signatures_to_distance",
    "build_corr_knn_graph", "node2vec_embeddings", "nodevec_distance",
    "train_contrastive_encoder", "contrastive_asset_emb", "contrastive_distance",
    "train_ts2vec_lite_encoder", "ts2vec_asset_emb", "ts2vec_distance",
]
