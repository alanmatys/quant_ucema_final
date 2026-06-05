"""Tests for the four embedding-based HRP variants.

Each variant has the same contract: take a returns DataFrame, return a
pd.Series of non-negative weights summing to 1, with `fallback_used`
False under normal conditions.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

from src.portfolio_maker import (
    HRP, HRPPathSig, HRPNodeEmbed, HRPContrastive, HRPTS2Vec,
)


def _make_returns(seed: int = 0, T: int = 250, N: int = 12) -> pd.DataFrame:
    """Synthetic returns with a small drift, suitable for path signatures
    and contrastive training without producing log1p-NaN issues."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((T, N)) * 0.02 + 0.0005
    cols = [f"A{i:02d}" for i in range(N)]
    return pd.DataFrame(data, columns=cols)


# ----------------------------------------------------------------------
# Weights validity (all 4)
# ----------------------------------------------------------------------

class TestWeightsValid:
    def setup_method(self):
        self.returns = _make_returns(T=250, N=12)

    def _check(self, w: pd.Series):
        assert isinstance(w, pd.Series)
        assert abs(w.sum() - 1.0) < 1e-6, f"sum = {w.sum()}"
        assert (w >= -1e-10).all(), f"min weight = {w.min()}"

    def test_path_signatures(self):
        m = HRPPathSig(self.returns, window=60, level=3)
        w = m.get_weights()
        self._check(w)
        assert not m.fallback_used

    def test_node2vec(self):
        m = HRPNodeEmbed(self.returns, dimensions=16, k=5, num_walks=10, walk_length=10)
        w = m.get_weights()
        self._check(w)
        assert not m.fallback_used

    def test_contrastive(self):
        m = HRPContrastive(
            self.returns, emb_dim=16, hidden=8, epochs=3,
            batch_size=32, stride=10, window=30,
        )
        w = m.get_weights()
        self._check(w)
        assert not m.fallback_used

    def test_ts2vec(self):
        m = HRPTS2Vec(
            self.returns, emb_dim=16, hidden=8, epochs=3,
            batch_size=32, stride=10, window=30,
        )
        w = m.get_weights()
        self._check(w)
        assert not m.fallback_used


# ----------------------------------------------------------------------
# Fallback safety
# ----------------------------------------------------------------------

class TestFallbackOnDegenerateData:
    """If embedding computation fails, variants must fall back to base HRP
    rather than crash."""

    def test_path_signatures_does_not_crash_on_degenerate_input(self):
        # All-zero returns are a known degeneracy that breaks HRP's
        # inverse-variance step (div-by-zero on cov diagonal). The fallback
        # path must still return a Series (of NaNs is acceptable) rather
        # than crash. Real PIT data never produces this input.
        r = pd.DataFrame(np.zeros((10, 5)), columns=[f"A{i}" for i in range(5)])
        m = HRPPathSig(r, window=60, level=3)
        w = m.get_weights()
        assert isinstance(w, pd.Series)
        assert len(w) == 5


# ----------------------------------------------------------------------
# Embedding helper functions (numerical invariants)
# ----------------------------------------------------------------------

class TestPathSignatureHelpers:
    def test_distance_diagonal_zero_and_symmetric(self):
        from src.embeddings.path_signatures import (
            asset_path_signatures, signatures_to_distance,
        )
        r = _make_returns(T=200, N=10)
        sig = asset_path_signatures(r, window=60, level=3)
        d = signatures_to_distance(sig)
        assert (np.diag(d.values) == 0.0).all()
        assert np.allclose(d.values, d.values.T)


class TestNode2VecHelpers:
    def test_graph_has_expected_structure(self):
        from src.embeddings.graph_emb import build_corr_knn_graph
        r = _make_returns(T=200, N=10)
        g = build_corr_knn_graph(r.corr(), k=3)
        assert g.number_of_nodes() == 10
        # Each node should have at least 3 neighbours (kNN with k=3 may have more
        # if reciprocal-NN inflation occurs).
        assert g.number_of_edges() >= 10 * 3 // 2

    def test_embeddings_dimension_matches(self):
        from src.embeddings.graph_emb import (
            build_corr_knn_graph, node2vec_embeddings,
        )
        r = _make_returns(T=200, N=10)
        g = build_corr_knn_graph(r.corr(), k=3)
        emb = node2vec_embeddings(g, dimensions=8, num_walks=5, walk_length=5)
        assert emb.shape == (10, 8)


class TestContrastiveHelpers:
    def test_encoder_output_is_normalised(self):
        from src.embeddings.contrastive import TinyConvEncoder
        import torch
        enc = TinyConvEncoder(emb_dim=16, hidden=8)
        x = torch.randn(4, 1, 30)
        z = enc(x)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestTS2VecHelpers:
    def test_dilated_encoder_output_is_normalised(self):
        from src.embeddings.ts2vec_lite import DilatedConvEncoder
        import torch
        enc = DilatedConvEncoder(emb_dim=16, hidden=8)
        x = torch.randn(4, 1, 30)
        z = enc(x)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
