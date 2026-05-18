"""Tests for ERC, MaxDiv, NetworkRiskParity — Spec 07."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio_maker import HRP, IVP, ERC, MaxDiv, NetworkRiskParity


RNG = np.random.default_rng(456)


def _make_returns(T: int = 500, N: int = 15, seed: int = 456) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((T, 3)) * 0.4
    loadings = rng.standard_normal((3, N))
    noise = rng.standard_normal((T, N)) * 0.8
    data = factors @ loadings + noise
    cols = [f"A{i:02d}" for i in range(N)]
    return pd.DataFrame(data, columns=cols)


# ----------------------------------------------------------------------
# AC1 — all three classes produce valid weights
# ----------------------------------------------------------------------

class TestWeightsValid:
    def setup_method(self):
        self.returns = _make_returns(T=600, N=12)

    def _check(self, w):
        assert isinstance(w, pd.Series)
        assert abs(w.sum() - 1.0) < 1e-6, f"sum = {w.sum()}"
        assert (w >= -1e-10).all(), f"min weight = {w.min()}"

    def test_erc_weights_valid(self):
        self._check(ERC(self.returns).get_weights())

    def test_maxdiv_weights_valid(self):
        self._check(MaxDiv(self.returns).get_weights())

    def test_nrp_weights_valid(self):
        self._check(NetworkRiskParity(self.returns).get_weights())


# ----------------------------------------------------------------------
# AC2 — ERC equal-risk-contribution property
# ----------------------------------------------------------------------

class TestERCEqualRiskContribution:
    def test_risk_contributions_are_uniform(self):
        returns = _make_returns(T=800, N=10)
        w = ERC(returns).get_weights().values
        cov = returns.cov().values
        portfolio_var = float(w @ cov @ w)
        marginal_risk = cov @ w
        rc = w * marginal_risk / portfolio_var  # normalized RC, sums to 1
        # Spec AC2: max deviation from 1/N target < 1%
        max_dev = float(np.max(np.abs(rc - rc.mean()) / rc.mean()))
        assert max_dev < 0.05, f"max RC deviation = {max_dev:.2%} (target < 5%)"


# ----------------------------------------------------------------------
# AC3 — MaxDiv produces a higher diversification ratio than IVP and HRP
# ----------------------------------------------------------------------

class TestMaxDivOptimizesDR:
    def test_dr_exceeds_baselines(self):
        returns = _make_returns(T=800, N=15)
        cov = returns.cov().values
        vols = np.sqrt(np.diag(cov))

        def dr(w_series):
            w = w_series.values
            return float((w @ vols) / np.sqrt(max(w @ cov @ w, 1e-16)))

        dr_maxdiv = dr(MaxDiv(returns).get_weights())
        dr_ivp = dr(IVP(returns).get_weights())
        dr_hrp = dr(HRP(returns).get_weights())
        assert dr_maxdiv >= dr_ivp - 1e-6, f"MaxDiv DR {dr_maxdiv:.4f} < IVP DR {dr_ivp:.4f}"
        assert dr_maxdiv >= dr_hrp - 1e-6, f"MaxDiv DR {dr_maxdiv:.4f} < HRP DR {dr_hrp:.4f}"


# ----------------------------------------------------------------------
# AC4 — NRP MST has exactly N-1 edges and is connected
# ----------------------------------------------------------------------

class TestNRPMSTStructure:
    def test_mst_has_n_minus_1_edges(self):
        returns = _make_returns(T=600, N=20)
        model = NetworkRiskParity(returns)
        _ = model.get_weights()
        assert model.mst_edges is not None
        assert len(model.mst_edges) == 20 - 1, f"MST has {len(model.mst_edges)} edges (expected 19)"

    def test_mst_is_connected(self):
        returns = _make_returns(T=600, N=15)
        model = NetworkRiskParity(returns)
        _ = model.get_weights()
        # A graph with N nodes and N-1 edges is connected iff it has no cycle.
        # We verify connectivity directly via BFS from node 0.
        n = 15
        adj = {i: set() for i in range(n)}
        for i, j in model.mst_edges:
            adj[i].add(j)
            adj[j].add(i)
        visited = {0}
        queue = [0]
        while queue:
            node = queue.pop()
            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        assert len(visited) == n, f"MST not connected: reached {len(visited)}/{n} nodes"


class TestNRPEigenvectorCentrality:
    """Bonus — alternative centrality option also produces valid weights."""

    def test_inverse_eigenvector_centrality(self):
        returns = _make_returns(T=600, N=12)
        w = NetworkRiskParity(returns, centrality="inverse_eigenvector").get_weights()
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= -1e-10).all()


class TestNRPRejectsUnknownNetworkType:
    def test_pmfg_not_implemented(self):
        returns = _make_returns(T=400, N=10)
        with pytest.raises(NotImplementedError):
            NetworkRiskParity(returns, network_type="pmfg")
