# Spec 07 — Additional Comparator Strategies

**Status:** Draft
**Owner:** Alan Matys, Federico Rodriguez

---

## 1. Motivation

The MACI 2025 paper compared HRP against IVP, MVP, and HODL baselines. To strengthen
the continuation paper's empirical comparison, three additional comparators are
added:

- **Equal Risk Contribution (ERC)** — standard risk-parity benchmark every
  practitioner expects to see.
- **Maximum Diversification (MaxDiv)** — Choueifaty & Coignard's classic;
  responds to volatility dispersion (relevant in heterogeneous crypto universe).
- **Network Risk Parity (NRP)** — Ciciretti & Pallotta (2024) report it
  outperforms HRP as N grows; allows the paper to position HRP within a richer
  set of clustering/network methods.

A simple **1/N (equal-weight)** baseline is also included (already specified in
Spec 03 R2) as a sanity floor.

## 2. Requirements

### 2.1 Equal Risk Contribution (ERC)

R1.1. Long-only, fully invested, no leverage.

R1.2. Solve for weights such that each asset contributes equally to portfolio
      risk: `w_i * (Σw)_i = constant` for all `i`.

R1.3. Implementation via sequential convex optimization or fixed-point iteration
      (Maillard, Roncalli & Teïletche 2010).

### 2.2 Maximum Diversification (MaxDiv)

R2.1. Maximize the diversification ratio `DR(w) = (w' · σ) / sqrt(w' · Σ · w)`
      subject to long-only, fully invested.

R2.2. Implementation via SLSQP optimizer (consistent with existing MVP code).

### 2.3 Network Risk Parity (NRP)

R3.1. Build a network from the correlation matrix using one of:
      - **MST** (Minimum Spanning Tree) — default
      - **PMFG** (Planar Maximally Filtered Graph) — optional sensitivity

R3.2. Allocate weights based on network centrality (inverse degree, inverse
      betweenness, or eigenvector centrality — default: inverse-degree-weighted
      risk parity within network communities).

R3.3. Implementation per Ciciretti & Pallotta (2024) §3.

## 3. Interface

### 3.1 `src/portfolio_maker.py` (additions)

```python
class ERC(PortfolioStrategy):
    def __init__(self, returns: pd.DataFrame): ...
    def get_weights(self) -> pd.Series: ...

class MaxDiv(PortfolioStrategy):
    def __init__(self, returns: pd.DataFrame): ...
    def get_weights(self) -> pd.Series: ...

class NetworkRiskParity(PortfolioStrategy):
    def __init__(self, returns: pd.DataFrame,
                 network_type: str = "mst",
                 centrality: str = "inverse_degree"): ...
    def get_weights(self) -> pd.Series: ...
```

### 3.2 `src/__init__.py`

Export `ERC`, `MaxDiv`, `NetworkRiskParity`.

## 4. Acceptance Criteria

AC1. All three classes produce weights that sum to 1.0 (tol 1e-8), non-negative.

AC2. ERC weights satisfy equal-risk-contribution property:
     `max_i |RC_i - RC_mean| / RC_mean < 0.01` on test data.

AC3. MaxDiv diversification ratio is strictly greater than that of IVP and HRP
     on the crypto dataset (sanity check that the optimizer maximizes the
     intended objective).

AC4. NRP MST has exactly `N-1` edges and is connected (graph-theoretic invariant).

AC5. Unit tests in `tests/test_comparators.py` cover AC1–AC4.

## 5. Out of Scope

- Black-Litterman with views (deferred to next paper).
- Risk-parity with custom risk budgets beyond equal.
- PMFG implementation tuning (MST is the default; PMFG kept as switch only).
- Path-signature embeddings (rejected as too speculative per deep-research review).

## 6. References

- Maillard, S., Roncalli, T., & Teïletche, J. (2010). "The properties of
  equally-weighted risk contribution portfolios." *Journal of Portfolio Management*.
- Choueifaty, Y., & Coignard, Y. (2008). "Toward Maximum Diversification."
  *Journal of Portfolio Management*.
- Ciciretti, V., & Pallotta, A. (2024). Network Risk Parity.
- Mantegna, R. N. (1999). "Hierarchical structure in financial markets" — MST
  in finance.
- Tumminello, M., Aste, T., Di Matteo, T., & Mantegna, R. N. (2005). PMFG.
