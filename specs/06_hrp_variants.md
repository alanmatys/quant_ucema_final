# Spec 06 — Additional HRP Variants

**Status:** Draft
**Owner:** Alan Matys, Federico Rodriguez
**Depends on:** [01_denoising.md](01_denoising.md) (uses denoised correlation as input)

---

## 1. Motivation

The deep-research review identifies several HRP "distance" variants worth comparing
beyond denoising/detoning. The recent comparative literature (e.g. Gómez et al. 2025)
finds that **correlation-based distances remain the standard to beat in HRP** — so we
focus on *enriched correlation* variants rather than exotic non-correlational metrics:

- **Partial correlation** — neutralizes shared factors (mathematical alternative
  to detoning); allows the paper to compare two different routes to "removing the
  market mode" in crypto.
- **EWMA dynamic correlation** — addresses regime-shifting correlations in crypto.
- **Lower-tail dependence** — addresses crypto's heavy left tails and joint-crash
  behaviour (appendix-grade; estimation is noisy).

## 2. Requirements

### 2.1 HRP Partial-Correlation (HRP_PartialCorr)

R1.1. Estimate sparse precision matrix `Θ` via graphical lasso on the sample
      correlation matrix.

R1.2. Compute partial correlation:
      `ρ_partial[i,j] = -Θ[i,j] / sqrt(Θ[i,i] * Θ[j,j])`.

R1.3. Feed partial correlation into the standard HRP pipeline (distance → linkage →
      quasi-diag → recursive bisection).

R1.4. Sparsity hyperparameter `alpha` configurable; default 0.05.

### 2.2 HRP EWMA-Dynamic (HRP_Dynamic)

R2.1. Compute EWMA correlation matrix with decay parameter `lambda`.

R2.2. At each rebalance date, use the **latest** EWMA correlation snapshot for HRP.

R2.3. `lambda` configurable; defaults to compare: 0.94 (RiskMetrics), 0.97, 0.99.

### 2.3 HRP Lower-Tail Dependence (HRP_TailDep)

R3.1. Estimate empirical lower-tail dependence coefficient for each asset pair:
      `λ_L[i,j] = P(F_i(X_i) ≤ q | F_j(X_j) ≤ q)` at threshold `q` (default 0.05).

R3.2. Convert tail-dependence matrix to distance:
      `d[i,j] = sqrt(1 - λ_L[i,j])`.

R3.3. Feed into HRP linkage and recursive bisection.

R3.4. **Headline strategy** in the paper (revised from "appendix only" after
      the Phase 2 smoke test confirmed it produces valid weights without
      fallback on the PIT universe, with T ≥ 730 daily observations and
      N ≈ 30 assets). Economic argument: for a long-only crypto portfolio
      the relevant risk is joint crash co-movement, which is exactly what
      lower-tail dependence measures and what Pearson correlation summarizes
      poorly. This is plausibly the most economically defensible HRP
      extension for crypto.

### 2.4 HRP Shrunk Covariance (HRP_ShrunkCov)

R4.1. Estimate the covariance matrix using **Ledoit-Wolf** shrinkage
      (sklearn's `LedoitWolf`), which optimally combines the sample
      covariance with a shrinkage target (scaled identity) to reduce
      estimation noise.

R4.2. Derive the corresponding shrunk correlation matrix and feed it
      into the standard HRP pipeline.

R4.3. Record the optimal shrinkage intensity `α ∈ [0, 1]` as a diagnostic
      output (informs the paper's stability discussion).

R4.4. Hyperparameter: none required — Ledoit-Wolf chooses `α` analytically.

### 2.5 HRP Vol-Standardized Returns (HRP_VolStd)

R5.1. Standardize each daily return by its per-asset **rolling window
      standard deviation** (default window = 30 days), shifted one day to
      avoid look-ahead bias, then compute Pearson correlation on the
      standardized series.

R5.2. Use the resulting correlation matrix for the dendrogram; keep the
      sample covariance for recursive bisection (the standardization is
      for cluster topology, not weight allocation).

R5.3. Rationale: crypto assets have wildly heterogeneous volatilities
      (BTC vs SHIB). Standardizing first surfaces co-movement structure
      that would otherwise be drowned out by the raw vol scale.

R5.4. Hyperparameter: vol_window (default 30 days; grid 21 / 30 / 60 for
      sensitivity in Backtest v2).

### 2.6 HRP Tail-Dep + Shrunk Cov Hybrid (HRP_TailDepShrunk)

R6.1. Inherit from `HRPTailDep` (so the dendrogram is built from lower-tail
      dependence distance per §2.3).

R6.2. In `__init__`, ALSO compute Ledoit-Wolf shrunk covariance and
      substitute it for `self.cov`. Recursive bisection then uses the
      shrunk covariance for the inverse-variance weight allocation.

R6.3. **Hybrid construction targets two distinct problems simultaneously:**
   - tail-dependence distance — cluster topology that reflects joint
     crash behaviour (the relevant risk for long-only crypto);
   - LW shrinkage — variance-allocation step that doesn't get whipsawed
     by sample-covariance noise.

R6.4. Report 3's literally #1 paper-extension recommendation:
   > "*el mejor 'default research paper extension' es... HRP de cola inferior
   > + covarianza regularizada*"

R6.5. Hyperparameter: `q` (tail quantile threshold; default 0.05). LW
      shrinkage is parameter-free.

**Empirical findings (Phase 3.5 smoke test, 2022-12-31 PIT snapshot, 27 assets, 730-day window):**
- LW shrinkage intensity inside HRP_TailDepShrunk: `α = 0.094` (same as
  HRP_ShrunkCov on the same snapshot — LW is purely a function of the
  covariance and doesn't see the tail-dep distance).
- **Spearman(HRP_TailDep weights, HRP_TailDepShrunk weights) = 0.995** —
  on this universe the shrinkage is a small refinement, not a topology-
  level change. Weight rankings essentially identical. The hybrid's value
  is variance stability at the per-cluster bisection step, not a
  re-ordering of the dendrogram.
- This pattern mirrors the Spec 02 §4.4 finding (HRP_Detoned vs
  HRP_PartialCorr Spearman = 0.99): when two HRP variants share their
  topology source, they converge in weight space.
- **Cross-topology pairs diverge more:** HRP_TailDep vs HRP_Detoned
  Spearman = 0.79; HRP vs HRP_TailDep = 0.67. Topology choice (Pearson
  vs detoned vs tail-dep) matters more for weights than the
  variance-allocation refinement.

**Empirical findings (Phase 2 smoke test, 2022-12-31 snapshot, 27 assets, 730-day window):**
- LW shrinkage intensity `α = 0.094` — modest shrinkage; sample covariance is
  reasonably well-conditioned at T/N ≈ 27, but shrinkage is non-trivially positive.
- Spearman(HRP, HRP_ShrunkCov) = **0.83** — weight rankings shift meaningfully
  but conservatively after shrinkage.
- Spearman(HRP_Detoned, HRP_ShrunkCov) = **0.66** vs Spearman(HRP_Detoned,
  HRP_PartialCorr) = **0.99** — confirms shrinkage and detoning are
  attacking different problems (variance stability vs market-mode removal),
  unlike detoning and partial correlation which converge to nearly the same
  solution. This is a clean separation worth reporting in the paper.

## 3. Interface

### 3.1 `src/hrp_variants.py` (new module)

```python
def partial_correlation_from_precision(theta: np.ndarray) -> np.ndarray:
    """Compute partial correlation matrix from precision matrix Θ."""

def estimate_partial_correlation(returns: pd.DataFrame,
                                  alpha: float = 0.05) -> pd.DataFrame:
    """Sparse partial correlation via graphical lasso."""

def ewma_correlation(returns: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """EWMA correlation matrix; uses latest snapshot."""

def lower_tail_dependence(returns: pd.DataFrame,
                          q: float = 0.05) -> pd.DataFrame:
    """Empirical lower-tail dependence coefficient matrix."""
```

### 3.2 `src/portfolio_maker.py` (additions)

```python
class HRPPartialCorr(HRP):
    def __init__(self, returns, alpha: float = 0.05): ...

class HRPDynamic(HRP):
    def __init__(self, returns, lam: float = 0.94): ...

class HRPTailDep(HRP):
    def __init__(self, returns, q: float = 0.05): ...

class HRPShrunkCov(HRP):
    def __init__(self, returns): ...   # Ledoit-Wolf, no hyperparameter

class HRPVolStd(HRP):
    def __init__(self, returns, vol_window: int = 30,
                 linkage_method: str = "single"): ...

class HRPTailDepShrunk(HRPTailDep):
    def __init__(self, returns, q: float = 0.05): ...   # LW inside
```

### 3.3 Linkage method robustness (cross-cutting)

The base `HRP` class accepts `linkage_method` (default `'single'`, preserving
the López de Prado convention and backward compatibility). All subclasses
inherit `self.linkage_method`; for the backtest robustness check, override
it after construction:
```python
model = HRPDetoned(returns)
model.linkage_method = "average"
```
Spec 03 R5b records cluster-stability metrics across linkage methods.

**Empirical sensitivity (Phase 3.5 smoke test, HRP_TailDepShrunk on 2022-12-31
PIT snapshot, 27 assets):** changing linkage from `'single'` produces
non-trivial weight shifts (L1 distance vs single-linkage):
- `average` → 0.13 (N_eff rises 21.9 → 22.8, slightly more diversified)
- `complete` → 0.25 (N_eff 21.8, similar concentration)
- `ward` → 0.20 (N_eff 20.7, max weight 12.3% — most concentrated)

Linkage choice has comparable magnitude to switching topology source. The
Backtest v2 robustness sweep should not skip this.

Each subclass overrides `self.corr` (and rebuilds `self.cov` consistently) before
delegating to the inherited HRP pipeline — same pattern as `HRPDenoised`/`HRPDetoned`.
`HRPShrunkCov` is the exception: it overrides `self.cov` directly and derives the
corresponding `self.corr` from the shrunk covariance.

## 4. Acceptance Criteria

AC1. All three variants produce weights that sum to 1.0 (tol 1e-8), non-negative.

AC2. `HRPPartialCorr` and `HRPDetoned` allocations on the crypto universe show
     **highly correlated** weight vectors — empirical Spearman rank correlation
     ≥ 0.80 on average across rebalance dates, evidence they target the same
     goal (market-mode removal) via different math.

     **Empirical finding (Phase 2 smoke test, 2022-12-31 PIT snapshot, 27 assets,
     730-day estimation window):** Spearman ρ = **0.990** — the two approaches
     converge to nearly identical weight rankings. This is a stronger
     equivalence than the spec's original guess of [0.4, 0.9] and **should be
     reported as a headline finding** in the continuation paper's §9.4 (see
     [04_paper.md](04_paper.md#7-acceptance-criteria) AC8). Original guess
     overestimated the divergence because synthetic random-loading data has a
     weaker market mode than real crypto.

AC3. `HRPDynamic` weights demonstrate measurable response to regime shifts:
     average absolute weight change between consecutive rebalances is **strictly
     greater** than that of static `HRP` (sanity check that the dynamic correlation
     actually drives weight movement).

AC4. `HRPTailDep` produces valid weights on at least 80% of rebalance dates;
     remaining 20% may fail due to insufficient tail observations and must fall
     back to base HRP (logged, not crashed).
     **Phase 2 result: 0% fallback rate on the PIT universe** — promoted to
     headline strategy.

AC5. `HRPShrunkCov` shrinkage intensity `α` is in [0, 1] on every rebalance
     date; reports the time-series of `α` as a paper diagnostic.

AC6. `HRPVolStd` produces a measurably different correlation matrix than
     vanilla sample correlation on heterogeneous-vol data (sanity check
     that the standardization actually changes the topology).

AC7. `HRPTailDepShrunk` records `shrinkage_intensity` in [0, 1] and overrides
     `self.cov` with the LW-shrunk version (verified by `!= sample_cov`).

AC8. Base `HRP.linkage_method` defaults to `'single'` and accepts any
     valid scipy linkage string ('average', 'complete', 'ward'). Subclasses
     inherit the attribute through `super().__init__()`.

AC9. Unit tests in `tests/test_hrp_variants.py` cover AC1–AC8 and basic
     numerical properties of each helper function.

## 5. Out of Scope

- DTW, mutual information, cointegration distance, NLP embeddings — explicitly
  rejected as low-ROI per the deep-research review and 2025 comparative evidence.
- Multi-factor partial correlation (only single-precision-matrix estimation).
- Time-varying tail-dependence estimation (rolling λ_L is appendix-only).

## 6. References

- Gómez, M. et al. (2025). Empirical comparison of HRP distance metrics.
- Lohre, H., Rother, C., & Schäfer, K. A. (2020). HRP with tail-dependence.
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). "Sparse inverse covariance
  estimation with the graphical lasso." *Biostatistics*.
- Engle, R. F. (2002). Dynamic conditional correlation (DCC) — EWMA as the simple
  special case used here.
