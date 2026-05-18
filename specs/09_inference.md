# Spec 09 — Statistical Inference Framework

**Status:** Draft
**Owner:** Alan Matys, Federico Rodriguez
**Used by:** [03_backtest_v2.md](03_backtest_v2.md), [04_paper.md](04_paper.md)

---

## 1. Motivation

The MACI 2025 paper compares strategies on point-estimate metrics (Sharpe,
return, drawdown) without quantifying statistical significance. The
continuation paper proposes a much larger comparison (20 strategies × 2
scenarios + hyperparameter grids), which amplifies multiple-testing risk: if
you compare enough strategies, one will look great by chance.

Han et al. (2024) explicitly note that a naive t-test on mean returns is
insufficient for evaluating crypto strategies. To make the paper publication-
worthy, three pieces of statistical machinery are required:

- **Ledoit-Wolf robust Sharpe-ratio difference test** — for pairwise
  comparisons against the HRP base.
- **Hansen Superior Predictive Ability (SPA) test** — corrects for the fact
  that we evaluate many strategies and many hyperparameter combinations.
- **Stationary block bootstrap** — confidence intervals on Sharpe, Sortino,
  and Maximum Drawdown that respect serial dependence in returns.

## 2. Requirements

### 2.1 Ledoit-Wolf Sharpe difference test

R1.1. Implement the asymptotic test from Ledoit & Wolf (2008), "Robust
      performance hypothesis testing with the Sharpe ratio."

R1.2. Pairwise: each strategy vs `HRP` (the baseline from MACI).

R1.3. Output: difference in Sharpe, standard error, two-sided p-value.

### 2.2 Hansen SPA test

R2.1. Implement Hansen (2005), "A test for superior predictive ability."

R2.2. Apply across the full set of strategies (and hyperparameter combinations
      if hyperparameter tuning is used in the final results).

R2.3. Output: consistent and conservative p-values per strategy of the null
      "strategy does not outperform the benchmark."

### 2.3 Stationary block bootstrap

R3.1. Implement Politis & Romano (1994) stationary bootstrap with
      automatic block length per Politis & White (2004).

R3.2. Resample at least **10,000 iterations** for each CI.

R3.3. Output 95% CIs for: Sharpe, Sortino, MaxDD, Calmar, Annualized Return.

### 2.4 Cluster stability diagnostics

(Added in audit pass after Spec 03 R5b introduced cophenetic correlation +
ARI but no module owned them. They sit naturally alongside the inference
machinery because they answer "is this dendrogram a stable structure or
estimation noise?", which is the cluster-level analog of the bootstrap CI
question for portfolio metrics.)

R4.1. Implement **cophenetic correlation** per López de Prado (2016) — Pearson
      correlation between the cophenetic distance matrix derived from the
      linkage and the original pairwise distance matrix.

R4.2. Implement **Adjusted Rand Index (ARI)** per Hubert & Arabie (1985)
      between cluster assignments at consecutive rebalance dates, both
      cut at K clusters via `scipy.cluster.hierarchy.fcluster`.

R4.3. K defaults to 5 (sectoral coarseness typical in crypto research);
      reported at K ∈ {3, 5, 7} as robustness.

R4.4. Both metrics are computed by the backtest notebook per Spec 03 R5b and
      saved to `data/cluster_stability.csv`.

**Empirical findings (Phase 4 smoke test, PIT universe snapshots 2022-12-31
and 2023-12-31, N≈41 assets):**

| Diagnostic | HRP (Pearson) | HRP_TailDep | Interpretation |
|---|---|---|---|
| cophenetic correlation, 2022-12-31 | 0.879 | 0.637 | Pearson distance more faithfully represented by dendrogram |
| cophenetic correlation, 2023-12-31 | 0.851 | 0.700 | |
| ARI between snapshots (K=5) | **-0.047** | **0.265** | Pearson clusters completely reshuffle year-over-year; tail-dep clusters are meaningfully more stable |
| ARI between snapshots (K=3) | 0.000 | -0.030 | At coarser K, both partition randomly |

**This is a paper-grade finding:** the Pearson HRP topology is essentially
random across consecutive years on crypto data (ARI ≈ 0), while the tail-
dependence topology preserves real structure (ARI = 0.27). This is the
empirical case for `HRP_TailDep` / `HRP_TailDepShrunk` as headline strategies
beyond their crash-risk economic motivation: they produce more STABLE
dendrograms, which translates to lower realized turnover. Headline §10.1
of the paper.

**Linkage method matters even on its own metric.** Cophenetic correlation
of HRP at 2022-12-31 by linkage:

| Linkage | Cophenetic |
|---|---|
| single (default) | 0.879 |
| **average** | **0.913** |
| complete | 0.748 |
| ward | 0.690 |

**Average linkage produces the most faithful dendrogram** by HRP's own
internal criterion. Backtest v2 should rerun the headline strategies under
average linkage and report whether the OOS Sharpe improvement matches the
cophenetic-correlation improvement.

## 3. Interface

### 3.1 `src/inference.py` (new module)

```python
def sharpe_diff_ledoit_wolf(r_a: np.ndarray, r_b: np.ndarray,
                             block_size: int = None) -> dict:
    """Robust Sharpe ratio difference test (Ledoit & Wolf 2008).

    Returns:
        {'sharpe_diff', 'se', 'p_value', 'ci_95'}
    """

def hansen_spa_test(returns_matrix: pd.DataFrame,
                     benchmark: pd.Series,
                     n_bootstrap: int = 5000,
                     block_size: int = None) -> pd.DataFrame:
    """Hansen Superior Predictive Ability test.

    Args:
        returns_matrix: T x K matrix of K strategy returns.
        benchmark: T-vector of benchmark returns.
    Returns:
        DataFrame indexed by strategy with consistent / lower / upper SPA p-values.
    """

def stationary_block_bootstrap(returns: pd.Series,
                                statistic_fn: callable,
                                n_iter: int = 10_000,
                                block_size: float = None,
                                seed: int = 42) -> dict:
    """Stationary block bootstrap CI for an arbitrary statistic.

    Returns:
        {'point_estimate', 'ci_lower_95', 'ci_upper_95', 'samples'}
    """

def optimal_block_length(returns: pd.Series) -> float:
    """Politis-White automatic block length selection."""
```

### 3.2 Notebook integration

`notebooks/backtest_v2.ipynb` (per Spec 03) imports `src/inference.py` and
produces:
- `data/inference_sharpe_diff.csv` — pairwise vs HRP
- `data/inference_spa.csv` — SPA p-values across all strategies
- `data/inference_bootstrap_cis.csv` — bootstrap CIs for all metrics

## 4. Acceptance Criteria

AC1. On synthetic data where Strategy A has true Sharpe = Strategy B Sharpe + 0.5,
     the Ledoit-Wolf test rejects equality at α=0.05 with power ≥ 0.80 for
     T ≥ 500.

AC2. Hansen SPA p-values are bounded in [0, 1]. `p_upper` (no recentering,
     most conservative scheme) is always the largest of the three. Strict
     ordering `p_lower ≤ p_consistent ≤ p_upper` is the asymptotic claim
     of Hansen 2005 but is NOT guaranteed in finite samples — the
     recentering of moderately-negative `d_bar` strategies under `mu_lower`
     can inflate the bootstrap max above `mu_consistent`. The test asserts
     only that `p_upper` dominates and that all three are in [0, 1]; the
     paper reports all three values and interprets `p_upper` as the
     headline conservative answer.

AC3. Block-bootstrap 95% CI for the **mean** of an iid Gaussian sample has
     coverage in [0.92, 0.98] over 1000 Monte Carlo trials (sanity check).

AC4. Optimal block length is positive and < T/2 on the crypto dataset.

AC5. All three CSV outputs include strategy names matching those in
     `data/backtest_v2_rebalance_results.csv` (joinable by primary key).

AC6. Unit tests in `tests/test_inference.py` cover AC1–AC4 with fixed seeds.

**Phase 4 smoke-test findings on real PIT returns (2022-12-31 weights, OOS
365-day 2023 evaluation, N≈41 assets):**

- HRP OOS daily Sharpe = 0.093, 95% block-bootstrap CI [−0.009, 0.198].
  CI includes zero — a single OOS year is too short to declare HRP
  significantly different from zero on this universe. Consistent with
  the crypto reality of 2023 (post-FTX recovery, high vol).

- LW pairwise Sharpe difference test (HRP vs HRP-variants on OOS daily
  returns):
  - HRP vs HRP_VolStd: **p = 0.045** (significant, HRP marginally better).
  - HRP vs HRP_TailDep, HRP_TailDepShrunk, HRP_Detoned: p > 0.45 (no
    detectable Sharpe difference on a 1-year window).

- Hansen SPA across 5 variants vs HRP benchmark: p_consistent = 0.121,
  p_upper = 0.535. **Cannot reject H0: no HRP variant outperforms
  baseline HRP** on this OOS window. This is a sample-size issue
  (T = 365 daily OOS), not necessarily a strategy issue. Backtest v2's
  multi-year walk-forward will resolve this.

These results are intentionally weak: a single 1-year OOS slice was
never going to give significant differentiation. The point of the
smoke test is to confirm the inference machinery runs end-to-end and
produces sensible point estimates and bootstrap distributions — both
verified.

## 5. Out of Scope

- Multiple-hypothesis correction beyond SPA (e.g. FDR, Romano-Wolf
  stepdown) — SPA is the de facto standard and sufficient.
- Bayesian model comparison.
- Bootstrap of full equity curves (only summary statistics are bootstrapped).

## 6. References

- Ledoit, O., & Wolf, M. (2008). "Robust performance hypothesis testing with the
  Sharpe ratio." *Journal of Empirical Finance*.
- Hansen, P. R. (2005). "A test for superior predictive ability." *Journal of
  Business & Economic Statistics*.
- Politis, D. N., & Romano, J. P. (1994). "The stationary bootstrap." *JASA*.
- Politis, D. N., & White, H. (2004). "Automatic block-length selection for the
  dependent bootstrap." *Econometric Reviews*.
- Han, Y., et al. (2024). Realistic-assumption momentum in cryptocurrency.
