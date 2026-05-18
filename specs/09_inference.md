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

**Empirical findings — REVISED in Phase 4.5 after dataset expansion.**

The Phase 4 finding that "TailDep produces more stable clusters than
Pearson" (ARI -0.047 vs 0.265) was **data-snooped from a single
year-pair (2022→2023)**. Re-running on 5 consecutive year-end pairs
across the expanded 2017-2026 dataset:

| Period | N_t → N_{t+1} | ARI Pearson | ARI TailDep | Winner |
|---|---|---|---|---|
| 2020-12-31 → 2021-12-31 | 32 → 51 | -0.075 | -0.037 | TailDep (barely) |
| 2021-12-31 → 2022-12-31 | 51 → 48 | **+0.470** | -0.039 | **Pearson** |
| 2022-12-31 → 2023-12-31 | 48 → 51 | +0.000 | -0.038 | Pearson |
| 2023-12-31 → 2024-12-31 | 51 → 50 | -0.075 | **+0.238** | TailDep |
| 2024-12-31 → 2025-12-31 | 50 → 48 | **+0.280** | -0.024 | **Pearson** |
| **mean** | | **+0.120** | **+0.020** | **Pearson** |

**Corrected finding for the paper:** TailDep is NOT more stable than
Pearson on average across the 2020-2025 window. Pearson ARI is higher
in 3 of 5 year-pairs and higher on average (+0.12 vs +0.02). The
original 2022→2023 pair was an unrepresentative single observation.

The paper should report the multi-pair table above honestly. The
economic argument for HRP_TailDep still stands (it captures joint-
crash co-movement that Pearson misses), but the "more stable clusters"
secondary argument does NOT hold up under multi-period scrutiny.

**Cophenetic correlation (dendrogram faithfulness) is also higher for
Pearson on most snapshots:**

| Snapshot | N | Pearson | TailDep |
|---|---|---|---|
| 2020-12-31 | 32 | 0.872 | 0.746 |
| 2021-12-31 | 51 | 0.864 | 0.787 |
| 2022-12-31 | 48 | 0.880 | 0.686 |
| 2023-12-31 | 51 | 0.788 | 0.715 |
| 2024-12-31 | 50 | 0.712 | 0.707 |
| 2025-12-31 | 48 | 0.885 | 0.792 |

Pearson dendrograms preserve their input distances more faithfully —
the tail-dependence distance has higher dimensionality / noise.

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

---

**Phase 4.5 re-analysis on the expanded 2024-2026 OOS window
(2.5 years, T = 868 daily observations, 13 candidate strategies vs HRP
baseline, static weights from 2023-12-31):**

OOS performance (annualized Sharpe via daily returns × √365):

| Strategy | Total return | Sharpe (ann.) | MaxDD | N_eff |
|---|---|---|---|---|
| MVP | **+58.4%** | **0.679** | -38.5% | 5.5 (concentrated) |
| MaxDiv | -4.0% | 0.283 | -64.1% | 9.9 |
| HRP_ShrunkCov | -14.9% | 0.170 | -64.9% | 27.2 |
| HRP | -17.3% | 0.161 | -66.9% | 28.4 |
| HRP_TailDep | -18.1% | 0.158 | -66.9% | 30.1 |
| HRP_Denoised | -18.8% | 0.156 | -67.5% | 31.3 |
| HRP_TailDepShrunk | -19.8% | 0.149 | -67.6% | 31.7 |
| HRP_VolStd | -20.0% | 0.138 | -66.8% | 30.1 |
| HRP_Detoned | -23.7% | 0.128 | -69.2% | 35.0 |
| HRP_PartialCorr | -23.3% | 0.130 | -69.1% | 34.8 |
| IVP | -23.3% | 0.130 | -69.1% | 34.8 |
| HRP_Dynamic_94 | -25.7% | 0.078 | -66.5% | 24.4 |
| NRP | -32.9% | 0.077 | -73.0% | 43.4 |
| ERC | -33.0% | 0.081 | -73.5% | 46.6 |

**Key finding for the paper — STATIC-WEIGHT OOS doesn't differentiate
HRP variants.** The 13 HRP-family and risk-based comparators bunch
in a narrow band (Sharpe 0.08–0.17). The two concentrated portfolios
(MVP, MaxDiv) appear to "win," but this is a regime artifact: MVP
concentrated 77% in BTC at end-2023, and BTC ran from $42k to $107k+
over 2024-2026. The static-weight test rewards lucky concentration,
not portfolio-construction skill.

LW pairwise Sharpe diff tests vs HRP (annualized ΔSharpe):
- HRP vs HRP_Dynamic_94: **p = 0.042** (HRP wins; dynamic correlation
  hurt in this stable-trending regime).
- HRP vs MVP: p = 0.062 (marginal; concentration won by luck).
- All other pairs: p > 0.08, no detectable difference.

Hansen SPA: across 13 candidates vs HRP, **p_consistent = 0.647,
p_upper = 0.886** — cannot reject H0 that no strategy outperforms
HRP after multiple-testing correction. Even MVP's apparent
outperformance is not significant once we account for trying 13
alternatives.

Bootstrap 95% CIs (annualized Sharpe) are still very wide on 2.5-yr
OOS: HRP CI [-1.13, +1.49]. Quarterly-rebalanced multi-year walk-
forward (Backtest v2 Scenario B) is the only way to get tight CIs;
single static-weight cuts can't.

**Bottom line for paper-grade write-up:** the long static-weight OOS
SHOULD NOT be the headline. Use it only as a "what if you bought and
held the 2023 weights" sanity check. The real comparison is the
monthly-rebalanced backtest with the inference framework wrapped
around each strategy's full return path.

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
