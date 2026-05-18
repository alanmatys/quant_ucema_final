# Spec 03 — Backtest v2 (Extended Comparison)

**Status:** Draft — v2 (revised after deep-research review)
**Owner:** Alan Matys, Federico Rodriguez
**Depends on:**
- [01_denoising.md](01_denoising.md), [02_detoning.md](02_detoning.md), [06_hrp_variants.md](06_hrp_variants.md) — HRP variants
- [07_comparators.md](07_comparators.md) — additional comparator strategies
- [08_universe.md](08_universe.md) — point-in-time universe (must be built first)
- [09_inference.md](09_inference.md) — statistical inference framework

---

## 1. Motivation

The MACI 2025 paper compared HRP, IVP, MVP, and HODL baselines on a single backtest.
The continuation paper adds:

- multiple HRP variants (denoised, detoned, partial-correlation, EWMA-dynamic, tail-dependence),
- three additional comparators (ERC, MaxDiv, NRP),
- momentum families (CS_MOM, TS_MOM, RM_MOM, MOM_HRP),
- a point-in-time universe (no survivorship bias),
- multi-cost-scenario sensitivity,
- statistical significance tests across all comparisons.

A single, reproducible backtesting protocol is required so every numeric claim in the
new paper is traceable to one notebook and one (or a small set of) results CSV(s).
**If a result is not in the CSVs emitted by this notebook, it does not go in the paper.**

## 2. Requirements

### 2.1 Universe and data

R1. **Point-in-time universe** per [08_universe.md](08_universe.md):
   - Monthly snapshots, **top-50** by rolling 30-day Binance USDT quote volume
     with entry/exit buffers (revised from top-30 per Spec 08 R2).
   - Source artifact: `data/pit_universe.csv`.
   - Includes assets later delisted (LUNA, FTT, etc.) to remove survivorship.
   - **91 unique symbols** in the underlying price dataset, **87 ever-included**
     in the PIT universe, **38 dropped during the window** (Phase 4.5 expansion).
   - **Date range: 2017-08-17 → 2026-05-18** (≈ 9 years; 100 monthly snapshots).

R1b. **Estimation window**: 365 days (rolling) at each rebalance snapshot,
    with a per-asset min-periods filter requiring ≥ 180 daily observations
    in the window. Assets failing the filter are dropped from that
    snapshot's strategy universe (logged, not crashed). Rationale:
    several newer assets are admitted via the 180-day age filter (Spec 08
    R2) but don't yet have 730 days of history; the 365-day window
    accommodates these without artificially shrinking the universe.

R2. **Price data**: `data/binance_usdt_pairs_pit_2019-2024_1d.csv` (PIT-augmented
    version of the existing daily CSV — extended to include delisted pairs).

### 2.2 Strategies in the comparison

**Risk-based / clustering:**
- HRP (baseline from MACI 2025)
- HRP_Denoised ([Spec 01](01_denoising.md))
- HRP_Detoned ([Spec 02](02_detoning.md))
- HRP_PartialCorr ([Spec 06](06_hrp_variants.md))
- HRP_Dynamic_94 ([Spec 06](06_hrp_variants.md), λ = 0.94 — RiskMetrics default)
- HRP_Dynamic_97 ([Spec 06](06_hrp_variants.md), λ = 0.97 — slower decay)
- HRP_Dynamic_99 ([Spec 06](06_hrp_variants.md), λ = 0.99 — very slow decay)
- HRP_TailDep ([Spec 06](06_hrp_variants.md)) — **headline** (promoted from appendix)
- HRP_ShrunkCov ([Spec 06](06_hrp_variants.md)) — Ledoit-Wolf shrunk covariance baseline
- HRP_VolStd ([Spec 06](06_hrp_variants.md) §2.5) — vol-standardized returns before correlation
- HRP_TailDepShrunk ([Spec 06](06_hrp_variants.md) §2.6) — hybrid (Report 3 #1 pick)
- IVP
- MVP
- ERC ([Spec 07](07_comparators.md))
- MaxDiv ([Spec 07](07_comparators.md))
- NetworkRiskParity ([Spec 07](07_comparators.md))

**Momentum:**
- CS_MOM
- TS_MOM
- RM_MOM
- MOM_HRP
- MOM_HRP_Detoned (bonus variant)

**Baselines:**
- HODL BTC, HODL ETH
- Equal-Weight (1/N)

Total: **24 strategies** in the headline comparison
(11 HRP variants — incl. 3 EWMA λ values + HRP_VolStd + HRP_TailDepShrunk —
+ 5 risk-based/network comparators + 5 momentum + 3 baselines).

Plus a **linkage-method robustness sweep** ([Spec 06](06_hrp_variants.md) §3.3):
selected HRP variants (HRP, HRP_Detoned, HRP_TailDep, HRP_TailDepShrunk) rerun
with `linkage_method ∈ {'single', 'average', 'complete', 'ward'}`. Reported in
a separate appendix table (not bloated into the headline comparison).

### 2.3 Scenarios

- **Scenario A — Static allocation**: weights set once at window start, no rebalancing.
  All risk-based and comparator strategies (momentum strategies skipped — meaningless without rebalancing).

- **Scenario B — Monthly calendar rebalancing**: 30-day cadence. All strategies.

- **Scenario C — Threshold rebalancing**: rebalance only when max absolute weight
  drift exceeds 5% from target. Applied to HRP, HRP_Dynamic, MOM_HRP, RM_MOM —
  the four strategies most sensitive to turnover.

- **Scenario C' — Smoothed rebalancing** (added per second-round deep-research):
  practical refinement of Scenario C with two extra knobs:
  - Linear smoothing: `w_traded = η · w_prev + (1 - η) · w_target` (default η = 0.25).
  - Minimum trade threshold: skip any per-asset weight delta below 25 bps.
  Reduces turnover at the cost of some tracking-error to the target weights.
  Reported alongside Scenario C for the same four strategies. Hyperparameter
  grid: η ∈ {0, 0.25, 0.5}, min-trade threshold ∈ {10, 25, 50} bps.

### 2.4 Cost model

R3. **Multi-scenario cost grid** (results reported for each):
   - **Zero costs** — academic-style headline, for direct comparison with MACI 2025.
   - **Conservative CEX**: 10 bps per side + linear slippage = 2 bps × (notional / 30-day median volume).
   - **Optimistic CEX**: 7.5 bps per side (Binance BNB-discounted) + 1 bp slippage.
   - **Stress**: 10 bps per side + 4 bps slippage (crisis approximation).
   - **Liquidation premium**: when a position is forced out by delisting (PIT), apply 2× the active cost-scenario slippage.

R4. The paper's headline tables use **Conservative CEX**; the other three are
    reported in a sensitivity appendix.

### 2.5 Metrics

R5. Same set across all strategies and scenarios:
   - Total Return (%), Annualized Return (%)
   - Annualized Volatility (%)
   - Sharpe (rf = 0), Sortino, Calmar
   - Maximum Drawdown (%)
   - Turnover (mean per rebalance), Total transaction costs (%)
   - Effective N (1 / Σwᵢ²), Max weight (%)
   - Skewness, Kurtosis
   - **Bootstrap 95% CIs** for Sharpe, Sortino, MaxDD (via [Spec 09](09_inference.md))

R5b. **Cluster stability diagnostics** (per snapshot, for every HRP-family
    strategy that builds a dendrogram):
   - **Cophenetic correlation** — Pearson correlation between the cophenetic
     distance matrix from the dendrogram and the original distance matrix.
     Measures how well the tree preserves pairwise distances (range [-1, 1];
     ≥ 0.7 is "good").
   - **Adjusted Rand Index (ARI)** between cluster assignments at consecutive
     rebalance dates (cutting both dendrograms at K=5 clusters via fcluster).
     Measures cluster stability over time (range [-1, 1]; ≥ 0.5 is "stable").
   - Both are written to `data/cluster_stability.csv` with columns
     `[date, strategy, cophenetic_corr, ari_vs_prev]`.

### 2.6 Statistical inference

R6. Per [Spec 09](09_inference.md):
   - Pairwise **Ledoit-Wolf Sharpe-difference p-value** vs `HRP` baseline.
   - **Hansen SPA p-value** across all 20 strategies vs HRP.
   - **Stationary block bootstrap** CIs for headline metrics.

### 2.7 Reproducibility

R7. All random seeds fixed (numpy seed = 42, bootstrap seed = 42).
R8. Notebook executes end-to-end without manual intervention.
R9. Output paths are deterministic and overwrite-safe.

## 3. Interface

### 3.1 Notebooks

- `notebooks/build_pit_universe.ipynb` ([Spec 08](08_universe.md)) — must run first.
- `notebooks/backtest_v2.ipynb` — runs Scenarios A/B/C across all cost variants;
  produces all CSVs and figures.

### 3.2 Outputs

```
# Metrics (committed)
data/backtest_v2_static_results.csv               # Scenario A
data/backtest_v2_rebalance_results.csv            # Scenario B, all cost scenarios
data/backtest_v2_threshold_results.csv            # Scenario C + C' (smoothed)
data/inference_sharpe_diff.csv                    # LW pairwise vs HRP
data/inference_spa.csv                            # Hansen SPA
data/inference_bootstrap_cis.csv                  # CIs for headline metrics
data/cluster_stability.csv                        # Cophenetic corr + ARI per snapshot
data/hrp_shrunkcov_intensity.csv                  # Time-series of Ledoit-Wolf α

# Time-series artifacts (gitignored, reproducible — CSV or parquet
# depending on whether pyarrow is installed; .gitignore covers both extensions)
data/backtest_v2_weights_history.csv
data/backtest_v2_returns_history.csv

# Figures (committed)
paper/figures/static_cumulative_returns.png
paper/figures/rebalance_cumulative_returns.png
paper/figures/metrics_heatmap.png
paper/figures/turnover_analysis.png
paper/figures/weight_concentration.png
paper/figures/return_distributions.png
paper/figures/risk_return_scatter.png
paper/figures/detoning_effect.png                 # Eigenvalue spectrum before/after
paper/figures/dendrograms_comparison.png          # HRP vs HRP_Detoned vs HRP_PartialCorr
paper/figures/cost_sensitivity_sharpe.png         # Sharpe across cost scenarios
paper/figures/threshold_vs_calendar_turnover.png  # Scenario C diagnostic
paper/figures/pit_universe_evolution.png          # Membership timeline
paper/figures/spa_pvalues_barplot.png             # Hansen SPA results
paper/figures/cluster_stability_timeseries.png    # Cophenetic + ARI over time per strategy
paper/figures/shrinkage_intensity_timeseries.png  # LW α evolution
paper/figures/turnover_smoothing_frontier.png     # Scenario C' η-vs-turnover-vs-net-Sharpe
paper/figures/detoning_vs_partial_corr.png        # Phase 2 finding (ρ ≈ 0.99)
```

## 4. Acceptance Criteria

AC1. Notebook runs top-to-bottom on a clean kernel without errors.

AC2. Every figure listed in §3.2 referenced from [04_paper.md](04_paper.md) is
     actually produced (no orphan figures, no missing figures).

AC3. Every numeric claim in the paper's results section is traceable to a (strategy,
     scenario, cost_scenario, metric) row in one of the committed CSVs.

AC4. Parquet artifacts are gitignored; CSVs and figures are committed.

AC5. For strategies present in both the MACI 2025 paper and Backtest v2 (HRP, IVP,
     MVP), Scenario B + Conservative-CEX Sharpe ratios match the MACI paper within
     **±0.10** (relaxed from ±0.05 because of PIT universe differences). **Any
     larger deviation must be explained explicitly in the paper.**

AC6. `HRP_Detoned` and `HRP_PartialCorr` produce **correlated but not identical**
     weight vectors per [Spec 06 AC2](06_hrp_variants.md#4-acceptance-criteria) —
     evidence that they target the market mode via different mathematical routes.

AC7. PIT universe contains at least one significant delisted asset (e.g. LUNA)
     and its position is liquidated cleanly with the liquidation premium applied.

AC8. SPA test is run across all comparators; results table includes consistent +
     lower + upper p-values per [Spec 09](09_inference.md).

AC9. Headline cost-scenario figures (`Conservative CEX`) replicate qualitatively
     under the other three cost scenarios; "qualitatively" meaning the ranking
     of the top-3 strategies by Sharpe is preserved.

## 5. Out of Scope

- Hyperparameter optimization on the test set (use defaults from the variant
  specs; grid search results are appendix-only, with SPA applied).
- Walk-forward retuning of momentum parameters.
- Tax/funding modeling.
- Intraday (hourly) data layer.

## 6. References

- MACI 2025 paper: [paper/MACI_latex_eng/MACIhrp2025final.tex](../paper/MACI_latex_eng/MACIhrp2025final.tex)
- Existing notebook [notebooks/backtesting.ipynb](../notebooks/backtesting.ipynb) — base loop to extend.
- Existing notebook [notebooks/momentum_backtest.ipynb](../notebooks/momentum_backtest.ipynb) — momentum metrics to merge in.
- Han, Y., et al. (2024). Realistic-assumption momentum in cryptocurrency.

---

## 7. Phase 5a results (headline run)

**Setup:** Scenario B (monthly rebalancing) on the expanded PIT universe,
2020-01-01 to 2026-05-18 (~6.3 years, 76 rebalances), Conservative CEX
cost (10 bps fee + 2 bps slippage + 2× liquidation premium).

| Strategy | Sharpe (ann) | Total Return | MaxDD | Avg N_eff | Avg Turnover |
|---|---|---|---|---|---|
| MVP | **0.950** | **+2035%** | -80.7% | 5.5 | 0.405 |
| HODL_BTC | 0.867 | +1002% | -76.6% | 1.0 | — |
| MaxDiv | 0.793 | +627% | -92.2% | 9.9 | 0.459 |
| **HRP_ShrunkCov** | **0.748** | +436% | -84.4% | 27.2 | 0.308 |
| HRP_TailDep | 0.733 | +396% | -85.2% | 30.1 | 0.317 |
| HRP | 0.732 | +393% | -85.3% | 28.4 | 0.314 |
| HRP_TailDepShrunk | 0.729 | +384% | -85.2% | 31.7 | 0.308 |
| HRP_VolStd | 0.728 | +381% | -84.9% | 30.1 | 0.307 |
| HRP_Detoned | 0.701 | +321% | -84.5% | 35.0 | 0.210 |
| HRP_PartialCorr / IVP | 0.696 | +311% | -84.5% | 34.8 | 0.140 |
| ERC | 0.686 | +277% | -85.9% | 46.6 | 0.131 |

**Statistical inference (Spec 09 framework wrapped around each result):**

- **Bootstrap 95% CIs (Sharpe, annualized):** only MVP [0.27, 1.58],
  MaxDiv [0.00, 1.46], and HODL_BTC [0.06, 1.63] have CIs cleanly above
  zero. All HRP-family variants have CIs straddling zero.

- **LW Sharpe difference test vs HRP baseline:** **NONE of 11 candidates
  reach p<0.05**. HRP_ShrunkCov beats HRP by 0.016 Sharpe (p=0.275, the
  best HRP-family result). MVP beats HRP by 0.217 (p=0.20). HODL_BTC
  beats HRP by 0.135 (p=0.56).

- **Hansen SPA across all candidates:** **p_consistent = 0.526**, p_upper
  = 0.870. **Cannot reject H0 that no strategy outperforms HRP** after
  multiple-testing correction, even with 6.3 years of monthly-rebalanced
  data and 11 candidates.

**Cost sensitivity (5 key strategies × 4 cost scenarios):**

Cost impact is small (Sharpe shifts <0.005 across {zero, conservative,
optimistic, stress} scenarios). Rankings are robust. Crypto's low
transaction costs at monthly cadence don't drive differentiation —
the paper can use Conservative CEX as headline and refer to
`data/backtest_v2_cost_sensitivity.csv` for the full grid.

**Paper-grade findings to report (§9 of [04_paper.md](04_paper.md)):**

1. **HRP_ShrunkCov is the best HRP-family variant** but improvement
   over HRP is small (+0.016 Sharpe) and not statistically significant.
   Validates the Phase 3.5 addition of LW shrinkage as a small refinement.

2. **HRP_TailDep ties HRP exactly** (0.733 vs 0.732). The economic story
   is validated (tail-dep doesn't hurt) but doesn't outperform — a
   neutral finding worth reporting honestly.

3. **HRP_TailDepShrunk does NOT beat HRP_TailDep** (0.729 vs 0.733).
   Report 3's #1 hybrid recommendation does not produce the expected
   improvement on this dataset. Contrary to the Phase 3.5 expectation.
   Adding LW shrinkage on top of tail-dep distance was redundant
   (per the 0.998-1.000 weight Spearman finding) and the small noise
   it introduces slightly hurt.

4. **Detoning/PartialCorr UNDERPERFORM HRP by ~0.03 Sharpe.** Removing
   the market mode in a BTC-led bull market gave up alpha. The
   economically-elegant approach paid the price for being too clever.

5. **MVP "wins" by concentration luck.** It concentrates ~77% in BTC
   throughout the period; BTC ran from $7k (2020) to $107k+ (2026).
   Sharpe 0.95 with avg N_eff = 5.5. **Not skill — regime dependence.**
   The paper must make this explicit.

6. **HODL_BTC beats every diversified portfolio except MVP.** 0.87
   Sharpe, 1002% total return. **A sobering finding the paper must
   address: a passive 100% BTC position outperforms every diversified
   risk-based portfolio considered.** Honest framing: "diversification
   in crypto pays a cost during BTC-dominated regimes that is not
   recovered by lower drawdowns on this sample."

7. **Statistical conclusion:** with 6.3 years and 11 candidates, we
   cannot reject the null that all HRP-family variants are equivalent
   to baseline HRP. The methodological contributions of denoising,
   detoning, partial correlation, EWMA, tail dependence, shrinkage,
   vol standardization, and their combinations do not produce
   statistically detectable Sharpe improvements at this sample size.
   Future work: longer history or higher rebalance frequency may
   resolve. For now, the paper's honest contribution is methodological
   rigor + the empirical findings about variant convergence, not a
   "HRP variant X beats baseline" claim.

**Artifacts produced (Phase 5a):**
- `data/backtest_v2_rebalance_results.csv` — 12-strategy summary metrics
- `data/inference_bootstrap_cis.csv` — 95% block-bootstrap Sharpe CIs
- `data/inference_sharpe_diff.csv` — LW pairwise p-values vs HRP
- `data/inference_spa.csv` — Hansen SPA across all candidates
- `data/hrp_shrunkcov_intensity.csv` — LW α time-series for ShrunkCov variants
- `data/backtest_v2_cost_sensitivity.csv` — 5 strategies × 4 cost scenarios
