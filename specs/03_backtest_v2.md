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
   - 78 unique symbols across the window (65 originally + 13 first-supplement
     + 15 second-supplement); steady-state ~49 included per snapshot.

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
