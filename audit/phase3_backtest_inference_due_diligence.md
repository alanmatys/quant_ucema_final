# Phase 3 Due Diligence

## Scope
This memo covers Phase 3 of the audit:

1. `src/backtest.py`
2. `src/inference.py`
3. `scripts/rerun_all.py`
4. Committed inference and headline result artifacts in `data/`

No external sources were used. All findings are based only on repository content.

## Executive Summary
The backtest and inference layers are materially better structured than the documentation layer, but they are not yet clean enough for publication-grade traceability.

The strongest issues are:

1. The committed headline performance table and the committed inference tables are not computed on exactly the same observation set.
2. The implemented cost model is materially simpler than the cost model described in the specs.
3. The inference implementation does not satisfy the spec's own stated bootstrap depth or output scope.
4. The backtest engine handles missing data by zero-filling estimation returns after filtering, which is methodologically consequential and not clearly defended.

The code is coherent enough to support a serious research workflow, but not yet precise enough in its contracts for publication review.

## Findings

### Finding 1. Headline performance metrics and inference metrics are computed on different sample lengths
Severity: High

Evidence:
- `data/backtest_v2_rebalance_results.csv` reports `n_days = 2299` for the main strategies.
- The same file reports `n_days = 2297` for `HODL_BTC`.
- The committed `data/backtest_v2_daily_returns.pkl` has shape `(2299, 23)`.
- After `dropna()`, the same daily panel has shape `(2297, 23)`.
- `scripts/rerun_all.py` computes inference using:
  - `rdf = pd.DataFrame(dailyB).dropna()` at line 166
  - `cand = rdf.drop(columns=["HRP"])` at line 187

Assessment:
- The headline table metrics are based on each strategy's own available daily series.
- The inference tables are based on the common intersection after `HODL_BTC` is included and `dropna()` is applied.
- This means the performance table and the inference table are not strictly computed on the same sample for all candidates.

Why it matters:
- The paper presents a unified headline comparison, but the repository currently supports two different effective sample definitions:
  - `2299` for strategy metrics,
  - `2297` for joint inference.

Required remediation:
- Freeze one explicit inference sample convention and state it clearly.
- Either compute all headline metrics on the common sample, or disclose that inference uses a common aligned subpanel.

### Finding 2. The implemented cost model is materially simpler than the cost model described in `specs/03_backtest_v2.md`
Severity: High

Evidence:
- The spec describes conservative CEX slippage as `2 bps × (notional / 30-day median volume)` and similar liquidity-aware formulations.
- `src/backtest.py` implements a flat linear cost model:
  - `base = (fee_bps + slippage_bps) / 10_000.0`
  - cost is `base * turnover` with a liquidation multiplier.
- No asset-level or volume-dependent slippage term is used.

Assessment:
- This is not a minor implementation detail. The live backtest uses a much simpler friction model than the spec describes.
- The paper's discussion of cost sensitivity should therefore be understood as sensitivity to flat turnover costs, not to liquidity-scaled execution costs.

Required remediation:
- Either implement the spec's liquidity-aware cost model or rewrite the spec and paper support text to reflect the simpler model actually used.

### Finding 3. The stationary bootstrap CI implementation does not satisfy the spec's own required iteration count or metric scope
Severity: High

Evidence:
- `specs/09_inference.md` requires at least `10,000` iterations for each CI and 95% CIs for `Sharpe, Sortino, MaxDD, Calmar, Annualized Return`.
- `scripts/rerun_all.py` computes CIs using `n_iter=2000` at lines 172-175.
- `data/inference_bootstrap_cis.csv` contains only:
  - `name`
  - `sharpe`
  - `ci_lower_95`
  - `ci_upper_95`

Assessment:
- The implemented CI pipeline does not satisfy the repository's own inference spec.
- This is currently a spec/implementation mismatch, not just a documentation typo.

Required remediation:
- Either raise the implementation to the spec, or reduce the spec to the implementation and explain the reason.

### Finding 4. `_returns_for_window` zero-fills missing returns after the min-period filter
Severity: High

Evidence:
- `src/backtest.py`:
  - computes `r = sub.pct_change()`
  - keeps columns with at least `min_periods`
  - returns `r[valid].fillna(0)`

Assessment:
- This is methodologically significant.
- For assets with partial histories inside the 365-day window, missing returns are converted to zero rather than the asset being truncated to its actual observed history.
- This can alter estimated covariance, correlation, volatility, and therefore clustering and weights.

Why it matters:
- The paper relies heavily on covariance- and clustering-based strategies, so this choice is not innocuous.
- The repository currently contains no explicit defense of this imputation choice in the paper-facing methodology.

Required remediation:
- Either justify zero-fill explicitly as a chosen convention, or replace it with a more conservative treatment such as pairwise-complete or post-entry truncation logic.

### Finding 5. The backtest engine has a clear fix for a prior skip-interval bug, but the implications are not surfaced in publication materials
Severity: Medium

Evidence:
- `src/backtest.py` lines 204-207 state that an earlier behavior skipped the entire OOS interval when too few assets passed the filter.
- The corrected behavior now holds previous weights instead.

Assessment:
- This is a positive implementation correction.
- However, because it affects OOS behavior and cumulative returns, it is the sort of backtest logic correction that should be traceable in the project narrative if any published results predate it.

Required remediation:
- Record whether all committed headline artifacts postdate this fix.

### Finding 6. The cost model applies rebalance cost as a subtraction from the first post-trade daily return
Severity: Medium

Evidence:
- `src/backtest.py` lines 307-311 subtract the entire rebalance cost from `net_daily.iloc[0]`.

Assessment:
- This is a coherent bookkeeping convention, but it should be documented as such.
- Otherwise readers may assume costs are distributed differently or embedded in execution price modeling.

Required remediation:
- Add one sentence to methodology or code documentation describing this accounting convention.

### Finding 7. `monthly_rebalance_dates` is not the operative driver of the walk-forward schedule
Severity: Low

Evidence:
- `monthly_rebalance_dates` exists in `src/backtest.py`, but `WalkForwardBacktest.run()` actually uses PIT snapshot dates from `self.pit`.

Assessment:
- Not a bug by itself.
- But it reinforces that the PIT artifact is the true schedule-defining object, which increases the importance of Phase 2 findings.

### Finding 8. `src/inference.py` is conceptually solid, but output semantics differ from the spec's phrasing
Severity: Medium

Evidence:
- `hansen_spa_test()` correctly documents that SPA p-values are grid-wide and therefore identical across rows.
- `specs/09_inference.md` phrasing suggests output `per strategy` p-values of the null.
- `data/inference_spa.csv` indeed repeats the same `p_lower`, `p_consistent`, `p_upper` on every row.

Assessment:
- The implementation is the more statistically honest artifact here.
- The spec should be revised to clarify that SPA is a joint test with per-candidate descriptive scores, not per-candidate distinct joint-test p-values.

### Finding 9. The inference implementation uses a common aligned panel that includes `HODL_BTC` and `NCO_RT`
Severity: Medium

Evidence:
- `scripts/rerun_all.py` builds `dailyB` from `FACTORIES` then adds `HODL_BTC`.
- `scripts/add_nco_rt.py` later splices `NCO_RT` into the same pickle and reruns inference.
- `data/inference_spa.csv` therefore covers all 22 candidates vs `HRP`, including `HODL_BTC` and `NCO_RT`.

Assessment:
- This is internally coherent.
- But it should be stated clearly in the paper-facing reproduction notes because the final inference set is assembled across two scripts.

### Finding 10. The backtest metrics and the paper's displayed `Arith. Sharpe` are not the same metric as `summarize_performance()['sharpe']`
Severity: Medium

Evidence:
- `summarize_performance()` computes Sharpe as `ann_return / ann_vol`.
- `scripts/rerun_all.py` separately computes `arith_sharpe_ann = mean / std * sqrt(365)`.
- `main.tex` explicitly labels the headline table column as `Arith. Sharpe`.

Assessment:
- This is acceptable if disclosed clearly.
- However, two Sharpe conventions coexist in the repo, and they should be consistently named everywhere.

Required remediation:
- Use one repository-wide naming convention distinguishing arithmetic Sharpe from CAGR/vol Sharpe.

## Positive Observations

1. `src/backtest.py` is relatively clean and readable for a research engine.
2. The no-lookahead OOS convention is explicit: out-of-sample returns start strictly after the rebalance date.
3. `src/inference.py` shows good methodological awareness, especially around studentization and SPA delegation to `arch`.
4. The code records strategy-specific diagnostics such as `shrinkage_intensity` and `fallback_used`.

## Publication Risk Assessment

### Main risk
The biggest publication risk at this layer is not that the inference is naive; it is that the repository's contracts do not clearly specify which sample, which cost model, and which bootstrap regime actually underlie the published claims.

### Most material gap
The `2299`-versus-`2297` split between headline metrics and inference is the kind of detail that should be made explicit before publication submission.

## Recommended Remediation Order

1. Freeze and document one aligned-sample policy for headline comparison and inference.
2. Reconcile the cost-model spec with the actual implemented cost model.
3. Reconcile the bootstrap spec with the actual iteration count and metric coverage.
4. Decide whether zero-filling estimation-window missing returns is a defended design choice or a legacy convenience.
5. Clarify the two-step inference-generation path involving `add_nco_rt.py`.

## Phase 3 Conclusion
The backtest and inference code are serious and technically competent, but they still fall short of publication-grade traceability because the methodological contract is not frozen tightly enough. The strongest remaining issue is alignment: sample alignment, cost-model alignment, and spec alignment.
