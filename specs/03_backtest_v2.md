# Spec 03 — Backtest v2

**Status:** Current contract for the final committed backtest artifacts

## 1. Scope

This spec defines the backtest protocol that underlies the final
committed result artifacts used by the UCEMA journal paper.

It supersedes earlier broader design drafts that included additional
candidate strategies and alternative universe configurations not present
in the final committed headline outputs.

## 2. Current Committed Universe and Data

The backtest consumes the committed PIT artifact defined in
`specs/08_universe.md`.

Current committed state:

- PIT source artifact: `data/pit_universe.csv`
- PIT artifact summary: `146` ever-included symbols, `97` dropped during
  the window, `100` monthly snapshots
- Strategy evaluation window: `2020-01-01` to `2026-05-18`
- Headline monthly-rebalance window: `76` rebalance snapshots
- Underlying price inputs used by the committed rerun scripts:
  - `data/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv`
  - `data/binance_pit_supplement_2019-2024_1d.csv`

## 3. Estimation and Rebalance Protocol

### 3.1 Estimation window

- Rolling lookback: `365` days
- Per-asset minimum observations: `180` daily returns

Assets failing the minimum-observation filter are dropped from that
snapshot's estimation universe.

### 3.2 Rebalance logic

The backtest engine supports:

1. `calendar`
2. `threshold`
3. `threshold_smoothed`

The headline paper uses the monthly calendar-rebalanced comparison.

### 3.3 Cost model

The committed implementation uses a **stylized flat turnover-based cost
model**:

- cost = `(fee_bps + slippage_bps) * L1 turnover`
- exiting-universe liquidations receive a `2x` liquidation premium

This is a coarse transaction-cost sensitivity model, not a full
liquidity-aware market-impact model.

## 4. Final Comparison Set

### 4.1 Constructed strategies in the committed final comparison

The final committed comparison set contains `27` constructed strategies,
grouped by family for clarity:

**HRP family (13)** — share inverse-variance recursive bisection; only
the dendrogram changes:

1. `HRP` (baseline)
2. `HRP_Detoned`
3. `HRP_Denoised`
4. `HRP_PartialCorr`
5. `HRP_Dynamic_94`
6. `HRP_TailDep`
7. `HRP_TailDepShrunk`
8. `HRP_ShrunkCov`
9. `HRP_VolStd`
10. `HRP_PathSig`
11. `HRP_NodeEmbed`
12. `HRP_Contrastive`
13. `HRP_TS2Vec`

**Risk-based comparators (5):**

14. `IVP`
15. `MVP`
16. `ERC`
17. `MaxDiv`
18. `CRISP`

**Nested-clustering optimisers (4):**

19. `HERC`
20. `NCO`
21. `NCO_RT` (return-tilted NCO with sample-mean tilt)
22. `NCO_CRISP` (NCO with CRISP-regularised allocation)

**Momentum (3):**

23. `CS_MOM_eq21`
24. `RM_MOM`
25. `MOM_HRP`

**Signal-aware extensions (2)** — fed by a walk-forward XGBoost return
forecast (`data/xgboost_mu_predictions.csv`):

26. `NCOML` (NCO with XGBoost-predicted mu replacing the sample-mean tilt)
27. `HRPSigmaMu` (Wuebben 2026 method A1 with L1 normalisation: HRP
    seriation + per-node Cramer's-rule 2x2 mean-variance optimisation
    on cluster representatives, then long-only projected)

### 4.2 Passive benchmark

The headline comparison also includes:

- `HODL_BTC`

It is a benchmark line in the paper and result CSVs, not a constructed
portfolio strategy.

## 5. Artifact Generation Path

The final committed headline artifacts are produced by a multi-step
pipeline. Each splice script backtests a single new strategy on the
headline + post-COVID windows, appends it to the daily-return pickle
and the results CSV, and re-runs Ledoit-Wolf / Hansen SPA / bootstrap-CI
inference on the augmented candidate set.

1. `scripts/rerun_all.py`
   - generates the 21-strategy main roster and the base inference outputs
2. `scripts/add_denoised.py`
   - appends `HRP_Denoised`
3. `scripts/add_nco_rt.py`
   - appends `NCO_RT`
4. `scripts/add_crisp.py`
   - appends `CRISP` (Wuebben 2026 signal-free correlation-shrinkage solver)
5. `scripts/add_ncocrisp.py`
   - appends `NCO_CRISP` (NCO with CRISP-regularised allocation)
6. `scripts/build_xgboost_signals.py`
   - precomputes the walk-forward XGBoost return-forecast panel
     `data/xgboost_mu_predictions.csv` (76 snapshots x 146 assets) with
     `TimeSeriesSplit(3)` CV inside every snapshot's training window;
     also writes `data/xgboost_best_params.csv` for audit
7. `scripts/add_signal_strategies.py`
   - appends `NCOML` and `HRPSigmaMu` (both consume the cached mu panel)

The pipeline is incremental by design: each step takes the previous
step's artifacts and extends them, so partial runs are usable. The
build_xgboost_signals.py step is the only slow one (~12 minutes on the
reference platform); every splice runs in seconds because the mu panel
is precomputed and the strategies only do a covariance solve per
snapshot.

## 6. Metrics and Output Conventions

### 6.1 Headline result CSV

`data/backtest_v2_rebalance_results.csv` stores per-series summary
metrics, including:

- `total_return`
- `ann_return`
- `ann_vol`
- `sharpe`
- `sortino`
- `max_drawdown`
- `calmar`
- `n_days`
- `arith_sharpe_ann`
- turnover/cost/concentration diagnostics

### 6.2 Sharpe conventions

Two Sharpe-style quantities coexist in the committed artifacts:

1. `sharpe`
   - CAGR-style annualized return divided by annualized volatility
2. `arith_sharpe_ann`
   - arithmetic mean divided by standard deviation times `sqrt(365)`

The paper's headline tables display the arithmetic Sharpe convention.

### 6.3 Inference alignment convention

Joint inference is computed on the **common aligned panel** across the
compared daily-return series.

This aligned panel is slightly shorter than some per-strategy daily
series once `HODL_BTC` is included.

## 7. Current Committed Outputs

- `data/backtest_v2_rebalance_results.csv`
- `data/backtest_v2_daily_returns.pkl`
- `data/inference_sharpe_diff.csv`
- `data/inference_spa.csv`
- `data/inference_bootstrap_cis.csv`
- `data/backtest_v2_cost_sensitivity.csv`
- `data/backtest_v2_threshold_results.csv`
- `data/backtest_v2_post_covid_results.csv`
- `data/backtest_v2_post_covid_daily_returns.pkl`
- `data/inference_post_covid_sharpe_diff.csv`
- `data/inference_post_covid_spa.csv`
- `data/backtest_v2_weekly_results.csv`
- `data/backtest_v2_weekly_daily_returns.pkl`
- `data/inference_weekly_spa.csv`
- `data/xgboost_mu_predictions.csv`
- `data/xgboost_best_params.csv`

## 8. Acceptance Criteria

AC1. The spec, `README.md`, `paper/ucema_journal/README.md`, and the
committed artifacts describe the same final comparison set (27
constructed strategies + `HODL_BTC`, 28-row headline table).

AC2. The full incremental generation path (`rerun_all.py` then the six
splice scripts: `add_denoised.py`, `add_nco_rt.py`, `add_crisp.py`,
`add_ncocrisp.py`, `build_xgboost_signals.py`, `add_signal_strategies.py`)
is documented wherever reproduction instructions are shown.

AC3. The paper's headline performance discussion is traceable to
`data/backtest_v2_rebalance_results.csv`.

AC4. The paper's inference discussion is traceable to the committed
`data/inference_*.csv` files and the aligned-panel convention is stated.

AC5. The cost-model description is aligned with the actual flat
turnover-based implementation unless the implementation changes.

## 9. Changelog (Recent Thesis Additions)

In chronological order of integration into the committed comparison set,
expanding the original 22-strategy contract:

- **`HRP_Denoised`** — first-class HRP variant with Marchenko-Pastur
  denoising applied to the correlation matrix before linkage.
- **`CRISP`** — Wuebben (2026) Correlation-Regularised Iterative
  Shrinkage Portfolio (signal-free form, `gamma=0.5`, long-only
  simplex projection). Closes the linear system
  `P_gamma w = 1` with `P_gamma = (1-gamma) diag(Sigma) + gamma Sigma`.
- **`NCO_CRISP`** — Wuebben-style hybrid: NCO's nested-clustering
  skeleton with each min-variance solve replaced by a CRISP solve on
  the corresponding block covariance.
- **`NCOML`** — Signal-aware NCO: keeps NCO's clustering and nesting
  but replaces each min-variance solve by a long-only max-Sharpe solve
  fed by the walk-forward XGBoost mu forecast. Directly tests whether
  `NCO_RT`'s -97% collapse was a signal-quality failure (it was).
- **`HRPSigmaMu`** — Wuebben (2026) method A1 with L1 normalisation:
  single bottom-up tree pass over the HRP dendrogram, each internal
  node solving a 2x2 Cramer's-rule mean-variance system on the left
  vs right cluster representatives. Signed weights are L1-normalised
  per node and long-only projected at the root for comparability.
- **XGBoost signal pipeline** — pooled cross-sectional regressor
  retuned per snapshot via `TimeSeriesSplit(3)` CV inside the
  training window (grid: `max_depth in {3, 5}`, `learning_rate in
  {0.03, 0.1}`, `n_estimators in {200, 500}`). Features: lagged
  returns at 1/5/21/63d, 21d realised vol, cross-sectional rank of
  21d return. Target: next-30-day cumulative log return.

Inference, power analysis and the Deflated Sharpe Ratio have been
re-run after each addition; current candidate count is 27 (28 with
HODL_BTC), SPA `p_consistent = 0.51`, DSR trial set `N = 27` with
`SR0 = 0.37` annualised. The headline thesis (no edge survives the
multiple-testing correction) is preserved across all additions.
