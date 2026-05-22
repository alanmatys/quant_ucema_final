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

The final committed comparison set contains `22` constructed strategies:

1. `HRP`
2. `HRP_Detoned`
3. `HRP_PartialCorr`
4. `HRP_Dynamic_94`
5. `HRP_TailDep`
6. `HRP_TailDepShrunk`
7. `HRP_ShrunkCov`
8. `HRP_VolStd`
9. `IVP`
10. `MVP`
11. `ERC`
12. `MaxDiv`
13. `CS_MOM_eq21`
14. `RM_MOM`
15. `MOM_HRP`
16. `HERC`
17. `NCO`
18. `HRP_PathSig`
19. `HRP_NodeEmbed`
20. `HRP_Contrastive`
21. `HRP_TS2Vec`
22. `NCO_RT`

### 4.2 Passive benchmark

The headline comparison also includes:

- `HODL_BTC`

It is a benchmark line in the paper and result CSVs, not a constructed
portfolio strategy.

## 5. Artifact Generation Path

The final committed headline artifacts are produced in two steps:

1. `scripts/rerun_all.py`
   - generates the main 21-strategy panel and the base inference outputs
2. `scripts/add_nco_rt.py`
   - appends `NCO_RT`
   - refreshes the final headline/post-COVID/weekly outputs and
     associated inference files

This two-step generation path is part of the current contract unless the
pipeline is later unified.

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

## 8. Acceptance Criteria

AC1. The spec, `README.md`, `paper/ucema_journal/README.md`, and the
committed artifacts describe the same final comparison set.

AC2. The two-step generation path (`rerun_all.py` then `add_nco_rt.py`)
is documented wherever reproduction instructions are shown.

AC3. The paper's headline performance discussion is traceable to
`data/backtest_v2_rebalance_results.csv`.

AC4. The paper's inference discussion is traceable to the committed
`data/inference_*.csv` files and the aligned-panel convention is stated.

AC5. The cost-model description is aligned with the actual flat
turnover-based implementation unless the implementation changes.
