# Spec 09 — Statistical Inference Framework

**Status:** Current contract for the final committed inference artifacts

## 1. Scope

This spec defines the inference layer used by the final committed result
artifacts and the UCEMA journal paper.

It supersedes earlier draft wording that described a broader candidate
set or a more expansive CI output contract than the currently committed
artifacts provide.

## 2. Implemented Components

The committed implementation in `src/inference.py` provides:

1. Stationary block bootstrap
2. Ledoit-Wolf Sharpe-difference test
3. Hansen SPA test
4. Cluster-stability utilities

## 3. Current Contract

### 3.1 Ledoit-Wolf Sharpe-difference test

- Baseline: `HRP`
- Comparison type: pairwise candidate vs baseline
- Output artifact: `data/inference_sharpe_diff.csv`
- Output columns:
  - `name`
  - `sharpe_diff_vs_hrp`
  - `lw_p_value`

### 3.2 Hansen SPA

- Joint null: no candidate outperforms the benchmark
- Benchmark: `HRP`
- Output artifact: `data/inference_spa.csv`
- Output columns:
  - `mean_diff`
  - `studentized`
  - `p_lower`
  - `p_consistent`
  - `p_upper`

Important contract note:
- SPA is a **joint** test.
- The three p-values are grid-wide and therefore repeat across rows in
  the committed CSV.
- Per-row `mean_diff` and `studentized` are descriptive candidate-level
  statistics, not distinct SPA joint-test p-values.

### 3.3 Bootstrap CIs

The committed headline CI artifact is:

- `data/inference_bootstrap_cis.csv`

Current committed output scope:

- Sharpe point estimate
- 95% lower bound
- 95% upper bound

Current committed implementation depth:

- `2000` bootstrap iterations in the headline rerun script

This is the active contract unless the implementation is later expanded.

### 3.4 Cluster stability

Cluster-stability outputs are handled in companion artifacts/scripts and
use the utilities in `src/inference.py` for:

- cophenetic correlation
- adjusted Rand index

## 4. Sample Alignment Convention

Joint inference is computed on the common aligned daily-return panel
across the compared series.

Because `HODL_BTC` is included in the final comparison set, this aligned
panel is slightly shorter than some per-strategy daily series used for
headline summary metrics.

## 5. Current Committed Candidate Set

The final committed inference layer covers the headline final comparison
set after all splice scripts have run (see `specs/03_backtest_v2.md` §5
for the full generation path). The candidate set used by all inference
artifacts (Ledoit-Wolf, Hansen SPA, bootstrap CIs) is:

- the 21-strategy main roster from `scripts/rerun_all.py`, plus
- `HRP_Denoised` (from `add_denoised.py`)
- `NCO_RT` (from `add_nco_rt.py`)
- `CRISP` (from `add_crisp.py`)
- `NCO_CRISP` (from `add_ncocrisp.py`)
- `NCOML` and `HRPSigmaMu` (from `add_signal_strategies.py`, both fed
  by the walk-forward XGBoost mu panel at
  `data/xgboost_mu_predictions.csv`)

That is **27 constructed strategies** including baseline `HRP`, with
`HODL_BTC` included in the aligned panel as a passive benchmark.

For the SPA the candidate set excludes the baseline `HRP` itself, so
`inference_spa.csv` carries 25 candidates (24 constructed candidates +
`HODL_BTC`) when produced by `scripts/rerun_all.py` and is augmented to
26 then 27 candidates by the splice scripts that follow.

## 6. Acceptance Criteria

AC1. `specs/09_inference.md`, `src/inference.py`, and the committed
`data/inference_*.csv` files describe the same output contract.

AC2. The paper states that joint inference is computed on the aligned
common panel.

AC3. The paper does not describe SPA as if it produced distinct
candidate-specific joint-test p-values.

AC4. Any future expansion of CI outputs or bootstrap iteration counts
must update both this spec and the committed generation script.

## 7. Changelog (Recent Thesis Additions)

Inference has been re-run after each new strategy was added to the
comparison set, growing the candidate count for the SPA and DSR:

- After `add_denoised.py`: candidate set grew to include `HRP_Denoised`.
- After `add_nco_rt.py`: candidate set grew to include `NCO_RT`; this
  strategy collapses to -97% total return in the headline window and
  is the most-negative entry in the Ledoit-Wolf table.
- After `add_crisp.py` and `add_ncocrisp.py`: both Wuebben (2026)
  strategies cleared the pairwise Ledoit-Wolf test against `HRP` at
  `p < 0.05` (`CRISP` p=0.012, `NCO_CRISP` p=0.011), but neither
  survives the Hansen SPA correction once it accounts for the search
  breadth.
- After `add_signal_strategies.py`: `NCOML` (Sharpe 0.80) sidesteps
  `NCO_RT`'s collapse but does not beat signal-blind `NCO`;
  `HRPSigmaMu` (Sharpe 0.61) underperforms baseline `HRP`. Both fail
  to clear the pairwise test (LW p = 0.80 and 0.20 respectively).

Current committed inference values for the joint test:

- Headline Hansen SPA: `p_lower = 0.40`, `p_consistent = 0.51`,
  `p_upper = 0.62`
- Post-COVID Hansen SPA: `p_consistent = 0.38`
- Weekly Hansen SPA (non-embedding subset): `p_consistent = 0.57`

The Deflated Sharpe Ratio (`data/deflated_sharpe.csv`) is computed on
the `N = 27` constructed-strategy trial set; the expected best-of-N
Sharpe under the null is `SR0 = 0.37` annualised. Only `MVP` clears the
conventional `DSR >= 0.95` bar; `CRISP`, `NCO_CRISP` and `NCO` fall
just short at `0.94`. None of these survive the wider `N = 547 + 27`
search-corrected hurdle.
