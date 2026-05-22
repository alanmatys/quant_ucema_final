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
set after `scripts/add_nco_rt.py` is applied:

- the 21-strategy main roster from `scripts/rerun_all.py`
- plus `NCO_RT`
- with `HODL_BTC` included in the aligned panel

## 6. Acceptance Criteria

AC1. `specs/09_inference.md`, `src/inference.py`, and the committed
`data/inference_*.csv` files describe the same output contract.

AC2. The paper states that joint inference is computed on the aligned
common panel.

AC3. The paper does not describe SPA as if it produced distinct
candidate-specific joint-test p-values.

AC4. Any future expansion of CI outputs or bootstrap iteration counts
must update both this spec and the committed generation script.
