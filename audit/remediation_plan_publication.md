# Publication Remediation Plan

## Purpose
This document converts the due diligence findings into a concrete remediation checklist, organized by file and ordered by publication impact.

Primary source memos:

- `audit/final_due_diligence_memo.md`
- `audit/phase1_consistency_due_diligence.md`
- `audit/phase2_pit_universe_due_diligence.md`
- `audit/phase3_backtest_inference_due_diligence.md`
- `audit/phase4_manuscript_due_diligence.md`

## Priority Levels

- `P0`: must fix before treating the paper as submission-ready.
- `P1`: strongly recommended before circulation to reviewers/advisors.
- `P2`: cleanup and hardening.

## P0 Changes

### 1. `paper/ucema_journal/main.tex`
Status: `P0`

Required changes:

1. Replace global claims about clustering irrelevance with design-scoped claims.
2. Replace language that treats NCO's edge as established-but-underpowered with unresolved-under-current-design language.
3. Add one explicit sentence clarifying that joint inference is computed on the common aligned panel, which is shorter than some per-strategy daily series.
4. Add one explicit sentence clarifying that the cost grid is a stylized flat turnover-based transaction-cost model with a liquidation premium.
5. Reconcile sample-size references:
   - `2297`
   - `2299`
6. Reconcile rounded return references:
   - `+417%`
   - `+418%`
7. Clarify the final comparison-set framing:
   - strategy vs benchmark
   - treatment of `HODL_BTC`
8. Replace `\date{\today}` with a fixed date or version string.

Suggested wording replacements:

- Replace:
  - `changing the clustering step does nothing`
- With:
  - `within this dataset, horizon, and search space, clustering-step modifications did not produce a detectable outperformance over baseline HRP`.

- Replace:
  - `time spent engineering a better distance metric ... is time wasted`
- With:
  - `on the evidence in this study, incremental effort on the clustering step appears lower-yield than effort on the allocation step`.

- Replace:
  - `is a limitation of sample size, not of the effect`
- With:
  - `is consistent with limited power under the current design, leaving the effect unresolved rather than established`.

Acceptance condition:
- The manuscript no longer overstates any conclusion beyond what committed artifacts support.

### 2. `specs/08_universe.md`
Status: `P0`

Required changes:

1. Rewrite the spec so it describes the current final PIT universe, not the obsolete top-30 / 2019-2024 build.
2. Remove active-contract language tied to CoinGecko historical market-cap ingestion.
3. Remove contradictory references to both top-30 and top-50 in the same spec.
4. Replace outdated acceptance criteria tied to a `30 ± 3` median universe size.
5. Replace obsolete interface examples that no longer match `src/universe.py`.
6. Add an explicit note that the manual listings file is sparse by design and that uncovered legacy symbols are assumed tradable unless explicitly constrained.

Acceptance condition:
- `specs/08_universe.md` becomes the authoritative description of the actual committed PIT artifact.

### 3. `notebooks/build_pit_universe.ipynb`
Status: `P0`

Required changes:

1. Update `WINDOW_START` / `WINDOW_END` to the final committed universe horizon.
2. Update `top_n` to the actual headline universe definition.
3. Remove obsolete acceptance checks and visuals centered on a target size of `30`.
4. Ensure the notebook's stated inputs/outputs and comments match the current committed `pit_universe.csv`.
5. If this notebook is no longer the authoritative builder, replace it with a new builder notebook or script and mark this notebook as historical.

Acceptance condition:
- The committed builder path can reproduce the committed PIT artifact or the repo clearly states which builder is authoritative.

### 4. `src/universe.py`
Status: `P0`

Required changes:

1. Update `top_n` default to the final headline choice, or remove the default and require explicit passing.
2. Rename `min_median_volume_usd` to a name aligned with the actual variable semantics.
3. Fix docstrings and comments that still say `median` where implementation uses `mean`.
4. Update comments describing the selection logic so they match the final method.

Acceptance condition:
- Implementation terminology is internally consistent and matches the rewritten spec.

### 5. `README.md`
Status: `P0`

Required changes:

1. Reconcile strategy-count language with the final comparison set.
2. Reconcile test-count language with the actual committed tests.
3. Clarify that the final headline artifacts are produced through `rerun_all.py` plus `add_nco_rt.py`, unless the pipeline is unified first.
4. Ensure the project-structure comments reflect the real current role of `paper`, `specs`, `scripts`, and `data`.

Acceptance condition:
- The repository front page no longer contradicts the final paper and artifact state.

### 6. `paper/ucema_journal/README.md`
Status: `P0`

Required changes:

1. Replace `79/79 must pass` with the real current test count.
2. Replace `19-strategy Scenario B table` with the actual final artifact framing.
3. Clarify which CSVs correspond to which final paper tables.
4. If inference uses a common aligned panel, state that briefly here too.

Acceptance condition:
- The paper-facing README is factually aligned with the committed paper artifacts.

## P1 Changes

### 7. `src/backtest.py`
Status: `P1`

Required changes:

1. Decide whether zero-filling missing estimation-window returns is a defended methodology choice.
2. If yes, document it explicitly in code and manuscript support text.
3. If no, replace it with the intended treatment and regenerate affected artifacts.
4. Add a short doc comment clarifying that rebalance costs are booked on the first post-trade day.

Acceptance condition:
- The estimation-window missing-data convention is explicit and defensible.

### 8. `specs/03_backtest_v2.md`
Status: `P1`

Required changes:

1. Reconcile the current committed comparison set against the spec's larger historical design.
2. Remove or archive strategies not present in the final committed headline roster.
3. Reconcile cost-model language with the implemented flat turnover-cost model unless implementation changes.
4. Reconcile output descriptions with the actual committed artifacts.
5. Clarify that headline metrics and joint inference may use different effective sample definitions if that remains true.

Acceptance condition:
- The backtest spec describes the committed final experiment, not a prior roadmap state.

### 9. `specs/09_inference.md`
Status: `P1`

Required changes:

1. Reconcile required bootstrap iterations with actual implementation, either by changing code or changing spec.
2. Reconcile requested CI outputs with the actual committed CI artifact.
3. Clarify that SPA is a joint test with repeated grid-wide p-values across rows, not distinct per-candidate p-values.
4. Remove or quarantine historical design language that no longer matches the committed final comparison set.

Acceptance condition:
- The inference spec matches what the repository actually computes.

### 10. `scripts/rerun_all.py` and `scripts/add_nco_rt.py`
Status: `P1`

Required changes:

Option A:
- Merge `NCO_RT` into `rerun_all.py` and retire `add_nco_rt.py`.

Option B:
- Keep the two-step pipeline but document it explicitly in `README.md`, `paper/ucema_journal/README.md`, and comments at the top of `rerun_all.py`.

Acceptance condition:
- The final comparison set has one unambiguous generation path.

### 11. `paper/ucema_journal/references.bib`
Status: `P1`

Required changes:

1. Normalize sparse entries where repository evidence already shows incomplete metadata.
2. Ensure comparator citations correspond to the final committed comparison set actually discussed in the paper.

Acceptance condition:
- Bibliography is submission-clean and consistent with manuscript scope.

## P2 Changes

### 12. `tests/`
Status: `P2`

Required additions:

1. Add dedicated PIT tests, ideally in a new `tests/test_universe.py`.
2. Add one integration-style test that validates a known PIT snapshot count or rule.
3. Add one integration-style test that validates one known headline artifact schema or row count.

Acceptance condition:
- The most important data-construction layer has direct test coverage.

### 13. `scripts/` and `notebooks/` comments
Status: `P2`

Required changes:

1. Remove comments and prose that refer to superseded dataset windows and strategy counts.
2. Mark any historical notebook or script clearly if it is no longer authoritative.

Acceptance condition:
- Historical residue no longer creates ambiguity about the live pipeline.

## Recommended Execution Order

1. `specs/08_universe.md`
2. `notebooks/build_pit_universe.ipynb`
3. `src/universe.py`
4. `README.md`
5. `paper/ucema_journal/README.md`
6. `specs/03_backtest_v2.md`
7. `specs/09_inference.md`
8. `scripts/rerun_all.py` and `scripts/add_nco_rt.py`
9. `paper/ucema_journal/main.tex`
10. `paper/ucema_journal/references.bib`
11. `tests/`

## Recommended Editorial Principle

When a choice exists between:

- changing the code to match old prose, or
- changing the prose/specs to match the actual committed final artifacts,

prefer updating the prose and specs first unless there is clear evidence that the implementation is wrong rather than merely under-documented.

## Definition of Done

The repository should be considered publication-ready only when all of the following are true:

1. The PIT artifact and its builder path are aligned.
2. `README`, specs, scripts, artifacts, and manuscript describe the same final experiment set.
3. The manuscript no longer makes claims stronger than the committed repository evidence.
4. The final comparison set has one unambiguous generation path.
5. The sample definition for headline metrics and inference is explicitly documented.
