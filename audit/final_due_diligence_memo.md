# Final Due Diligence Memo

## Project
`quant_ucema_final`

## Branch Audited
`feature/embedding-hrp-variants`

## Scope
This memo consolidates the findings from Phases 1-4 of the repository audit, using only repository contents.

Covered areas:

1. Repository consistency and traceability.
2. Point-in-time universe construction.
3. Backtest and inference pipeline.
4. Manuscript quality and publication readiness.

Supporting phase memos:

- `audit/phase1_consistency_due_diligence.md`
- `audit/phase2_pit_universe_due_diligence.md`
- `audit/phase3_backtest_inference_due_diligence.md`
- `audit/phase4_manuscript_due_diligence.md`

## Executive Judgment
The project is serious, technically competent, and materially stronger than a typical academic quant repository. Its central research framing is credible, and the manuscript contains real scientific strengths: negative-result honesty, explicit bug disclosure, an explicit retraction, and meaningful use of Ledoit-Wolf, SPA, and power analysis.

The project is not yet publication-grade.

The main blockers are not missing models or missing results. They are:

1. The repository does not currently expose one frozen, internally consistent methodological contract.
2. The point-in-time universe artifact and the committed builder path are not aligned.
3. The backtest/inference pipeline uses conventions that are not fully reconciled with the specs or clearly disclosed in the manuscript.
4. Several manuscript conclusions are stronger than the committed repository evidence strictly supports.

The project has a credible path to publication quality, but it needs a focused reconciliation and redline pass before it should be treated as submission-ready.

## Overall Assessment

### What is strong
- Repository structure is clear and research-usable.
- The codebase distinguishes reusable modules, orchestration scripts, data artifacts, tests, and paper assets cleanly.
- Environment reproducibility is relatively strong (`pyproject.toml`, `uv.lock`, pinned requirements).
- The inferential stack is much more thoughtful than average for this type of work.
- The manuscript's central conceptual distinction, clustering step vs allocation step, is valuable.
- The project demonstrates healthy scientific behavior by preserving corrections and retractions instead of hiding them.

### What is weak
- Specs, scripts, artifacts, and paper were not fully re-frozen after major scope expansions.
- The PIT universe, the paper's most important methodological dependency, is not currently backed by a committed builder path that matches the final artifact.
- The implemented cost and bootstrap conventions differ materially from the written specs.
- The manuscript's interpretation occasionally exceeds the evidence.

## Core Findings

### 1. The repository lacks a single current-state contract
Severity: High

Across the repo there are multiple overlapping definitions of:

- number of strategies,
- number of tests,
- sample window,
- PIT universe size,
- headline roster,
- inference scope.

Examples found during the audit:

- `20 strategies` in specs.
- `24 strategies` in `specs/03_backtest_v2.md`.
- `22 portfolio strategies` in `README.md` and `main.tex`.
- `19-strategy` language in `paper/ucema_journal/README.md`.
- `89 tests` in `README.md` vs `79/79` in the paper README.

Interpretation:
- The project evolved materially, but the documentation and contract layers were not synchronized after those changes.

### 2. The PIT universe is the highest methodological risk
Severity: High

The committed `data/pit_universe.csv` is consistent with the manuscript's expanded 2026 framing:

- `146` ever-included symbols
- `97` dropped during window
- `100` snapshots
- end date `2026-05-31`

But the committed builder notebook still describes and builds an older universe:

- `2019-01-01 -> 2024-01-01`
- `top_n=30`
- acceptance criteria centered around `30`

Interpretation:
- The repository contains a plausible final PIT artifact, but not a committed builder path that reproduces that final artifact.

This is the single most important publication-risk issue because the paper's survivorship-bias and PIT claims depend on it.

### 3. The manual listing/delisting file is a methodological assumption, not just metadata
Severity: High

`data/binance_listings_manual.json` contains only partial coverage by design.

Observed state:

- `65` manual rows total.
- `59` of the `146` ever-included symbols are covered.
- `87` ever-included symbols have no manual row.

The repository's effective rule is:
- if a symbol is not covered in the manual file, it is treated as always tradable in the relevant window.

This may be practically acceptable, but it must be documented explicitly as a simplifying assumption.

### 4. The backtest and inference pipelines are serious, but not yet contract-clean
Severity: High

Key issues:

#### 4.1 Sample alignment split
- Headline performance metrics use per-strategy daily series of length `2299` for most strategies.
- Joint inference is computed on the common aligned panel of length `2297` after adding `HODL_BTC` and calling `dropna()`.

The manuscript currently mixes those sample descriptions.

#### 4.2 Cost model mismatch
- Spec language describes liquidity-aware slippage.
- Implementation uses a flat turnover-based cost model with a liquidation premium.

#### 4.3 Bootstrap mismatch
- Spec requires at least `10,000` bootstrap iterations and multiple CI outputs.
- Implementation uses `2000` iterations for CIs and only writes Sharpe CIs.

#### 4.4 Estimation-window zero filling
- Missing returns inside the estimation window are zero-filled after filtering.
- This is a consequential methodological choice for covariance- and clustering-based strategies.

Interpretation:
- The code is coherent enough to run a serious study.
- The methodological contract around the code is not yet sufficiently frozen for publication review.

### 5. The headline results depend on a two-step generation path
Severity: Medium

The final headline artifacts are not generated by `scripts/rerun_all.py` alone.

- `rerun_all.py` builds the main panel.
- `add_nco_rt.py` later splices `NCO_RT` into the result files and reruns inference.

This is not inherently invalid, but it needs to be documented clearly because the published comparison set depends on that second step.

### 6. The manuscript is structurally strong but interpretively over-assertive in several places
Severity: High

Best parts of the manuscript:

- abstract and structure,
- inferential framing,
- explicit bug correction,
- explicit retraction,
- power-analysis section.

Weakest parts:

- statements that clustering-step work `does nothing`,
- statements that further clustering/embedding work is `time wasted`,
- statements implying NCO's edge is real and merely underpowered,
- phrasing such as `not of the effect` rather than `unresolved under current design`.

Interpretation:
- The manuscript is strongest when careful and scoped.
- It becomes vulnerable when it turns universal or interpretive.

## Publication Readiness Rating

### Current state
`Promising but not submission-ready`

### Reason
The work has enough technical and empirical substance to justify further investment, but it still lacks:

- a frozen methodology contract,
- a reproducible PIT builder path matching the final artifact,
- fully reconciled result-generation documentation,
- final calibrated manuscript language.

## Risk Ranking

### Highest risk
1. PIT universe builder/artifact mismatch.
2. Repository-wide contract drift across specs, scripts, data, and paper.
3. Manuscript overclaim around clustering irrelevance and NCO interpretation.

### Medium risk
4. Sample alignment ambiguity between performance tables and inference.
5. Cost-model mismatch between spec and implementation.
6. Insufficient dedicated PIT tests.

### Lower risk
7. Bibliography normalization.
8. Editorial cleanup (`\today`, rounding mismatches, benchmark/strategy labeling).

## Recommended Remediation Plan

### Phase A. Freeze the current-state contract
Create one canonical current-state document with:

- final strategy roster,
- benchmark designation,
- PIT universe definition,
- final sample window,
- artifact generation order,
- test count.

This should become the authoritative reference for the repo and paper.

### Phase B. Rebuild the PIT universe layer into a reproducible final path
Required actions:

1. Update or replace `notebooks/build_pit_universe.ipynb` so it reproduces the committed final PIT artifact.
2. Reconcile `src/universe.py` defaults and naming.
3. Add dedicated PIT tests.
4. Document the sparse manual-listing coverage assumption.

### Phase C. Reconcile the backtest/inference contract
Required actions:

1. Decide and document the effective aligned sample used for inference.
2. Reconcile the cost model description with the implementation.
3. Reconcile bootstrap iteration count and CI scope.
4. Decide whether zero-filling estimation-window missing returns is a defended methodology choice.
5. Make the `rerun_all.py` plus `add_nco_rt.py` workflow explicit.

### Phase D. Redline the manuscript
Required actions:

1. Replace global claims with design-scoped claims.
2. Reframe NCO as unresolved under current power, not established-but-undercertified.
3. Clarify sample alignment and cost-model simplification.
4. Reconcile visible count and rounding inconsistencies.
5. Normalize bibliography and freeze version/date.

## Suggested Final Positioning of the Paper
If revised properly, the strongest defensible positioning is:

`Within a survivorship-bias-corrected Binance PIT universe and the study's search space, clustering-step modifications to HRP did not produce a detectable outperformance over baseline HRP, whereas allocation-step modifications produced materially different outcomes, though even the strongest positive result remained statistically unresolved under the current sample size.`

That positioning is strong, honest, and substantially safer than the manuscript's more universal formulations.

## Final Conclusion
This is a real research project, not a superficial experiment bundle. The paper's core contribution is worth preserving. The work now required is not additional modeling breadth but methodological freezing and evidentiary discipline.

If the repository is reconciled and the manuscript is redlined accordingly, the project has a credible path to publication-grade quality.
