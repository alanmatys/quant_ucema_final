# Phase 4 Due Diligence

## Scope
This memo covers the manuscript audit for `paper/ucema_journal/main.tex` against committed repository artifacts only.

Focus:

1. Internal consistency of the manuscript.
2. Alignment between manuscript claims and committed data artifacts.
3. Publication-risk language, especially where claims may exceed the demonstrated evidence.
4. Bibliographic and editorial readiness.

No external sources were used.

## Executive Summary
The manuscript is strong in structure, unusually honest in several places, and materially above the level of a typical student quant paper. Its central framing is valuable: distinguishing clustering-step changes from allocation-step changes.

The manuscript is not yet publication-grade. The main issues are not literary polish, but evidentiary calibration and traceability. Several passages make stronger claims than the repository's own artifacts cleanly support, and there are still internal inconsistencies in sample framing, counts, and metric conventions.

The largest manuscript risk is not a false result but an overstated conclusion.

## Findings

### Finding 1. The manuscript's headline sample description is internally inconsistent with the committed performance and inference artifacts
Severity: High

Evidence:
- `main.tex` line 518 says the headline table uses `2,297 daily out-of-sample observations`.
- `data/backtest_v2_rebalance_results.csv` reports `n_days = 2299` for the main strategies.
- `data/backtest_v2_daily_returns.pkl` contains a `2299 x 23` panel.
- `scripts/rerun_all.py` computes inference on the common aligned panel after `dropna()`, which yields `2297` rows once `HODL_BTC` is included.
- `main.tex` line 736 later refers to `2,299-day return series` in the sweeps section.

Assessment:
- The manuscript currently mixes at least two different effective sample lengths:
  - `2299` for per-strategy daily panels,
  - `2297` for joint aligned inference.
- This is not fatal, but it must be stated clearly and consistently.

Required remediation:
- Explicitly distinguish:
  - per-strategy OOS sample length,
  - aligned common-sample length used for inference.

### Finding 2. The manuscript's strategy framing is still mixed between `strategy` and `benchmark`
Severity: High

Evidence:
- `main.tex` lines 37-43 and 1145-1148 state `22 portfolio strategies`.
- `main.tex` line 528 says `HODL_BTC is a passive benchmark, not a strategy`.
- The committed headline CSV has 23 rows total: 21 strategy rows plus `HODL_BTC` plus `NCO_RT`.

Assessment:
- The paper uses `22 strategies` as a narrative shorthand, but the benchmark/strategy boundary is not fully normalized.
- This is manageable, but it should be made precise.

Required remediation:
- Add one explicit sentence defining the headline roster, for example:
  - `21 constructed strategies plus one passive BTC benchmark, with NCO_RT included in the final comparison set`.

### Finding 3. The manuscript makes stronger claims about clustering irrelevance than the evidence strictly certifies
Severity: High

Evidence:
- `main.tex` lines 613-615: `changing the clustering step does nothing; changing the allocation step is what moves performance`.
- `main.tex` lines 1082-1084: `time spent engineering a better distance metric or a learned embedding is ... time wasted`.
- `main.tex` lines 1155-1159: `the clustering step ... does not matter`.

Assessment:
- The repository supports a narrower claim:
  - within this dataset, sample, and search space, no clustering-step variant achieved a detectable outperformance over baseline HRP after correction.
- It does not support a general statement that clustering-step work is globally irrelevant.

Required remediation:
- Downgrade the language to design-scoped claims.
- Replace universal formulations with statements tied to this universe, horizon, and search space.

### Finding 4. The manuscript sometimes treats NCO's non-significant edge as effectively real
Severity: High

Evidence:
- `main.tex` lines 55-61 and 1095-1098 frame NCO's edge as economically meaningful but statistically unresolved.
- `main.tex` line 961 says non-significance `is a limitation of sample size, not of the effect`.

Assessment:
- This overreaches.
- Low power can explain non-rejection, but it does not justify treating the effect as established.
- The right interpretation is unresolved, not likely-true-by-default.

Required remediation:
- Rephrase these passages to say the effect remains unresolved under the current design.

### Finding 5. The manuscript's cost-language overstates what the implemented cost model actually is
Severity: High

Evidence:
- The paper repeatedly refers to multi-cost scenarios in a realistic execution setting.
- Phase 3 audit showed that the implemented cost model is a flat turnover-based model with a liquidation premium, not a volume-scaled slippage model.

Assessment:
- The cost discussion is acceptable if framed as a coarse sensitivity analysis.
- It is overstated if framed as a liquidity-aware execution model.

Required remediation:
- Clarify in the manuscript that the cost grid is a stylized flat-turnover sensitivity model, not a full liquidity impact model.

### Finding 6. The manuscript does not disclose the sample-alignment convention used for inference
Severity: High

Evidence:
- `main.tex` presents Ledoit-Wolf and SPA results as if they refer transparently to the headline table.
- The committed pipeline computes inference on the common non-missing panel after adding `HODL_BTC` and, later, `NCO_RT`.

Assessment:
- This is an important reproducibility detail that belongs in a publication-grade methods section or appendix.

Required remediation:
- Add a sentence stating that joint inference is computed on the common aligned daily panel across all compared series.

### Finding 7. The manuscript reflects the current expanded universe better than the specs, but not all supportive repository files agree with it
Severity: Medium

Evidence:
- `main.tex` consistently uses the expanded `146 ever-included` framing.
- This matches the committed PIT artifact.
- But the repository's committed builder notebook and universe spec still describe an older top-30/2019-2024 build path.

Assessment:
- The manuscript is closer to the final experiment state than the specs are.
- This helps the paper's coherence, but it means the paper currently outruns the reproducible builder path in the repo.

Required remediation:
- Reconcile the repo build path to the paper, not the reverse.

### Finding 8. The manuscript contains a visible rounding inconsistency on baseline HRP total return
Severity: Medium

Evidence:
- `main.tex` line 539: baseline HRP total return `+417%`.
- `main.tex` line 772: baseline HRP total return `+418%`.
- The committed CSV value supports `+417%` under conventional rounding.

Assessment:
- Small but publication-visible.

Required remediation:
- Reconcile all table values directly against committed CSVs.

### Finding 9. The manuscript's citation set is usable but not fully publication-polished
Severity: Medium

Evidence:
- `references.bib` includes several sparse entries:
  - `ciciretti2024nrp` has only author, title, year, and `Working paper`.
  - `lohre2020tailhrp` lacks volume/issue/pages/DOI.
  - `liu2022cryptofactors` lacks volume/pages.
  - `ledoitwolf2003shrunk` is sparse.
- `main.tex` cites `Network Risk Parity` in the introduction contribution framing, but the committed headline comparison does not include it in the final headline CSV.

Assessment:
- Bibliography quality is acceptable for a draft but not fully normalized for formal review.
- The introductory comparator framing should also be aligned with the final committed roster.

Required remediation:
- Normalize incomplete bibliography entries.
- Reconcile introduction comparator language with the committed final comparison set.

### Finding 10. The manuscript title block is not version-frozen
Severity: Low

Evidence:
- `main.tex` line 27 uses `\date{\today}`.

Assessment:
- Fine for drafting, not ideal for a stable working-paper or submission artifact.

Required remediation:
- Replace `\today` with a fixed date or version tag.

### Finding 11. The manuscript is strongest when it is cautious, and weakest when it turns interpretive
Severity: Medium

Evidence:
- Strong sections:
  - the abstract's broad framing,
  - the explicit bug correction,
  - the retraction section,
  - the power-analysis logic.
- Weaker sections:
  - lines 952-961 on NCO,
  - lines 963-974 on embedding reversal,
  - lines 1081-1084 in Discussion.

Assessment:
- The paper's honest mode is strong.
- The paper's interpretive mode occasionally runs ahead of what the artifacts certify.

Required remediation:
- Keep the paper in its strongest mode: careful, scoped, and explicit about what remains unresolved.

## Positive Observations

1. The manuscript structure is strong and reviewable.
2. The central conceptual contrast is valuable and clear.
3. The retraction section materially improves credibility.
4. The power-analysis section is one of the paper's best assets.
5. The manuscript does a better job than the specs of reflecting the current expanded experiment state.

## Recommended Redline Priorities

1. Replace all global claims about clustering irrelevance with design-scoped claims.
2. Remove or soften language implying that NCO's edge is real but merely underpowered.
3. Add one explicit note on sample alignment for inference.
4. Add one explicit note on the stylized flat-turnover cost model.
5. Reconcile `2297` vs `2299` and `+417%` vs `+418%`.
6. Clarify the final roster definition: strategy vs benchmark.
7. Normalize bibliography entries and freeze the manuscript date.

## Suggested Replacement Language

Instead of:
- `changing the clustering step does nothing`

Use:
- `within this dataset, horizon, and search space, clustering-step modifications did not produce a detectable outperformance over baseline HRP`.

Instead of:
- `time spent engineering a better distance metric ... is time wasted`

Use:
- `on the evidence in this study, additional effort on the clustering step appears lower-yield than effort on the allocation step`.

Instead of:
- `is a limitation of sample size, not of the effect`

Use:
- `is consistent with limited power under the current design, leaving the effect unresolved rather than established`.

## Phase 4 Conclusion
The manuscript has a credible path to publication quality, but it needs one serious redline pass focused on calibration, not expansion. The core value of the paper is already present. The work now is to ensure that every strong statement is no stronger than the committed repository evidence behind it.
