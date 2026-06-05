# Phase 2 Due Diligence

## Scope
This memo covers Phase 2 of the audit:

1. `src/universe.py`
2. `data/pit_universe.csv`
3. `data/pit_universe_summary.csv`
4. `data/binance_listings_manual.json`
5. `notebooks/build_pit_universe.ipynb`

No external sources were used. All findings are based only on repository content.

## Executive Summary
The point-in-time universe is the most important methodological dependency of the paper, and it is not yet publication-grade as a documented pipeline.

The core problem is not that the PIT artifact is obviously wrong. The committed artifact appears internally plausible and consistent with the paper's current 2026-expanded narrative. The problem is that the committed builder notebook and the live implementation/documentation do not describe a single frozen, reproducible universe-generation process.

In practical terms:

1. The committed `pit_universe.csv` reflects a later, expanded universe.
2. The committed `notebooks/build_pit_universe.ipynb` still builds an older top-30, 2019-2024 universe.
3. `src/universe.py` contains terminology and defaults from the older design.
4. The manual listings file is only partial coverage by construction, which may be acceptable, but it is not documented rigorously enough for publication scrutiny.

## Observed Repository State

### Committed PIT artifact
From `data/pit_universe.csv` and `data/pit_universe_summary.csv`:

- Start snapshot: `2018-02-28`
- End snapshot: `2026-05-31`
- Number of snapshots: `100`
- Ever-included symbols: `146`
- Included in last snapshot: `49`
- Dropped during window: `97`
- Maximum included universe size: `51`

This aligns with the current manuscript framing much more closely than with the builder notebook or the stale specs.

### Committed builder notebook
`notebooks/build_pit_universe.ipynb` still defines:

- `WINDOW_START = '2019-01-01'`
- `WINDOW_END = '2024-01-01'`
- `top_n=30`
- acceptance checks around a target of `30`
- a figure line at `30`

This notebook therefore cannot generate the currently committed `pit_universe.csv`.

## Findings

### Finding 1. The committed PIT artifact and the committed builder notebook are from different project generations
Severity: High

Evidence:
- `data/pit_universe.csv` covers `2018-02-28 -> 2026-05-31` with `100` snapshots and `146` ever-included symbols.
- `notebooks/build_pit_universe.ipynb` still sets:
  - `WINDOW_START = '2019-01-01'`
  - `WINDOW_END = '2024-01-01'`
  - `top_n=30`
- The notebook writes `data/pit_universe.csv` directly.

Assessment:
- The repository does not currently contain a committed, canonical builder that can reproduce the committed PIT artifact.
- This is the most serious reproducibility problem inside the PIT pipeline.

Required remediation:
- Either update the notebook to the current final universe definition or replace it with a script/notebook that matches the committed artifact exactly.

### Finding 2. `src/universe.py` still encodes stale defaults from the older top-30 design
Severity: High

Evidence:
- `build_pit_universe(..., top_n: int = 30)` at `src/universe.py:253`.
- The current manuscript and committed artifact reflect a top-50 style steady-state universe, not a top-30 one.
- The notebook also still calls `build_pit_universe(..., top_n=30)`.

Assessment:
- The current default implementation does not match the current paper narrative or committed PIT artifact.
- Even if downstream callers override this explicitly in an uncommitted workflow, the committed code does not express the final headline method.

Required remediation:
- Freeze the headline top-N in one place and make the implementation reflect it explicitly.

### Finding 3. The implementation's own terminology is internally inconsistent
Severity: Medium

Evidence:
- `to_monthly_snapshots` docstring says `rolling_quote_vol_30d is the median`.
- The actual code uses `.rolling(...).mean()`.
- `build_pit_universe` still refers to `30d-median-volume` in the step description.
- The parameter name `min_median_volume_usd` is retained even though the current signal is not a rolling median.

Assessment:
- This weakens auditability because the meaning of the core ranking variable is no longer stable inside the implementation itself.

Required remediation:
- Rename and rewrite comments/docstrings to match the implemented variable.

### Finding 4. The manual listings file is partial by design, but that partiality is a methodological assumption, not just metadata
Severity: High

Evidence:
- `data/binance_listings_manual.json` contains `65` manual rows.
- Of the `146` ever-included symbols in the committed PIT artifact, only `59` have a corresponding manual row.
- `87` ever-included symbols do not have a manual listing row.
- The file description explicitly says that assets not listed there are assumed always tradable during the window.

Assessment:
- This is a strong simplifying assumption.
- It may be acceptable in practice, especially for older symbols, but it is not a neutral omission. It is an active methodological rule.
- The paper currently discusses manual treatment of certain assets, but the repository should describe this assumption much more directly as a universe-construction policy.

Required remediation:
- Add an explicit appendix note or methodology note stating:
  - the manual file is sparse by design,
  - unlisted assets are assumed always tradable over the relevant window,
  - this creates residual model risk in listing-date accuracy for older names.

### Finding 5. The manual listings file successfully enforces the implemented listing window for the symbols it covers
Severity: Low

Evidence:
- No symbol was found with first inclusion earlier than its `listed_at` among the manually covered rows.
- No included row was found with `date > delisted_at` among manually covered rows.

Assessment:
- This is a positive result.
- The listing-window filter appears to be working correctly for the manually specified subset.

Implication:
- The key issue is not obvious malfunction of the listing filter, but partial coverage and stale documentation.

### Finding 6. There is no dedicated PIT universe test coverage in `tests/`
Severity: Medium

Evidence:
- Searches in `tests/` only surfaced PIT usage inside synthetic backtest tests.
- No dedicated `test_universe.py` or equivalent acceptance-style PIT test file was found.

Assessment:
- The most important data-construction layer in the repository lacks targeted test coverage.
- This is misaligned with the importance of the PIT universe to the paper's claims.

Required remediation:
- Add targeted tests for:
  - no inclusion before `listed_at` for covered symbols,
  - no inclusion after `delisted_at` for covered symbols,
  - age filter enforcement,
  - exclusion-set enforcement,
  - stable reproduction of a known snapshot count.

### Finding 7. The committed notebook still embeds obsolete acceptance criteria
Severity: Medium

Evidence:
- `notebooks/build_pit_universe.ipynb` still checks `AC2` against a steady-state median around `30`.
- The visual diagnostic still draws a dashed line at `30`.
- The committed artifact reaches `49-51` included symbols in steady state.

Assessment:
- This is direct evidence that the notebook was not updated after the universe expansion.

Required remediation:
- Replace the acceptance criteria in the notebook with checks aligned to the final committed universe.

### Finding 8. The universe start-date narrative is not frozen
Severity: Medium

Evidence:
- The committed PIT artifact starts in `2018-02`.
- `specs/08_universe.md` still says monthly snapshots span `Jan 2019 -> Dec 2023` in one place.
- The builder notebook uses `2019-01-01 -> 2024-01-01`.
- The manuscript uses the full expanded 2017/2018-to-2026 framing.

Assessment:
- The repository currently contains three different temporal narratives for the PIT universe.

Required remediation:
- Freeze one final narrative and align all files to it.

## Positive Observations

1. The committed artifact does appear structurally plausible.
2. The inclusion logic around `listed_at` and `delisted_at` appears consistent for the manually covered subset.
3. The manual file contains clear per-symbol notes and distinguishes effectively important listing events.

## Publication Risk Assessment

### Main risk
The PIT universe may be empirically fine, but the repository does not currently provide a single, committed, reproducible generation path that matches the paper's final artifact.

### Why this matters
For publication review, a point-in-time universe is not just an input file. It is a core methodological object. If the builder, implementation, specs, and artifact do not agree, reviewers can question whether the survivorship-bias fix is fully auditable.

## Recommended Remediation Order

1. Update or replace `notebooks/build_pit_universe.ipynb` so it can reproduce the current committed PIT artifact.
2. Reconcile `src/universe.py` defaults and terminology with the final headline method.
3. Add dedicated PIT tests.
4. Add a short methodology appendix note on the sparse-by-design manual listing file and the assumption for uncovered legacy names.
5. Rewrite `specs/08_universe.md` to match the final artifact and builder.

## Phase 2 Conclusion
The repository's current PIT universe should be treated as a valid research artifact candidate, but not yet as a publication-grade reproducible pipeline. The evidence points to project evolution without a final refreeze of the universe-construction layer.

The next audit phase should move to the backtest engine and its dependence on this PIT artifact, but only after acknowledging that the PIT build path itself needs cleanup and formalization.
