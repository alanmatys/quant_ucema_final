# Specifications

Spec-driven development for the UCEMA continuation paper.
Each spec follows the same template: **Motivation → Requirements → Interface → Acceptance Criteria → References**.

## Index

| # | Spec | Purpose |
|---|------|---------|
| 01 | [Denoising](01_denoising.md) | Marchenko–Pastur eigenvalue denoising; `HRPDenoised` |
| 02 | [Detoning](02_detoning.md) | Market-mode removal; `HRPDetoned` |
| 03 | [Backtest v2](03_backtest_v2.md) | Current committed backtest contract for the final headline artifacts |
| 04 | [Paper](04_paper.md) | Continuation-paper outline (UCEMA journal) — 25–35 pages |
| 05 | [Cleanup](05_cleanup.md) | Repo hygiene — branches, stash, duplicate folders |
| 06 | [HRP Variants](06_hrp_variants.md) | `HRPPartialCorr`, `HRPDynamic`, `HRPTailDep` |
| 07 | [Comparators](07_comparators.md) | `ERC`, `MaxDiv`, `NetworkRiskParity` |
| 08 | [Universe](08_universe.md) | Point-in-time monthly universe reconstruction |
| 09 | [Inference](09_inference.md) | Ledoit-Wolf, Hansen SPA, stationary block bootstrap |

## Execution order

```
Phase 0: Specs locked (this commit)
Phase 1: Universe (08)                    ← blocks everything else
Phase 2: Math kernels (01, 02, 06)        ← can run in parallel
Phase 3: Comparators (07)                 ← can run in parallel with Phase 2
Phase 4: Inference framework (09)         ← unit-tested standalone
Phase 5: Backtest v2 (03)                 ← consumes Phases 1–4
Phase 6: Paper (04)                       ← consumes Phase 5
Phase 7: Cleanup (05)                     ← last
```

## Conventions

- Each spec is the **contract** for one workstream. If implementation diverges,
  amend the spec first, then re-implement.
- Cross-references between specs use the file-anchor syntax
  `[NN_name.md#section](NN_name.md#section)` so they remain clickable in GitHub.
- Acceptance criteria are numbered `AC1`, `AC2`, … and each PR description
  must claim which ACs are met.
