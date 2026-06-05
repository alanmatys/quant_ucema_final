# Spec 04 — Continuation Paper

**Status:** Draft — v2 (revised after deep-research review)
**Owner:** Alan Matys, Federico Rodriguez
**Depends on:** [01](01_denoising.md), [02](02_detoning.md), [03](03_backtest_v2.md), [06](06_hrp_variants.md), [07](07_comparators.md), [08](08_universe.md), [09](09_inference.md)

---

## 1. Motivation

The MACI 2025 paper introduced HRP for crypto portfolios and identified two open
items: denoising and detoning. The continuation paper:

- closes those two future-work items quantitatively,
- introduces **three additional HRP variants** (partial-correlation, EWMA-dynamic,
  lower-tail-dependence — last as appendix),
- benchmarks against a broader comparator set (IVP, MVP, ERC, MaxDiv, Network
  Risk Parity),
- introduces **momentum strategies** and a hybrid **Momentum + HRP** allocation,
- fixes the **survivorship-bias** issue of the original dataset via point-in-time
  universe reconstruction,
- adopts a **statistical inference framework** (Ledoit-Wolf, Hansen SPA, block
  bootstrap) so results are publication-defensible.

## 2. Target Venue & Format

- **Primary target:** UCEMA finance journal / working paper series.
- **Language:** English (consistent with MACI 2025 English version).
- **Format:** Journal-style LaTeX (longer than MACI's 5-page conference format),
  **target 25–35 pages** including figures, references, and appendix
  (revised upward from 15–25 to accommodate the expanded scope).
- **Class:** `article` with a clean preamble (move away from `maciarticle.cls`).
  If UCEMA provides an official template, prefer that.

## 3. Contributions (the paper must make these explicit in the intro)

C1. **Close the MACI 2025 future-work items**: implement and quantitatively
    evaluate **Marchenko–Pastur denoising** and **market-mode detoning** of
    the correlation matrix in HRP.

C2. **Compare two routes to neutralizing the crypto market mode** —
    spectral (detoning) vs. precision-matrix (partial correlation) — and
    report when they agree and when they diverge.

C3. **Extend HRP with regime-aware correlation** (EWMA-dynamic) and **tail-aware
    correlation** (lower-tail dependence, appendix), addressing two well-known
    crypto features.

C4. **Introduce momentum strategies for crypto** (CS_MOM, TS_MOM, RM_MOM) and
    propose a hybrid **Momentum + HRP** that combines alpha generation with
    robust diversification.

C5. **Methodological rigor**: point-in-time universe (no survivorship), multi-cost
    sensitivity, and full statistical-inference testing (Ledoit-Wolf, Hansen SPA,
    stationary block bootstrap).

C6. **Unified, fully reproducible backtest** with single-notebook output and
    every figure/number traceable to a committed CSV.

## 4. Section Outline

1. **Abstract** (≤200 words; write last)
2. **Introduction**
   - 2.1 Motivation & recap of MACI 2025
   - 2.2 Limitations addressed in this work
   - 2.3 Contributions (C1–C6)
   - 2.4 Paper structure
3. **Related Work**
   - 3.1 HRP and its extensions (López de Prado lineage; Lohre tail-aware; Gómez et al. comparative)
   - 3.2 Network and risk-parity literature (Maillard et al.; Ciciretti & Pallotta)
   - 3.3 Crypto momentum literature (Liu et al.; Moskowitz et al.; Barroso & Santa-Clara)
4. **Methodology — Correlation / covariance matrix variants**
   - 4.1 HRP recap (compact)
   - 4.2 Denoising (Marchenko–Pastur)
   - 4.3 Detoning (market-mode removal)
   - 4.4 Partial correlation (precision-matrix-based market-mode removal)
   - 4.5 EWMA-dynamic correlation
   - 4.6 **Lower-tail dependence** (headline — promoted from appendix; rationale:
        economically the most defensible HRP extension for long-only crypto)
   - 4.7 **Ledoit-Wolf shrunk covariance** (variance-stability story)
5. **Methodology — Comparator strategies**
   - 5.1 IVP, MVP (recap)
   - 5.2 Equal Risk Contribution (ERC)
   - 5.3 Maximum Diversification (MaxDiv)
   - 5.4 Network Risk Parity (NRP)
6. **Methodology — Momentum**
   - 6.1 Cross-sectional momentum
   - 6.2 Time-series momentum
   - 6.3 Risk-managed momentum (Barroso & Santa-Clara)
   - 6.4 Momentum + HRP hybrid (and detoned variant)
7. **Data & Universe**
   - 7.1 Source (Binance USDT spot, 2018-12-31 → 2024-01-01)
   - 7.2 Point-in-time universe construction (motivates survivorship-bias fix)
   - 7.3 Stablecoin and wrapper exclusion rules
   - 7.4 Descriptive statistics (BTC dominance, average correlation, eigenvalue spectrum)
8. **Backtest Protocol** (summarize Spec 03)
   - 8.1 Scenarios A (static), B (monthly), C (threshold rebalancing)
   - 8.2 Multi-cost-scenario sensitivity grid
   - 8.3 Evaluation metrics
   - 8.4 Statistical inference (LW Sharpe, Hansen SPA, block bootstrap)
9. **Results**
   - 9.1 Static allocation results (Scenario A)
   - 9.2 Monthly rebalancing (Scenario B) — risk-based strategies
   - 9.3 **Denoising and detoning effects on HRP** (closes MACI future-work)
   - 9.4 **Detoning vs partial correlation** — empirical comparison of the two
        routes to market-mode neutralization (C2).
        **Updated finding (Phase 4.5 re-analysis across 6 year-end snapshots
        in the expanded 2017-2026 dataset):** Spearman ρ between HRPDetoned
        and HRPPartialCorr weight vectors is **0.87-0.998** depending on
        regime — mostly 0.99+ but drops to ~0.87 during the 2021 bull peak
        and 2024 ETF-approval transition where the dominant eigenvector
        shifts dramatically. The convergence is structural but not
        invariant. Backtest v2 must reproduce the time-series of the ρ
        statistic across all rebalance dates (not a single point estimate)
        and report it as `paper/figures/detoning_vs_partial_corr.png`.
   - 9.5 EWMA-dynamic HRP and regime sensitivity (C3)
   - 9.5b **Lower-tail-dependence HRP** — headline result for the crypto
        crash-risk story (revised promotion)
   - 9.5c **Shrunk-covariance HRP** — does Ledoit-Wolf shrinkage alone
        explain the gains, or is denoising/detoning adding value beyond it?
        **Preliminary finding (Phase 2 smoke test, 2022-12-31 PIT snapshot,
        N=27):** LW shrinkage intensity α = 0.094. Spearman(HRP, HRP_ShrunkCov)
        = 0.83 vs Spearman(HRP_Detoned, HRP_ShrunkCov) = 0.66 at that single
        snapshot. **Phase 4.5 multi-snapshot re-analysis confirmed the
        topology vs refinement distinction holds across regimes:**
        HRP_TailDep vs HRP_TailDepShrunk Spearman = 0.998-1.000 across all
        6 snapshots tested (shrinkage is a refinement; topology source
        is what reorders weights). Backtest v2 must report the multi-snapshot
        Sharpe-gap analysis (Spec 09 §4 mini-backtest documented this is
        not testable on static-weight OOS alone).
   - 9.6 Comparator benchmarks (IVP, MVP, ERC, MaxDiv, NRP)
   - 9.7 Momentum strategies
   - 9.8 Hybrid Momentum + HRP (and detoned variant)
   - 9.9 Threshold rebalancing diagnostic (Scenario C) — including the
        turnover-smoothed variant (`w_traded = η·w_prev + (1-η)·w_target`
        with min-trade threshold)
   - 9.9b **Cluster stability diagnostics** — cophenetic correlation per
        snapshot and Adjusted Rand Index between consecutive cluster
        assignments. Addresses "which distance metric produces *stable*
        clusters vs noise". Stability matters as much as point-estimate
        Sharpe for a paper that argues a methodological change.
        **Phase 4.5 re-analysis correction:** earlier Phase 4 finding
        that "TailDep is more stable than Pearson" was data-snooped
        from a single year-pair (2022→2023). Across 5 year-pairs
        (2020-2025), Pearson ARI (mean +0.120) > TailDep ARI (mean
        +0.020) — Pearson wins on 3 of 5 pairs. The paper must report
        the full 5-pair table and NOT claim tail-dep produces more
        stable clusters. The economic argument for HRP_TailDep stands
        (joint-crash co-movement), but the secondary cluster-stability
        argument does not hold up.
   - 9.10 Cost sensitivity (4 cost scenarios)
   - 9.11 Statistical inference results (LW pairwise, Hansen SPA, bootstrap CIs)
10. **Discussion**
    - 10.1 Why detoning helps (or doesn't) in crypto; comparison with partial correlation
    - 10.2 Momentum crashes and risk-managed momentum behavior
    - 10.3 Cost sensitivity — when do net returns disappear?
    - 10.4 What the SPA tells us about multiple testing
    - 10.5 Limitations
11. **Conclusion**
12. **Future Work**
    - Time-varying covariance (DCC-GARCH), regime detection, on-chain factors,
      Black-Litterman with views, network embeddings / path signatures.
13. **References**
14. **Appendix A** — Algorithmic pseudocode (HRP, denoising, detoning,
    partial correlation, EWMA, NRP, MOM_HRP).
15. **Appendix B** — Lower-tail-dependence HRP results.
16. **Appendix C** — Full cost-sensitivity tables.
17. **Appendix D** — Hyperparameter grid results with SPA correction.

## 5. Bibliography (extend MACI 2025 bib)

Already cited:
- López de Prado (2016) — original HRP paper
- López de Prado (2018) — *Advances in Financial Machine Learning*
- López de Prado (2020) — *Machine Learning for Asset Managers*

To add:
- Marchenko & Pastur (1967)
- Laloux et al. (1999) — noise dressing of correlation matrices
- Plerou et al. (2002) — random matrix theory in finance
- Friedman, Hastie & Tibshirani (2008) — graphical lasso
- Maillard, Roncalli & Teïletche (2010) — ERC
- Choueifaty & Coignard (2008) — MaxDiv
- Mantegna (1999) — MST in finance
- Tumminello et al. (2005) — PMFG
- Ciciretti & Pallotta (2024) — Network Risk Parity
- Lohre, Rother & Schäfer (2020) — HRP with tail dependence
- Gómez et al. (2025) — empirical HRP distance comparison
- Liu, Tsyvinski & Wu (2022) — crypto factor model
- Moskowitz, Ooi & Pedersen (2012) — time-series momentum
- Barroso & Santa-Clara (2015) — risk-managed momentum
- Jegadeesh & Titman (1993) — original momentum
- Han, Y. et al. (2024) — realistic-assumption crypto momentum
- Ledoit & Wolf (2003) — "Honey, I Shrunk the Sample Covariance Matrix"
  (the original analytic shrinkage estimator, used by HRPShrunkCov per §4.7)
- Ledoit & Wolf (2004) — "A well-conditioned estimator for large-dimensional
  covariance matrices" (multi-target shrinkage refinement)
- Ledoit & Wolf (2008) — robust Sharpe difference test (used in §8.4 inference)
- Hansen (2005) — SPA test
- Politis & Romano (1994) — stationary block bootstrap
- Politis & White (2004) — automatic block length

## 6. Interface

### 6.1 Directory layout

```
paper/ucema_journal/
├── main.tex                # Top-level document
├── sections/
│   ├── 01_intro.tex
│   ├── 02_related_work.tex
│   ├── 03_methodology_corr.tex
│   ├── 04_methodology_comparators.tex
│   ├── 05_methodology_momentum.tex
│   ├── 06_data.tex
│   ├── 07_backtest.tex
│   ├── 08_results.tex
│   ├── 09_discussion.tex
│   ├── 10_conclusion.tex
│   ├── appendix_a_pseudocode.tex
│   ├── appendix_b_taildep.tex
│   ├── appendix_c_costs.tex
│   └── appendix_d_hyperparams.tex
├── figures/                # Symlinks or copies from paper/figures/
├── references.bib
└── README.md
```

### 6.2 Compilation

`latexmk -pdf main.tex` from `paper/ucema_journal/` produces `main.pdf`.

## 7. Acceptance Criteria

AC1. `main.tex` compiles cleanly (no LaTeX errors, no undefined references,
     no overfull/underfull warnings beyond minor unavoidable cases).

AC2. Every numeric claim in §9 (Results) is traceable to a row in one of the
     committed CSVs from [Spec 03](03_backtest_v2.md) §3.2.

AC3. Every figure is produced by `notebooks/backtest_v2.ipynb` per
     [Spec 03 AC2](03_backtest_v2.md#4-acceptance-criteria).

AC4. Page count within 25–35 pages including refs and appendices.

AC5. Each contribution C1–C6 is addressed by at least one §9 subsection.

AC6. MACI 2025 future-work items are explicitly resolved in §9.3.

AC7. **All headline Sharpe comparisons in §9 are accompanied by a Ledoit-Wolf
     p-value or a Hansen SPA p-value** — no unqualified Sharpe rankings.

AC8. The "detoning vs partial correlation" comparison in §9.4 includes a
     scatter/correlation analysis of the two weight vectors over time
     (per [Spec 03 AC6](03_backtest_v2.md#4-acceptance-criteria) and
     [Spec 06 AC2](06_hrp_variants.md#4-acceptance-criteria)).

AC9. Independent re-read by both authors confirms claims, math, and citations.

## 8. Out of Scope (for this paper; candidates for next continuation)

- Live / paper trading implementation.
- Multi-asset extensions beyond crypto.
- On-chain or alternative-data factors.
- Regime-switching / HMM portfolio models.
- Reinforcement-learning meta-allocators.
- Path-signature or graph-embedding HRP variants (mentioned in future work).

## 9. Open Questions

OQ1. Does UCEMA provide an official journal/working-paper LaTeX template?
     If yes, use it; if no, adopt clean `article` class.

OQ2. Risk-free rate assumption: keep at 0 (matches MACI) or use a stablecoin
     yield proxy? **Default: keep at 0** for consistency.

OQ3. Should the appendix include full Python listings or only pseudocode?
     **Default: pseudocode in appendix**, full code referenced via repo URL.

OQ4. Which cost scenario is the "headline" for the abstract and conclusion?
     **Default: Conservative CEX** (10 bps + linear slippage).

OQ5. Network Risk Parity choice: MST or PMFG for headline results?
     **Default: MST** (simpler; PMFG as a robustness sensitivity).
