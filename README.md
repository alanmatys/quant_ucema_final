# HRP Variants for Cryptocurrency Portfolios — UCEMA Continuation Paper

Spec-driven extension of the MACI 2025 paper *"Hierarchical Risk Parity for
Cryptocurrency Portfolios: A Comparative Analysis"*. Implements **9 HRP
variants**, **4 risk-based comparators**, **7 momentum strategies**, a
**point-in-time universe** across 2017-2026, and a **statistical inference
framework** (Ledoit-Wolf Sharpe / Hansen SPA / stationary block bootstrap)
wrapped around every comparison.

## Authors

- Alan Matys
- Federico Martin Rodriguez

*Universidad del CEMA — Quantitative Finance*

📄 **Compiled paper draft:** [paper/ucema_journal/main.pdf](paper/ucema_journal/main.pdf) (14 pages, builds via `latexmk -pdf main.tex`)
🔀 **Pull request:** [#9 — Continuation paper](https://github.com/alanmatys/quant_ucema_final/pull/9)
📚 **Original MACI paper:** [paper/MACI_latex_eng/MACIhrp2025final.pdf](paper/MACI_latex_eng/MACIhrp2025final.pdf)

---

## What's new vs. the MACI 2025 paper

| Dimension | MACI 2025 | This work |
|---|---|---|
| Universe | 5 currently-trading assets | 91 symbols, **point-in-time** (includes LUNA, FTT, etc. before delisting) |
| Window | 2021-2023 (3 yrs) | **2017-08-17 → 2026-05-18** (~9 yrs) |
| Snapshots | 1 backtest | **100 monthly snapshots**, walk-forward |
| Strategies | HRP, IVP, MVP, HODL | 9 HRP variants + 5 risk-based + 7 momentum |
| Cost model | flat 0.1% | **4 cost scenarios** + 2× liquidation premium |
| Inference | Sharpe point estimates | **Bootstrap CIs + LW Sharpe diff + Hansen SPA** |
| Reproducibility | one notebook | 9 specs + 79 unit tests + figure-gen script + reproducibility notebook |

---

## Project Structure

```
quant_ucema_final/
├── specs/                          # 9 spec docs (the contracts)
│   ├── README.md
│   ├── 01_denoising.md             # Marchenko-Pastur
│   ├── 02_detoning.md              # Market-mode removal
│   ├── 03_backtest_v2.md           # Backtest protocol + §7/§8 result tables
│   ├── 04_paper.md                 # Paper outline (Spec 04)
│   ├── 05_cleanup.md               # Repo cleanup
│   ├── 06_hrp_variants.md          # PartialCorr, Dynamic, TailDep, ShrunkCov, VolStd, TailDepShrunk
│   ├── 07_comparators.md           # ERC, MaxDiv, NetworkRiskParity
│   ├── 08_universe.md              # Point-in-time universe
│   └── 09_inference.md             # Bootstrap + LW Sharpe + SPA + cluster stability
├── src/
│   ├── portfolio_maker.py          # HRP, IVP, MVP + 9 HRP variants + 3 comparators + 5 momentum
│   ├── denoising.py                # mp_pdf, denoise_corr_constant_residual, detone_corr
│   ├── hrp_variants.py             # partial_correlation, ewma_correlation, lower_tail_dependence
│   ├── backtest.py                 # WalkForwardBacktest engine + cost models + scenarios A/B/C/C'
│   ├── inference.py                # block bootstrap + LW Sharpe diff + Hansen SPA + cluster stability
│   ├── universe.py                 # PIT universe builder (Binance quote-volume ranking)
│   ├── binance_data.py             # Binance API ingestion (existing)
│   ├── coingecko_data.py           # CoinGecko fallback (existing)
│   └── agent.py                    # Portfolio transition analysis (existing)
├── tests/                          # 79 pytest tests
│   ├── test_denoising.py           (15 tests)
│   ├── test_hrp_variants.py        (24 tests)
│   ├── test_comparators.py         (9 tests)
│   ├── test_inference.py           (13 tests)
│   ├── test_backtest.py            (11 tests)
│   └── test_universe.py            (basic invariants)
├── notebooks/
│   ├── build_pit_universe.ipynb    # PIT universe construction (Phase 1)
│   ├── backtest_v2.ipynb           # Phase 5 reproducibility notebook
│   ├── hrp.ipynb                   # Original static HRP analysis
│   ├── backtesting.ipynb           # Original monthly-rebal backtest
│   ├── corr_matrix_dendrogam.ipynb # Correlation / dendrogram exploration
│   ├── data_collection.ipynb       # Historical data fetching
│   ├── momentum_backtest.ipynb     # Original momentum analysis
│   └── slides_examples.ipynb       # Educational examples
├── scripts/
│   └── generate_paper_figures.py   # Regenerates all 11 paper figures
├── data/                           # All committed
│   ├── binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv  (existing daily)
│   ├── binance_pit_supplement_2019-2024_1d.csv          (extended history — 91 symbols, 2017-2026)
│   ├── pit_universe.csv                                  (Phase 1 output)
│   ├── pit_universe_summary.csv                          (per-snapshot summary)
│   ├── coingecko_candidates.json                         (97 candidate symbols)
│   ├── binance_listings_manual.json                      (Binance listing/delisting dates)
│   ├── backtest_v2_rebalance_results.csv                 (19 strategies × Scenario B)
│   ├── backtest_v2_static_results.csv                    (Scenario A)
│   ├── backtest_v2_threshold_results.csv                 (Scenarios B/C/C')
│   ├── backtest_v2_cost_sensitivity.csv                  (5 strategies × 4 cost scenarios)
│   ├── cluster_stability.csv                             (76 snapshots × cophenetic + ARI)
│   ├── inference_bootstrap_cis.csv                       (95% Sharpe CIs per strategy)
│   ├── inference_sharpe_diff.csv                         (LW pairwise vs HRP)
│   ├── inference_spa.csv                                 (Hansen SPA)
│   ├── hrp_shrunkcov_intensity.csv                       (LW α time series)
│   └── detoned_vs_partialcorr_rho_timeseries.csv         (Spearman ρ per snapshot)
└── paper/
    ├── MACI_latex_eng/             # Original MACI 2025 submission (kept as-published)
    ├── ucema_journal/              # NEW: continuation paper
    │   ├── main.tex
    │   ├── main.pdf                (14 pages, compiled)
    │   ├── references.bib          (25 cited works)
    │   └── README.md               (compile instructions)
    └── figures/                    # 11 paper figures from scripts/generate_paper_figures.py
```

---

## Strategies implemented

### 9 HRP variants (`src/portfolio_maker.py`)

| Variant | Distance source | Bisection covariance | Spec |
|---|---|---|---|
| `HRP` | sample Pearson correlation | sample | López de Prado 2016 |
| `HRPDenoised` | Marchenko–Pastur denoised correlation | sample | [Spec 01](specs/01_denoising.md) |
| `HRPDetoned` | denoised + top eigenvector removed | sample | [Spec 02](specs/02_detoning.md) |
| `HRPPartialCorr` | sparse precision matrix via graphical lasso | sample | [Spec 06 §2.1](specs/06_hrp_variants.md) |
| `HRPDynamic` | EWMA correlation (λ = 0.94/0.97/0.99) | sample | [Spec 06 §2.2](specs/06_hrp_variants.md) |
| `HRPTailDep` | empirical lower-tail dependence (q=0.05) | sample | [Spec 06 §2.3](specs/06_hrp_variants.md) |
| `HRPShrunkCov` | Ledoit-Wolf shrunk correlation | LW shrunk | [Spec 06 §2.4](specs/06_hrp_variants.md) |
| `HRPVolStd` | Pearson on vol-standardised returns | sample | [Spec 06 §2.5](specs/06_hrp_variants.md) |
| `HRPTailDepShrunk` | tail-dependence distance | **LW shrunk** | [Spec 06 §2.6](specs/06_hrp_variants.md) |

### 5 risk-based comparators

`IVP`, `MVP` (existing), plus `ERC` (equal risk contribution),
`MaxDiv` (maximum diversification), `NetworkRiskParity` (MST-based).
Specs: [Spec 07](specs/07_comparators.md).

### 7 momentum strategies

`CrossSectionalMomentum` (equal / inverse-vol / momentum weighted),
`TimeSeriesMomentum` (MA crossover / absolute), `RiskManagedMomentum`
(Barroso & Santa-Clara), `MomentumHRP`, `MomentumHRPDetoned`.

---

## Methodological framework

**Point-in-time universe** ([Spec 08](specs/08_universe.md)).
Monthly snapshots ranked by rolling 30-day Binance USDT quote volume,
top-50 per snapshot. Includes 87 distinct symbols across the window,
38 of which were dropped during 2020-2026 (LUNA, FTT, MIR, SRM, etc.).
Pivoted from CoinGecko market cap (free demo tier caps history at 365
days, blocking 9-year window).

**Walk-forward backtest engine** (`src/backtest.py`).
365-day estimation window with 180-day per-asset min-periods filter.
Four scenarios:
- **A** static (fit once, hold)
- **B** monthly calendar rebalancing
- **C** threshold (5% drift trigger)
- **C′** smoothed (threshold + η blending + 25 bps min-trade filter)

**Cost model**: linear `(fee_bps + slippage_bps) × turnover` with
2× liquidation premium. Four cost scenarios:
zero / conservative_cex (10 + 2 bps) / optimistic_cex (7.5 + 1) / stress (10 + 4).

**Statistical inference** (`src/inference.py`, [Spec 09](specs/09_inference.md)):
- **Stationary block bootstrap** (Politis-Romano 1994) with automatic
  block length (Politis-White 2004)
- **Ledoit-Wolf studentised Sharpe-difference test** (LW 2008) for pairwise
- **Hansen SPA test** (Hansen 2005) with `p_lower`, `p_consistent`, `p_upper`
- **Cluster stability**: cophenetic correlation + ARI between consecutive snapshots

---

## Headline results (Scenario B, Conservative CEX cost, 2020-2026, 76 rebalances)

| Strategy | Arith. Sharpe | Total Return | Max DD | N_eff | Turnover |
|---|---|---|---|---|---|
| MVP | 0.950 | +2035% | -81% | 5.5 | 0.405 |
| **HODL_BTC** | **0.867** | **+1003%** | -77% | 1.0 | — |
| MaxDiv | 0.793 | +627% | -92% | 9.9 | 0.459 |
| **HRP_ShrunkCov** | **0.748** | +436% | -84% | 27.2 | 0.308 |
| HRP_TailDep | 0.733 | +396% | -85% | 30.1 | 0.317 |
| HRP (baseline) | 0.732 | +393% | -85% | 28.4 | 0.314 |
| HRP_TailDepShrunk | 0.729 | +384% | -85% | 31.7 | 0.308 |
| HRP_VolStd | 0.728 | +381% | -85% | 30.1 | 0.307 |
| HRP_Detoned | 0.701 | +321% | -85% | 35.0 | 0.210 |
| HRP_PartialCorr / IVP | 0.696 | +311% | -85% | 34.8 | 0.140 |
| ERC | 0.686 | +277% | -86% | 46.6 | 0.131 |
| MOM_HRP_Detoned (best momentum) | 0.305 | +351% | -88% | 12.0 | 1.420 |

### Key findings
1. **HODL_BTC beats every diversified portfolio** except MVP — a sobering
   reality the paper addresses honestly. Diversification in crypto paid
   a real cost during 2020-2026's BTC-led bull run.
2. **MVP "wins" by accident** — concentrates ~77% in BTC; not skill, regime-dependent.
3. **HRP_ShrunkCov is the best HRP variant** but only +0.016 Sharpe over baseline (p=0.275).
4. **Hansen SPA cannot reject** the null that no variant outperforms baseline HRP
   (p_consistent = 0.526) after multiple-testing correction.
5. **Detoning HURT** (-0.03 Sharpe) — removing the market mode in a BTC-led
   bull market gave up alpha.
6. **MOM_HRP_Detoned more than doubles MOM_HRP** (0.305 vs 0.136) — detoning
   helps the momentum-selected basket where pre-concentrated trends co-move heavily.
7. **Two structural convergences**:
   - `HRP_Detoned` vs `HRP_PartialCorr`: mean Spearman 0.89 across 76 snapshots
     (mostly 0.99+, regime dips to 0.48 at 2024 ETF transition)
   - `HRP_TailDep` vs `HRP_TailDepShrunk`: Spearman 0.998-1.000 in every
     snapshot (most robust convergence finding)
8. **Retracted intermediate finding**: an earlier draft claimed TailDep
   produces more stable clusters than Pearson. That was based on a single
   year-pair; across 76 snapshots Pearson wins 72% of month-pairs.
   The full retraction is in the paper's §6.4.

The paper's contribution is **methodological rigour + structural
characterisation of HRP variant equivalence classes**, NOT a
"variant X beats baseline" claim.

---

## Spec-driven phases (paper-execution branch)

| Phase | Commit | What it added |
|---|---|---|
| 0 | `aabe6fc` | 9 specs + .gitignore audit |
| 1 | `1561031` | PIT universe (Binance quote-volume ranking) |
| 2 | `bf32476` | 6 HRP correlation variants + Ledoit-Wolf shrinkage |
| 3 | `e319eae` | ERC, MaxDiv, NetworkRiskParity |
| 3.5 | `8e7822b` | HRP_VolStd + HRP_TailDepShrunk + linkage parameter |
| 3.6 | `7c7edc7` | PIT top_n → 50, +15 candidates |
| 4 | `2222cd1` | Inference framework (bootstrap, LW, SPA, cluster stability) |
| 4.5 | `88b3c7b` | Dataset extended to 2017-08-17 → 2026-05-18 (91 symbols) |
| 4.5b | `7a10902` | Re-analysis corrections (retracted single-pair findings) |
| 5a | `f0c97be` | Headline backtest: 11 strategies × Scenario B × 6.3 years |
| 5b | `c3d3642` | Momentum + Scenarios A/C/C' + cluster stability ×76 snapshots |
| 5c | `6cbad08` | 11 paper figures + reproducibility notebook |
| 6 | `979909f` | LaTeX paper draft (14 pages, compiles) |
| 7 | `4c486b6` | Repo cleanup per Spec 05 (branches, stash, paper/) |

---

## Reproducing the work

### Setup
```bash
# Option A: conda
conda env create -f environment.yml
conda activate quant_ucema

# Option B: venv + pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn pytest  # added in Phase 2/4
```

### Run the test suite
```bash
PYTHONPATH=. venv/bin/python -m pytest tests/ -q
# Expected: 79 passed
```

### Re-run the headline backtest
```bash
jupyter notebook notebooks/backtest_v2.ipynb
```

### Regenerate the paper figures
```bash
PYTHONPATH=. venv/bin/python scripts/generate_paper_figures.py
# Writes 11 figures to paper/figures/
```

### Compile the paper
```bash
cd paper/ucema_journal
latexmk -pdf main.tex
# Produces main.pdf (14 pages)
```

### Build the PIT universe (heavy — fetches from Binance)
```bash
jupyter notebook notebooks/build_pit_universe.ipynb
```

---

## Reference Papers

**Foundational HRP**
- López de Prado, M. (2016). [Building Diversified Portfolios that Outperform Out-of-Sample](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678). *Journal of Portfolio Management*.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.

**Shrinkage and inference**
- Ledoit, O. & Wolf, M. (2003, 2004, 2008). Honey-I-shrunk-the-sample-covariance + Sharpe-difference trilogy.
- Hansen, P. R. (2005). A test for superior predictive ability. *JBES*.
- Politis & Romano (1994); Politis & White (2004). Stationary block bootstrap.

**Crypto factors and momentum**
- Liu, Y., Tsyvinski, A. & Wu, X. (2022). [Common Risk Factors in Cryptocurrency](https://www.sciencedirect.com/science/article/abs/pii/S1544612325011377). *Journal of Finance*.
- Barroso, P. & Santa-Clara, P. (2015). [Momentum Has Its Moments](https://www.sciencedirect.com/science/article/abs/pii/S154461232030177X). *JFE*.
- Moskowitz, T. J., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *JFE*.

**Comparators and network methods**
- Maillard, Roncalli & Teïletche (2010). ERC properties. *JPM*.
- Choueifaty & Coignard (2008). Maximum Diversification. *JPM*.
- Ciciretti & Pallotta (2024). Network Risk Parity.

Full bibliography (25 entries) in [paper/ucema_journal/references.bib](paper/ucema_journal/references.bib).

---

## License

Academic / research use as part of UCEMA's Quantitative Finance programme.
