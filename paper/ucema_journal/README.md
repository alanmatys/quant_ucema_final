# UCEMA Journal Paper — Compile Instructions

Continuation of the MACI 2025 HRP paper. Spec: [`specs/04_paper.md`](../../specs/04_paper.md).

## Compile

```bash
cd paper/ucema_journal
latexmk -pdf main.tex
```

Or one-shot:
```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Files
- `main.tex` — single-file paper draft, ~25-30 pages
- `references.bib` — 25 cited works
- Figures: `../figures/*.png` (built by `scripts/generate_paper_figures.py`)

## Result tables
All numbers in §6 (Results) trace to committed CSVs in `data/`:
- `backtest_v2_rebalance_results.csv` — 23-row Scenario B table
  (22 constructed strategies including baseline HRP and the return-tilted
  NCO candidate, plus the passive `HODL_BTC` benchmark)
- `backtest_v2_static_results.csv` — Scenario A
- `backtest_v2_threshold_results.csv` — Scenarios B/C/C′
- `backtest_v2_cost_sensitivity.csv` — cost grid
- `cluster_stability.csv` — 76-snapshot cophenetic + ARI
- `detoned_vs_partialcorr_rho_timeseries.csv` — convergence finding
- `inference_*.csv` — bootstrap CIs, LW Sharpe diff, Hansen SPA
- `hrp_shrunkcov_intensity.csv` — LW α time series

## Reproducing the figures
```bash
PYTHONPATH=. venv/bin/python scripts/generate_paper_figures.py
```

## Reproducing the backtest
```bash
PYTHONPATH=. venv/bin/python -m pytest tests/  # 89 tests must pass
PYTHONPATH=. venv/bin/python scripts/build_pit_universe_final.py
PYTHONPATH=. venv/bin/python scripts/rerun_all.py
PYTHONPATH=. venv/bin/python scripts/add_nco_rt.py
```

Inference is computed on the common aligned daily-return panel after all
compared series are loaded. This is slightly shorter than some
per-strategy daily series once `HODL_BTC` is included.
