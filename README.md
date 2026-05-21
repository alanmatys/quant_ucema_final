# HRP Variants for Cryptocurrency Portfolios — UCEMA Continuation Paper

A continuation of the MACI 2025 paper *"Hierarchical Risk Parity for
Cryptocurrency Portfolios: A Comparative Analysis"*. It compares **22
portfolio strategies** — twelve HRP-family variants (including four
embedding-based ones), four risk-based comparators, three nested-clustering
optimisers (HERC, NCO, return-tilted NCO), and three crypto-momentum
strategies — on a **survivorship-bias-corrected, point-in-time Binance
universe**, with Ledoit–Wolf and Hansen SPA inference wrapped around every
comparison.

The headline finding: across an exhaustive hyperparameter search the
*clustering* step never separates from baseline HRP, while the *allocation*
step does — NCO reaches Sharpe 0.93 — though no edge clears statistical
significance on a 6.3-year sample.

## People

- **Authors:** Alan Matys, Federico Martin Rodriguez
- **Tutor:** Emiliano Delfau — Universidad del CEMA, Quantitative Finance
  faculty; main tutor of this paper

*Universidad del CEMA — Master in Quantitative Finance*

- Continuation paper: [`paper/ucema_journal/main.pdf`](paper/ucema_journal/main.pdf)
- Original MACI 2025 paper: [`paper/maci_2025/MACIhrp2025final.pdf`](paper/maci_2025/MACIhrp2025final.pdf)

---

## Quick start

The project uses [**uv**](https://docs.astral.sh/uv/) for dependency
management. `uv` installs the correct Python (3.11), every dependency at
the exact versions in `uv.lock`, and the project itself — in one command.

```bash
# 1. install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. from the repo root: create the environment from the lockfile
uv sync

# 3. run anything inside that environment with `uv run`
uv run pytest -q                       # test suite — expect 89 passed
uv run python scripts/power_analysis.py
```

`uv sync` is fully reproducible: it reads `pyproject.toml` + `uv.lock` and
recreates the exact tested environment (159 packages, hash-pinned) in
`.venv/`. No `PYTHONPATH`, no manual Python install, no `conda`.

**Note on `esig`.** The path-signature dependency builds a small C++
extension; on a fresh machine you may need a compiler toolchain (Xcode
Command Line Tools on macOS, `build-essential` on Linux). `uv` handles the
build automatically once the toolchain is present.

**Without uv.** A pinned, hash-locked `requirements.txt` is exported from
the lockfile as a fallback: `pip install -r requirements.txt` into a
Python 3.11 virtual environment, then run scripts with `PYTHONPATH=.`.

---

## Reproducing the results

Every script resolves the repo root from its own location, so it can be
run from any working directory.

```bash
# Test suite (89 tests)
uv run pytest -q

# Full re-run: all 22 strategies x scenarios + inference (~1-2 h; the
# embedding strategies retrain neural encoders at every rebalance)
uv run python scripts/rerun_all.py

# Add the return-tilted NCO candidate and re-run inference
uv run python scripts/add_nco_rt.py

# Power analysis (detectable Sharpe edge at 80% power)
uv run python scripts/power_analysis.py

# Regenerate all paper figures into paper/figures/
uv run python scripts/generate_paper_figures.py

# Combined Hansen SPA across the 547-cell hyperparameter grid
uv run python scripts/hp_sweeps/spa_combined.py

# Compile the paper
cd paper/ucema_journal && latexmk -pdf main.tex
```

All result CSVs and pickled daily-return series live in `data/` and are
committed, so the figures and paper compile without re-running the
backtests.

---

## Project structure

```
quant_ucema_final/
├── pyproject.toml / uv.lock     # dependency + reproducibility manifest
├── requirements.txt             # pinned fallback for non-uv users
├── src/                         # importable package (`from src.X import ...`)
│   ├── portfolio_maker.py        # HRP + variants, NCO/HERC, comparators, momentum
│   ├── hrp_variants.py           # correlation estimators (partial, EWMA, tail-dep)
│   ├── denoising.py              # Marchenko-Pastur denoising + detoning
│   ├── embeddings/               # path-signature / node2vec / contrastive encoders
│   ├── backtest.py               # walk-forward engine + cost models + scenarios
│   ├── inference.py              # block bootstrap, studentized LW, Hansen SPA (arch)
│   └── universe.py               # point-in-time universe builder
├── tests/                        # 89 pytest tests
├── scripts/
│   ├── rerun_all.py              # regenerate every backtest + inference CSV
│   ├── add_nco_rt.py             # backtest the return-tilted NCO
│   ├── power_analysis.py         # statistical power of the design
│   ├── generate_paper_figures.py # regenerate paper figures
│   └── hp_sweeps/                # hyperparameter-sweep scripts + combined SPA
├── data/                         # committed inputs + all result CSVs/pickles
├── notebooks/                    # exploratory + data-collection notebooks
├── specs/                        # design specs (01-09)
└── paper/
    ├── ucema_journal/             # the continuation paper (main.tex, main.pdf)
    └── maci_2025/                 # the original MACI 2025 paper
```

---

## Reference papers

- López de Prado, M. (2016). *Building Diversified Portfolios that
  Outperform Out-of-Sample.* Journal of Portfolio Management.
- López de Prado, M. (2019). *A Robust Estimator of the Efficient
  Frontier* (Nested Clustered Optimization).
- Raffinot, T. (2018). *The Hierarchical Equal Risk Contribution Portfolio.*
- Ledoit, O. & Wolf, M. (2008). *Robust performance hypothesis testing
  with the Sharpe ratio.*
- Hansen, P. R. (2005). *A test for superior predictive ability.*

Full bibliography in [`paper/ucema_journal/references.bib`](paper/ucema_journal/references.bib).

## License

Academic / research use, UCEMA Master in Quantitative Finance.
