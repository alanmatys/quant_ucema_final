"""Hansen SPA across the combined hyperparameter-sweep grid.

The honest multiple-testing question: across EVERY hyperparameter cell tried
in Phase 8 (correlation / distance / linkage / strategy-specific params /
embedding channels), does the single best configuration significantly beat
baseline HRP — once we correct for having searched hundreds of cells?

Loads every data/hp_sweep_*_results.pkl, extracts each cell's net daily
returns, aligns on common dates, and runs the Hansen (2005) SPA test with
HRP as the benchmark.
"""

from __future__ import annotations

import glob
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd

import sys
# --- resolve the repo root so this script runs from any cwd / machine ---
import os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)

from src.backtest import WalkForwardBacktest, COST_SCENARIOS
from src.portfolio_maker import HRP
from src.inference import hansen_spa_test, sharpe_diff_ledoit_wolf
from src.universe import load_pit_universe

warnings.filterwarnings("ignore")

START, END = "2020-01-01", "2026-05-18"
ANN = np.sqrt(365)


def _daily_returns(cell_val):
    """Extract a daily-returns Series from either pickle structure."""
    if isinstance(cell_val, pd.Series):
        return cell_val
    if isinstance(cell_val, dict) and "daily_returns" in cell_val:
        return cell_val["daily_returns"]
    return None


# ---------- canonical HRP baseline ----------
print("Computing canonical HRP baseline...")
pit = load_pit_universe("data/pit_universe.csv")
existing = pd.read_csv("data/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv", parse_dates=["open_time"])
supp = pd.read_csv("data/binance_pit_supplement_2019-2024_1d.csv", parse_dates=["open_time"])
prices_long = pd.concat([existing, supp], ignore_index=True).drop_duplicates(subset=["symbol", "open_time"])
prices_long["close"] = prices_long["close"].astype(float)
prices = prices_long.pivot(index="open_time", columns="symbol", values="close").sort_index()
bt = WalkForwardBacktest(prices, pit, lambda r: HRP(r), cost_model=COST_SCENARIOS["conservative_cex"])
hrp_ret = bt.run(START, END)["daily_returns"]
sharpe_hrp = hrp_ret.mean() / hrp_ret.std() * ANN
print(f"  HRP baseline: Sharpe={sharpe_hrp:+.4f}  ({len(hrp_ret)} days)")

# ---------- collect every sweep cell ----------
print("\nLoading sweep pickles...")
series: dict[str, pd.Series] = {}
for fn in sorted(glob.glob("data/hp_sweep_*_results.pkl")):
    tag = os.path.basename(fn).replace("hp_sweep_", "").replace("_results.pkl", "")
    d = pickle.load(open(fn, "rb"))
    cell_dict = d.get("returns") or d.get("cells") or d.get("results_avg") or {}
    n_ok = 0
    for name, val in cell_dict.items():
        r = _daily_returns(val)
        if r is None or len(r) < 100:
            continue
        key = f"{tag}::{name.strip()}"
        series[key] = r
        n_ok += 1
    print(f"  {tag:20s}  {n_ok} cells")

print(f"\nTotal cells collected: {len(series)}")

# ---------- align on common dates ----------
mat = pd.DataFrame(series)
mat = mat.join(hrp_ret.rename("__HRP__"), how="inner").dropna()
hrp_aligned = mat.pop("__HRP__")
print(f"Aligned matrix: {mat.shape[0]} days x {mat.shape[1]} cells")

# drop exact-duplicate HRP cells (a cell literally identical to the benchmark)
is_hrp_dup = mat.apply(lambda c: np.allclose(c.values, hrp_aligned.values, atol=1e-12))
if is_hrp_dup.any():
    print(f"  dropping {int(is_hrp_dup.sum())} cells identical to HRP benchmark")
    mat = mat.loc[:, ~is_hrp_dup]

# ---------- per-cell summary ----------
sharpes = mat.apply(lambda c: c.mean() / c.std() * ANN)
delta = sharpes - sharpe_hrp
n_beat = int((sharpes > sharpe_hrp).sum())
print(f"\nCells beating HRP baseline (Sharpe): {n_beat}/{mat.shape[1]}")
top = delta.sort_values(ascending=False).head(10)
print("\nTop 10 cells by Δ Sharpe vs HRP:")
for k, v in top.items():
    print(f"  {k:48s}  Sharpe={sharpes[k]:+.4f}  Δ={v:+.4f}")

# ---------- Hansen SPA ----------
print(f"\nRunning Hansen SPA ({mat.shape[1]} candidates, 5000 bootstrap)...")
t0 = time.time()
spa = hansen_spa_test(mat, hrp_aligned, n_bootstrap=5000, seed=42)
print(f"  done [{time.time()-t0:.0f}s]")

best_cell = sharpes.idxmax()
spa_best = spa.loc[best_cell]
spa_max_stud = spa.loc[spa["studentized"].idxmax()]
print("\n========== Hansen SPA — combined grid ==========")
print(f"  candidates K            : {mat.shape[1]}")
print(f"  cells with +Δ Sharpe    : {n_beat}")
print(f"  max studentized stat    : {spa['studentized'].max():+.3f}  ({spa['studentized'].idxmax()})")
print(f"  SPA p_lower             : {spa['p_lower'].iloc[0]:.4f}")
print(f"  SPA p_consistent        : {spa['p_consistent'].iloc[0]:.4f}")
print(f"  SPA p_upper             : {spa['p_upper'].iloc[0]:.4f}")
print(f"  best cell by Sharpe     : {best_cell}")
print(f"    studentized={spa_best['studentized']:+.3f}  mean_diff={spa_best['mean_diff']:+.2e}")

# ---------- pairwise LW on the best cell ----------
lw = sharpe_diff_ledoit_wolf(mat[best_cell].values, hrp_aligned.values)
print(f"\nLedoit-Wolf pairwise Sharpe-diff test, best cell vs HRP:")
print(f"  Δ Sharpe = {sharpes[best_cell]-sharpe_hrp:+.4f}   p = {lw['p_value']:.4f}")

spa.to_csv("data/hp_sweep_combined_spa.csv")
print("\nSaved → data/hp_sweep_combined_spa.csv")
