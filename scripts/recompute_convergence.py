"""Recompute the HRP_Detoned vs HRP_PartialCorr weight-vector convergence.

The original detoned_vs_partialcorr_rho_timeseries.csv was produced when
HRP_PartialCorr still degenerated to IVP (bug #5). This recomputes the
per-snapshot Spearman rank correlation of the two weight vectors with the
bug-fixed PartialCorr (alpha = 1e-3), and also reports Detoned-vs-IVP.
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
import sys
# --- resolve the repo root so this script runs from any cwd / machine ---
import os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
from scipy.stats import spearmanr
from src.backtest import _returns_for_window
from src.portfolio_maker import HRP, HRPDetoned, HRPPartialCorr, IVP
from src.universe import load_pit_universe
warnings.filterwarnings("ignore")
DATA = str(_ROOT / "data")

pit = load_pit_universe(f"{DATA}/pit_universe.csv")
ex = pd.read_csv(f"{DATA}/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv", parse_dates=["open_time"])
sp = pd.read_csv(f"{DATA}/binance_pit_supplement_2019-2024_1d.csv", parse_dates=["open_time"])
pl = pd.concat([ex, sp], ignore_index=True).drop_duplicates(subset=["symbol", "open_time"])
pl["close"] = pl["close"].astype(float)
prices = pl.pivot(index="open_time", columns="symbol", values="close").sort_index()

snaps = sorted(pd.Timestamp(x) for x in pit["date"].unique()
               if pd.Timestamp("2020-01-01") <= pd.Timestamp(x) <= pd.Timestamp("2026-05-18"))

rows = []
for snap in snaps:
    inc = sorted(pit[(pit["date"] == snap) & pit["included"]]["symbol"].tolist())
    train = _returns_for_window(prices, snap, inc, 365, 180)
    if train.shape[1] < 5:
        continue
    wd = HRPDetoned(train).get_weights()
    wp = HRPPartialCorr(train).get_weights()
    wi = IVP(train).get_weights()
    idx = wd.index
    rho_dp = spearmanr(wd.values, wp.reindex(idx).values).correlation
    rho_di = spearmanr(wd.values, wi.reindex(idx).values).correlation
    rho_pi = spearmanr(wp.values, wi.reindex(idx).values).correlation
    rows.append({"date": snap.date(), "n": train.shape[1],
                 "rho_detoned_partialcorr": rho_dp,
                 "rho_detoned_ivp": rho_di,
                 "rho_partialcorr_ivp": rho_pi})

df = pd.DataFrame(rows)
df.to_csv(f"{DATA}/detoned_vs_partialcorr_rho_timeseries.csv", index=False)
print(df.describe().round(3).to_string())
print()
for c in ["rho_detoned_partialcorr", "rho_detoned_ivp", "rho_partialcorr_ivp"]:
    print(f"  {c:28s} mean={df[c].mean():.3f}  median={df[c].median():.3f}  "
          f"min={df[c].min():.3f}  max={df[c].max():.3f}")
print(f"\nSaved -> detoned_vs_partialcorr_rho_timeseries.csv ({len(df)} snapshots)")
