"""HP sweep #15: extended encoders — full OHLCV-derived channel set.

Completes the encoder picture from sweep #13 (ret / ret+vol / ret+vol+ntr)
by adding the widest channel set: returns + all six
features.derive_channel_panels() channels.

Grid: encoder × linkage = 2 × 3 = 6 cells, channel_set fixed to ret+full6.
  encoder ∈ {Contrastive, TS2Vec}   linkage ∈ {single, average, complete}
Fixed (identical to sweep #13 so the cells are directly comparable):
  window=40, stride=8, epochs=8, emb_dim=32, hidden=16, batch_size=64.

The ret / ret+vol / ret+vol+ntr cells are NOT re-run — reuse sweep #13.
Reports Sharpe AND total return + saves daily returns to a pickle.
"""

from __future__ import annotations

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
from src.portfolio_maker import HRP, HRPContrastive, HRPTS2Vec
from src.embeddings.features import derive_channel_panels
from src.universe import load_pit_universe

warnings.filterwarnings("ignore")

START = "2020-01-01"
END   = "2026-05-18"

# ---------- Data load ----------
print("Loading data...")
pit = load_pit_universe("data/pit_universe.csv")
existing = pd.read_csv(
    "data/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv", parse_dates=["open_time"],
)
supp = pd.read_csv(
    "data/binance_pit_supplement_2019-2024_1d.csv", parse_dates=["open_time"],
)
prices_long = pd.concat([existing, supp], ignore_index=True).drop_duplicates(
    subset=["symbol", "open_time"]
)
prices_long["close"] = prices_long["close"].astype(float)
prices = prices_long.pivot(index="open_time", columns="symbol", values="close").sort_index()
panels = derive_channel_panels(prices_long)
FULL6 = [panels["log_qvol"], panels["log_ntrades"], panels["log_trade_size"],
         panels["intraday_range"], panels["taker_buy_ratio"], panels["log_amihud"]]
print(f"  prices: {prices.shape}  full channel set: ret + {len(FULL6)} OHLCV-derived")

cost = COST_SCENARIOS["conservative_cex"]

# ---------- Baseline HRP ----------
print("\n>>> Baseline HRP")
t0 = time.time()
bt = WalkForwardBacktest(prices, pit, lambda r: HRP(r), cost_model=cost)
res_base = bt.run(START, END)
ret_base = res_base["daily_returns"]
sharpe_base = ret_base.mean() / ret_base.std() * np.sqrt(365)
total_ret_base = (1 + ret_base).prod() - 1
print(f"  HRP baseline: Sharpe={sharpe_base:+.4f}  TR={total_ret_base*100:+.1f}%  [{time.time()-t0:.0f}s]")

# ---------- Sweep grid ----------
LINKAGES = ["single", "average", "complete"]
FIXED = dict(window=40, stride=8, epochs=8, emb_dim=32, hidden=16, batch_size=64)
ENCODERS = {"Contrastive": HRPContrastive, "TS2Vec": HRPTS2Vec}

results: dict[str, pd.Series] = {}
cells = []
t_all = time.time()

for enc_name, enc_cls in ENCODERS.items():
    print(f"\n{'='*60}\n>>> {enc_name}: ret+full6 × {len(LINKAGES)} linkages\n{'='*60}")
    for i, link in enumerate(LINKAGES, 1):
        name = f"{enc_name}X_ret+full6_{link}"
        t0 = time.time()
        fb = {"count": 0, "total": 0}

        def factory(r, enc_cls=enc_cls, link=link, fb=fb):
            strat = enc_cls(
                r, linkage_method=link, extra_channels=FULL6,
                log_transform=False, **FIXED,
            )
            orig = strat.get_weights
            def wrapped():
                w = orig()
                fb["total"] += 1
                if strat.fallback_used:
                    fb["count"] += 1
                return w
            strat.get_weights = wrapped
            return strat

        try:
            bt = WalkForwardBacktest(prices, pit, factory, cost_model=cost)
            res = bt.run(START, END)
            ret = res["daily_returns"]
            sh = ret.mean() / ret.std() * np.sqrt(365) if ret.std() > 0 else 0.0
            tr = (1 + ret).prod() - 1
            results[name] = ret
            dt = time.time() - t0
            cells.append((name, enc_name, link, sh, tr, fb["count"], fb["total"], dt))
            print(f"  [{i}/3] {name:32s}  Sharpe={sh:+.4f}  TR={tr*100:+8.1f}%  "
                  f"Δ={sh - sharpe_base:+.4f}  fb={fb['count']}/{fb['total']}  [{dt:.0f}s]")
        except Exception as e:
            print(f"  [{i}/3] {name:32s}  ERROR: {e}")
            cells.append((name, enc_name, link, np.nan, np.nan,
                          fb["count"], fb["total"], time.time() - t0))

        # incremental save
        df_p = pd.DataFrame(cells, columns=[
            "name", "encoder", "linkage", "sharpe", "total_return",
            "fallback", "n_snaps", "secs"])
        with open("/tmp/hp_sweep_encoders_ext.pkl", "wb") as f:
            pickle.dump({"returns": results, "summary": df_p,
                         "sharpe_base": sharpe_base,
                         "total_return_base": total_ret_base, "fixed": FIXED}, f)

print(f"\nTotal sweep time: {time.time() - t_all:.0f}s")

# ---------- Summary ----------
df = pd.DataFrame(cells, columns=[
    "name", "encoder", "linkage", "sharpe", "total_return",
    "fallback", "n_snaps", "secs"])
df["delta_sharpe"] = df["sharpe"] - sharpe_base
df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

print(f"\n========== extended encoder sweep — all cells (HRP base {sharpe_base:+.4f}) ==========")
print(df[["name", "sharpe", "total_return", "delta_sharpe", "fallback", "secs"]].to_string(
    index=False, float_format=lambda x: f"{x:+.4f}"))

n_beat = int((df["sharpe"] > sharpe_base).sum())
print(f"\nCells beating HRP baseline: {n_beat}/{len(df)}")
best = df.iloc[0]
print(f"Best cell: {best['name']}  Sharpe={best['sharpe']:+.4f}  "
      f"TR={best['total_return']*100:+.1f}%  Δ={best['delta_sharpe']:+.4f}")

with open("/tmp/hp_sweep_encoders_ext.pkl", "wb") as f:
    pickle.dump({"returns": results, "summary": df, "sharpe_base": sharpe_base,
                 "total_return_base": total_ret_base, "fixed": FIXED}, f)
print("\nSaved → /tmp/hp_sweep_encoders_ext.pkl")
