"""HP sweep #14: extended multi-channel HRP_PathSig — full OHLCV-derived set.

Follows sweep #12 (ret / ret+vol / ret+vol+ntrades). Here the widest channel
set adds the four further OHLCV-derived channels — intraday range, taker buy
ratio, Amihud illiquidity, average trade size — every one orthogonal to the
close-to-close return series.

Grid: channel_set × level × window × linkage = 3 × 2 × 2 × 3 = 36 cells.
  channel_set ∈ {ret, ret+vol+ntr, ret+full6}
  level ∈ {2,3}  window ∈ {60,90}  linkage ∈ {single,average,complete}

Reports Sharpe AND total return per cell + saves daily returns to a pickle.
"""

from __future__ import annotations

import pickle
import time
import warnings

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/Users/alanmatys/Repos/quant_ucema_final")

from src.backtest import WalkForwardBacktest, COST_SCENARIOS
from src.portfolio_maker import HRP, HRPPathSig
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

# Derive the OHLCV channel panels (pre-tamed → log_transform=False downstream)
panels = derive_channel_panels(prices_long)
print(f"  prices: {prices.shape}  channels: {list(panels)}")

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
CHANNEL_SETS = {
    "ret":          [],
    "ret+vol+ntr":  [panels["log_qvol"], panels["log_ntrades"]],
    "ret+full6":    [panels["log_qvol"], panels["log_ntrades"],
                     panels["log_trade_size"], panels["intraday_range"],
                     panels["taker_buy_ratio"], panels["log_amihud"]],
}
LEVELS   = [2, 3]
WINDOWS  = [60, 90]
LINKAGES = ["single", "average", "complete"]

total = len(CHANNEL_SETS) * len(LEVELS) * len(WINDOWS) * len(LINKAGES)
print(f"\n>>> Running extended HRP_PathSig sweep: {total} cells")

cells = []
results: dict[str, pd.Series] = {}

idx = 0
t_all = time.time()
for chan_name, chan_list in CHANNEL_SETS.items():
    for lvl in LEVELS:
        for win in WINDOWS:
            for link in LINKAGES:
                idx += 1
                name = f"PathSigX_{chan_name}_L{lvl}_w{win}_{link}"
                t0 = time.time()
                fb = {"count": 0, "total": 0}

                def factory(r, lvl=lvl, win=win, link=link, chan_list=chan_list, fb=fb):
                    strat = HRPPathSig(
                        r, window=win, level=lvl, linkage_method=link,
                        extra_channels=chan_list or None,
                        log_transform=False,  # panels are pre-tamed in features.py
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
                    cells.append((name, chan_name, lvl, win, link, sh, tr,
                                  fb["count"], fb["total"], dt))
                    print(f"  [{idx:2d}/{total}] {name:38s}  Sharpe={sh:+.4f}  TR={tr*100:+8.1f}%  "
                          f"Δ={sh - sharpe_base:+.4f}  fb={fb['count']}/{fb['total']}  [{dt:.1f}s]")
                except Exception as e:
                    print(f"  [{idx:2d}/{total}] {name:38s}  ERROR: {e}")
                    cells.append((name, chan_name, lvl, win, link, np.nan, np.nan,
                                  fb["count"], fb["total"], time.time() - t0))

print(f"\nTotal sweep time: {time.time() - t_all:.1f}s")

# ---------- Summary ----------
df = pd.DataFrame(cells, columns=[
    "name", "channels", "level", "window", "linkage", "sharpe",
    "total_return", "fallback", "n_snaps", "secs"])
df["delta_sharpe"] = df["sharpe"] - sharpe_base
df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

print(f"\n========== extended HRP_PathSig sweep — top 12 (HRP base {sharpe_base:+.4f}) ==========")
print(df.head(12)[["name", "sharpe", "total_return", "delta_sharpe", "fallback"]].to_string(
    index=False, float_format=lambda x: f"{x:+.4f}"))

print("\n========== by channel set ==========")
for cs in CHANNEL_SETS:
    sub = df[df["channels"] == cs]
    print(f"  {cs:14s}  mean={sub['sharpe'].mean():+.4f}  max={sub['sharpe'].max():+.4f}  "
          f"meanTR={sub['total_return'].mean()*100:+.0f}%  "
          f"n_beat={int((sub['sharpe']>sharpe_base).sum())}/{len(sub)}")

n_beat = int((df["sharpe"] > sharpe_base).sum())
print(f"\nCells beating HRP baseline: {n_beat}/{len(df)}")
best = df.iloc[0]
print(f"Best cell: {best['name']}  Sharpe={best['sharpe']:+.4f}  "
      f"TR={best['total_return']*100:+.1f}%  Δ={best['delta_sharpe']:+.4f}")

with open("/tmp/hp_sweep_pathsig_ext.pkl", "wb") as f:
    pickle.dump({
        "returns": results, "summary": df,
        "sharpe_base": sharpe_base, "total_return_base": total_ret_base,
        "grid": {"channel_sets": list(CHANNEL_SETS), "levels": LEVELS,
                 "windows": WINDOWS, "linkages": LINKAGES},
    }, f)
print("\nSaved → /tmp/hp_sweep_pathsig_ext.pkl")
