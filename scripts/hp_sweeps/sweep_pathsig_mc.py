"""HP sweep #12: multi-channel HRP_PathSig grid.

Tests whether appending orthogonal channels (quote volume, trade count) to
the path lets the signature distance beat HRP baseline — the single-channel
sweep #11 found 0/108 cells beat baseline, the hypothesis being that a
log-price-only path is information-equivalent to the return series.

Grid: channel_set × level × window × linkage.
  channel_set ∈ {ret, ret+vol, ret+vol+ntrades}
  level       ∈ {2, 3}
  window      ∈ {60, 90}
  linkage     ∈ {single, average, complete}
  → 3 × 2 × 2 × 3 = 36 cells.

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
prices_long["quote_volume"] = pd.to_numeric(prices_long["quote_volume"], errors="coerce")
prices_long["num_trades"] = pd.to_numeric(prices_long["num_trades"], errors="coerce")

prices  = prices_long.pivot(index="open_time", columns="symbol", values="close").sort_index()
qvol    = prices_long.pivot(index="open_time", columns="symbol", values="quote_volume").sort_index()
ntrades = prices_long.pivot(index="open_time", columns="symbol", values="num_trades").sort_index()
print(f"  prices:  {prices.shape}")
print(f"  qvol:    {qvol.shape}  (non-null {qvol.notna().mean().mean()*100:.0f}%)")
print(f"  ntrades: {ntrades.shape}  (non-null {ntrades.notna().mean().mean()*100:.0f}%)")

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
    "ret":            [],
    "ret+vol":        [qvol],
    "ret+vol+ntr":    [qvol, ntrades],
}
LEVELS   = [2, 3]
WINDOWS  = [60, 90]
LINKAGES = ["single", "average", "complete"]

total = len(CHANNEL_SETS) * len(LEVELS) * len(WINDOWS) * len(LINKAGES)
print(f"\n>>> Running multi-channel HRP_PathSig sweep: {total} cells")

cells = []
results: dict[str, pd.Series] = {}
fb_counts: dict[str, int] = {}

idx = 0
t_all = time.time()
for chan_name, chan_list in CHANNEL_SETS.items():
    for lvl in LEVELS:
        for win in WINDOWS:
            for link in LINKAGES:
                idx += 1
                name = f"PathSigMC_{chan_name}_L{lvl}_w{win}_{link}"
                t0 = time.time()
                fb = {"count": 0, "total": 0}

                def factory(r, lvl=lvl, win=win, link=link, chan_list=chan_list, fb=fb):
                    strat = HRPPathSig(
                        r, window=win, level=lvl, linkage_method=link,
                        extra_channels=chan_list or None,
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
                    fb_counts[name] = fb["count"]
                    dt = time.time() - t0
                    cells.append((name, sh, tr, fb["count"], fb["total"], dt))
                    print(f"  [{idx:2d}/{total}] {name:38s}  Sharpe={sh:+.4f}  TR={tr*100:+8.1f}%  "
                          f"Δ={sh - sharpe_base:+.4f}  fb={fb['count']}/{fb['total']}  [{dt:.1f}s]")
                except Exception as e:
                    print(f"  [{idx:2d}/{total}] {name:38s}  ERROR: {e}")
                    cells.append((name, np.nan, np.nan, fb["count"], fb["total"], time.time() - t0))

print(f"\nTotal sweep time: {time.time() - t_all:.1f}s")

# ---------- Summary ----------
df = pd.DataFrame(cells, columns=["name", "sharpe", "total_return", "fallback", "n_snaps", "secs"])
df["delta_sharpe"] = df["sharpe"] - sharpe_base
df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

print("\n========== multi-channel HRP_PathSig sweep — top 12 cells ==========")
print(df.head(12).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

print("\n========== by channel set (mean Sharpe) ==========")
for cs in CHANNEL_SETS:
    # regex=False — channel-set labels contain '+', a regex metacharacter
    sub = df[df["name"].str.contains(f"_{cs}_", regex=False)]
    print(f"  {cs:14s}  mean={sub['sharpe'].mean():+.4f}  "
          f"max={sub['sharpe'].max():+.4f}  n_beat={int((sub['sharpe']>sharpe_base).sum())}/{len(sub)}")

n_beat = int((df["sharpe"] > sharpe_base).sum())
print(f"\nCells beating HRP baseline ({sharpe_base:+.4f}): {n_beat}/{len(df)}")
print(f"Best cell: {df.iloc[0]['name']}  Sharpe={df.iloc[0]['sharpe']:+.4f}  "
      f"TR={df.iloc[0]['total_return']*100:+.1f}%  Δ={df.iloc[0]['delta_sharpe']:+.4f}")

# ---------- Save ----------
out_path = "/tmp/hp_sweep_pathsig_mc.pkl"
with open(out_path, "wb") as f:
    pickle.dump({
        "returns": results,
        "summary": df,
        "sharpe_base": sharpe_base,
        "total_return_base": total_ret_base,
        "fallback_counts": fb_counts,
        "grid": {
            "channel_sets": list(CHANNEL_SETS), "levels": LEVELS,
            "windows": WINDOWS, "linkages": LINKAGES,
        },
    }, f)
print(f"\nSaved → {out_path}")
