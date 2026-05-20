"""HP sweep #13: multi-channel HRP_Contrastive + HRP_TS2Vec.

Tests whether feeding the torch encoders orthogonal channels (quote volume,
trade count) on top of returns lets the learned embedding beat HRP — the
single-channel Contrastive run scored +0.7144, below HRP +0.7415.

Grid per encoder: channel_set × linkage = 3 × 3 = 9 cells.
  channel_set ∈ {ret, ret+vol, ret+vol+ntrades}
  linkage     ∈ {single, average, complete}
Fixed (held constant so the channel comparison is clean):
  window=40, stride=8, epochs=8, emb_dim=32, hidden=16, batch_size=64.

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
from src.portfolio_maker import HRP, HRPContrastive, HRPTS2Vec
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
print(f"  prices: {prices.shape}")

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
    "ret":         [],
    "ret+vol":     [qvol],
    "ret+vol+ntr": [qvol, ntrades],
}
LINKAGES = ["single", "average", "complete"]
FIXED = dict(window=40, stride=8, epochs=8, emb_dim=32, hidden=16, batch_size=64)

ENCODERS = {"Contrastive": HRPContrastive, "TS2Vec": HRPTS2Vec}

results: dict[str, pd.Series] = {}
cells = []
t_all = time.time()

for enc_name, enc_cls in ENCODERS.items():
    print(f"\n{'='*60}\n>>> {enc_name}: {len(CHANNEL_SETS)*len(LINKAGES)} cells\n{'='*60}")
    idx = 0
    for chan_name, chan_list in CHANNEL_SETS.items():
        for link in LINKAGES:
            idx += 1
            name = f"{enc_name}MC_{chan_name}_{link}"
            t0 = time.time()
            fb = {"count": 0, "total": 0}

            def factory(r, enc_cls=enc_cls, chan_list=chan_list, link=link, fb=fb):
                strat = enc_cls(
                    r, linkage_method=link,
                    extra_channels=chan_list or None, **FIXED,
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
                cells.append((name, enc_name, chan_name, link, sh, tr,
                              fb["count"], fb["total"], dt))
                print(f"  [{idx}/9] {name:34s}  Sharpe={sh:+.4f}  TR={tr*100:+8.1f}%  "
                      f"Δ={sh - sharpe_base:+.4f}  fb={fb['count']}/{fb['total']}  [{dt:.0f}s]")
            except Exception as e:
                print(f"  [{idx}/9] {name:34s}  ERROR: {e}")
                cells.append((name, enc_name, chan_name, link, np.nan, np.nan,
                              fb["count"], fb["total"], time.time() - t0))

            # incremental save
            df_partial = pd.DataFrame(cells, columns=[
                "name", "encoder", "channels", "linkage", "sharpe",
                "total_return", "fallback", "n_snaps", "secs"])
            with open("/tmp/hp_sweep_encoders_mc.pkl", "wb") as f:
                pickle.dump({
                    "returns": results, "summary": df_partial,
                    "sharpe_base": sharpe_base, "total_return_base": total_ret_base,
                    "fixed": FIXED,
                }, f)

print(f"\nTotal sweep time: {time.time() - t_all:.0f}s")

# ---------- Summary ----------
df = pd.DataFrame(cells, columns=[
    "name", "encoder", "channels", "linkage", "sharpe",
    "total_return", "fallback", "n_snaps", "secs"])
df["delta_sharpe"] = df["sharpe"] - sharpe_base
df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

print(f"\n========== multi-channel encoder sweep — all cells (HRP base {sharpe_base:+.4f}) ==========")
print(df[["name", "sharpe", "total_return", "delta_sharpe", "fallback", "secs"]].to_string(
    index=False, float_format=lambda x: f"{x:+.4f}"))

print("\n========== by encoder × channel set (mean Sharpe) ==========")
for enc_name in ENCODERS:
    for cs in CHANNEL_SETS:
        sub = df[(df["encoder"] == enc_name) & (df["channels"] == cs)]
        if len(sub):
            print(f"  {enc_name:12s} {cs:14s}  mean={sub['sharpe'].mean():+.4f}  "
                  f"max={sub['sharpe'].max():+.4f}")

n_beat = int((df["sharpe"] > sharpe_base).sum())
print(f"\nCells beating HRP baseline: {n_beat}/{len(df)}")
best = df.iloc[0]
print(f"Best cell: {best['name']}  Sharpe={best['sharpe']:+.4f}  "
      f"TR={best['total_return']*100:+.1f}%  Δ={best['delta_sharpe']:+.4f}")

with open("/tmp/hp_sweep_encoders_mc.pkl", "wb") as f:
    pickle.dump({
        "returns": results, "summary": df,
        "sharpe_base": sharpe_base, "total_return_base": total_ret_base,
        "fixed": FIXED,
    }, f)
print("\nSaved → /tmp/hp_sweep_encoders_mc.pkl")
