"""Regenerate cluster_stability.csv on the current 146-symbol PIT universe.

Per monthly snapshot: build the estimation-window returns, form the Pearson
and lower-tail-dependence distance matrices, single-linkage cluster each,
and record cophenetic correlation and the Adjusted Rand Index (K=5) versus
the previous snapshot. Fast (no backtest).
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
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from src.backtest import _returns_for_window
from src.inference import cophenetic_correlation, adjusted_rand_between_snapshots
from src.hrp_variants import lower_tail_dependence, tail_dependence_distance
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


def pearson_distance(r):
    c = r.corr().values
    d = np.sqrt(np.clip((1.0 - c) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return (d + d.T) / 2.0


rows = []
prev = None
for snap in snaps:
    inc = sorted(pit[(pit["date"] == snap) & pit["included"]]["symbol"].tolist())
    train = _returns_for_window(prices, snap, inc, 365, 180)
    if train.shape[1] < 6:
        continue
    labels = list(train.columns)
    dp = pearson_distance(train)
    dt = tail_dependence_distance(lower_tail_dependence(train, q=0.05)).values.copy()
    np.fill_diagonal(dt, 0.0); dt = (dt + dt.T) / 2.0
    lp = linkage(squareform(dp, checks=False), "single")
    lt = linkage(squareform(dt, checks=False), "single")
    row = {"date": snap.date(), "N": train.shape[1],
           "pearson_cophenetic": cophenetic_correlation(dp, lp),
           "taildep_cophenetic": cophenetic_correlation(dt, lt),
           "pearson_ari_vs_prev": np.nan, "taildep_ari_vs_prev": np.nan}
    if prev is not None:
        row["pearson_ari_vs_prev"] = adjusted_rand_between_snapshots(
            prev["lp"], lp, prev["labels"], labels, k=5)
        row["taildep_ari_vs_prev"] = adjusted_rand_between_snapshots(
            prev["lt"], lt, prev["labels"], labels, k=5)
    rows.append(row)
    prev = {"lp": lp, "lt": lt, "labels": labels}

df = pd.DataFrame(rows)
df.to_csv(f"{DATA}/cluster_stability.csv", index=False)
ari = df.dropna(subset=["pearson_ari_vs_prev"])
n_td_wins = int((ari["taildep_ari_vs_prev"] > ari["pearson_ari_vs_prev"]).sum())
print(f"snapshots: {len(df)}  (ARI pairs: {len(ari)})")
print(f"  pearson cophenetic  mean={df['pearson_cophenetic'].mean():.3f}")
print(f"  taildep cophenetic  mean={df['taildep_cophenetic'].mean():.3f}")
print(f"  pearson ARI  mean={ari['pearson_ari_vs_prev'].mean():.3f}  median={ari['pearson_ari_vs_prev'].median():.3f}")
print(f"  taildep ARI  mean={ari['taildep_ari_vs_prev'].mean():.3f}  median={ari['taildep_ari_vs_prev'].median():.3f}")
print(f"  TailDep ARI > Pearson ARI in {n_td_wins}/{len(ari)} month-pairs")
print(f"Saved -> cluster_stability.csv")
