"""HP sweep #11: HRP_PathSig full grid (level x window x distance x linkage).

Goal: find the absolute-best HRP_PathSig configuration via systematic search
over signature level, rolling window, distance function (for the corr-fallback
inside the HRP base), and linkage method.

Reports Sharpe AND total return per cell + saves daily returns to a pickle.
"""

from __future__ import annotations

import pickle
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

import sys
# --- resolve the repo root so this script runs from any cwd / machine ---
import os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)

from src.backtest import WalkForwardBacktest, COST_SCENARIOS
from src.portfolio_maker import HRP, HRPPathSig
from src.universe import load_pit_universe
from src.embeddings.path_signatures import (
    asset_path_signatures, signatures_to_distance,
)

warnings.filterwarnings("ignore")

START = "2020-01-01"
END   = "2026-05-18"

# Custom HRP_PathSig subclass to inject distance/linkage choice for the
# fallback path (when signature computation succeeds, distance is fixed to
# cosine on the signature; when it fails, the base HRP distance is used).
# For the SWEEP, we vary the level/window/linkage and use the standard
# cosine distance on the signature. Distance choice modulates how we
# transform the signature-based similarity into a distance metric, and
# we test 3 alternative distance constructions on top of the signature.


class HRPPathSigCustom(HRP):
    """HRP with path-signature embedding distance, with a configurable
    distance construction from the signature similarity matrix.

    Args:
        returns: T x N returns DataFrame.
        window: rolling-window length for path signatures.
        level: signature truncation level.
        linkage_method: hierarchical-clustering linkage.
        distance: how to convert cosine similarity in [-1, 1] to distance
            in [0, *]:
            - 'cosine'   : sqrt(1 - sim)             [default, in [0, sqrt(2)]]
            - 'angular'  : arccos(sim) / pi          [in [0, 1]]
            - 'one_minus': 1 - sim                   [in [0, 2]]
            - 'sqrt_lp'  : sqrt(0.5 * (1 - sim))     [Lopez-de-Prado-style on sim]
    """

    def __init__(self, returns: pd.DataFrame, window: int = 60, level: int = 3,
                 linkage_method: str = "single", distance: str = "cosine") -> None:
        super().__init__(returns, linkage_method=linkage_method)
        self.window = window
        self.level = level
        self.distance = distance
        self.fallback_used = False
        self.sig_df: Optional[pd.DataFrame] = None

    def _signature_distance(self, sig_df: pd.DataFrame) -> pd.DataFrame:
        X = sig_df.values.astype(float)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms <= 0, 1e-12, norms)
        Xn = X / norms
        sim = Xn @ Xn.T
        sim = np.clip(sim, -1.0, 1.0)
        if self.distance == "cosine":
            dist = np.sqrt(np.maximum(1.0 - sim, 0.0))
        elif self.distance == "angular":
            dist = np.arccos(sim) / np.pi
        elif self.distance == "one_minus":
            dist = 1.0 - sim
        elif self.distance == "sqrt_lp":
            dist = np.sqrt(np.maximum(0.5 * (1.0 - sim), 0.0))
        else:
            raise ValueError(f"unknown distance: {self.distance}")
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0
        return pd.DataFrame(dist, index=sig_df.index, columns=sig_df.index)

    def get_weights(self) -> pd.Series:
        try:
            self.sig_df = asset_path_signatures(
                self.returns, window=self.window, level=self.level,
            )
            dist = self._signature_distance(self.sig_df)
            d = dist.reindex(index=self.cov.index, columns=self.cov.index).fillna(1.0)
            d_arr = d.values.copy()
            d_arr = (d_arr + d_arr.T) / 2
            np.fill_diagonal(d_arr, 0.0)
            condensed = squareform(d_arr, checks=False)
            link = linkage(condensed, self.linkage_method)
            sort_ix = HRP.get_quasi_diag(link)
            sort_ix = self.cov.index[sort_ix].tolist()
            self.weights = HRP.get_rec_bipart(self.cov, sort_ix)
            return self.weights
        except Exception:
            self.fallback_used = True
            return super().get_weights()


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
print(f"  prices: {prices.shape[0]} dates x {prices.shape[1]} symbols")

cost = COST_SCENARIOS["conservative_cex"]

# Baseline HRP for delta-Sharpe context
print("\n>>> Baseline HRP")
t0 = time.time()
bt = WalkForwardBacktest(prices, pit, lambda r: HRP(r), cost_model=cost)
res_base = bt.run(START, END)
ret_base = res_base["daily_returns"]
sharpe_base = ret_base.mean() / ret_base.std() * np.sqrt(365)
total_ret_base = (1 + ret_base).prod() - 1
print(f"  HRP baseline: Sharpe={sharpe_base:.4f}  TR={total_ret_base*100:+.1f}%  [{time.time()-t0:.0f}s]")

# ---------- Sweep grid ----------
LEVELS    = [2, 3, 4]
WINDOWS   = [40, 60, 90]
DISTANCES = ["cosine", "angular", "one_minus", "sqrt_lp"]
LINKAGES  = ["single", "average", "complete"]

cells = []
total = len(LEVELS) * len(WINDOWS) * len(DISTANCES) * len(LINKAGES)
print(f"\n>>> Running HRP_PathSig sweep: {total} cells")
print(f"    Levels={LEVELS}  Windows={WINDOWS}  Distances={DISTANCES}  Linkages={LINKAGES}")

results: dict[str, pd.Series] = {}
fb_counts: dict[str, int] = {}
n_snaps: dict[str, int] = {}

idx = 0
t_all = time.time()
for lvl in LEVELS:
    for win in WINDOWS:
        for dist_kind in DISTANCES:
            for link in LINKAGES:
                idx += 1
                name = f"PathSig_L{lvl}_w{win}_{dist_kind}_{link}"
                t0 = time.time()
                fb = {"count": 0, "total": 0}

                def factory(r, lvl=lvl, win=win, dist_kind=dist_kind, link=link, fb=fb):
                    strat = HRPPathSigCustom(
                        r, window=win, level=lvl,
                        linkage_method=link, distance=dist_kind,
                    )
                    # wrap get_weights to count fallbacks
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
                    n_snaps[name] = fb["total"]
                    dt = time.time() - t0
                    cells.append((name, sh, tr, fb["count"], fb["total"], dt))
                    print(f"  [{idx:2d}/{total}] {name:42s}  Sharpe={sh:+.4f}  TR={tr*100:+8.1f}%  "
                          f"Δ={sh - sharpe_base:+.4f}  fb={fb['count']}/{fb['total']}  [{dt:.1f}s]")
                except Exception as e:
                    print(f"  [{idx:2d}/{total}] {name:42s}  ERROR: {e}")
                    cells.append((name, np.nan, np.nan, fb["count"], fb["total"], time.time() - t0))

print(f"\nTotal sweep time: {time.time() - t_all:.1f}s")

# ---------- Summary ----------
df = pd.DataFrame(cells, columns=["name", "sharpe", "total_return", "fallback", "n_snaps", "secs"])
df["delta_sharpe"] = df["sharpe"] - sharpe_base
df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

print("\n========== HRP_PathSig sweep — top 10 cells ==========")
print(df.head(10).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

print("\n========== bottom 5 cells ==========")
print(df.tail(5).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

n_beat = int((df["sharpe"] > sharpe_base).sum())
print(f"\nCells beating HRP baseline ({sharpe_base:+.4f}): {n_beat}/{len(df)}")
print(f"Best cell: {df.iloc[0]['name']}  Sharpe={df.iloc[0]['sharpe']:+.4f}  "
      f"TR={df.iloc[0]['total_return']*100:+.1f}%  Δ={df.iloc[0]['delta_sharpe']:+.4f}")

# ---------- Save daily returns + summary ----------
out_path = "/tmp/hp_sweep_pathsig.pkl"
with open(out_path, "wb") as f:
    pickle.dump({
        "returns": results,
        "summary": df,
        "sharpe_base": sharpe_base,
        "total_return_base": total_ret_base,
        "fallback_counts": fb_counts,
        "n_snaps": n_snaps,
        "grid": {
            "levels": LEVELS, "windows": WINDOWS,
            "distances": DISTANCES, "linkages": LINKAGES,
        },
    }, f)
print(f"\nSaved → {out_path}")
