"""Verify the two signal-aware strategies are deterministic.

Three checks:

  (A) XGBoost run-to-run bit-identity on a synthetic dataset. Confirms
      the chosen XGBoost settings (tree_method='hist', n_jobs=1, fixed
      seed) plus sklearn's GridSearchCV+TimeSeriesSplit are deterministic
      end-to-end.
  (B) Strategy run-to-run bit-identity. NCOML and HRPSigmaMu, called
      twice on the same returns window, must produce byte-identical
      weight vectors.
  (C) Strategy reproduces committed CSV. Re-run the headline backtest
      for both strategies and compare arith_sharpe_ann + total_return
      to data/backtest_v2_rebalance_results.csv.

Exit code 0 iff all three pass.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.backtest import COST_SCENARIOS, WalkForwardBacktest  # noqa: E402
from src.portfolio_maker import HRPSigmaMu, NCOML  # noqa: E402
from src.universe import load_pit_universe  # noqa: E402

import xgboost as xgb  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402

warnings.filterwarnings("ignore")
ANN = np.sqrt(365)
DATA = _ROOT / "data"
TOL = 1e-6
SEED = 42

# data
pit = load_pit_universe(f"{DATA}/pit_universe.csv")
ex = pd.read_csv(f"{DATA}/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv",
                 parse_dates=["open_time"])
sp = pd.read_csv(f"{DATA}/binance_pit_supplement_2019-2024_1d.csv",
                 parse_dates=["open_time"])
pl = pd.concat([ex, sp], ignore_index=True).drop_duplicates(
    subset=["symbol", "open_time"])
pl["close"] = pl["close"].astype(float)
prices = pl.pivot(index="open_time", columns="symbol",
                  values="close").sort_index()
committed = pd.read_csv(f"{DATA}/backtest_v2_rebalance_results.csv"
                        ).set_index("name")


# ---------------- (A) XGBoost run-to-run on a synthetic dataset
print("=== (A) XGBoost run-to-run ===", flush=True)
rng = np.random.default_rng(0)
X = rng.standard_normal((600, 6)).astype(np.float32)
y = (X[:, 0] * 0.3 - X[:, 1] * 0.2 +
     rng.standard_normal(600) * 0.5).astype(np.float32)
GRID = {"max_depth": [3, 5], "learning_rate": [0.03, 0.1],
        "n_estimators": [100]}


def _fit_predict() -> np.ndarray:
    base = xgb.XGBRegressor(tree_method="hist", n_jobs=1,
                            random_state=SEED, verbosity=0)
    gs = GridSearchCV(base, GRID, cv=TimeSeriesSplit(3),
                      scoring="neg_mean_squared_error",
                      n_jobs=1, refit=True)
    gs.fit(X, y)
    return gs.best_estimator_.predict(X)


p1, p2 = _fit_predict(), _fit_predict()
a_ok = bool(np.all(p1 == p2))
print(f"  [{'OK' if a_ok else 'FAIL'}] max|p1-p2| = "
      f"{float(np.abs(p1 - p2).max()):.2e}", flush=True)


# ---------------- (B) Strategy run-to-run on one returns window
print("\n=== (B) Strategy run-to-run ===", flush=True)
# the latest snapshot is well-covered by the mu panel
snap_dates = sorted(pd.Timestamp(d) for d in pit["date"].unique()
                    if pd.Timestamp("2025-01-01") <= pd.Timestamp(d))
snap = snap_dates[0]
universe = sorted(pit[(pit["date"] == snap) & pit["included"]]
                  ["symbol"].tolist())
sample = prices.loc[:snap, universe].pct_change().iloc[-365:].fillna(0)
print(f"  sample at {snap.date()}: {sample.shape[0]}d x {sample.shape[1]} assets",
      flush=True)
b_ok = True
for name, cls in [("NCOML", NCOML), ("HRPSigmaMu", HRPSigmaMu)]:
    s1 = cls(sample.copy()); w1 = s1.get_weights().sort_index()
    s2 = cls(sample.copy()); w2 = s2.get_weights().sort_index()
    fell = getattr(s1, "fallback_used", False) or \
           getattr(s2, "fallback_used", False)
    dw = float(np.abs(w1.values - w2.values).max())
    ok = (not fell) and dw == 0.0
    b_ok &= ok
    extra = "  (FELL BACK!)" if fell else ""
    print(f"  [{'OK' if ok else 'FAIL'}] {name:12s} "
          f"max|w1-w2|={dw:.2e}{extra}", flush=True)


# ---------------- (C) Headline backtest reproduces committed
print("\n=== (C) Headline backtest vs committed CSV ===", flush=True)
cm = COST_SCENARIOS["conservative_cex"]
c_ok = True
for name, cls in [("NCOML", NCOML), ("HRPSigmaMu", HRPSigmaMu)]:
    t0 = time.time()
    bt = WalkForwardBacktest(prices, pit, lambda r, c=cls: c(r), cost_model=cm)
    res = bt.run("2020-01-01", "2026-05-18")
    d = res["daily_returns"]
    sharpe = float(d.mean() / d.std(ddof=1) * ANN)
    tr = float((1 + d).prod() - 1)
    cs = float(committed.loc[name, "arith_sharpe_ann"])
    ct = float(committed.loc[name, "total_return"])
    ds, dtr = sharpe - cs, tr - ct
    ok = abs(ds) < TOL and abs(dtr) < TOL
    c_ok &= ok
    print(f"  [{'MATCH' if ok else 'SHIFTED'}] {name:12s} "
          f"Sharpe {sharpe:.10f} "
          f"(committed {cs:.10f}, d={ds:+.2e})", flush=True)
    print(f"          {'':12s} "
          f"TotRet {tr:.10f} (committed {ct:.10f}, d={dtr:+.2e})  "
          f"[{time.time() - t0:.0f}s]", flush=True)


print("\n=== VERDICT ===", flush=True)
print(f"  (A) XGBoost determinism       : {'PASS' if a_ok else 'FAIL'}",
      flush=True)
print(f"  (B) Strategy run-to-run       : {'PASS' if b_ok else 'FAIL'}",
      flush=True)
print(f"  (C) Reproduces committed CSV  : {'PASS' if c_ok else 'FAIL'}",
      flush=True)
sys.exit(0 if (a_ok and b_ok and c_ok) else 1)
