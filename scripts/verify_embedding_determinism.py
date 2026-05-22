"""Verify the two PyTorch-trained embedding strategies are deterministic.

Context: an external reviewer re-ran the pipeline on a different machine
and reported HRP_Contrastive / HRP_TS2Vec Sharpes ~0.018 below the
committed artifacts. Those two trained-encoder strategies were the only
ones that drifted — HRP_PathSig (deterministic geometry), HRP_NodeEmbed
(node2vec) and all 21 closed-form strategies reproduced exactly. The
cause: multi-threaded BLAS reductions in PyTorch are not deterministic,
and the small floating-point differences propagate through single-linkage
clustering into materially different HRP weights.

After the determinism hardening in src/embeddings/{contrastive,ts2vec_lite}.py
(single-thread + use_deterministic_algorithms), this script checks:

  (A) run-to-run bit-identity: train each encoder twice on the same
      returns window and confirm the HRP weights are identical;
  (B) reproduction of the committed headline numbers: re-run the
      Scenario-B backtest for the two strategies and compare
      arith_sharpe_ann / total_return against
      data/backtest_v2_rebalance_results.csv.

Exit code 0 iff both checks pass for both strategies.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from src.backtest import WalkForwardBacktest, COST_SCENARIOS
from src.portfolio_maker import HRPContrastive, HRPTS2Vec
from src.universe import load_pit_universe

warnings.filterwarnings("ignore")
ANN = np.sqrt(365)
DATA = _ROOT / "data"
TOL = 1e-6  # arith_sharpe_ann is reported to ~15 sig-figs; this is generous

STRATS = {"HRP_Contrastive": HRPContrastive, "HRP_TS2Vec": HRPTS2Vec}

# ---------------------------------------------------------------- data
print("Loading prices + PIT universe...", flush=True)
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

committed = pd.read_csv(f"{DATA}/backtest_v2_rebalance_results.csv").set_index(
    "name")

# =================================================================
# (A) run-to-run bit-identity on a single returns window
# =================================================================
print("\n=== (A) run-to-run determinism ===", flush=True)
sample = prices.pct_change().iloc[-420:].dropna(axis=1, how="any")
print(f"  sample window: {sample.shape[0]} days x {sample.shape[1]} assets",
      flush=True)

a_ok = True
for name, cls in STRATS.items():
    s1 = cls(sample.copy())
    w1 = s1.get_weights()
    s2 = cls(sample.copy())
    w2 = s2.get_weights()
    fellback = getattr(s1, "fallback_used", False) or getattr(
        s2, "fallback_used", False)
    w1, w2 = w1.sort_index(), w2.sort_index()
    max_dw = float(np.abs(w1.values - w2.values).max())
    identical = (not fellback) and max_dw == 0.0
    a_ok &= identical
    flag = "OK" if identical else "FAIL"
    extra = "  (FELL BACK TO PLAIN HRP!)" if fellback else ""
    print(f"  [{flag}] {name:16s} max|w1-w2| = {max_dw:.2e}{extra}", flush=True)

# =================================================================
# (B) reproduction of the committed headline backtest
# =================================================================
print("\n=== (B) headline backtest vs committed CSV ===", flush=True)
cm = COST_SCENARIOS["conservative_cex"]
b_ok = True
for name, cls in STRATS.items():
    t0 = time.time()
    bt = WalkForwardBacktest(prices, pit, lambda r, c=cls: c(r), cost_model=cm)
    res = bt.run("2020-01-01", "2026-05-18")
    d = res["daily_returns"]
    sharpe = float(d.mean() / d.std(ddof=1) * ANN)
    tr = float((1 + d).prod() - 1)
    c_sharpe = float(committed.loc[name, "arith_sharpe_ann"])
    c_tr = float(committed.loc[name, "total_return"])
    d_sharpe, d_tr = sharpe - c_sharpe, tr - c_tr
    match = abs(d_sharpe) < TOL and abs(d_tr) < TOL
    b_ok &= match
    flag = "MATCH" if match else "SHIFTED"
    print(f"  [{flag}] {name:16s} "
          f"Sharpe {sharpe:.10f} (committed {c_sharpe:.10f}, d={d_sharpe:+.2e})",
          flush=True)
    print(f"          {'':16s} "
          f"TotRet {tr:.10f} (committed {c_tr:.10f}, d={d_tr:+.2e})  "
          f"[{time.time()-t0:.0f}s]", flush=True)

# ---------------------------------------------------------------- verdict
print("\n=== VERDICT ===", flush=True)
print(f"  (A) run-to-run determinism : {'PASS' if a_ok else 'FAIL'}", flush=True)
print(f"  (B) reproduces committed   : {'PASS' if b_ok else 'FAIL'}", flush=True)
if a_ok and b_ok:
    print("  -> hardening holds; committed embedding numbers stand.", flush=True)
    sys.exit(0)
if a_ok and not b_ok:
    print("  -> deterministic now, but numbers shifted vs committed: "
          "a full re-run + paper update is required.", flush=True)
sys.exit(1)
