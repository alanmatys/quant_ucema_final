"""Backtest NCOML and HRPSigmaMu and splice them into the result files.

The two signal-aware comparators (Wuebben 2026 items 1 and 2) are added
as the 26th and 27th strategies in the headline comparison. The
walk-forward XGBoost mu panel (data/xgboost_mu_predictions.csv) must
exist on disk; run scripts/build_xgboost_signals.py first if not.

Coverage:
  - Scenario B (headline, monthly, 2020-01-01 to 2026-05-18)
  - Post-COVID regime (2022-01-01 to 2026-05-18)
  - Weekly cadence is intentionally skipped: the mu panel is monthly,
    and retraining XGBoost at ~330 weekly dates is out of scope for
    this paper (matches the convention used for the embedding family).

After backtesting we splice each strategy into:
  - backtest_v2_daily_returns.pkl and backtest_v2_post_covid_daily_returns.pkl
  - backtest_v2_rebalance_results.csv and backtest_v2_post_covid_results.csv
  - inference_*.csv: Ledoit-Wolf, Hansen SPA, bootstrap CIs (both windows)
"""
from __future__ import annotations

import os as _os
import pickle
import sys
import warnings
from pathlib import Path as _Path

import numpy as np
import pandas as pd

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)

from src.backtest import COST_SCENARIOS, WalkForwardBacktest, summarize_performance
from src.inference import (
    hansen_spa_test, sharpe_diff_ledoit_wolf, stationary_block_bootstrap,
)
from src.portfolio_maker import HRPSigmaMu, NCOML
from src.universe import load_pit_universe

warnings.filterwarnings("ignore")
ANN = np.sqrt(365)
DATA = str(_ROOT / "data")

STRATEGIES = {
    "NCOML":      lambda r: NCOML(r),
    "HRPSigmaMu": lambda r: HRPSigmaMu(r),
}

_MU = _ROOT / "data" / "xgboost_mu_predictions.csv"
if not _MU.exists():
    print(f"ERROR: {_MU} missing.\n"
          f"  Run: uv run python scripts/build_xgboost_signals.py first.")
    sys.exit(1)

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
cm = COST_SCENARIOS["conservative_cex"]

COLS = ["name", "total_return", "ann_return", "ann_vol", "sharpe", "sortino",
        "max_drawdown", "calmar", "n_days", "arith_sharpe_ann", "rebalances",
        "avg_turnover", "total_costs", "avg_max_weight", "avg_n_eff"]


def metrics_row(name: str, daily: pd.Series, res: dict) -> dict:
    m = summarize_performance(daily, name=name)
    row = {k: m.get(k) for k in ("total_return", "ann_return", "ann_vol",
           "sharpe", "sortino", "max_drawdown", "calmar", "n_days")}
    row["name"] = name
    row["arith_sharpe_ann"] = float(daily.mean() / daily.std(ddof=1) * ANN) \
        if daily.std(ddof=1) > 0 else 0.0
    wh = res["weights_history"]
    th, ch = res["turnover_history"], res["cost_history"]
    row["rebalances"] = float(len(res["snapshot_dates"]))
    row["avg_turnover"] = float(np.mean(np.asarray(th))) if len(th) else 0.0
    row["total_costs"] = float(np.sum(np.asarray(ch))) if len(ch) else 0.0
    W = wh.fillna(0.0)
    row["avg_max_weight"] = float(W.max(axis=1).mean())
    row["avg_n_eff"] = float((1.0 / (W ** 2).sum(axis=1)).mean())
    return row


def splice_pickle(path: str, name_to_series: dict) -> dict:
    d = pickle.load(open(path, "rb"))
    idx = pd.DataFrame({k: v for k, v in d.items() if k != "HODL_BTC"}
                       ).dropna().index
    for name, series in name_to_series.items():
        d[name] = series.reindex(idx)
    pickle.dump(d, open(path, "wb"))
    return d


def update_results(path: str, rows: list) -> None:
    df = pd.read_csv(path)
    new_names = [r["name"] for r in rows]
    df = df[~df["name"].isin(new_names)]
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df[COLS].to_csv(path, index=False)


def run_window(label: str, start: str, end: str) -> dict:
    runs, dailies = {}, {}
    print(f"\n=== {label} ({start} to {end}) ===", flush=True)
    for name, fac in STRATEGIES.items():
        res = WalkForwardBacktest(prices, pit, fac, cost_model=cm).run(start, end)
        d = res["daily_returns"]
        runs[name] = res
        dailies[name] = d
        sh = d.mean() / d.std(ddof=1) * ANN if d.std(ddof=1) > 0 else 0.0
        print(f"  {name:12s} Sharpe={sh:+.4f}  "
              f"TR={((1+d).prod()-1)*100:+.0f}%", flush=True)
    return {"runs": runs, "dailies": dailies}


# ---------- Scenario B headline ----------
B = run_window("Scenario B", "2020-01-01", "2026-05-18")
dailyB = splice_pickle(f"{DATA}/backtest_v2_daily_returns.pkl", B["dailies"])
update_results(f"{DATA}/backtest_v2_rebalance_results.csv",
               [metrics_row(n, B["dailies"][n], B["runs"][n]) for n in STRATEGIES])

print("\n  re-running Scenario B inference...", flush=True)
rdfB = pd.DataFrame(dailyB).dropna()
hrp = rdfB["HRP"]
sh = {c: rdfB[c].mean() / rdfB[c].std(ddof=1) * ANN for c in rdfB.columns}
lw_rows, ci_rows = [], []
for c in rdfB.columns:
    boot = stationary_block_bootstrap(
        rdfB[c].values,
        lambda r: float(np.mean(r) / np.std(r, ddof=1) * ANN)
        if np.std(r, ddof=1) > 0 else 0.0,
        n_iter=2000, seed=42)
    ci_rows.append({"name": c, "sharpe": sh[c],
                    "ci_lower_95": boot["ci_lower_95"],
                    "ci_upper_95": boot["ci_upper_95"]})
    if c == "HRP":
        continue
    lw = sharpe_diff_ledoit_wolf(rdfB[c].values, hrp.values, n_iter=3000, seed=42)
    lw_rows.append({"name": c, "sharpe_diff_vs_hrp": sh[c] - sh["HRP"],
                    "lw_p_value": lw["p_value"]})
pd.DataFrame(lw_rows).to_csv(f"{DATA}/inference_sharpe_diff.csv", index=False)
pd.DataFrame(ci_rows).to_csv(f"{DATA}/inference_bootstrap_cis.csv", index=False)
spa = hansen_spa_test(rdfB.drop(columns=["HRP"]), hrp.values,
                      n_bootstrap=5000, seed=42)
spa.to_csv(f"{DATA}/inference_spa.csv")
print(f"  SPA p_consistent = {spa['p_consistent'].iloc[0]:.4f}", flush=True)

# ---------- Post-COVID ----------
P = run_window("Post-COVID", "2022-01-01", "2026-05-18")
dailyP = splice_pickle(f"{DATA}/backtest_v2_post_covid_daily_returns.pkl",
                       P["dailies"])
update_results(f"{DATA}/backtest_v2_post_covid_results.csv",
               [metrics_row(n, P["dailies"][n], P["runs"][n]) for n in STRATEGIES])

print("\n  re-running post-COVID inference...", flush=True)
rdfP = pd.DataFrame(dailyP).dropna()
hrpP = rdfP["HRP"]
shP = {c: rdfP[c].mean() / rdfP[c].std(ddof=1) * ANN for c in rdfP.columns}
lwP = []
for c in rdfP.columns:
    if c == "HRP":
        continue
    lw = sharpe_diff_ledoit_wolf(rdfP[c].values, hrpP.values,
                                 n_iter=3000, seed=42)
    lwP.append({"name": c, "sharpe_diff_vs_hrp": shP[c] - shP["HRP"],
                "lw_p_value": lw["p_value"]})
pd.DataFrame(lwP).to_csv(f"{DATA}/inference_post_covid_sharpe_diff.csv",
                        index=False)
spaP = hansen_spa_test(rdfP.drop(columns=["HRP"]), hrpP.values,
                       n_bootstrap=5000, seed=42)
spaP.to_csv(f"{DATA}/inference_post_covid_spa.csv")
print(f"  Post-COVID SPA p_consistent = {spaP['p_consistent'].iloc[0]:.4f}",
      flush=True)

print("\nDONE.")
