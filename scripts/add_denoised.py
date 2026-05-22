"""Backtest HRP_Denoised and splice it into the existing result files.

The paper claims to evaluate Marchenko-Pastur denoising as a standalone
contribution, but HRP_Denoised (denoise the correlation matrix, no
detoning) was only ever used internally as the input stage of
HRP_Detoned. This adds it as a first-class strategy --- the clean
ablation HRP (neither) / HRP_Denoised (denoise) / HRP_Detoned
(denoise + detone) --- backtesting it for the headline, post-COVID and
weekly windows and re-running the inference on the augmented set.
"""
from __future__ import annotations
import pickle, warnings
import numpy as np, pandas as pd
import sys
# --- resolve the repo root so this script runs from any cwd / machine ---
import os as _os
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
from src.backtest import WalkForwardBacktest, COST_SCENARIOS, summarize_performance
from src.portfolio_maker import HRP, HRPDenoised
from src.inference import (stationary_block_bootstrap, sharpe_diff_ledoit_wolf,
                           hansen_spa_test)
from src.universe import load_pit_universe
warnings.filterwarnings("ignore")
ANN = np.sqrt(365)
DATA = str(_ROOT / "data")
NAME = "HRP_Denoised"

pit = load_pit_universe(f"{DATA}/pit_universe.csv")
ex = pd.read_csv(f"{DATA}/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv", parse_dates=["open_time"])
sp = pd.read_csv(f"{DATA}/binance_pit_supplement_2019-2024_1d.csv", parse_dates=["open_time"])
pl = pd.concat([ex, sp], ignore_index=True).drop_duplicates(subset=["symbol", "open_time"])
pl["close"] = pl["close"].astype(float)
prices = pl.pivot(index="open_time", columns="symbol", values="close").sort_index()
cm = COST_SCENARIOS["conservative_cex"]
fac = lambda r: HRPDenoised(r)
cols = ["name", "total_return", "ann_return", "ann_vol", "sharpe", "sortino",
        "max_drawdown", "calmar", "n_days", "arith_sharpe_ann", "rebalances",
        "avg_turnover", "total_costs", "avg_max_weight", "avg_n_eff"]


def metrics_row(name, daily, res):
    m = summarize_performance(daily, name=name)
    row = {k: m.get(k) for k in ("total_return", "ann_return", "ann_vol",
           "sharpe", "sortino", "max_drawdown", "calmar", "n_days")}
    row["name"] = name
    row["arith_sharpe_ann"] = float(daily.mean() / daily.std(ddof=1) * ANN)
    wh, th, ch = res["weights_history"], res["turnover_history"], res["cost_history"]
    row["rebalances"] = float(len(res["snapshot_dates"]))
    row["avg_turnover"] = float(np.mean(np.asarray(th))) if len(th) else 0.0
    row["total_costs"] = float(np.sum(np.asarray(ch))) if len(ch) else 0.0
    W = wh.fillna(0.0)
    row["avg_max_weight"] = float(W.max(axis=1).mean())
    row["avg_n_eff"] = float((1.0 / (W ** 2).sum(axis=1)).mean())
    return row


def splice_pickle(path, daily_series):
    d = pickle.load(open(path, "rb"))
    idx = pd.DataFrame({k: v for k, v in d.items() if k != "HODL_BTC"}).dropna().index
    d[NAME] = daily_series.reindex(idx)
    pickle.dump(d, open(path, "wb"))
    return d


# ---------- Scenario B headline ----------
print(f"{NAME} Scenario B...", flush=True)
resB = WalkForwardBacktest(prices, pit, fac, cost_model=cm).run("2020-01-01", "2026-05-18")
dB = resB["daily_returns"]
print(f"  Sharpe={dB.mean()/dB.std(ddof=1)*ANN:+.4f}  TR={((1+dB).prod()-1)*100:+.0f}%", flush=True)
dailyB = splice_pickle(f"{DATA}/backtest_v2_daily_returns.pkl", dB)

res_csv = pd.read_csv(f"{DATA}/backtest_v2_rebalance_results.csv")
res_csv = res_csv[res_csv["name"] != NAME]
res_csv = pd.concat([res_csv, pd.DataFrame([metrics_row(NAME, dB, resB)])], ignore_index=True)
res_csv[cols].to_csv(f"{DATA}/backtest_v2_rebalance_results.csv", index=False)

rdfB = pd.DataFrame(dailyB).dropna()
hrp = rdfB["HRP"]
sh = {c: rdfB[c].mean() / rdfB[c].std(ddof=1) * ANN for c in rdfB.columns}
lw_rows, ci_rows = [], []
for c in rdfB.columns:
    boot = stationary_block_bootstrap(
        rdfB[c].values,
        lambda r: float(np.mean(r) / np.std(r, ddof=1) * ANN) if np.std(r, ddof=1) > 0 else 0.0,
        n_iter=2000, seed=42)
    ci_rows.append({"name": c, "sharpe": sh[c],
                    "ci_lower_95": boot["ci_lower_95"], "ci_upper_95": boot["ci_upper_95"]})
    if c == "HRP":
        continue
    lw = sharpe_diff_ledoit_wolf(rdfB[c].values, hrp.values, n_iter=3000, seed=42)
    lw_rows.append({"name": c, "sharpe_diff_vs_hrp": sh[c] - sh["HRP"],
                    "lw_p_value": lw["p_value"]})
pd.DataFrame(lw_rows).to_csv(f"{DATA}/inference_sharpe_diff.csv", index=False)
pd.DataFrame(ci_rows).to_csv(f"{DATA}/inference_bootstrap_cis.csv", index=False)
spa = hansen_spa_test(rdfB.drop(columns=["HRP"]), hrp.values, n_bootstrap=5000, seed=42)
spa.to_csv(f"{DATA}/inference_spa.csv")
print(f"  headline SPA p_consistent={spa['p_consistent'].iloc[0]:.4f}", flush=True)

# ---------- Post-COVID ----------
print(f"{NAME} post-COVID...", flush=True)
resP = WalkForwardBacktest(prices, pit, fac, cost_model=cm).run("2022-01-01", "2026-05-18")
dP = resP["daily_returns"]
print(f"  Sharpe={dP.mean()/dP.std(ddof=1)*ANN:+.4f}  TR={((1+dP).prod()-1)*100:+.0f}%", flush=True)
dailyP = splice_pickle(f"{DATA}/backtest_v2_post_covid_daily_returns.pkl", dP)

pc_csv = pd.read_csv(f"{DATA}/backtest_v2_post_covid_results.csv")
pc_csv = pc_csv[pc_csv["name"] != NAME]
pc_csv = pd.concat([pc_csv, pd.DataFrame([metrics_row(NAME, dP, resP)])], ignore_index=True)
pc_csv[cols].to_csv(f"{DATA}/backtest_v2_post_covid_results.csv", index=False)

rdfP = pd.DataFrame(dailyP).dropna()
hrpP = rdfP["HRP"]
shP = {c: rdfP[c].mean() / rdfP[c].std(ddof=1) * ANN for c in rdfP.columns}
lwP = []
for c in rdfP.columns:
    if c == "HRP":
        continue
    r = sharpe_diff_ledoit_wolf(rdfP[c].values, hrpP.values, n_iter=3000, seed=42)
    lwP.append({"name": c, "sharpe_diff_vs_hrp": shP[c] - shP["HRP"], "lw_p_value": r["p_value"]})
pd.DataFrame(lwP).to_csv(f"{DATA}/inference_post_covid_sharpe_diff.csv", index=False)
spaP = hansen_spa_test(rdfP.drop(columns=["HRP"]), hrpP.values, n_bootstrap=5000, seed=42)
spaP.to_csv(f"{DATA}/inference_post_covid_spa.csv")
print(f"  post-COVID SPA p_consistent={spaP['p_consistent'].iloc[0]:.4f}", flush=True)

# ---------- Weekly ----------
print(f"{NAME} weekly...", flush=True)
weekly_dates = pd.date_range("2020-01-01", "2026-05-18", freq="W-FRI")
month_ends = sorted(pd.Timestamp(d) for d in pit["date"].unique())
wk_rows = []
ps = pit.sort_values("date")
for d in weekly_dates:
    prior = [m for m in month_ends if m <= d]
    if not prior:
        continue
    for _, r in ps[(ps["date"] == prior[-1]) & ps["included"]].iterrows():
        wk_rows.append({"date": d, "symbol": r["symbol"], "included": True})
pit_weekly = pd.DataFrame(wk_rows)
resW = WalkForwardBacktest(prices, pit_weekly, fac, cost_model=cm).run("2020-01-01", "2026-05-18")
dailyW = splice_pickle(f"{DATA}/backtest_v2_weekly_daily_returns.pkl", resW["daily_returns"])
rdfW = pd.DataFrame(dailyW).dropna()
shW = float(rdfW[NAME].mean() / rdfW[NAME].std(ddof=1) * ANN)
boot = stationary_block_bootstrap(
    rdfW[NAME].values,
    lambda r: float(np.mean(r) / np.std(r, ddof=1) * ANN) if np.std(r, ddof=1) > 0 else 0.0,
    n_iter=2000, seed=42)
wk_csv = pd.read_csv(f"{DATA}/backtest_v2_weekly_results.csv")
wk_csv = wk_csv[wk_csv["strategy"] != NAME]
wk_csv = pd.concat([wk_csv, pd.DataFrame([{"strategy": NAME, "sharpe_ann": shW,
    "total_return": float((1 + rdfW[NAME]).prod() - 1),
    "ci_lower_95": boot["ci_lower_95"], "ci_upper_95": boot["ci_upper_95"]}])], ignore_index=True)
wk_csv.to_csv(f"{DATA}/backtest_v2_weekly_results.csv", index=False)
print(f"  weekly Sharpe={shW:+.4f}", flush=True)
print("DONE.", flush=True)
