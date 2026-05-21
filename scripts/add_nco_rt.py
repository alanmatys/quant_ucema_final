"""Backtest NCOReturnTilted and splice it into the existing result files.

NCO_RT is a new candidate added after the main re-run; rather than re-run
every strategy, we backtest only NCO_RT (it is fast --- no embedding) for
the headline, post-COVID and weekly windows, append it to the metric
CSVs and the daily-return pickles, and re-run the inference (Ledoit-Wolf,
Hansen SPA, bootstrap CIs) on the augmented candidate set.
"""
from __future__ import annotations
import pickle, warnings
import numpy as np, pandas as pd
import sys
sys.path.insert(0, "/Users/alanmatys/Repos/quant_ucema_final")
from src.backtest import WalkForwardBacktest, COST_SCENARIOS, summarize_performance
from src.portfolio_maker import HRP, NCOReturnTilted
from src.inference import (stationary_block_bootstrap, sharpe_diff_ledoit_wolf,
                           hansen_spa_test)
from src.universe import load_pit_universe
warnings.filterwarnings("ignore")
ANN = np.sqrt(365)
DATA = "/Users/alanmatys/Repos/quant_ucema_final/data"

pit = load_pit_universe(f"{DATA}/pit_universe.csv")
ex = pd.read_csv(f"{DATA}/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv", parse_dates=["open_time"])
sp = pd.read_csv(f"{DATA}/binance_pit_supplement_2019-2024_1d.csv", parse_dates=["open_time"])
pl = pd.concat([ex, sp], ignore_index=True).drop_duplicates(subset=["symbol", "open_time"])
pl["close"] = pl["close"].astype(float)
prices = pl.pivot(index="open_time", columns="symbol", values="close").sort_index()
cm = COST_SCENARIOS["conservative_cex"]
fac = lambda r: NCOReturnTilted(r)
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


def run_inference(rdf, tag):
    """Re-run LW + SPA + CIs on an augmented daily-return panel."""
    hrp = rdf["HRP"]
    sh = {c: rdf[c].mean() / rdf[c].std(ddof=1) * ANN for c in rdf.columns}
    lw_rows, ci_rows = [], []
    for c in rdf.columns:
        boot = stationary_block_bootstrap(
            rdf[c].values,
            lambda r: float(np.mean(r) / np.std(r, ddof=1) * ANN) if np.std(r, ddof=1) > 0 else 0.0,
            n_iter=2000, seed=42)
        ci_rows.append({"name": c, "sharpe": sh[c],
                        "ci_lower_95": boot["ci_lower_95"], "ci_upper_95": boot["ci_upper_95"]})
        if c == "HRP":
            continue
        lw = sharpe_diff_ledoit_wolf(rdf[c].values, hrp.values, n_iter=3000, seed=42)
        lw_rows.append({"name": c, "sharpe_diff_vs_hrp": sh[c] - sh["HRP"],
                        "lw_p_value": lw["p_value"]})
    spa = hansen_spa_test(rdf.drop(columns=["HRP"]), hrp.values, n_bootstrap=5000, seed=42)
    return pd.DataFrame(lw_rows), pd.DataFrame(ci_rows), spa


# ---------- Scenario B headline ----------
print("NCO_RT Scenario B...", flush=True)
resB = WalkForwardBacktest(prices, pit, fac, cost_model=cm).run("2020-01-01", "2026-05-18")
dB = resB["daily_returns"]
print(f"  Sharpe={dB.mean()/dB.std(ddof=1)*ANN:+.4f}  TR={((1+dB).prod()-1)*100:+.0f}%", flush=True)

dailyB = pickle.load(open(f"{DATA}/backtest_v2_daily_returns.pkl", "rb"))
dailyB["NCO_RT"] = dB.reindex(pd.DataFrame({k: v for k, v in dailyB.items() if k != "HODL_BTC"}).dropna().index)
pickle.dump(dailyB, open(f"{DATA}/backtest_v2_daily_returns.pkl", "wb"))

res_csv = pd.read_csv(f"{DATA}/backtest_v2_rebalance_results.csv")
res_csv = res_csv[res_csv["name"] != "NCO_RT"]
res_csv = pd.concat([res_csv, pd.DataFrame([metrics_row("NCO_RT", dB, resB)])], ignore_index=True)
res_csv[cols].to_csv(f"{DATA}/backtest_v2_rebalance_results.csv", index=False)

rdfB = pd.DataFrame(dailyB).dropna()
lw, ci, spa = run_inference(rdfB, "B")
lw.to_csv(f"{DATA}/inference_sharpe_diff.csv", index=False)
ci.to_csv(f"{DATA}/inference_bootstrap_cis.csv", index=False)
spa.to_csv(f"{DATA}/inference_spa.csv")
print(f"  headline SPA p_consistent={spa['p_consistent'].iloc[0]:.4f}", flush=True)

# ---------- Post-COVID ----------
print("NCO_RT post-COVID...", flush=True)
resP = WalkForwardBacktest(prices, pit, fac, cost_model=cm).run("2022-01-01", "2026-05-18")
dP = resP["daily_returns"]
print(f"  Sharpe={dP.mean()/dP.std(ddof=1)*ANN:+.4f}  TR={((1+dP).prod()-1)*100:+.0f}%", flush=True)

dailyP = pickle.load(open(f"{DATA}/backtest_v2_post_covid_daily_returns.pkl", "rb"))
dailyP["NCO_RT"] = dP.reindex(pd.DataFrame({k: v for k, v in dailyP.items() if k != "HODL_BTC"}).dropna().index)
pickle.dump(dailyP, open(f"{DATA}/backtest_v2_post_covid_daily_returns.pkl", "wb"))

pc_csv = pd.read_csv(f"{DATA}/backtest_v2_post_covid_results.csv")
pc_csv = pc_csv[pc_csv["name"] != "NCO_RT"]
pc_csv = pd.concat([pc_csv, pd.DataFrame([metrics_row("NCO_RT", dP, resP)])], ignore_index=True)
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
print("NCO_RT weekly...", flush=True)
weekly_dates = pd.date_range("2020-01-01", "2026-05-18", freq="W-FRI")
month_ends = sorted(pd.Timestamp(d) for d in pit["date"].unique())
wk_rows = []
ps = pit.sort_values("date")
for d in weekly_dates:
    prior = [m for m in month_ends if m <= d]
    if not prior:
        continue
    inc = ps[(ps["date"] == prior[-1]) & ps["included"]]
    for _, r in inc.iterrows():
        wk_rows.append({"date": d, "symbol": r["symbol"], "included": True})
pit_weekly = pd.DataFrame(wk_rows)
resW = WalkForwardBacktest(prices, pit_weekly, fac, cost_model=cm).run("2020-01-01", "2026-05-18")
dW = resW["daily_returns"]

dailyW = pickle.load(open(f"{DATA}/backtest_v2_weekly_daily_returns.pkl", "rb"))
dailyW["NCO_RT"] = dW.reindex(pd.DataFrame({k: v for k, v in dailyW.items() if k != "HODL_BTC"}).dropna().index)
pickle.dump(dailyW, open(f"{DATA}/backtest_v2_weekly_daily_returns.pkl", "wb"))
rdfW = pd.DataFrame(dailyW).dropna()
shW = {c: rdfW[c].mean() / rdfW[c].std(ddof=1) * ANN for c in rdfW.columns}
wk_csv = pd.read_csv(f"{DATA}/backtest_v2_weekly_results.csv")
wk_csv = wk_csv[wk_csv["strategy"] != "NCO_RT"]
boot = stationary_block_bootstrap(
    rdfW["NCO_RT"].values,
    lambda r: float(np.mean(r) / np.std(r, ddof=1) * ANN) if np.std(r, ddof=1) > 0 else 0.0,
    n_iter=2000, seed=42)
wk_csv = pd.concat([wk_csv, pd.DataFrame([{"strategy": "NCO_RT", "sharpe_ann": shW["NCO_RT"],
    "total_return": float((1 + rdfW["NCO_RT"]).prod() - 1),
    "ci_lower_95": boot["ci_lower_95"], "ci_upper_95": boot["ci_upper_95"]}])], ignore_index=True)
wk_csv.to_csv(f"{DATA}/backtest_v2_weekly_results.csv", index=False)
print(f"  weekly Sharpe={shW['NCO_RT']:+.4f}", flush=True)
print("DONE.", flush=True)
