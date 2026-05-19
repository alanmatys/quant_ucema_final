"""Generate all paper figures (Phase 5c).

Saves figures to paper/figures/. Reads from committed CSVs + reproduces
the equity curves by running the engine for the headline strategies.

Run: venv/bin/python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.universe import load_pit_universe
from src.backtest import WalkForwardBacktest, COST_SCENARIOS
from src.portfolio_maker import (
    HRP, HRPDetoned, HRPPartialCorr, HRPTailDep, HRPTailDepShrunk,
    HRPShrunkCov, HRPVolStd, IVP, MVP, ERC, MaxDiv, NetworkRiskParity,
)
from src.denoising import denoise_corr_constant_residual, detone_corr

OUT = "paper/figures"
START, END = "2020-01-01", "2026-05-18"

# Style
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 10})

# --- Data load ---
print("Loading data...")
pit = load_pit_universe("data/pit_universe.csv")
existing = pd.read_csv("data/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv", parse_dates=["open_time"])
supp = pd.read_csv("data/binance_pit_supplement_2019-2024_1d.csv", parse_dates=["open_time"])
prices_long = pd.concat([existing, supp], ignore_index=True).drop_duplicates(subset=["symbol","open_time"])
prices_long["close"] = prices_long["close"].astype(float)
prices = prices_long.pivot(index="open_time", columns="symbol", values="close").sort_index()

# --- Run backtests to get equity curves ---
print("Running backtests for equity curves...")
factories = {
    "HRP":                lambda r: HRP(r),
    "HRP_Detoned":        lambda r: HRPDetoned(r),
    "HRP_TailDep":        lambda r: HRPTailDep(r, q=0.05),
    "HRP_TailDepShrunk":  lambda r: HRPTailDepShrunk(r, q=0.05),
    "HRP_ShrunkCov":      lambda r: HRPShrunkCov(r),
    "MVP":                lambda r: MVP(r),
    "ERC":                lambda r: ERC(r),
}
cost = COST_SCENARIOS["conservative_cex"]
returns_dict = {}
for name, factory in factories.items():
    t0 = time.time()
    bt = WalkForwardBacktest(prices, pit, factory, cost_model=cost)
    res = bt.run(START, END)
    returns_dict[name] = res["daily_returns"]
    print(f"  {name:20s} done [{time.time()-t0:.1f}s]")

# BTC HODL
returns_dict["HODL_BTC"] = prices["BTCUSDT"].loc[START:END].pct_change().dropna()
returns_df = pd.DataFrame(returns_dict).dropna()

# ====================================================================
# Fig 1: cumulative returns (monthly rebal)
# ====================================================================
print("\nGenerating figures...")
fig, ax = plt.subplots(figsize=(11, 5.5))
cum = (1 + returns_df).cumprod()
palette = sns.color_palette("tab10", n_colors=len(cum.columns))
for i, c in enumerate(cum.columns):
    style = "--" if c in ("HODL_BTC", "MVP") else "-"
    lw = 1.6 if c in ("HRP", "HRP_TailDepShrunk", "HODL_BTC") else 1.0
    ax.plot(cum.index, cum[c], style, label=c, color=palette[i], linewidth=lw)
ax.set_yscale("log")
ax.set_ylabel("Cumulative return (log scale)")
ax.set_title("Cumulative returns — monthly rebalancing (Conservative CEX cost), 2020-01 → 2026-05")
ax.legend(loc="upper left", ncol=2, fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/rebalance_cumulative_returns.png", bbox_inches="tight")
plt.close()
print("  rebalance_cumulative_returns.png")

# ====================================================================
# Fig 2: risk-return scatter (all strategies in summary CSV)
# ====================================================================
results_df = pd.read_csv("data/backtest_v2_rebalance_results.csv").set_index("name")
fig, ax = plt.subplots(figsize=(10, 6.5))
for name, row in results_df.iterrows():
    color = ("orange" if "MOM" in name or name in ("CS_MOM_eq21","CS_MOM_vol21","TS_MOM_ma","TS_MOM_abs","RM_MOM") else
             "tab:red" if name in ("MVP","MaxDiv","HODL_BTC") else
             "tab:blue")
    marker = "o" if name.startswith("HRP") else ("s" if name in ("MVP","ERC","IVP","MaxDiv") else "^")
    ax.scatter(row["ann_vol"] * 100, row["ann_return"] * 100,
               s=120, alpha=0.7, color=color, marker=marker, edgecolor="black", linewidth=0.5)
    ax.annotate(name, (row["ann_vol"]*100, row["ann_return"]*100),
                xytext=(5, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("Annualized volatility (%)")
ax.set_ylabel("Annualized return (%)")
ax.set_title("Risk-return scatter (Scenario B, Conservative CEX, 2020-2026)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/risk_return_scatter.png", bbox_inches="tight")
plt.close()
print("  risk_return_scatter.png")

# ====================================================================
# Fig 3: metrics heatmap
# ====================================================================
metrics_for_heatmap = results_df[["sharpe", "sortino", "calmar", "max_drawdown", "ann_vol", "avg_turnover"]].copy()
metrics_for_heatmap["max_drawdown"] = -metrics_for_heatmap["max_drawdown"]  # show positive magnitude
metrics_for_heatmap = metrics_for_heatmap.rename(columns={
    "sharpe":"Sharpe", "sortino":"Sortino", "calmar":"Calmar",
    "max_drawdown":"|MaxDD|", "ann_vol":"AnnVol", "avg_turnover":"AvgTurn"
})
# Normalize each column to [0,1] for color scale (separately for each metric since units differ)
norm = (metrics_for_heatmap - metrics_for_heatmap.min()) / (metrics_for_heatmap.max() - metrics_for_heatmap.min())
fig, ax = plt.subplots(figsize=(8, 9))
sns.heatmap(norm, annot=metrics_for_heatmap.round(2).values, fmt=".2f",
            cmap="RdYlGn", ax=ax, cbar_kws={"label":"per-metric rank (0=worst, 1=best)"},
            linewidths=0.5)
ax.set_title("Strategy × Metric heatmap (Scenario B)")
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(f"{OUT}/metrics_heatmap.png", bbox_inches="tight")
plt.close()
print("  metrics_heatmap.png")

# ====================================================================
# Fig 4: cluster stability timeseries (Phase 5b finding)
# ====================================================================
cs = pd.read_csv("data/cluster_stability.csv", parse_dates=["date"])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
ax1.plot(cs["date"], cs["pearson_cophenetic"], label="Pearson", color="tab:blue", linewidth=1.4)
ax1.plot(cs["date"], cs["taildep_cophenetic"], label="TailDep", color="tab:orange", linewidth=1.4)
ax1.set_ylabel("Cophenetic correlation")
ax1.set_title("Cluster stability across 76 monthly snapshots")
ax1.legend(loc="lower left"); ax1.grid(True, alpha=0.3)
ax1.axhline(0.7, color="grey", linestyle=":", alpha=0.5)

ax2.plot(cs["date"], cs["pearson_ari_vs_prev"], label="Pearson", color="tab:blue", linewidth=1.4)
ax2.plot(cs["date"], cs["taildep_ari_vs_prev"], label="TailDep", color="tab:orange", linewidth=1.4)
ax2.set_ylabel("ARI vs previous snapshot (K=5)")
ax2.set_xlabel("Date")
ax2.axhline(0, color="grey", linestyle=":", alpha=0.5)
ax2.legend(loc="lower left"); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/cluster_stability_timeseries.png", bbox_inches="tight")
plt.close()
print("  cluster_stability_timeseries.png")

# ====================================================================
# Fig 5: SPA p-values bar plot
# ====================================================================
spa = pd.read_csv("data/inference_spa.csv", index_col=0)
fig, ax = plt.subplots(figsize=(10, 5))
spa[["p_lower", "p_consistent", "p_upper"]].plot(kind="bar", ax=ax,
    color=["tab:green", "tab:blue", "tab:red"], width=0.78)
ax.axhline(0.05, color="black", linestyle="--", linewidth=1, label="α=0.05")
ax.set_ylabel("Hansen SPA p-value (vs HRP)")
ax.set_title("Hansen SPA test (Scenario B) — cannot reject 'no strategy beats HRP'")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/spa_pvalues_barplot.png", bbox_inches="tight")
plt.close()
print("  spa_pvalues_barplot.png")

# ====================================================================
# Fig 6: cost sensitivity
# ====================================================================
cost_df = pd.read_csv("data/backtest_v2_cost_sensitivity.csv")
fig, ax = plt.subplots(figsize=(9, 5))
cost_pivot = cost_df.pivot(index="strategy", columns="cost_scenario", values="sharpe_ann")
cost_pivot = cost_pivot[["zero", "conservative_cex", "optimistic_cex", "stress"]]
cost_pivot.plot(kind="bar", ax=ax, colormap="viridis")
ax.set_ylabel("Annualized Sharpe")
ax.set_title("Cost sensitivity (5 strategies × 4 cost scenarios)")
ax.legend(title="cost scenario", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(True, alpha=0.3, axis="y")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/cost_sensitivity_sharpe.png", bbox_inches="tight")
plt.close()
print("  cost_sensitivity_sharpe.png")

# ====================================================================
# Fig 7: detoning vs partial correlation (the 0.99 finding)
# ====================================================================
print("Computing detoning vs partial-correlation Spearman across all snapshots...")
from src.backtest import _returns_for_window
snap_dates = sorted([pd.Timestamp(d) for d in pit["date"].unique()
                     if pd.Timestamp(START) <= pd.Timestamp(d) <= pd.Timestamp(END)])
rho_rows = []
for snap in snap_dates:
    inc = sorted(pit[(pit["date"]==snap) & pit["included"]]["symbol"].tolist())
    r = _returns_for_window(prices, snap, inc, lookback=365, min_periods=180)
    if r.shape[1] < 5: continue
    try:
        w_d = HRPDetoned(r).get_weights()
        w_p = HRPPartialCorr(r, alpha=0.10).get_weights()
        common = w_d.index.intersection(w_p.index)
        rho = float(w_d[common].corr(w_p[common], method="spearman"))
        rho_rows.append({"date": snap, "N": r.shape[1], "rho": rho})
    except Exception:
        continue
rho_df = pd.DataFrame(rho_rows)
rho_df.to_csv(f"{OUT}/../../data/detoned_vs_partialcorr_rho_timeseries.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(rho_df["date"], rho_df["rho"], color="tab:blue", linewidth=1.4, marker="o", markersize=3)
ax.axhline(0.99, color="grey", linestyle=":", alpha=0.5, label="ρ=0.99 (typical)")
ax.set_ylim(0.7, 1.01)
ax.set_ylabel("Spearman ρ (weight rankings)")
ax.set_title(f"HRP_Detoned vs HRP_PartialCorr weight Spearman across {len(rho_df)} snapshots — mostly 0.99+ with regime exceptions")
ax.grid(True, alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/detoning_vs_partial_corr.png", bbox_inches="tight")
plt.close()
print(f"  detoning_vs_partial_corr.png  (mean ρ = {rho_df['rho'].mean():.4f}, min = {rho_df['rho'].min():.4f})")

# ====================================================================
# Fig 8: shrinkage intensity timeseries
# ====================================================================
shrink = pd.read_csv("data/hrp_shrunkcov_intensity.csv", parse_dates=["date"])
fig, ax = plt.subplots(figsize=(11, 4))
for s in shrink["strategy"].unique():
    sub = shrink[shrink["strategy"]==s]
    ax.plot(sub["date"], sub["alpha"], label=s, linewidth=1.4)
ax.set_ylabel("Ledoit-Wolf shrinkage intensity α")
ax.set_title("LW shrinkage intensity across rebalances (lower α = sample cov better conditioned)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/shrinkage_intensity_timeseries.png", bbox_inches="tight")
plt.close()
print("  shrinkage_intensity_timeseries.png")

# ====================================================================
# Fig 9: weight concentration histogram (avg N_eff per strategy)
# ====================================================================
fig, ax = plt.subplots(figsize=(10, 5))
neff = results_df["avg_n_eff"].dropna().sort_values()
colors = ["tab:red" if v < 5 else "tab:orange" if v < 15 else "tab:blue" for v in neff]
ax.barh(neff.index, neff.values, color=colors, edgecolor="black", linewidth=0.5)
ax.axvline(50, color="grey", linestyle=":", label="N=50 (max diversified)")
ax.set_xlabel("Average effective N (1/Σwᵢ²)")
ax.set_title("Average portfolio concentration (Scenario B, 2020-2026)")
ax.legend(); ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(f"{OUT}/weight_concentration.png", bbox_inches="tight")
plt.close()
print("  weight_concentration.png")

# ====================================================================
# Fig 10: turnover analysis
# ====================================================================
fig, ax = plt.subplots(figsize=(10, 5))
turn = results_df["avg_turnover"].dropna().sort_values()
colors2 = ["tab:green" if v < 0.2 else "tab:orange" if v < 0.5 else "tab:red" for v in turn]
ax.barh(turn.index, turn.values, color=colors2, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Average L1 turnover per rebalance")
ax.set_title("Strategy turnover (Scenario B) — momentum strategies fully reconstruct each month")
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(f"{OUT}/turnover_analysis.png", bbox_inches="tight")
plt.close()
print("  turnover_analysis.png")

# ====================================================================
# Fig 11: PIT universe evolution
# ====================================================================
summary = pd.read_csv("data/pit_universe_summary.csv", parse_dates=["date"])
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(summary["date"], summary["n_eligible"], label="eligible (passed filters)", color="tab:blue", linewidth=1.4)
ax.plot(summary["date"], summary["n_included"], label="included (after buffers)", color="tab:orange", linewidth=1.4)
ax.axhline(50, color="grey", linestyle=":", label="top_n cap = 50")
ax.set_ylabel("Number of symbols")
ax.set_title("PIT universe size over time (2017-08-17 → 2026-05-18)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/pit_universe_evolution.png", bbox_inches="tight")
plt.close()
print("  pit_universe_evolution.png")

print(f"\nAll figures written to {OUT}/")
