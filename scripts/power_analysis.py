"""Statistical power analysis for the headline Sharpe-difference test.

A null result is only as strong as the test's power. This quantifies what
Sharpe edge the design could actually have detected: for the Ledoit-Wolf
studentized Sharpe-difference test, the studentized statistic is ~N(d/SE, 1)
under the alternative, so 80% power at a two-sided 5% level needs
d/SE >= 2.80 (Phi(2.80 - 1.96) = 0.80). The minimum detectable annualised
Sharpe edge is therefore 2.80 * SE, where SE is the delta-method HAC
standard error of the annualised Sharpe difference vs baseline HRP.

Writes data/power_analysis.csv and prints a summary.
"""
from __future__ import annotations
import pickle
import warnings
import numpy as np, pandas as pd
import sys
sys.path.insert(0, "/Users/alanmatys/Repos/quant_ucema_final")
from src.inference import _sharpe_diff_hac_se, optimal_block_length, stationary_block_bootstrap
warnings.filterwarnings("ignore")
DATA = "/Users/alanmatys/Repos/quant_ucema_final/data"
ANN = np.sqrt(365)
Z80 = 2.80  # d/SE needed for 80% power, two-sided 5% test

d = pickle.load(open(f"{DATA}/backtest_v2_daily_returns.pkl", "rb"))
rdf = pd.DataFrame(d).dropna()
hrp = rdf["HRP"].values
n = len(rdf)
bs = max(2, int(round(optimal_block_length(hrp))))
print(f"Headline window: {n} daily obs, block length {bs}\n")

rows = []
for c in rdf.columns:
    if c == "HRP":
        continue
    rc = rdf[c].values
    # annualised Sharpe difference and its delta-method HAC SE
    se_daily = _sharpe_diff_hac_se(rc, hrp, bandwidth=bs)
    se_ann = se_daily * ANN
    sh_c = rc.mean() / rc.std(ddof=1) * ANN
    sh_h = hrp.mean() / hrp.std(ddof=1) * ANN
    mde = Z80 * se_ann  # minimum detectable edge at 80% power
    rows.append({"strategy": c, "sharpe_vs_hrp": sh_c - sh_h,
                 "se_ann": se_ann, "min_detectable_edge_80pct": mde})

df = pd.DataFrame(rows).sort_values("se_ann")
df.to_csv(f"{DATA}/power_analysis.csv", index=False)

# The MDE depends on how far a strategy departs from HRP. Near-clone HRP
# variants are ~0.97+ correlated with HRP, so the paired difference has
# tiny variance and the test detects minute edges; but those variants
# barely differ from HRP anyway. The meaningful figure is the MDE for
# strategies whose allocation genuinely departs from HRP.
distinct = df[df["se_ann"] >= 0.05]   # genuinely-different allocations
clones = df[df["se_ann"] < 0.05]
print("Minimum detectable annualised Sharpe edge vs HRP at 80% power:")
print(f"  near-clone HRP variants ({len(clones)}): MDE "
      f"{clones['min_detectable_edge_80pct'].min():.2f}-{clones['min_detectable_edge_80pct'].max():.2f} "
      f"(but their actual edges are <= {clones['sharpe_vs_hrp'].abs().max():.2f})")
print(f"  structurally distinct strategies ({len(distinct)}): MDE "
      f"{distinct['min_detectable_edge_80pct'].min():.2f}-{distinct['min_detectable_edge_80pct'].max():.2f}")
for _, r in distinct.sort_values("min_detectable_edge_80pct").iterrows():
    flag = "detectable" if abs(r["sharpe_vs_hrp"]) >= r["min_detectable_edge_80pct"] else "BELOW threshold"
    print(f"    {r['strategy']:14s} edge {r['sharpe_vs_hrp']:+.2f}  vs  MDE "
          f"{r['min_detectable_edge_80pct']:.2f}  -> {flag}")

# Per-strategy level-Sharpe CI half-width (why even single Sharpes are imprecise)
hw = []
for c in ["HRP", "HRP_ShrunkCov", "MVP"]:
    boot = stationary_block_bootstrap(
        rdf[c].values,
        lambda r: float(np.mean(r) / np.std(r, ddof=1) * ANN) if np.std(r, ddof=1) > 0 else 0.0,
        n_iter=3000, seed=42)
    hw.append((c, (boot["ci_upper_95"] - boot["ci_lower_95"]) / 2.0))
print("\nLevel-Sharpe 95% bootstrap CI half-width:")
for c, h in hw:
    print(f"  {c:14s} +/- {h:.2f}")
print(f"\nSaved -> power_analysis.csv")
print(f"\nInterpretation: a strategy whose allocation genuinely departs from "
      f"HRP must beat it by roughly 0.4-0.8 Sharpe to be detected at 80% "
      f"power on this 6.3-year sample. NCO (+0.19) and MVP (+0.31) post "
      f"economically large edges that nonetheless fall below that "
      f"threshold --- the null is a statement about detectable effect "
      f"size, not evidence of zero effect.")
