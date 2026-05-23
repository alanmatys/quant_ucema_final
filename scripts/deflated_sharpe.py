"""Deflated Sharpe Ratio for the headline strategies.

The DSR (Bailey & Lopez de Prado 2014) deflates an observed Sharpe for
(i) selection across the strategies tried, (ii) non-normality of the
return series, and (iii) sample length. It is a second multiple-testing
correction alongside the Hansen SPA. We compute it for the strategies
that beat baseline HRP, treating the 25 constructed strategies as the
trial set.
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from src.inference import deflated_sharpe_ratio, expected_max_sharpe

DATA = _ROOT / "data"
ANN = np.sqrt(365)

panel = pickle.load(open(DATA / "backtest_v2_daily_returns.pkl", "rb"))
rdf = pd.DataFrame(panel).dropna()
# the 25 constructed strategies (HODL_BTC is a passive benchmark, not a trial)
constructed = [c for c in rdf.columns if c != "HODL_BTC"]
N = len(constructed)
trial_sr = np.array([rdf[c].mean() / rdf[c].std(ddof=1) for c in constructed])

print(f"Trial set: {N} constructed strategies, "
      f"{rdf.shape[0]} daily observations")
sr0 = expected_max_sharpe(trial_sr, N)
print(f"Deflation hurdle SR0 (per-day) = {sr0:.5f}  "
      f"= {sr0*ANN:.4f} annualised\n")

# N = 25 (headline search) and N = 572 (headline + the 547-cell sweep).
N_FULL = N + 547
print(f"Full-search hurdle SR0 (N={N_FULL}) = "
      f"{expected_max_sharpe(trial_sr, N_FULL)*ANN:.4f} annualised\n")

rows = []
report = ["MVP", "CRISP", "NCO_CRISP", "NCO", "NCOML", "HRPSigmaMu",
          "HRP_Dynamic_94", "HRP"]
for name in report:
    d = deflated_sharpe_ratio(rdf[name].values, trial_sr, n_trials=N)
    d_full = deflated_sharpe_ratio(rdf[name].values, trial_sr, n_trials=N_FULL)
    rows.append({"strategy": name, "sharpe_ann": d["sr_hat"] * ANN,
                 "skew": d["skew"], "kurtosis": d["kurtosis"],
                 "sr0_ann": d["sr0"] * ANN, f"dsr_n{N}": d["dsr"],
                 f"dsr_n{N_FULL}": d_full["dsr"]})
    f1 = "clears" if d["dsr"] >= 0.95 else "below"
    f2 = "clears" if d_full["dsr"] >= 0.95 else "below"
    print(f"  {name:16s} Sharpe={d['sr_hat']*ANN:+.3f}  skew={d['skew']:+.2f}  "
          f"kurt={d['kurtosis']:5.1f}  DSR(N={N})={d['dsr']:.3f} ({f1} 0.95)  "
          f"DSR(N={N_FULL})={d_full['dsr']:.3f} ({f2})")

out = pd.DataFrame(rows)
out.to_csv(DATA / "deflated_sharpe.csv", index=False)
print(f"\nSaved -> data/deflated_sharpe.csv")
print("\nInterpretation: DSR is the probability the strategy's true Sharpe "
      "exceeds the\nexpected best-of-N under the null. DSR >= 0.95 means the "
      "edge survives the\nselection + non-normality correction.")
