"""Walk-forward XGBoost forecasts of next-30-day asset returns.

For each of the 76 monthly PIT rebalance snapshots:

  1. Build a per-asset feature panel from data observable strictly before
     the snapshot (date <= snap - 2 days). Features:
       r1   : 1-day return
       r5   : 5-day cumulative log return
       r21  : 21-day cumulative log return
       r63  : 63-day cumulative log return
       vol21: 21-day realised volatility
       xrank: cross-sectional rank of r21 on that date
  2. Target: next-30-day cumulative log return. Trainable rows are those
     whose target is fully observable by the snapshot cutoff.
  3. Pooled cross-sectional XGBoost regressor (one model across all
     assets per snapshot). Hyperparameters tuned via TimeSeriesSplit(3)
     CV inside the training window with a small deterministic grid:
       max_depth in {3, 5}, learning_rate in {0.03, 0.1},
       n_estimators in {200, 500}.
  4. Best estimator refit on the full training window, then used to
     predict mu for every asset in the PIT universe at the snapshot.

Outputs:
  data/xgboost_mu_predictions.csv  (wide: snapshot x asset)
  data/xgboost_best_params.csv     (snapshot x best params, for audit)

Determinism: tree_method='hist', n_jobs=1, single-thread BLAS via the
imports below, fixed seed=42. Bit-identical within a fixed XGBoost +
scikit-learn build.
"""
from __future__ import annotations

import os
# Pin BLAS / OMP to single-thread BEFORE numpy/xgboost import.
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
os.chdir(_ROOT)

from src.universe import load_pit_universe  # noqa: E402

warnings.filterwarnings("ignore")

import xgboost as xgb  # noqa: E402
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit  # noqa: E402

DATA = _ROOT / "data"
SEED = 42
HORIZON = 30        # forward-return target window (calendar days)
SNAP_BUFFER = 2     # use only data <= snap - SNAP_BUFFER (no look-ahead)
FEATURES = ["r1", "r5", "r21", "r63", "vol21", "xrank"]

GRID = {
    "max_depth":     [3, 5],
    "learning_rate": [0.03, 0.1],
    "n_estimators":  [200, 500],
}
TS_SPLITS = 3
START = pd.Timestamp("2020-01-01")
END   = pd.Timestamp("2026-05-18")


def _load_prices() -> pd.DataFrame:
    ex = pd.read_csv(f"{DATA}/binance_usdt_pairs_2018-12-31_2024-01-01_1d.csv",
                     parse_dates=["open_time"])
    sp = pd.read_csv(f"{DATA}/binance_pit_supplement_2019-2024_1d.csv",
                     parse_dates=["open_time"])
    pl = pd.concat([ex, sp], ignore_index=True).drop_duplicates(
        subset=["symbol", "open_time"])
    pl["close"] = pl["close"].astype(float)
    return pl.pivot(index="open_time", columns="symbol",
                    values="close").sort_index()


def _long(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.stack().rename(name).to_frame().reset_index()
    out.columns = ["date", "asset", name]
    return out


def _build_panels(prices: pd.DataFrame):
    """Long-form (date, asset, features..., y) and feature-only panels."""
    rets = prices.pct_change()
    lret = np.log1p(rets)
    r5   = lret.rolling(5).sum()
    r21  = lret.rolling(21).sum()
    r63  = lret.rolling(63).sum()
    vol21 = rets.rolling(21).std()
    xrank = r21.rank(axis=1, pct=True)
    fwd  = lret.rolling(HORIZON).sum().shift(-HORIZON)

    # feature-only panel (used at prediction time, before target is observable)
    feat = _long(rets, "r1")
    for nm, df in [("r5", r5), ("r21", r21), ("r63", r63),
                   ("vol21", vol21), ("xrank", xrank)]:
        feat = feat.merge(_long(df, nm), on=["date", "asset"], how="inner")
    feat = feat.dropna(subset=FEATURES).sort_values(["date", "asset"])

    # full panel (features + target) for training; drops rows without target
    train = feat.merge(_long(fwd, "y"), on=["date", "asset"], how="inner")
    train = train.dropna(subset=FEATURES + ["y"])
    return feat, train


def main() -> None:
    print("Loading data...", flush=True)
    prices = _load_prices()
    pit = load_pit_universe(f"{DATA}/pit_universe.csv")
    print(f"  prices {prices.shape}", flush=True)

    feat, train = _build_panels(prices)
    print(f"  feature panel {len(feat):,} rows  /  train panel {len(train):,} "
          f"rows ({train['date'].min().date()}..{train['date'].max().date()})",
          flush=True)

    snap_dates = sorted(pd.Timestamp(d) for d in pit["date"].unique()
                        if START <= pd.Timestamp(d) <= END)
    print(f"  snapshots: {len(snap_dates)}\n", flush=True)

    rng_seed_master = np.random.RandomState(SEED)  # reserved for any later use
    mu_rows: list[dict] = []
    param_rows: list[dict] = []

    t_all = time.time()
    for i, snap in enumerate(snap_dates, 1):
        cutoff = snap - pd.Timedelta(days=SNAP_BUFFER)
        universe = sorted(pit[(pit["date"] == snap) & pit["included"]]
                          ["symbol"].tolist())

        # trainable rows: feature date <= cutoff AND target observable by cutoff
        tr = train[train["date"] <= cutoff - pd.Timedelta(days=HORIZON)]
        if len(tr) < 200 or len(universe) == 0:
            print(f"[{i}/{len(snap_dates)}] {snap.date()}  "
                  f"SKIP (n_train={len(tr)}, n_univ={len(universe)})", flush=True)
            continue
        X = tr[FEATURES].values.astype(np.float32)
        y = tr["y"].values.astype(np.float32)

        # prediction features: latest available feature row <= cutoff per asset
        f = feat[(feat["date"] <= cutoff) & feat["asset"].isin(universe)]
        f = f.sort_values("date").groupby("asset", as_index=False).tail(1)
        if f.empty:
            print(f"[{i}/{len(snap_dates)}] {snap.date()}  SKIP (no pred features)",
                  flush=True)
            continue

        t0 = time.time()
        base = xgb.XGBRegressor(tree_method="hist", n_jobs=1,
                                random_state=SEED, verbosity=0)
        cv = TimeSeriesSplit(n_splits=TS_SPLITS)
        gs = GridSearchCV(base, GRID, cv=cv,
                          scoring="neg_mean_squared_error",
                          n_jobs=1, refit=True)
        gs.fit(X, y)
        mu = gs.best_estimator_.predict(f[FEATURES].values.astype(np.float32))
        dt = time.time() - t0

        row = {"snapshot_date": snap}
        row.update(dict(zip(f["asset"].tolist(), mu.tolist())))
        mu_rows.append(row)
        param_rows.append({"snapshot_date": snap, **gs.best_params_,
                           "cv_score": float(gs.best_score_),
                           "n_train": int(len(tr)), "n_univ": int(len(universe))})

        print(f"[{i}/{len(snap_dates)}] {snap.date()}  n_train={len(tr):,} "
              f"n_univ={len(universe)}  best={gs.best_params_}  "
              f"mu_mean={float(np.mean(mu)):+.4f}  mu_std={float(np.std(mu)):.4f}"
              f"  [{dt:.1f}s]", flush=True)

    mu_df = pd.DataFrame(mu_rows).set_index("snapshot_date").sort_index()
    mu_df.to_csv(DATA / "xgboost_mu_predictions.csv")
    pd.DataFrame(param_rows).to_csv(DATA / "xgboost_best_params.csv", index=False)
    print(f"\nDONE in {time.time()-t_all:.0f}s.  "
          f"wrote {DATA / 'xgboost_mu_predictions.csv'} ({mu_df.shape})",
          flush=True)


if __name__ == "__main__":
    main()
