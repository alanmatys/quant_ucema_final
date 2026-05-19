"""Walk-forward backtest engine for the continuation paper (Spec 03).

Handles:
- Monthly rebalancing on a PIT universe (different included assets per snapshot).
- 365-day estimation window with per-asset 180-day min-periods filter (Spec 03 R1b).
- Transaction costs proportional to L1 turnover, parameterized for the cost grid.
- Strategy weight history, daily portfolio return series, turnover per rebalance.

The engine is strategy-agnostic: takes any `PortfolioStrategy` subclass and its
constructor kwargs, fits at each rebalance, and tracks results.

Output of `WalkForwardBacktest.run(...)`:
    {
        "daily_returns": pd.Series of net daily portfolio returns (after costs),
        "gross_daily_returns": pd.Series of gross daily returns (before costs),
        "weights_history": pd.DataFrame [rebalance_date x asset] of post-trade weights,
        "turnover_history": pd.Series of L1 turnover per rebalance,
        "cost_history": pd.Series of % cost per rebalance,
        "snapshot_dates": list of rebalance dates,
        "extra": dict of strategy-specific diagnostics (e.g. LW shrinkage intensity).
    }
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.portfolio_maker import PortfolioStrategy


@dataclass
class CostModel:
    """Linear cost: cost_pct = (fee_bps + slippage_bps) * turnover_L1.

    For more sophisticated cost models, subclass and override `cost_pct`.

    Args:
        fee_bps: per-side execution fee in bps (e.g. 10 = 0.10%).
        slippage_bps: linear slippage in bps on top of the fee.
        liquidation_premium: multiplier applied to costs for assets exiting the
            universe (must be sold at any price). Default 2.0 per Spec 03 R3.
    """
    fee_bps: float = 10.0
    slippage_bps: float = 2.0
    liquidation_premium: float = 2.0

    def cost_pct(self, turnover: float, liquidation_share: float = 0.0) -> float:
        """Compute total cost as a fraction of portfolio value.

        Args:
            turnover: L1 turnover (sum of absolute weight changes).
            liquidation_share: fraction of turnover that is forced liquidation
                (assets exiting the PIT universe).
        """
        base = (self.fee_bps + self.slippage_bps) / 10_000.0
        normal_turn = turnover - liquidation_share
        liquidated_turn = liquidation_share
        return base * normal_turn + base * self.liquidation_premium * liquidated_turn


# Standard cost scenarios from Spec 03 §2.4
COST_SCENARIOS: dict[str, CostModel] = {
    "zero": CostModel(fee_bps=0.0, slippage_bps=0.0, liquidation_premium=1.0),
    "conservative_cex": CostModel(fee_bps=10.0, slippage_bps=2.0, liquidation_premium=2.0),
    "optimistic_cex": CostModel(fee_bps=7.5, slippage_bps=1.0, liquidation_premium=2.0),
    "stress": CostModel(fee_bps=10.0, slippage_bps=4.0, liquidation_premium=2.0),
}


def monthly_rebalance_dates(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[pd.Timestamp]:
    """Generate month-end rebalance dates between start and end (inclusive)."""
    return [d for d in pd.date_range(start=start, end=end, freq="ME") if d <= end]


def _returns_for_window(
    panel: pd.DataFrame,
    end_date: pd.Timestamp,
    included: list[str],
    lookback: int = 365,
    min_periods: int = 180,
) -> pd.DataFrame:
    """Build the estimation-window returns DataFrame for `included` assets.

    Applies Spec 03 R1b: 365-day lookback, per-asset min 180 daily returns.
    Assets failing the filter are dropped (logged via stderr by the caller).
    """
    start = end_date - pd.Timedelta(days=lookback)
    sub = panel.loc[start:end_date, [c for c in included if c in panel.columns]]
    r = sub.pct_change()
    valid = r.columns[r.notna().sum() >= min_periods].tolist()
    return r[valid].fillna(0)


def _next_day_returns(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Out-of-sample daily returns between (start, end] (exclusive of start)."""
    sub = panel.loc[start:end].pct_change().dropna(how="all")
    # Drop the start row (it's the rebalance close — not a return)
    sub = sub[sub.index > start]
    return sub


class WalkForwardBacktest:
    """Monthly-rebalanced walk-forward backtest on a PIT universe.

    Args:
        prices: T x N wide DataFrame of close prices (index = date, columns = symbol).
        pit_universe: long-format PIT DataFrame with [date, symbol, included] columns.
        strategy_factory: callable (returns_df) -> PortfolioStrategy instance.
            Called fresh at each rebalance date with the current estimation window.
        cost_model: CostModel instance (default = conservative_cex).
        lookback_days: estimation-window lookback (default 365 per Spec 03 R1b).
        min_periods: per-asset min daily-returns count (default 180).
        verbose: print per-snapshot progress.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        pit_universe: pd.DataFrame,
        strategy_factory: Callable[[pd.DataFrame], PortfolioStrategy],
        cost_model: CostModel | None = None,
        lookback_days: int = 365,
        min_periods: int = 180,
        rebalance_mode: str = "calendar",
        drift_threshold: float = 0.05,
        smoothing_eta: float = 0.0,
        min_trade_bps: float = 0.0,
        capture_cluster_stability: bool = False,
        verbose: bool = False,
    ):
        """
        Args:
            rebalance_mode: 'calendar' (every snapshot, current default),
                'threshold' (only when max(|w_drift|) > drift_threshold), or
                'threshold_smoothed' (threshold + linear blending via eta and
                min-trade filter).
            drift_threshold: triggers rebal in 'threshold' / 'threshold_smoothed'.
            smoothing_eta: in [0, 1). w_traded = eta * w_prev + (1 - eta) * w_target.
                Default 0 = no smoothing.
            min_trade_bps: minimum per-asset weight delta in bps to actually trade.
                Smaller deltas are reverted to prev weight.
            capture_cluster_stability: if True, save linkage matrix per snapshot
                so cophenetic + ARI can be computed downstream.
        """
        self.prices = prices.sort_index()
        self.pit = pit_universe.sort_values("date")
        self.strategy_factory = strategy_factory
        self.cost_model = cost_model or COST_SCENARIOS["conservative_cex"]
        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.rebalance_mode = rebalance_mode
        self.drift_threshold = drift_threshold
        self.smoothing_eta = smoothing_eta
        self.min_trade_bps = min_trade_bps
        self.capture_cluster_stability = capture_cluster_stability
        self.verbose = verbose

    def _included_at(self, snap: pd.Timestamp) -> list[str]:
        return sorted(
            self.pit[(self.pit["date"] == snap) & self.pit["included"]]["symbol"].tolist()
        )

    def run(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> dict:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        # Use PIT snapshots that fall in the window
        snap_dates = sorted(d for d in self.pit["date"].unique()
                            if pd.Timestamp(d) >= start and pd.Timestamp(d) <= end)
        if len(snap_dates) < 2:
            raise ValueError(f"Need >=2 snapshots in [{start.date()}, {end.date()}]; found {len(snap_dates)}")

        weights_history: dict[pd.Timestamp, pd.Series] = {}
        turnover_history: dict[pd.Timestamp, float] = {}
        cost_history: dict[pd.Timestamp, float] = {}
        extra: dict[pd.Timestamp, dict] = {}
        prev_weights = pd.Series(dtype=float)
        prev_universe: set[str] = set()
        daily_returns: list[pd.Series] = []
        gross_daily_returns: list[pd.Series] = []

        for i, snap in enumerate(snap_dates):
            snap = pd.Timestamp(snap)
            included = self._included_at(snap)
            if len(included) < 5:
                if self.verbose:
                    print(f"  {snap.date()}: only {len(included)} included, skipping")
                continue

            # Build estimation window
            train_returns = _returns_for_window(
                self.prices, snap, included,
                lookback=self.lookback_days, min_periods=self.min_periods,
            )
            if train_returns.shape[1] < 5:
                if self.verbose:
                    print(f"  {snap.date()}: only {train_returns.shape[1]} assets pass min-periods, holding previous weights")
                # Previously skipped the entire OOS interval (bug caught in
                # 2026-05-19 code review). Correct behaviour: hold previous
                # weights through the OOS interval, then move to the next
                # snapshot.
                if not prev_weights.empty:
                    next_snap = snap_dates[i + 1] if i + 1 < len(snap_dates) else end
                    if pd.Timestamp(next_snap) > snap:
                        oos_window = _next_day_returns(self.prices, snap, pd.Timestamp(next_snap))
                        if not oos_window.empty:
                            common = prev_weights.index.intersection(oos_window.columns)
                            if len(common) > 0:
                                held_returns = (oos_window[common] * prev_weights[common]).sum(axis=1)
                                daily_returns.append(held_returns)
                                gross_daily_returns.append(held_returns)
                continue

            # Fit strategy and get target weights
            try:
                model = self.strategy_factory(train_returns)
                target_w = model.get_weights()
            except Exception as exc:  # noqa: BLE001
                if self.verbose:
                    print(f"  {snap.date()}: strategy failed: {exc}; holding previous weights")
                target_w = prev_weights.copy() if not prev_weights.empty else pd.Series(
                    1.0 / train_returns.shape[1], index=train_returns.columns
                )

            # Apply rebalance-mode adjustments to compute traded weights
            current_universe = set(target_w.index[target_w > 0])
            if prev_weights.empty:
                traded_w = target_w
                turnover = 1.0
                liquidation_share = 0.0
            else:
                full_idx = sorted(set(target_w.index) | set(prev_weights.index))
                p = prev_weights.reindex(full_idx).fillna(0.0)
                t = target_w.reindex(full_idx).fillna(0.0)
                exited = prev_universe - set(included)
                forced_sale_pct = float(p.loc[list(exited)].sum()) if exited else 0.0

                if self.rebalance_mode == "threshold":
                    # Only rebal if max abs weight drift exceeds threshold OR if
                    # forced liquidation is required.
                    max_drift = float((t - p).abs().max())
                    if max_drift < self.drift_threshold and forced_sale_pct == 0.0:
                        # Skip rebalance — hold previous weights
                        traded_w = prev_weights
                        turnover = 0.0
                        liquidation_share = 0.0
                    else:
                        traded_w = target_w
                        turnover = float((t - p).abs().sum())
                        liquidation_share = forced_sale_pct
                elif self.rebalance_mode == "threshold_smoothed":
                    max_drift = float((t - p).abs().max())
                    if max_drift < self.drift_threshold and forced_sale_pct == 0.0:
                        traded_w = prev_weights
                        turnover = 0.0
                        liquidation_share = 0.0
                    else:
                        # Linear smoothing toward target
                        smoothed = self.smoothing_eta * p + (1 - self.smoothing_eta) * t
                        # Min-trade filter: revert per-asset deltas below threshold
                        min_trade = self.min_trade_bps / 10_000.0
                        delta = smoothed - p
                        keep = delta.abs() >= min_trade
                        smoothed = p + delta.where(keep, 0.0)
                        # Renormalize
                        smoothed = smoothed.clip(lower=0)
                        if smoothed.sum() > 0:
                            smoothed = smoothed / smoothed.sum()
                        traded_w = smoothed.reindex(target_w.index, fill_value=0.0)
                        turnover = float((smoothed - p).abs().sum())
                        liquidation_share = forced_sale_pct
                else:  # 'calendar' (default)
                    traded_w = target_w
                    turnover = float((t - p).abs().sum())
                    liquidation_share = forced_sale_pct

            cost = self.cost_model.cost_pct(turnover, liquidation_share)
            weights_history[snap] = traded_w
            turnover_history[snap] = turnover
            cost_history[snap] = cost

            # Strategy-specific diagnostics
            d = {}
            if hasattr(model, "shrinkage_intensity"):
                d["shrinkage_intensity"] = float(model.shrinkage_intensity)
            if hasattr(model, "fallback_used"):
                d["fallback_used"] = bool(model.fallback_used)
            extra[snap] = d

            # OOS returns: hold target_w from snap+1day to next snap (or end if last)
            next_snap = snap_dates[i + 1] if i + 1 < len(snap_dates) else end
            if pd.Timestamp(next_snap) <= snap:
                continue
            oos_window = _next_day_returns(self.prices, snap, pd.Timestamp(next_snap))
            if oos_window.empty:
                continue

            common = traded_w.index.intersection(oos_window.columns)
            if len(common) == 0:
                continue
            gross_daily = (oos_window[common] * traded_w[common]).sum(axis=1)
            # Apply cost on the FIRST day of holding (transaction settles at rebalance)
            net_daily = gross_daily.copy()
            if len(net_daily) > 0:
                net_daily.iloc[0] = net_daily.iloc[0] - cost
            gross_daily_returns.append(gross_daily)
            daily_returns.append(net_daily)

            if self.verbose:
                print(f"  {snap.date()}: N={train_returns.shape[1]}, "
                      f"turn={turnover:.3f}, cost={cost:.4%}, max_wt={traded_w.max():.4f}")

            prev_weights = traded_w
            prev_universe = set(traded_w.index[traded_w > 0])

        full_daily = pd.concat(daily_returns).sort_index() if daily_returns else pd.Series(dtype=float)
        full_gross = pd.concat(gross_daily_returns).sort_index() if gross_daily_returns else pd.Series(dtype=float)

        wh = pd.DataFrame(weights_history).T.fillna(0.0)
        return {
            "daily_returns": full_daily,
            "gross_daily_returns": full_gross,
            "weights_history": wh,
            "turnover_history": pd.Series(turnover_history),
            "cost_history": pd.Series(cost_history),
            "snapshot_dates": list(weights_history.keys()),
            "extra": extra,
        }


# ----------------------------------------------------------------------
# Performance metrics
# ----------------------------------------------------------------------

def summarize_performance(daily_returns: pd.Series, name: str = "strategy") -> dict:
    """Compute the headline metrics from a series of daily portfolio returns.

    Returns: dict with total_return, ann_return, ann_vol, sharpe, sortino,
    max_drawdown, calmar.
    """
    r = daily_returns.dropna()
    if r.empty:
        return {"name": name}
    cum = (1 + r).cumprod()
    total_return = float(cum.iloc[-1] - 1)
    n_days = len(r)
    ann_return = float((1 + total_return) ** (365.0 / n_days) - 1) if n_days > 0 else 0.0
    ann_vol = float(r.std(ddof=1) * np.sqrt(365)) if r.std(ddof=1) > 0 else 0.0
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    downside = r[r < 0]
    sortino = float(ann_return / (downside.std(ddof=1) * np.sqrt(365))) if (
        len(downside) > 1 and downside.std(ddof=1) > 0
    ) else 0.0
    drawdown = (cum / cum.cummax()) - 1
    max_dd = float(drawdown.min())
    calmar = float(ann_return / abs(max_dd)) if max_dd < 0 else 0.0
    return {
        "name": name,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "n_days": n_days,
    }
